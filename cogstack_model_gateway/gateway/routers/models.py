import json
import logging
from typing import Annotated

import requests
from docker.errors import DockerException
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile

from cogstack_model_gateway.common.config import Config, get_config
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, TaskManager
from cogstack_model_gateway.common.tracking import TrackingClient
from cogstack_model_gateway.gateway.core.models import (
    get_model_meta,
    get_running_models,
    run_model_container,
)
from cogstack_model_gateway.gateway.core.priority import calculate_task_priority
from cogstack_model_gateway.gateway.routers.utils import (
    get_content_type,
    get_query_params,
    validate_model_name,
)

SUPPORTED_ENDPOINTS = {
    "info": {"method": "GET", "url": "/info", "content_type": "application/json"},
    "process": {"method": "POST", "url": "/process", "content_type": "text/plain"},
    "process_jsonl": {
        "method": "POST",
        "url": "/process_jsonl",
        "content_type": "application/x-ndjson",
    },
    "process_bulk": {"method": "POST", "url": "/process_bulk", "content_type": "application/json"},
    "process_bulk_file": {
        "method": "POST",
        "url": "/process_bulk_file",
        "content_type": "multipart/form-data",
    },
    "redact": {"method": "POST", "url": "/redact", "content_type": "text/plain"},
    "redact_with_encryption": {
        "method": "POST",
        "url": "/redact_with_encryption",
        "content_type": "application/json",
    },
    "preview": {"method": "POST", "url": "/preview", "content_type": "text/plain"},
    "preview_trainer_export": {
        "method": "POST",
        "url": "/preview_trainer_export",
        "content_type": "multipart/form-data",
    },
    "train_supervised": {
        "method": "POST",
        "url": "/train_supervised",
        "content_type": "multipart/form-data",
        "extra_params": {"tracking_id"},
    },
    "train_unsupervised": {
        "method": "POST",
        "url": "/train_unsupervised",
        "content_type": "multipart/form-data",
        "extra_params": {"tracking_id"},
    },
    "train_unsupervised_with_hf_hub_dataset": {
        "method": "POST",
        "url": "/train_unsupervised_with_hf_hub_dataset",
        "content_type": "multipart/form-data",
        "extra_params": {"tracking_id"},
    },
    "train_metacat": {
        "method": "POST",
        "url": "/train_metacat",
        "content_type": "multipart/form-data",
        "extra_params": {"tracking_id"},
    },
    "evaluate": {
        "method": "POST",
        "url": "/evaluate",
        "content_type": "multipart/form-data",
        "extra_params": {"tracking_id"},
    },
    "sanity-check": {
        "method": "POST",
        "url": "/sanity-check",
        "content_type": "multipart/form-data",
    },
    "iaa-scores": {"method": "POST", "url": "/iaa-scores", "content_type": "multipart/form-data"},
    "concat_trainer_exports": {
        "method": "POST",
        "url": "/concat_trainer_exports",
        "content_type": "multipart/form-data",
    },
    "annotation-stats": {
        "method": "POST",
        "url": "/annotation-stats",
        "content_type": "multipart/form-data",
    },
}

log = logging.getLogger("cmg.gateway")
router = APIRouter()


@router.get("/models/", response_model=list[dict], tags=["models"])
async def get_models():
    models = get_running_models()
    for model in models:
        if model["uri"]:
            if model_info := get_model_meta(model["uri"]):
                model["info"] = model_info
    return models


@router.get("/models/{model_name}/info", response_model=dict, tags=["models"])
async def get_model_info(model_name: str):
    response = requests.get(f"http://{model_name}:8000/info")
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. You can list all available models at /models",
        )

    try:
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response.json()


@router.post("/models/{model_name}", response_model=dict, tags=["models"])
async def deploy_model(
    model_name: Annotated[str, Depends(validate_model_name)],
    tracking_id: Annotated[
        str | None,
        Body(
            description=(
                "The tracking ID of the run that generated the model to serve (e.g. MLflow run ID),"
                " used to fetch the model URI (optional if model_uri is provided explicitly)"
            )
        ),
    ] = None,
    model_uri: Annotated[
        str | None,
        Body(description="The URI of the model to serve (optional if run_id is provided)"),
    ] = None,
    ttl: Annotated[
        int | None,
        Body(
            description=(
                "The deployed model will be deleted after TTL seconds (defaults to 86400, i.e. 1d)."
                " Set -1 as the TTL value to protect the model from being deleted."
            )
        ),
    ] = 86400,
):
    if not tracking_id and not model_uri:
        raise HTTPException(
            status_code=400, detail="At least one of tracking_id or model_uri must be provided."
        )

    if not model_uri and tracking_id:
        tc = TrackingClient()
        model_uri = tc.get_model_uri(tracking_id)
        if not model_uri:
            raise HTTPException(
                status_code=404, detail=f"Model not found for tracking ID '{tracking_id}'."
            )

    if any(model["name"] == model_name for model in get_running_models()):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model '{model_name}' is already running, please choose a different name."
                " You can list all available models at /models"
            ),
        )

    try:
        container = run_model_container(model_name, model_uri, ttl)
    except DockerException as e:
        log.error(f"Failed to deploy model '{model_name}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to deploy model '{model_name}': {str(e)}"
        )

    log.info(f"Model '{model_name}' deployed successfully with container ID {container.id}")
    return {
        "message": f"Model '{model_name}' deployed successfully",
        "model_uri": model_uri,
        "container_id": container.id,
        "container_name": container.name,
        "ttl": ttl,
    }


@router.post("/models/{model_name}/tasks/{task}", response_model=dict, tags=["models"])
async def execute_task(
    model_name: str,
    task: str,
    request: Request,
    content_type: Annotated[str, Header()],
    parsed_content_type: Annotated[str, Depends(get_content_type)],
    query_params: Annotated[dict[str, str], Depends(get_query_params)],
    config: Annotated[Config, Depends(get_config)],
):
    endpoint = SUPPORTED_ENDPOINTS.get(task)
    if not endpoint:
        supported_endpoints_str = ", ".join(SUPPORTED_ENDPOINTS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task}' not found. Supported tasks are: {supported_endpoints_str}",
        )

    if parsed_content_type != endpoint["content_type"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: expected {endpoint['content_type']}",
        )

    references = []
    osm: ObjectStoreManager = config.task_object_store_manager

    tm: TaskManager = config.task_manager
    task_uuid = tm.create_task(Status.PENDING)

    # FIXME: Extract task metadata (e.g. type, payload size) for priority calculation
    if parsed_content_type in ("text/plain", "application/x-ndjson"):
        payload = await request.body()
        file_extension = "txt" if parsed_content_type == "text/plain" else "ndjson"
        object_key = osm.upload_object(payload, f"payload.{file_extension}", prefix=task_uuid)
        references.append({"key": object_key, "content_type": content_type})

    elif parsed_content_type == "application/json":
        payload = await request.json()
        object_key = osm.upload_object(
            json.dumps(payload).encode(), "payload.json", prefix=task_uuid
        )
        references.append({"key": object_key, "content_type": content_type})

    elif parsed_content_type == "multipart/form-data":
        form = await request.form()
        for field, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                object_key = osm.upload_object(await value.read(), value.filename, prefix=task_uuid)
                references.append(
                    {
                        "field": field,
                        "key": object_key,
                        "filename": value.filename,
                        "content_type": f"{content_type}; part=file",
                    }
                )
            else:
                # FIXME: This field might still hold the raw contents of a file
                references.append(
                    {"field": field, "value": value, "content_type": f"{content_type}; part=field"}
                )

    if "extra_params" in endpoint and "tracking_id" in endpoint["extra_params"]:
        query_params["tracking_id"] = task_uuid

    task = {
        "uuid": task_uuid,
        "method": endpoint["method"],
        "url": f"http://{model_name}:8000{endpoint['url']}",
        "content_type": content_type,
        "params": query_params,
        "refs": references,
    }
    priority = calculate_task_priority(task, config)

    log.info(f"Executing task: {task}")
    qm: QueueManager = config.queue_manager
    qm.publish(task, priority)

    return {"uuid": task_uuid, "status": "Task submitted successfully"}
