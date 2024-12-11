import json
import logging
from typing import Annotated

import requests
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from starlette.datastructures import UploadFile as StarletteUploadFile

from cogstack_model_gateway.common.config import Config, get_config
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, TaskManager
from cogstack_model_gateway.gateway.core.models import get_model_meta, get_running_models
from cogstack_model_gateway.gateway.core.priority import calculate_task_priority
from cogstack_model_gateway.gateway.routers.utils import get_content_type, get_query_params

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
    },
    "train_unsupervised": {
        "method": "POST",
        "url": "/train_unsupervised",
        "content_type": "multipart/form-data",
    },
    "train_unsupervised_with_hf_hub_dataset": {
        "method": "POST",
        "url": "/train_unsupervised_with_hf_hub_dataset",
        "content_type": "multipart/form-data",
    },
    "train_metacat": {
        "method": "POST",
        "url": "/train_metacat",
        "content_type": "multipart/form-data",
    },
    "evaluate": {"method": "POST", "url": "/evaluate", "content_type": "multipart/form-data"},
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


@router.post("/models/{model_name}/{task}", response_model=dict, tags=["models"])
async def execute_task(
    model_name: str,
    task: str,
    request: Request,
    # content_type: str = Depends(get_content_type),
    content_type: Annotated[str, Depends(get_content_type)],
    # query_params: dict[str, str] = Depends(get_query_params),
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

    if content_type != endpoint["content_type"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: expected {endpoint['content_type']}",
        )

    references = []
    osm: ObjectStoreManager = config.object_store_manager

    if content_type in ("text/plain", "application/x-ndjson"):
        payload = await request.body()
        file_extension = "txt" if content_type == "text/plain" else "ndjson"
        object_key = await osm.upload_object(payload, f"payload.{file_extension}")
        references.append({"key": object_key, "content_type": content_type})

    elif content_type == "application/json":
        payload = await request.json()
        object_key = await osm.upload_object(json.dumps(payload).encode(), "payload.json")
        references.append({"key": object_key, "content_type": content_type})

    elif content_type == "multipart/form-data":
        form = await request.form()
        for field, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                object_key = osm.upload_object(await value.read(), value.filename)
                references.append(
                    {
                        "field": field,
                        "key": object_key,
                        "filename": value.filename,
                        "content_type": f"{content_type}; file",
                    }
                )
            else:
                # FIXME: This field might still hold the raw contents of a file
                references.append(
                    {"field": field, "value": value, "content_type": f"{content_type}; field"}
                )

    priority = calculate_task_priority(task, config)

    tm: TaskManager = config.task_manager
    task_uuid = tm.create_task(Status.PENDING, priority)
    task = {
        "uuid": task_uuid,
        "method": endpoint["method"],
        "url": f"http://{model_name}:8000{endpoint['url']}",
        "content_type": content_type,
        "params": query_params,
        "refs": references,
    }

    log.info(f"Executing task: {task}")
    qm: QueueManager = config.queue_manager
    qm.publish(task, priority)

    return {"uuid": task_uuid, "status": "Task submitted successfully"}
