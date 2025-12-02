import json
import logging
from typing import Annotated

import requests
from docker.errors import DockerException
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request
from starlette.datastructures import UploadFile as StarletteUploadFile

from cogstack_model_gateway.common.config import Config, get_config
from cogstack_model_gateway.common.models import ModelDeploymentType, ModelManager
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import TaskManager
from cogstack_model_gateway.common.tracking import TrackingClient
from cogstack_model_gateway.gateway.core.auto_deploy import ensure_model_available
from cogstack_model_gateway.gateway.core.models import get_running_models, run_model_container
from cogstack_model_gateway.gateway.core.priority import calculate_task_priority
from cogstack_model_gateway.gateway.prometheus.metrics import (
    gateway_models_deployed_total,
    gateway_tasks_processed_total,
)
from cogstack_model_gateway.gateway.routers.utils import (
    get_cms_url,
    get_content_type,
    get_query_params,
    validate_model_name,
)
from cogstack_model_gateway.gateway.schemas import ModelResponse, ModelsListResponse

DEFAULT_CONTENT_TYPE = "text/plain"
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


async def ensure_model_dependency(
    model_name: str,
    config: Annotated[Config, Depends(get_config)],
) -> None:
    """FastAPI dependency to ensure a model is available before processing request.

    This handles:
    - Checking if model is running and healthy (all deployment types)
    - Auto-creating STATIC entries for untracked models
    - Auto-deploying on-demand models if configured

    Raises:
        HTTPException(503): If model unavailable.
    """
    is_available = await ensure_model_available(
        model_name=model_name,
        config=config,
        model_manager=config.model_manager,
    )

    if not is_available:
        running = [m["service_name"] for m in get_running_models()]
        on_demand = [m.service_name for m in config.list_on_demand_models()]
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{model_name}' is not available. "
                f"Running models: {running}. "
                f"On-demand models: {on_demand}. "
                "List all models at /models"
            ),
        )


def _build_model_response(
    model_dict: dict,
    is_running: bool,
    tracking_client: TrackingClient,
    model_manager: ModelManager,
    verbose: bool = False,
) -> ModelResponse:
    """Build unified model response from model dictionary.

    Args:
        model_dict: Model data (from get_running_models() or OnDemandModel.model_dump())
        is_running: Whether the model is currently running
        tracking_client: Tracking client for accessing tracking metadata
        model_manager: Model manager for database operations
        verbose: Whether to include tracking metadata and runtime info
    """
    name = model_dict.get("service_name") or model_dict.get("name")
    uri = model_dict.get("model_uri") or model_dict.get("uri")
    minimal_response = {"name": name, "uri": uri, "is_running": is_running}

    if not verbose:
        return ModelResponse(**minimal_response)

    cms_info = None
    idle_ttl = model_dict.get("idle_ttl")
    description = model_dict.get("description")
    resources = model_dict.get("deploy", {}).get("resources")
    model_type = tracking_client.get_model_type(uri) if uri else None
    tracking_metadata = tracking_client.get_model_metadata(uri) if uri else None
    deployment_type = model_dict.get("deployment_type", ModelDeploymentType.AUTO.value)

    if is_running:
        try:
            # FIXME: Enable SSL verification when certificates are properly set up
            cms_response = requests.get(get_cms_url(name, "info"), verify=False)
            cms_response.raise_for_status()
            cms_info = cms_response.json()

            if not model_type:
                model_type = cms_info.get("model_type")
            if not description:
                description = cms_info.get("model_description")

            deployment_type = (
                model_manager.get_model_deployment_type(name) or ModelDeploymentType.STATIC.value
            )

        except requests.RequestException as e:
            log.warning(f"Failed to fetch CMS info for model '{name}': {e}")

    return ModelResponse(
        **{
            **minimal_response,
            "description": description,
            "model_type": model_type,
            "deployment_type": deployment_type,
            "idle_ttl": idle_ttl,
            "resources": resources,
            "runtime": cms_info,
            "tracking": tracking_metadata,
        }
    )


@router.get(
    "/models/",
    response_model=ModelsListResponse,
    response_model_exclude_none=True,
    tags=["models"],
    name="List running and on-demand CogStack Model Serve instances",
)
async def get_models(
    config: Annotated[Config, Depends(get_config)],
    verbose: Annotated[
        bool | None, Query(description="Include tracking metadata and runtime info")
    ] = False,
):
    """List running model servers and on-demand models that can be auto-deployed.

    Returns a dictionary with two keys:
    - 'running': List of currently running model containers
    - 'on_demand': List of models that can be deployed on-demand (excludes already running models)

    When verbose=false (default):
    - Returns minimal info: name, uri, is_running

    When verbose=true:
    - Includes description, model_type, deployment_type, idle_ttl, resources
    - Includes 'tracking': Model metadata from tracking server (e.g. uuid, run_id, signature)
    - Includes 'runtime': CMS /info response (for running models only)
    """
    running_models = get_running_models()
    running_model_names = {m["service_name"] for m in running_models}

    return ModelsListResponse(
        running=[
            _build_model_response(
                model, True, config.tracking_client, config.model_manager, verbose
            )
            for model in running_models
        ],
        on_demand=[
            _build_model_response(
                model.model_dump(), False, config.tracking_client, config.model_manager, verbose
            )
            for model in config.list_on_demand_models()
            if model.service_name not in running_model_names
        ],
    )


@router.get(
    "/models/{model_name}",
    response_model=ModelResponse,
    response_model_exclude_none=True,
    tags=["models"],
    name="Get information about a model (running or on-demand)",
)
async def get_model(
    model_name: str,
    config: Annotated[Config, Depends(get_config)],
    verbose: Annotated[
        bool | None, Query(description="Include tracking metadata and runtime info")
    ] = False,
):
    """Get information about a model (running or on-demand).

    When verbose=false (default):
    - Returns minimal info: name, uri, is_running

    When verbose=true:
    - Includes description, model_type, deployment_type, idle_ttl, resources
    - Includes 'tracking': Model metadata from tracking server (e.g. uuid, run_id, signature)
    - Includes 'runtime': CMS /info response (for running models only)

    Does not trigger auto-deployment for on-demand models.
    """
    running_models = {m["service_name"]: m for m in get_running_models()}
    on_demand_models = {m.service_name: m for m in config.list_on_demand_models()}

    if model_name in running_models:
        return _build_model_response(
            running_models[model_name], True, config.tracking_client, config.model_manager, verbose
        )
    elif model_name in on_demand_models:
        return _build_model_response(
            on_demand_models[model_name].model_dump(),
            False,
            config.tracking_client,
            config.model_manager,
            verbose,
        )
    else:
        running_models_names = list(running_models.keys())
        on_demand_models_names = [m for m in on_demand_models.keys() if m not in running_models]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' not found:"
                f" * Running models: {running_models_names}."
                f" * On-demand models: {on_demand_models_names}."
                "You can list all available models at /models."
            ),
        )


@router.get(
    "/models/{model_name}/info",
    response_model=dict,
    tags=["models"],
    name="Get information about a running CogStack Model Serve instance",
    dependencies=[Depends(ensure_model_dependency)],
)
async def get_model_info(model_name: str, config: Annotated[Config, Depends(get_config)]):
    """Get information about a running model server through its `/info` API."""
    gateway_tasks_processed_total.labels(model=model_name, task="info").inc()
    # FIXME: Enable SSL verification when certificates are properly set up
    response = requests.get(get_cms_url(model_name, "info"), verify=False)
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. You can list all available models at /models",
        )

    try:
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

    model_manager: ModelManager = config.model_manager
    model_manager.record_model_usage(model_name)
    return response.json()


@router.post(
    "/models/{model_name}",
    response_model=dict,
    tags=["models"],
    name="Deploy a CogStack Model Serve instance with a given model URI or tracking ID",
)
async def deploy_model(
    config: Annotated[Config, Depends(get_config)],
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
        Body(description="The URI of the model to serve (optional if tracking_id is provided)"),
    ] = None,
    ttl: Annotated[
        int | None,
        Body(
            description=(
                "The deployed model will be deleted after TTL seconds."
                " Set -1 as the TTL value to protect the model from being deleted."
                " If not provided, uses the default from manual deployment config."
            )
        ),
    ] = None,
):
    """Deploy a CogStack Model Serve instance with a given model URI or tracking ID.

    The model URI refers to the location of the model artifact to be served, which can be found on
    the tracking server (e.g. MLflow). The tracking ID, on the other hand, refers to the ID of the
    run that generated the model artifact (e.g. MLflow run for model training) and is only used to
    fetch the model URI if not provided explicitly. The model is deployed as a Docker container
    with the specified name and the CogStack Model Serve image. The container is labelled with the
    model URI, the project name, and the TTL value to determine its expiration time.

    A corresponding Model database entry is created to track usage and enable lifecycle management.
    """
    tc: TrackingClient = config.tracking_client
    manual_config = config.get_manual_deployment_config()

    if not tracking_id and not model_uri:
        raise HTTPException(
            status_code=400, detail="At least one of tracking_id or model_uri must be provided."
        )

    if not model_uri and tracking_id:
        model_uri = tc.get_model_uri(tracking_id)
        if not model_uri:
            raise HTTPException(
                status_code=404, detail=f"Model not found for tracking ID '{tracking_id}'."
            )

    if manual_config.require_model_uri_validation and model_uri:
        model_metadata = tc.get_model_metadata(model_uri)
        if model_metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model URI '{model_uri}' could not be validated. Model may not exist.",
            )
        log.debug(f"Validated model URI '{model_uri}': {model_metadata}")

    if any(model["service_name"] == model_name for model in get_running_models()):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model '{model_name}' is already running, please choose a different name."
                " You can list all available models at /models"
            ),
        )

    if ttl is None:
        ttl = manual_config.default_ttl
    elif not manual_config.allow_ttl_override:
        raise HTTPException(
            status_code=403,
            detail="TTL override is not allowed. Remove ttl parameter or contact administrator.",
        )
    elif manual_config.max_ttl is not None and ttl > manual_config.max_ttl and ttl != -1:
        raise HTTPException(
            status_code=400,
            detail=f"TTL exceeds maximum allowed value of {manual_config.max_ttl} seconds.",
        )

    try:
        container = run_model_container(
            model_name=model_name,
            model_uri=model_uri,
            # FIXME: add model type
            model_type="medcat_umls",
            deployment_type=ModelDeploymentType.MANUAL,
            ttl=ttl,
            resources=None,  # TODO: Add resource limits support for manual deployments
        )

        model_manager: ModelManager = config.model_manager
        model_manager.create_model(
            model_name=model_name, deployment_type=ModelDeploymentType.MANUAL
        )

    except DockerException as e:
        log.error(f"Failed to deploy model '{model_name}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to deploy model '{model_name}': {str(e)}"
        )

    gateway_models_deployed_total.labels(model=model_name, model_uri=model_uri).inc()

    log.info(f"Model '{model_name}' deployed successfully with container ID {container.id}")
    return {
        "message": f"Model '{model_name}' deployed successfully",
        "model_uri": model_uri,
        "container_id": container.id,
        "container_name": container.name,
        "ttl": ttl,
    }


@router.post(
    "/models/{model_name}/tasks/{task}",
    response_model=dict,
    tags=["models"],
    name="Schedule a task for execution on a running CogStack Model Serve instance",
    dependencies=[Depends(ensure_model_dependency)],
)
async def execute_task(
    model_name: str,
    task: str,
    request: Request,
    content_type: Annotated[str, Header()],
    parsed_content_type: Annotated[str, Depends(get_content_type)],
    query_params: Annotated[dict[str, str], Depends(get_query_params)],
    config: Annotated[Config, Depends(get_config)],
):
    """Schedule a task for execution on a running model server.

    The task is assigned a tracking ID used throughout the stack and is scheduled by publishing a
    message to the task queue with the task details (e.g. method, URL, payload, content type). This
    process varies depending on the content type of the original request. Payloads in the form of
    textual data or JSON are serialized and uploaded to the object store, while a reference to the
    uploaded object is included in the task details. For multipart requests, each part is processed
    separately; files are uploaded to the object store, while other fields are included as-is in the
    task details.
    """
    endpoint = SUPPORTED_ENDPOINTS.get(task)
    if not endpoint:
        supported_endpoints_str = ", ".join(SUPPORTED_ENDPOINTS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task}' not found. Supported tasks are: {supported_endpoints_str}",
        )

    if (
        parsed_content_type != endpoint["content_type"]
        and parsed_content_type != DEFAULT_CONTENT_TYPE
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: expected {endpoint['content_type']}",
        )

    references = []
    osm: ObjectStoreManager = config.task_object_store_manager

    tm: TaskManager = config.task_manager

    client_ip = request.client.host if request.client else "N/A"
    user_agent = request.headers.get("user-agent", "N/A")
    source = f"ip={client_ip}; ua={user_agent}"

    submitted_task = tm.create_task(model=model_name, type=task, source=source)
    task_uuid = submitted_task.uuid

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

    task_dict = {
        "uuid": task_uuid,
        "method": endpoint["method"],
        "url": get_cms_url(model_name, endpoint["url"]),
        "content_type": content_type,
        "params": query_params,
        "refs": references,
    }
    priority = calculate_task_priority(task, config)

    log.info(f"Executing task '{task_dict['uuid']}': {task_dict['method']} {task_dict['url']}")
    log.debug(f"Task details: {task_dict}")
    qm: QueueManager = config.queue_manager
    qm.publish(task_dict, priority)

    model_manager: ModelManager = config.model_manager
    model_manager.record_model_usage(model_name)
    gateway_tasks_processed_total.labels(model=model_name, task=task).inc()

    return {"uuid": task_uuid, "status": "Task submitted successfully"}
