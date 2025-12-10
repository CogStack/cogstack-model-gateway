import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from cogstack_model_gateway.common.config import Config, get_config
from cogstack_model_gateway.common.containers import get_models
from cogstack_model_gateway.common.exceptions import ConfigConflictError, ConfigValidationError
from cogstack_model_gateway.common.models import ModelManager, OnDemandModelConfig
from cogstack_model_gateway.gateway.routers.utils import resolve_and_validate_model_uri
from cogstack_model_gateway.gateway.schemas import (
    OnDemandModelCreate,
    OnDemandModelListResponse,
    OnDemandModelUpdate,
)

log = logging.getLogger("cmg.gateway.admin")
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/on-demand",
    response_model=OnDemandModelListResponse,
    response_model_exclude_none=True,
    name="List on-demand model configurations",
)
async def list_on_demand_configs(
    config: Annotated[Config, Depends(get_config)],
    include_disabled: Annotated[
        bool, Query(description="Include disabled (soft-deleted) configurations")
    ] = False,
):
    """List all on-demand model configurations.

    By default, only enabled configurations are returned. Use `include_disabled=true`
    to also include historical/disabled configurations.
    """
    model_manager: ModelManager = config.model_manager
    model_configs = model_manager.list_on_demand_configs(include_disabled=include_disabled)

    return OnDemandModelListResponse(configs=model_configs, total=len(model_configs))


@router.post(
    "/on-demand",
    response_model=OnDemandModelConfig,
    response_model_exclude_none=True,
    status_code=201,
    name="Create on-demand model configuration",
)
async def create_on_demand_config(
    body: OnDemandModelCreate, config: Annotated[Config, Depends(get_config)]
):
    """Create a new on-demand model configuration.

    The model will be available for auto-deployment when requests target its `model_name`.
    Only one **enabled** configuration can exist per `model_name` at a time.

    You can specify the model using either `tracking_id` (e.g. MLflow run ID) or `model_uri`
    (direct artifact URI), or both. While `model_uri` takes precedence, if only `tracking_id` is
    provided, it will be resolved to a `model_uri`. If `require_model_uri_validation` is set to true
    in the config, the resolved or explicit URI will be validated against the tracking server.

    Set `replace_enabled=true` to atomically disable any existing config and create the new one,
    preserving the old config in history for potential rollback and replacing the original as the
    enabled config. If `replace_enabled=false` and an enabled config already exists for the same
    `model_name`, a `409 Conflict` error is returned.

    Set `inherit_config=true` to copy settings from the currently enabled config for this model.
    If no enabled config exists, a new config will be created as long as the mandatory fields are
    provided (i.e. `model_uri` or `tracking_id`). Any explicitly provided fields will override the
    inherited values. When creating a config from scratch is desired, it is recommended to set
    `inherit_config=false` to avoid confusion and explicitly provide configuration fields as needed.
    """
    model_manager: ModelManager = config.model_manager

    # Resolve and validate if any model field is provided to catch errors with provided values
    # Let ModelManager catch missing model reference if not able to inherit
    if body.tracking_id is not None or body.model_uri is not None:
        model_uri, tracking_id = resolve_and_validate_model_uri(
            tracking_id=body.tracking_id,
            model_uri=body.model_uri,
            tracking_client=config.tracking_client,
            require_validation=config.models.deployment.auto.require_model_uri_validation,
        )
    else:
        model_uri, tracking_id = None, None

    try:
        model_config = model_manager.create_on_demand_config(
            model_name=body.model_name,
            model_uri=model_uri,
            tracking_id=tracking_id,
            idle_ttl=body.idle_ttl,
            description=body.description,
            deploy_spec=body.deploy.model_dump(exclude_none=True) if body.deploy else None,
            replace_enabled=body.replace_enabled,
            inherit_config=body.inherit_config,
        )
    except ConfigValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConfigConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))

    log.info("Created on-demand config via API: %s", body.model_name)
    return model_config


@router.get(
    "/on-demand/{model_name}",
    response_model=OnDemandModelConfig,
    response_model_exclude_none=True,
    name="Get on-demand model configuration",
)
async def get_on_demand_config(
    model_name: str,
    config: Annotated[Config, Depends(get_config)],
):
    """Get the enabled on-demand model configuration for a specific model name."""
    model_manager: ModelManager = config.model_manager
    model_config = model_manager.get_on_demand_config(model_name)

    if not model_config:
        raise HTTPException(
            status_code=404, detail=f"No enabled on-demand config found for '{model_name}'"
        )

    return model_config


@router.get(
    "/on-demand/{model_name}/history",
    response_model=OnDemandModelListResponse,
    response_model_exclude_none=True,
    name="Get on-demand model configuration history",
)
async def get_on_demand_config_history(
    model_name: str, config: Annotated[Config, Depends(get_config)]
):
    """Get all versions (enabled and disabled) of an on-demand model configuration.

    Returns configurations ordered by creation date, newest first.
    """
    model_manager: ModelManager = config.model_manager
    model_configs = model_manager.get_on_demand_config_history(model_name)

    if not model_configs:
        raise HTTPException(
            status_code=404, detail=f"No on-demand configurations found for '{model_name}'"
        )

    return OnDemandModelListResponse(configs=model_configs, total=len(model_configs))


@router.put(
    "/on-demand/{model_name}",
    response_model=OnDemandModelConfig,
    response_model_exclude_none=True,
    name="Update on-demand model configuration",
)
async def update_on_demand_config(
    model_name: str, body: OnDemandModelUpdate, config: Annotated[Config, Depends(get_config)]
):
    """Update an existing on-demand model configuration in-place without creating a new version.

    Only provided fields will be updated. Use `clear_*` flags to explicitly unset optional fields.
    This does NOT create version history. To preserve history, use POST with `replace_enabled=true`.

    You can update the model reference using `tracking_id`, `model_uri`, or both. While `model_uri`
    takes precedence, if only `tracking_id` is provided, it will be resolved to a `model_uri`. If
    `require_model_uri_validation` is set to true in the config, the resolved or explicit URI will
    be validated against the tracking server. Note that updating `tracking_id` results in updating
    `model_uri` as well if the latter is not explicitly provided.

    Note: If a model is currently running, changes will only take effect on the next deployment.
    """
    model_manager: ModelManager = config.model_manager

    if body.tracking_id is not None or body.model_uri is not None:
        model_uri, tracking_id = resolve_and_validate_model_uri(
            tracking_id=body.tracking_id,
            model_uri=body.model_uri,
            tracking_client=config.tracking_client,
            require_validation=config.models.deployment.auto.require_model_uri_validation,
        )
    else:
        model_uri, tracking_id = None, None

    deploy_spec_dict = body.deploy.model_dump(exclude_none=True) if body.deploy else None

    updated_config = model_manager.update_on_demand_config(
        model_name=model_name,
        model_uri=model_uri,
        tracking_id=tracking_id,
        idle_ttl=body.idle_ttl,
        description=body.description,
        deploy_spec=deploy_spec_dict,
        clear_tracking_id=body.clear_tracking_id,
        clear_idle_ttl=body.clear_idle_ttl,
        clear_description=body.clear_description,
        clear_deploy_spec=body.clear_deploy,
    )

    if not updated_config:
        raise HTTPException(
            status_code=404, detail=f"No enabled on-demand config found for '{model_name}'"
        )

    models = get_models(all=False, managed_only=False)
    if model_name in {m["service_name"] for m in models} | {m["ip_address"] for m in models}:
        log.warning(
            "Updated on-demand config for '%s' which is currently running. Changes will take effect"
            " on next deployment.",
            model_name,
        )

    log.info("Updated on-demand config: %s", model_name)
    return updated_config


@router.delete(
    "/on-demand/{model_name}",
    status_code=204,
    name="Delete on-demand model configuration",
)
async def delete_on_demand_config(model_name: str, config: Annotated[Config, Depends(get_config)]):
    """Soft-delete an on-demand model configuration by disabling it.

    The configuration is not permanently deleted; it can be viewed via the
    `/on-demand/{model_name}/history` endpoint and re-enabled if needed.

    **Note**: This does NOT stop a currently running instance of the model. The running model will
        continue until it's stopped or its idle TTL expires.
    """
    model_manager: ModelManager = config.model_manager
    success = model_manager.disable_on_demand_config(model_name)

    if not success:
        raise HTTPException(
            status_code=404, detail=f"No enabled on-demand config found for '{model_name}'"
        )

    models = get_models(all=False, managed_only=False)
    if model_name in {m["service_name"] for m in models} | {m["ip_address"] for m in models}:
        log.warning(
            "Disabled on-demand config for '%s' which is currently running. The running instance"
            " will not be affected.",
            model_name,
        )

    log.info("Disabled on-demand config: %s", model_name)
    return None


@router.post(
    "/on-demand/{config_id}/enable",
    response_model=OnDemandModelConfig,
    response_model_exclude_none=True,
    name="Enable an on-demand model configuration",
)
async def enable_on_demand_config(config_id: int, config: Annotated[Config, Depends(get_config)]):
    """Enable an on-demand model configuration by its ID.

    This will disable any currently enabled configuration for the same `model_name` and enable the
    specified configuration. Useful for rolling back to a previous model version.
    """
    model_manager: ModelManager = config.model_manager
    enabled_config = model_manager.enable_on_demand_config(config_id)

    if enabled_config is None:
        raise HTTPException(status_code=404, detail=f"On-demand config '{config_id}' not found")

    log.info("Enabled on-demand config: %s (id=%d)", enabled_config.model_name, config_id)
    return enabled_config
