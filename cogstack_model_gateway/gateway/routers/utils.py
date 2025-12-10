import logging
import re
from typing import Annotated

from fastapi import Header, HTTPException, Path, Request

from cogstack_model_gateway.common.config import get_config
from cogstack_model_gateway.common.containers import get_models
from cogstack_model_gateway.common.tracking import TrackingClient
from cogstack_model_gateway.common.utils import parse_content_type_header

MODEL_NAME_REGEX = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]+$"
VALID_MODEL_DESCRIPTION = (
    "The model name must start with an alphanumeric character and can only contain"
    " alphanumeric characters, underscores (_), dots (.), and dashes (-)."
)

log = logging.getLogger("cmg.gateway")


def get_query_params(request: Request) -> dict[str, str]:
    """Get query parameters from a request."""
    return dict(request.query_params)


def get_content_type(content_type: Annotated[str, Header()]) -> str:
    """Get the content type from the request headers."""
    parsed_content_type, _ = parse_content_type_header(content_type)
    return parsed_content_type


def validate_model_name(
    model_name: Annotated[
        str,
        Path(
            ...,
            description=(
                "The name of the model to deploy (used as the container name)."
                f" {VALID_MODEL_DESCRIPTION}"
            ),
        ),
    ],
) -> str:
    """Validate the model name path parameter."""
    if not re.match(MODEL_NAME_REGEX, model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. {VALID_MODEL_DESCRIPTION}",
        )
    return model_name


def get_cms_url(model_name: str, endpoint: str = None) -> str:
    """Get the URL of a CogStack Model Serve instance endpoint."""
    config = get_config()
    host_url = config.cms.host_url.rstrip("/") if config.cms.host_url else ""
    server_port = config.cms.server_port
    base_url = f"{host_url}/{model_name}" if host_url else f"http://{model_name}:{server_port}"
    endpoint = endpoint.lstrip("/") if endpoint else ""
    return f"{base_url}/{endpoint}" if endpoint else base_url


def resolve_model_host(model_name: str) -> str:
    """Resolve model name to IP address if needed for network communication."""
    if not get_config().models.deployment.use_ip_addresses:
        log.debug("Using model name '%s' for network communication", model_name)
        return model_name

    log.debug("Attempting to resolve model '%s' to IP address", model_name)
    models = get_models(all=False, managed_only=False)
    model_data = next(
        (m for m in models if m["service_name"] == model_name or m["ip_address"] == model_name),
        None,
    )

    if model_data and model_data.get("ip_address"):
        log.info("Resolved model '%s' to IP '%s'", model_name, model_data["ip_address"])
        return model_data["ip_address"]

    log.warning("Unable to resolve name to IP for model '%s', using name as fallback", model_name)
    return model_name


def resolve_and_validate_model_uri(
    tracking_id: str | None,
    model_uri: str | None,
    tracking_client: TrackingClient,
    require_validation: bool,
) -> tuple[str, str | None]:
    """Resolve tracking_id to model_uri and optionally validate.

    Args:
        tracking_id: Optional tracking ID to resolve
        model_uri: Optional explicit model URI
        tracking_client: Tracking client for resolution/validation
        require_validation: Whether to validate the model URI exists

    Returns:
        Tuple of (resolved_model_uri, tracking_id)

    Raises:
        HTTPException(400): If neither `tracking_id` nor `model_uri` provided
        HTTPException(404): If `tracking_id` doesn't resolve or validation fails
    """
    if not tracking_id and not model_uri:
        raise HTTPException(
            status_code=400, detail="At least one of `tracking_id` or `model_uri` must be provided."
        )

    if not model_uri and tracking_id:
        model_uri = tracking_client.get_model_uri(tracking_id)
        if not model_uri:
            raise HTTPException(
                status_code=404, detail=f"Model not found for tracking ID '{tracking_id}'."
            )

    if require_validation and model_uri:
        model_metadata = tracking_client.get_model_metadata(model_uri)
        if model_metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model URI '{model_uri}' could not be validated. Model may not exist.",
            )
        log.debug(f"Validated model URI '{model_uri}': {model_metadata}")

    return model_uri, tracking_id
