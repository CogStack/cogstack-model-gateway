import re
from typing import Annotated

from fastapi import Header, HTTPException, Path, Request

from cogstack_model_gateway.common.utils import parse_content_type_header

MODEL_NAME_REGEX = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]+$"
VALID_MODEL_DESCRIPTION = (
    "The model name must start with an alphanumeric character and can only contain"
    " alphanumeric characters, underscores (_), dots (.), and dashes (-)."
)


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
