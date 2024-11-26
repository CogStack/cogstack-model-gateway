from typing import Annotated

from fastapi import Header, Request

from cogstack_model_gateway.common.utils import parse_content_type_header


def get_query_params(request: Request) -> dict[str, str]:
    return dict(request.query_params)


def get_content_type(content_type: Annotated[str, Header()]) -> str:
    parsed_content_type, _ = parse_content_type_header(content_type)
    return parsed_content_type
