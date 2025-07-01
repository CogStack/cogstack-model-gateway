import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
)

log = logging.getLogger("cmg.client")


def is_network_error(exception: Exception):
    """Check if the exception is a network-related error."""
    return isinstance(
        exception,
        httpx.RemoteProtocolError
        | httpx.ConnectError
        | httpx.TimeoutException
        | httpx.NetworkError,
    )


retry_if_network_error = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception(is_network_error),
    before_sleep=before_sleep_log(log, logging.DEBUG),
)
