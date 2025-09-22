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


class TaskFailedError(Exception):
    """Raised when a Gateway task fails during execution.

    Attributes:
        task_uuid: The UUID of the failed task
        error_message: The error message from the task
        task_info: The full task information dict (optional)
    """

    def __init__(self, task_uuid: str, error_message: str, task_info: dict = None):
        self.task_uuid = task_uuid
        self.error_message = error_message
        self.task_info = task_info
        super().__init__(f"Task '{task_uuid}' failed: {error_message}")


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
