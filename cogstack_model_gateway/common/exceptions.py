import logging

from requests import HTTPError
from requests.exceptions import ConnectionError, Timeout
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
    wait_random_exponential,
)

log = logging.getLogger("cmg.common")


def is_rate_limited(exception: Exception):
    """Check if the exception is a rate limit error."""
    return (
        isinstance(exception, HTTPError)
        and exception.response is not None
        and exception.response.status_code == 429  # Too Many Requests
    )


def is_connection_error(exception: Exception):
    """Check if the exception is a connection or a bad gateway error."""
    return (
        exception.response.status_code == 502  # Bad Gateway
        if isinstance(exception, HTTPError) and exception.response is not None
        else isinstance(exception, ConnectionError)
    )


def is_timeout_error(exception: Exception):
    """Check if the exception is a timeout, request timeout, or gateway timeout error."""
    return (
        exception.response.status_code in (408, 504)  # Request Timeout, Gateway Timeout
        if isinstance(exception, HTTPError) and exception.response is not None
        else isinstance(exception, Timeout)
    )


retry_if_rate_limited = retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(min=5, max=60),
    retry=retry_if_exception(is_rate_limited),
    before_sleep=before_sleep_log(log, logging.WARNING),
)

retry_if_connection_error = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception(is_connection_error),
    before_sleep=before_sleep_log(log, logging.WARNING),
)

retry_if_timeout_error = retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(5),
    retry=retry_if_exception(is_timeout_error),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
