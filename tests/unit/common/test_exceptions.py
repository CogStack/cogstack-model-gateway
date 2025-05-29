from unittest.mock import MagicMock

import pytest
from requests import HTTPError, Response
from requests.exceptions import ConnectionError, Timeout
from tenacity import RetryError

from cogstack_model_gateway.common.exceptions import (
    is_connection_error,
    is_rate_limited,
    is_timeout_error,
    retry_if_connection_error,
    retry_if_rate_limited,
    retry_if_timeout_error,
)


def test_is_rate_limited():
    mock_response_429 = MagicMock(spec=Response, status_code=429)
    error_429 = HTTPError(response=mock_response_429)
    assert is_rate_limited(error_429)

    mock_response_200 = MagicMock(spec=Response, status_code=200)
    error_200 = HTTPError(response=mock_response_200)
    assert not is_rate_limited(error_200)

    other_exception = ValueError("Something else")
    assert not is_rate_limited(other_exception)


def test_is_connection_error():
    connection_error = ConnectionError("Connection failed")
    assert is_connection_error(connection_error)

    mock_response_502 = MagicMock(spec=Response, status_code=502)
    error_502 = HTTPError(response=mock_response_502)
    assert is_connection_error(error_502)

    mock_response_200 = MagicMock(spec=Response, status_code=200)
    error_200 = HTTPError(response=mock_response_200)
    assert not is_connection_error(error_200)

    timeout_error = Timeout("Request timed out")
    assert not is_connection_error(timeout_error)


def test_is_timeout_error():
    timeout_error = Timeout("Request timed out")
    assert is_timeout_error(timeout_error)

    mock_response_408 = MagicMock(spec=Response, status_code=408)
    error_408 = HTTPError(response=mock_response_408)
    assert is_timeout_error(error_408)

    mock_response_504 = MagicMock(spec=Response, status_code=504)
    error_504 = HTTPError(response=mock_response_504)
    assert is_timeout_error(error_504)

    mock_response_200 = MagicMock(spec=Response, status_code=200)
    error_200 = HTTPError(response=mock_response_200)
    assert not is_timeout_error(error_200)

    connection_error = ConnectionError("Connection failed")
    assert not is_timeout_error(connection_error)


def test_retry_if_rate_limited():
    mock_func = MagicMock(
        side_effect=[
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            "Success",
        ]
    )

    wrapped_func = retry_if_rate_limited(mock_func)
    assert wrapped_func() == "Success"
    assert mock_func.call_count == 3

    mock_func_fail = MagicMock(
        side_effect=HTTPError(response=MagicMock(spec=Response, status_code=429))
    )
    wrapped_func_fail = retry_if_rate_limited(mock_func_fail)
    with pytest.raises(RetryError):
        wrapped_func_fail()
    assert mock_func_fail.call_count == 10  # Stops after 10 attempts

    mock_func_no_retry = MagicMock(side_effect=ValueError("Something else"))
    wrapped_func_no_retry = retry_if_rate_limited(mock_func_no_retry)
    with pytest.raises(ValueError):
        wrapped_func_no_retry()
    assert mock_func_no_retry.call_count == 1


def test_retry_if_connection_error():
    mock_func = MagicMock(
        side_effect=[
            ConnectionError("Failed"),
            HTTPError(response=MagicMock(spec=Response, status_code=502)),
            "Success",
        ]
    )

    wrapped_func = retry_if_connection_error(mock_func)
    assert wrapped_func() == "Success"
    assert mock_func.call_count == 3

    mock_func_fail = MagicMock(
        side_effect=[
            ConnectionError("Failed"),
            HTTPError(response=MagicMock(spec=Response, status_code=502)),
            ConnectionError("Failed"),
        ]
    )
    wrapped_func_fail = retry_if_connection_error(mock_func_fail)
    with pytest.raises(RetryError):
        wrapped_func_fail()
    assert mock_func_fail.call_count == 3  # Stops after 3 attempts

    mock_func_no_retry = MagicMock(side_effect=ValueError("Something else"))
    wrapped_func_no_retry = retry_if_connection_error(mock_func_no_retry)
    with pytest.raises(ValueError):
        wrapped_func_no_retry()
    assert mock_func_no_retry.call_count == 1


def test_retry_if_timeout_error():
    mock_func = MagicMock(
        side_effect=[
            Timeout("Timed out"),
            "Success",
        ]
    )

    wrapped_func = retry_if_timeout_error(mock_func)
    assert wrapped_func() == "Success"
    assert mock_func.call_count == 2

    mock_func_fail = MagicMock(
        side_effect=[
            Timeout("Timed out"),
            HTTPError(response=MagicMock(spec=Response, status_code=408)),
        ]
    )
    wrapped_func_fail = retry_if_timeout_error(mock_func_fail)
    with pytest.raises(RetryError):
        wrapped_func_fail()
    assert mock_func_fail.call_count == 2  # Stops after 2 attempts

    mock_func_no_retry = MagicMock(side_effect=ValueError("Something else"))
    wrapped_func_no_retry = retry_if_timeout_error(mock_func_no_retry)
    with pytest.raises(ValueError):
        wrapped_func_no_retry()
    assert mock_func_no_retry.call_count == 1


def test_combined_retry_decorators():
    mock_func = MagicMock(
        side_effect=[
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            ConnectionError("Failed"),
            Timeout("Timed out"),
            "Success",
        ]
    )

    @retry_if_rate_limited
    @retry_if_connection_error
    @retry_if_timeout_error
    def wrapped_combined():
        return mock_func()

    assert wrapped_combined() == "Success"
    assert mock_func.call_count == 4

    mock_func_fail = MagicMock(
        side_effect=[
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            HTTPError(response=MagicMock(spec=Response, status_code=429)),
            ConnectionError("Failed"),
            ConnectionError("Failed"),
            Timeout("Timed out"),
            ValueError("Something else"),
        ]
    )

    @retry_if_rate_limited
    @retry_if_connection_error
    @retry_if_timeout_error
    def wrapped_combined_fail():
        return mock_func_fail()

    with pytest.raises(ValueError) as excinfo:
        wrapped_combined_fail()
    assert str(excinfo.value) == "Something else"
    assert mock_func_fail.call_count == 8
