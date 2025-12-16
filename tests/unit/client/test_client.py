import asyncio
import concurrent.futures
import signal
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from cogstack_model_gateway_client.client import GatewayClient, GatewayClientSync
from cogstack_model_gateway_client.exceptions import TaskFailedError


@pytest.fixture
def mock_httpx_async_client(mocker):
    """Fixture to mock httpx.AsyncClient and its methods.

    Returns a tuple: (mocked_httpx_async_client_class, mocked_httpx_async_client_instance)
    """
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_class = mocker.patch("httpx.AsyncClient", return_value=mock_client_instance)
    return mock_client_class, mock_client_instance


@pytest.mark.asyncio
async def test_gateway_client_init():
    """Test GatewayClient initialization."""
    client = GatewayClient(
        base_url="http://localhost:8888/",
        default_model="test-model",
        polling_interval=0.5,
        polling_timeout=0.1,
        request_timeout=120.0,
    )
    assert client.base_url == "http://localhost:8888"
    assert client.default_model == "test-model"
    assert client.polling_interval == 0.5
    assert client.polling_timeout == 0.1
    assert client.request_timeout == 120.0
    assert client._client is None

    client = GatewayClient(
        base_url="http://localhost:8888/",
        polling_interval=10,
    )
    assert client.polling_interval == 3.0  # Maximum 3.0 seconds
    assert client.polling_timeout is None  # Default polling_timeout should be None
    assert client.request_timeout == 300.0  # Default request_timeout

    client.polling_interval = 0.05
    assert client.polling_interval == 0.5  # Minimum is 0.5 seconds


@pytest.mark.asyncio
async def test_gateway_client_aenter_aexit(mock_httpx_async_client):
    """Test __aenter__ and __aexit__ methods."""
    mock_client_class, mock_client_instance = mock_httpx_async_client

    client = GatewayClient(base_url="http://test-gateway.com")
    assert client._client is None
    async with client:
        mock_client_class.assert_called_once()
        assert client._client is mock_client_instance
        assert mock_client_instance.aclose.call_count == 0
    assert client._client is None
    mock_client_instance.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_client_decorator_raises_error():
    """Test that require_client decorator raises RuntimeError if client not entered."""
    client = GatewayClient(base_url="http://test-gateway.com")
    with pytest.raises(
        RuntimeError, match="GatewayClient must be used as an async context manager"
    ):
        await client.get_models()


@pytest.mark.asyncio
async def test_submit_task_success(mock_httpx_async_client):
    """Test submit_task for a successful submission."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"uuid": "task-123", "status": "pending"}
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        task_info = await client.submit_task(
            model_name="my_model", task="process", data="some text"
        )

    assert task_info == {"uuid": "task-123", "status": "pending"}
    mock_client_instance.request.assert_awaited_once_with(
        method="POST",
        url="http://test-gateway.com/models/my_model/tasks/process",
        data="some text",
        json=None,
        files=None,
        params=None,
        headers=None,
    )
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_submit_task_with_default_model(mock_httpx_async_client):
    """Test submit_task using the default model."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"uuid": "task-456", "status": "pending"}
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(
        base_url="http://test-gateway.com", default_model="default_model"
    ) as client:
        task_info = await client.submit_task(task="process", data="some text")

    assert task_info == {"uuid": "task-456", "status": "pending"}
    mock_client_instance.request.assert_awaited_once_with(
        method="POST",
        url="http://test-gateway.com/models/default_model/tasks/process",
        data="some text",
        json=None,
        files=None,
        params=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_submit_task_no_model_name_raises_value_error():
    """Test submit_task raises ValueError if no model name is provided."""
    with pytest.raises(ValueError, match="Please provide a model name or set a default model"):
        async with GatewayClient(base_url="http://test-gateway.com") as client:
            await client.submit_task(task="process", data="some text")


@pytest.mark.asyncio
async def test_submit_task_http_error(mock_httpx_async_client):
    """Test submit_task handles HTTP errors."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=httpx.Request("POST", "url"), response=httpx.Response(400)
    )
    mock_client_instance.request.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError):
        async with GatewayClient(base_url="http://test-gateway.com") as client:
            await client.submit_task(model_name="my_model", task="process")


@pytest.mark.asyncio
async def test_submit_task_wait_for_completion_and_return_result(mock_httpx_async_client, mocker):
    """Test submit_task with wait_for_completion and return_result."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"uuid": "task-123", "status": "pending"}
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    mock_wait_for_task = mocker.patch(
        "cogstack_model_gateway_client.client.GatewayClient.wait_for_task",
        new=AsyncMock(return_value={"uuid": "task-123", "status": "succeeded"}),
    )
    mock_get_task_result = mocker.patch(
        "cogstack_model_gateway_client.client.GatewayClient.get_task_result",
        new=AsyncMock(return_value="processed text"),
    )

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        result = await client.submit_task(
            model_name="my_model",
            task="process",
            data="text",
            wait_for_completion=True,
            return_result=True,
        )

    assert result == "processed text"
    mock_wait_for_task.assert_awaited_once_with("task-123")
    mock_get_task_result.assert_awaited_once_with("task-123")

    mock_wait_for_task = mocker.patch(
        "cogstack_model_gateway_client.client.GatewayClient.wait_for_task",
        new=AsyncMock(
            return_value={
                "uuid": "task-123",
                "status": "failed",
                "error_message": "Processing failed",
            }
        ),
    )

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        with pytest.raises(TaskFailedError, match="Task 'task-123' failed: Processing failed"):
            await client.submit_task(
                model_name="my_model",
                task="process",
                data="text",
                wait_for_completion=True,
                return_result=True,
            )

    mock_wait_for_task.assert_awaited_once_with("task-123")


@pytest.mark.asyncio
async def test_process_method(mocker):
    """Test the process method."""
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        mock_submit_task = mocker.patch.object(client, "submit_task", new=AsyncMock())
        await client.process(text="test text", model_name="annotation_model")
        mock_submit_task.assert_awaited_once_with(
            model_name="annotation_model",
            task="process",
            data="test text",
            headers={"Content-Type": "text/plain"},
            wait_for_completion=True,
            return_result=True,
        )


@pytest.mark.asyncio
async def test_process_bulk_method(mocker):
    """Test the process_bulk method."""
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        mock_submit_task = mocker.patch.object(client, "submit_task", new=AsyncMock())
        await client.process_bulk(texts=["text1", "text2"], model_name="bulk_model")
        mock_submit_task.assert_awaited_once_with(
            model_name="bulk_model",
            task="process_bulk",
            json=["text1", "text2"],
            headers={"Content-Type": "application/json"},
            wait_for_completion=True,
            return_result=True,
        )


@pytest.mark.asyncio
async def test_redact_method(mocker):
    """Test the redact method."""
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        mock_submit_task = mocker.patch.object(client, "submit_task", new=AsyncMock())
        await client.redact(
            text="sensitive text", concepts_to_keep=["label1", "label2"], model_name="deid_model"
        )
        mock_submit_task.assert_awaited_once_with(
            model_name="deid_model",
            task="redact",
            data="sensitive text",
            params={"concepts_to_keep": ["label1", "label2"]},
            headers={"Content-Type": "text/plain"},
            wait_for_completion=True,
            return_result=True,
        )


@pytest.mark.asyncio
async def test_get_task_success(mock_httpx_async_client):
    """Test get_task for a successful task retrieval."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = mock_response
    mock_response.json.return_value = {"uuid": "task-123", "status": "succeeded"}
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        task_info = await client.get_task("task-123")

    assert task_info == {"uuid": "task-123", "status": "succeeded"}
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/tasks/task-123",
        params={"detail": True, "download": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_get_task_result_json(mock_httpx_async_client):
    """Test get_task_result with JSON content."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.content = b'{"key": "value"}'
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        result = await client.get_task_result("task-123")
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_get_task_result_jsonl(mock_httpx_async_client):
    """Test get_task_result with JSONL content."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.content = b'{"item": 1}\n{"item": 2}\n'
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        result = await client.get_task_result("task-123")
    assert result == [{"item": 1}, {"item": 2}]


@pytest.mark.asyncio
async def test_get_task_result_text(mock_httpx_async_client):
    """Test get_task_result with plain text content."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.content = b"plain text result"
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        result = await client.get_task_result("task-123")
    assert result == "plain text result"


@pytest.mark.asyncio
async def test_get_task_result_binary(mock_httpx_async_client):
    """Test get_task_result with binary content."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.content = b"\x80\x01\x02\x03"  # Example binary data
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        result = await client.get_task_result("task-123")
    assert result == b"\x80\x01\x02\x03"


@pytest.mark.asyncio
async def test_get_task_result_no_parse(mock_httpx_async_client):
    """Test get_task_result with parse=False."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.content = b'{"key": "value"}'
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        result = await client.get_task_result("task-123", parse=False)
    assert result == b'{"key": "value"}'


@pytest.mark.asyncio
async def test_wait_for_task_succeeded(mock_httpx_async_client, mocker):
    """Test wait_for_task when task succeeds."""
    mock_get_task = AsyncMock()
    mock_get_task.side_effect = [
        {"uuid": "task-polling", "status": "pending"},
        {"uuid": "task-polling", "status": "succeeded", "result": "done"},
    ]
    mocker.patch("cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        task_info = await client.wait_for_task("task-polling")

    assert task_info == {"uuid": "task-polling", "status": "succeeded", "result": "done"}
    assert mock_get_task.await_count == 2
    asyncio.sleep.assert_awaited_once_with(client.polling_interval)


@pytest.mark.asyncio
async def test_wait_for_task_timeout(mock_httpx_async_client, mocker):
    """Test wait_for_task raises TimeoutError."""
    mock_get_task = AsyncMock(return_value={"uuid": "task-polling", "status": "pending"})
    mocker.patch("cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(
        base_url="http://test-gateway.com", polling_timeout=0.05, polling_interval=0.5
    ) as client:
        with pytest.raises(
            TimeoutError, match="Timed out waiting for task 'task-polling' to complete"
        ):
            await client.wait_for_task("task-polling")

        assert mock_get_task.await_count >= (client.polling_timeout / client.polling_interval)


@pytest.mark.asyncio
async def test_wait_for_task_no_timeout(mock_httpx_async_client, mocker):
    """Test wait_for_task doesn't timeout when timeout is None."""
    call_count = 0

    async def mock_get_task_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 5:  # Return pending for first 4 calls
            return {"uuid": "task-polling", "status": "pending"}
        else:  # Return succeeded on 5th call
            return {"uuid": "task-polling", "status": "succeeded"}

    mock_get_task = AsyncMock(side_effect=mock_get_task_side_effect)
    mocker.patch("cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(
        base_url="http://test-gateway.com", polling_timeout=None, polling_interval=0.01
    ) as client:
        result = await client.wait_for_task("task-polling")

        assert result["status"] == "succeeded"
        assert mock_get_task.await_count == 5


@pytest.mark.asyncio
async def test_wait_for_task_failed_raise_on_error(mock_httpx_async_client, mocker):
    """Test wait_for_task raises TaskFailedError on task failure with raise_on_error."""
    mock_get_task = AsyncMock()
    mock_get_task.side_effect = [
        {"uuid": "task-polling", "status": "pending"},
        {"uuid": "task-polling", "status": "failed", "error_message": "Something went wrong"},
    ]
    mocker.patch("cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        with pytest.raises(
            TaskFailedError, match="Task 'task-polling' failed: Something went wrong"
        ):
            await client.wait_for_task("task-polling", raise_on_error=True)

        assert mock_get_task.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_task_failed_no_raise_on_error(mock_httpx_async_client, mocker):
    """Test wait_for_task returns task info on failure when raise_on_error is False."""
    mock_get_task = AsyncMock()
    mock_get_task.side_effect = [
        {"uuid": "task-polling", "status": "pending"},
        {"uuid": "task-polling", "status": "failed", "error_message": "Something went wrong"},
    ]
    mocker.patch("cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        task_info = await client.wait_for_task("task-polling", raise_on_error=False)

    assert task_info == {
        "uuid": "task-polling",
        "status": "failed",
        "error_message": "Something went wrong",
    }
    assert mock_get_task.await_count == 2


@pytest.mark.asyncio
async def test_get_models_success(mock_httpx_async_client):
    """Test get_models for successful retrieval."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = ["model_a", "model_b"]
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        models = await client.get_models()
    assert models == ["model_a", "model_b"]
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/models/",
        params={"verbose": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_get_model_success(mock_httpx_async_client):
    """Test get_model for successful retrieval."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "my_model",
        "uri": "models:/my_model/1",
        "is_running": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        model_info = await client.get_model(model_name="my_model")
    assert model_info == {"name": "my_model", "uri": "models:/my_model/1", "is_running": True}
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/models/my_model",
        params={"verbose": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_get_model_with_default_model(mock_httpx_async_client):
    """Test get_model using the default model."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "default_model",
        "uri": "models:/default_model/1",
        "is_running": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(
        base_url="http://test-gateway.com", default_model="default_model"
    ) as client:
        model_info = await client.get_model()
    assert model_info == {
        "name": "default_model",
        "uri": "models:/default_model/1",
        "is_running": True,
    }
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/models/default_model",
        params={"verbose": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_get_model_no_model_name_raises_value_error():
    """Test get_model raises ValueError if no model name is provided."""
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        with pytest.raises(ValueError, match="Please provide a model name or set a default model"):
            await client.get_model()


@pytest.mark.asyncio
async def test_deploy_model_success(mock_httpx_async_client):
    """Test deploy_model for successful deployment."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "new_model", "status": "deploying"}
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        deploy_info = await client.deploy_model(
            model_name="new_model",
            model_uri="mlflow-artifacts:/1/runidabcd1234/artifacts/new_model",
            ttl=3600,
        )

    assert deploy_info == {"name": "new_model", "status": "deploying"}
    mock_client_instance.request.assert_awaited_once_with(
        method="POST",
        url="http://test-gateway.com/models/new_model",
        json={
            "tracking_id": None,
            "model_uri": "mlflow-artifacts:/1/runidabcd1234/artifacts/new_model",
            "ttl": 3600,
        },
        params=None,
        data=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_remove_model(mock_httpx_async_client):
    """Test remove_model."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        await client.remove_model(model_name="my_model", force=False)

    mock_client_instance.request.assert_awaited_once_with(
        method="DELETE",
        url="http://test-gateway.com/models/my_model",
        params={"force": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )

    # Test remove_model with force flag
    mock_client_instance.request.reset_mock()
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        await client.remove_model(model_name="my_model", force=True)

    mock_client_instance.request.assert_awaited_once_with(
        method="DELETE",
        url="http://test-gateway.com/models/my_model",
        params={"force": True},
        data=None,
        json=None,
        files=None,
        headers=None,
    )

    # Test remove_model using default model
    mock_client_instance.request.reset_mock()
    async with GatewayClient(
        base_url="http://test-gateway.com", default_model="default_model"
    ) as client:
        await client.remove_model()

    mock_client_instance.request.assert_awaited_once_with(
        method="DELETE",
        url="http://test-gateway.com/models/default_model",
        params={"force": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )

    # Test remove_model raises ValueError if no model name is provided
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        with pytest.raises(ValueError, match="Please provide a model name or set a default model"):
            await client.remove_model()


@pytest.mark.asyncio
async def test_list_on_demand_configs(mock_httpx_async_client):
    """Test list_on_demand_configs."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "configs": [{"model_name": "model1"}, {"model_name": "model2"}],
        "total": 2,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        configs = await client.list_on_demand_configs(include_disabled=False)

    assert configs == {"configs": [{"model_name": "model1"}, {"model_name": "model2"}], "total": 2}
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/admin/on-demand",
        params={"include_disabled": False},
        data=None,
        json=None,
        files=None,
        headers=None,
    )

    # Test list_on_demand_configs including disabled configs
    mock_client_instance.request.reset_mock()
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        configs = await client.list_on_demand_configs(include_disabled=True)

    assert configs["total"] == 2
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/admin/on-demand",
        params={"include_disabled": True},
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_create_on_demand_config(mock_httpx_async_client):
    """Test create_on_demand_config."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": 1,
        "model_name": "my_model",
        "model_uri": "models:/my_model/1",
        "enabled": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        config = await client.create_on_demand_config(
            model_name="my_model",
            model_uri="models:/my_model/1",
            idle_ttl=3600,
            description="Test model",
            replace_enabled=True,
            inherit_config=False,
        )

    assert config["model_name"] == "my_model"
    mock_client_instance.request.assert_awaited_once_with(
        method="POST",
        url="http://test-gateway.com/admin/on-demand",
        json={
            "model_name": "my_model",
            "tracking_id": None,
            "model_uri": "models:/my_model/1",
            "idle_ttl": 3600,
            "description": "Test model",
            "deploy": None,
            "replace_enabled": True,
            "inherit_config": False,
        },
        params=None,
        data=None,
        files=None,
        headers=None,
    )

    # Test create_on_demand_config with tracking_id
    mock_client_instance.request.reset_mock()
    mock_response.json.return_value = {
        "id": 1,
        "model_name": "my_model",
        "tracking_id": "run123",
        "model_uri": "models:/my_model/1",
        "enabled": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        config = await client.create_on_demand_config(
            model_name="my_model",
            tracking_id="run123",
        )

    assert config["tracking_id"] == "run123"
    mock_client_instance.request.assert_awaited_once_with(
        method="POST",
        url="http://test-gateway.com/admin/on-demand",
        json={
            "model_name": "my_model",
            "tracking_id": "run123",
            "model_uri": None,
            "idle_ttl": None,
            "description": None,
            "deploy": None,
            "replace_enabled": True,
            "inherit_config": True,
        },
        params=None,
        data=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_get_on_demand_config(mock_httpx_async_client):
    """Test get_on_demand_config."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": 1,
        "model_name": "my_model",
        "model_uri": "models:/my_model/1",
        "enabled": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        config = await client.get_on_demand_config(model_name="my_model")

    assert config["model_name"] == "my_model"
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/admin/on-demand/my_model",
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_update_on_demand_config(mock_httpx_async_client):
    """Test update_on_demand_config."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": 1,
        "model_name": "my_model",
        "model_uri": "models:/my_model/2",
        "idle_ttl": 7200,
        "enabled": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        config = await client.update_on_demand_config(
            model_name="my_model",
            model_uri="models:/my_model/2",
            idle_ttl=7200,
        )

    assert config["model_uri"] == "models:/my_model/2"
    assert config["idle_ttl"] == 7200
    mock_client_instance.request.assert_awaited_once_with(
        method="PUT",
        url="http://test-gateway.com/admin/on-demand/my_model",
        json={
            "tracking_id": None,
            "model_uri": "models:/my_model/2",
            "idle_ttl": 7200,
            "description": None,
            "deploy": None,
            "clear_tracking_id": False,
            "clear_idle_ttl": False,
            "clear_description": False,
            "clear_deploy": False,
        },
        params=None,
        data=None,
        files=None,
        headers=None,
    )

    # Test update_on_demand_config with clear flags
    mock_client_instance.request.reset_mock()
    mock_response.json.return_value = {
        "id": 1,
        "model_name": "my_model",
        "model_uri": "models:/my_model/2",
        "tracking_id": None,
        "description": None,
        "enabled": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        config = await client.update_on_demand_config(
            model_name="my_model",
            clear_tracking_id=True,
            clear_description=True,
        )

    mock_client_instance.request.assert_awaited_once_with(
        method="PUT",
        url="http://test-gateway.com/admin/on-demand/my_model",
        json={
            "tracking_id": None,
            "model_uri": None,
            "idle_ttl": None,
            "description": None,
            "deploy": None,
            "clear_tracking_id": True,
            "clear_idle_ttl": False,
            "clear_description": True,
            "clear_deploy": False,
        },
        params=None,
        data=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_delete_on_demand_config(mock_httpx_async_client):
    """Test delete_on_demand_config."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        await client.delete_on_demand_config(model_name="my_model")

    mock_client_instance.request.assert_awaited_once_with(
        method="DELETE",
        url="http://test-gateway.com/admin/on-demand/my_model",
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_get_on_demand_config_history(mock_httpx_async_client):
    """Test get_on_demand_config_history."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "configs": [
            {"id": 2, "model_name": "my_model", "enabled": True},
            {"id": 1, "model_name": "my_model", "enabled": False},
        ],
        "total": 2,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        history = await client.get_on_demand_config_history(model_name="my_model")

    assert history["total"] == 2
    assert len(history["configs"]) == 2
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/admin/on-demand/my_model/history",
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_enable_on_demand_config(mock_httpx_async_client):
    """Test enable_on_demand_config."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": 1,
        "model_name": "my_model",
        "enabled": True,
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        config = await client.enable_on_demand_config(config_id=1)

    assert config["enabled"] is True
    assert config["id"] == 1
    mock_client_instance.request.assert_awaited_once_with(
        method="POST",
        url="http://test-gateway.com/admin/on-demand/1/enable",
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_health_check_success(mock_httpx_async_client):
    """Test health_check for successful health status retrieval."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "healthy",
        "components": {
            "database": "healthy",
            "task_object_store": "healthy",
            "results_object_store": "healthy",
            "queue": "healthy",
        },
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        health_status = await client.health_check()

    assert health_status["status"] == "healthy"
    assert "components" in health_status
    assert health_status["components"]["database"] == "healthy"
    mock_client_instance.request.assert_awaited_once_with(
        method="GET",
        url="http://test-gateway.com/health",
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    )


@pytest.mark.asyncio
async def test_health_check_unhealthy_503_with_json(mock_httpx_async_client):
    """Test health_check when service is unhealthy (503) but returns health details."""
    _, mock_client_instance = mock_httpx_async_client

    mock_503_response = MagicMock()
    mock_503_response.status_code = 503
    mock_503_response.json.return_value = {
        "status": "unhealthy",
        "components": {
            "database": "unhealthy",
            "task_object_store": "healthy",
            "results_object_store": "healthy",
            "queue": "healthy",
        },
    }

    mock_response = MagicMock()
    http_error = httpx.HTTPStatusError(
        "Service Unavailable", request=httpx.Request("GET", "url"), response=mock_503_response
    )
    mock_response.raise_for_status.side_effect = http_error
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        health_status = await client.health_check()

    assert health_status["status"] == "unhealthy"
    assert health_status["components"]["database"] == "unhealthy"
    assert health_status["components"]["task_object_store"] == "healthy"


@pytest.mark.asyncio
async def test_health_check_unhealthy_503_without_json(mock_httpx_async_client):
    """Test health_check when service returns 503 without valid JSON."""
    _, mock_client_instance = mock_httpx_async_client

    mock_503_response = MagicMock()
    mock_503_response.status_code = 503
    mock_503_response.json.side_effect = Exception("Invalid JSON")

    mock_response = MagicMock()
    http_error = httpx.HTTPStatusError(
        "Service Unavailable", request=httpx.Request("GET", "url"), response=mock_503_response
    )
    mock_response.raise_for_status.side_effect = http_error
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        health_status = await client.health_check()

    assert health_status["status"] == "unhealthy"
    assert health_status["error"] == "Service unavailable"


@pytest.mark.asyncio
async def test_health_check_other_http_error(mock_httpx_async_client):
    """Test health_check when service returns non-503 HTTP error."""
    _, mock_client_instance = mock_httpx_async_client

    mock_response = MagicMock()
    http_error = httpx.HTTPStatusError(
        "Internal Server Error", request=httpx.Request("GET", "url"), response=httpx.Response(500)
    )
    mock_response.raise_for_status.side_effect = http_error
    mock_client_instance.request.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        # Should re-raise non-503 HTTP errors
        with pytest.raises(httpx.HTTPStatusError):
            await client.health_check()


@pytest.mark.asyncio
async def test_is_healthy(mock_httpx_async_client, mocker):
    """Test is_healthy convenience method."""
    mock_health_check = mocker.patch(
        "cogstack_model_gateway_client.client.GatewayClient.health_check",
        new=AsyncMock(return_value={"status": "healthy", "components": {}}),
    )

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        is_healthy = await client.is_healthy()

    assert is_healthy is True
    mock_health_check.assert_awaited_once()

    # Test unhealthy status
    mock_health_check = mocker.patch(
        "cogstack_model_gateway_client.client.GatewayClient.health_check",
        new=AsyncMock(return_value={"status": "unhealthy", "components": {}}),
    )

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        is_healthy = await client.is_healthy()

    assert is_healthy is False
    mock_health_check.assert_awaited_once()

    # Test exception handling
    mock_health_check = mocker.patch(
        "cogstack_model_gateway_client.client.GatewayClient.health_check",
        new=AsyncMock(side_effect=Exception("Network error")),
    )

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        is_healthy = await client.is_healthy()

    assert is_healthy is False
    mock_health_check.assert_awaited_once()


def test_sync_client_without_event_loop(mock_httpx_async_client):
    """Test GatewayClientSync works when no event loop is running."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = ["model1", "model2"]
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    try:
        # Should work without a running event loop
        client = GatewayClientSync(base_url="http://test-gateway.com")
        models = client.get_models()
        assert models == ["model1", "model2"]
        assert hasattr(client, "_client")
    finally:
        del client


def test_sync_client_with_existing_event_loop(mock_httpx_async_client):
    """Test GatewayClientSync works when an event loop is already running in a different thread."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = ["model1", "model2"]
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    result = {}
    exception = {}

    def test_from_different_thread():
        try:
            client = GatewayClientSync(base_url="http://test-gateway.com")
            models = client.get_models()
            result["models"] = models
            del client
        except Exception as e:
            exception["error"] = e

    # Run the test in a separate thread
    test_thread = threading.Thread(target=test_from_different_thread)
    test_thread.start()
    test_thread.join(timeout=10)

    assert "error" not in exception, f"Test failed with error: {exception.get('error')}"
    assert result["models"] == ["model1", "model2"]


@pytest.mark.asyncio
async def test_sync_client_in_async_context(mock_httpx_async_client):
    """Test that GatewayClientSync correctly rejects usage from within an async context."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = ["model1", "model2"]
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    # This test is running in an async context (due to @pytest.mark.asyncio)
    with pytest.raises(RuntimeError, match="can't be used inside an async context"):
        _ = GatewayClientSync(base_url="http://test-gateway.com")


def test_sync_client_multiple_requests_reuse_resources(mock_httpx_async_client):
    """Test that multiple requests reuse the same background resources."""
    _, mock_client_instance = mock_httpx_async_client

    # Mock different responses for different calls and use a function to return them
    mock_response1 = MagicMock()
    mock_response1.json.return_value = ["model1", "model2"]
    mock_response1.raise_for_status.return_value = mock_response1

    mock_response2 = MagicMock()
    mock_response2.json.return_value = {"name": "model1", "status": "active"}
    mock_response2.raise_for_status.return_value = mock_response2

    responses = [mock_response1, mock_response2]
    response_index = 0

    def mock_request(*args, **kwargs):
        nonlocal response_index
        if response_index < len(responses):
            response = responses[response_index]
            response_index += 1
            return response
        else:
            # If we run out of responses, return the last one
            return responses[-1]

    mock_client_instance.request.side_effect = mock_request

    try:
        client = GatewayClientSync(base_url="http://test-gateway.com")

        # First request
        models = client.get_models()
        assert models == ["model1", "model2"]

        # Get references to the background resources if available
        background_loop = getattr(client, "_background_loop", None)
        background_thread = getattr(client, "_thread", None)

        # Second request
        model_info = client.get_model(model_name="model1")
        assert model_info == {"name": "model1", "status": "active"}

        # If we have background resources, verify they're reused
        if background_loop is not None:
            assert client._background_loop is background_loop
            assert client._thread is background_thread

        # Verify both requests were made
        assert mock_client_instance.request.call_count == 2

        # Verify the specific URLs were called
        call_args = [call[1]["url"] for call in mock_client_instance.request.call_args_list]
        assert "http://test-gateway.com/models/" in call_args[0]  # get_models
        assert "http://test-gateway.com/models/model1" in call_args[1]  # get_model

    finally:
        del client


def test_sync_client_error_handling(mock_httpx_async_client):
    """Test that errors from async operations are properly propagated."""
    _, mock_client_instance = mock_httpx_async_client
    mock_client_instance.request.side_effect = httpx.HTTPStatusError(
        "Server Error", request=httpx.Request("GET", "url"), response=httpx.Response(500)
    )

    try:
        client = GatewayClientSync(base_url="http://test-gateway.com")
        with pytest.raises(httpx.HTTPStatusError):
            client.get_models()
    finally:
        del client


def test_sync_client_health_check(mock_httpx_async_client):
    """Test health_check method for sync client."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "healthy",
        "components": {
            "database": "healthy",
            "task_object_store": "healthy",
            "results_object_store": "healthy",
            "queue": "healthy",
        },
    }
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    try:
        client = GatewayClientSync(base_url="http://test-gateway.com")
        health_status = client.health_check()

        assert health_status["status"] == "healthy"
        assert "components" in health_status
        assert health_status["components"]["database"] == "healthy"

        call_args = mock_client_instance.request.call_args_list
        assert len(call_args) == 1
        assert call_args[0][1]["url"] == "http://test-gateway.com/health"
        assert call_args[0][1]["method"] == "GET"
    finally:
        del client

    # Test unhealthy 503 response with JSON body
    mock_503_response = MagicMock()
    mock_503_response.status_code = 503
    mock_503_response.json.return_value = {
        "status": "unhealthy",
        "components": {"database": "unhealthy"},
    }

    mock_response = MagicMock()
    http_error = httpx.HTTPStatusError(
        "Service Unavailable", request=httpx.Request("GET", "url"), response=mock_503_response
    )
    mock_response.raise_for_status.side_effect = http_error
    mock_client_instance.request.return_value = mock_response

    try:
        client = GatewayClientSync(base_url="http://test-gateway.com")
        health_status = client.health_check()

        assert health_status["status"] == "unhealthy"
        assert health_status["components"]["database"] == "unhealthy"
    finally:
        del client


def test_sync_client_is_healthy(mock_httpx_async_client):
    """Test is_healthy method for sync client."""
    _, mock_client_instance = mock_httpx_async_client

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "healthy", "components": {}}
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    try:
        client = GatewayClientSync(base_url="http://test-gateway.com")
        is_healthy = client.is_healthy()
        assert is_healthy is True
    finally:
        del client


def test_sync_client_timeout_handling(mock_httpx_async_client):
    """Test that polling_timeout works correctly in the sync client."""
    _, mock_client_instance = mock_httpx_async_client

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(2)
        mock_response = MagicMock()
        mock_response.json.return_value = ["model1"]
        mock_response.raise_for_status.return_value = mock_response
        return mock_response

    mock_client_instance.request.side_effect = slow_response

    try:
        client = GatewayClientSync(base_url="http://test-gateway.com")
        start_time = time.time()

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)

        try:
            with pytest.raises((TimeoutError, concurrent.futures.TimeoutError)):
                client.get_models()
        finally:
            signal.alarm(0)  # Cancel the alarm

        elapsed = time.time() - start_time
        assert elapsed < 2.0  # Should fail faster than the 2 second sleep

    finally:
        del client


def test_no_event_loop_conflict_error(mock_httpx_async_client):
    """Test that we don't get the event loop conflict errors."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = ["model1", "model2"]
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.request.return_value = mock_response

    loop = asyncio.new_event_loop()
    errors = []

    def run_with_existing_loop():
        asyncio.set_event_loop(loop)
        try:
            client = GatewayClientSync(base_url="http://test-gateway.com")
            models = client.get_models()
            assert models == ["model1", "model2"]
            del client
        except Exception as e:
            errors.append(str(e))
        finally:
            loop.call_soon_threadsafe(loop.stop)

    def keep_loop_running():
        loop.run_forever()

    # Start loop in one thread
    loop_thread = threading.Thread(target=keep_loop_running, daemon=True)
    loop_thread.start()

    # Run client in another thread (simulating web app scenario)
    client_thread = threading.Thread(target=run_with_existing_loop)
    client_thread.start()
    client_thread.join(timeout=10)

    loop_thread.join(timeout=2)

    assert len(errors) == 0, f"Unexpected errors: {errors}"
