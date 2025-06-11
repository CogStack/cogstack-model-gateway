import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from client.cogstack_model_gateway_client.client import GatewayClient


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
        polling_interval=0.01,
        timeout=0.1,
    )
    assert client.base_url == "http://localhost:8888"
    assert client.default_model == "test-model"
    assert client.polling_interval == 0.01
    assert client.timeout == 0.1
    assert client._client is None


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
    mock_client_instance.post.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        task_info = await client.submit_task(
            model_name="my_model", task="process", data="some text"
        )

    assert task_info == {"uuid": "task-123", "status": "pending"}
    mock_client_instance.post.assert_awaited_once_with(
        "http://test-gateway.com/models/my_model/tasks/process",
        data="some text",
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
    mock_client_instance.post.return_value = mock_response

    async with GatewayClient(
        base_url="http://test-gateway.com", default_model="default_model"
    ) as client:
        task_info = await client.submit_task(task="process", data="some text")

    assert task_info == {"uuid": "task-456", "status": "pending"}
    mock_client_instance.post.assert_awaited_once_with(
        "http://test-gateway.com/models/default_model/tasks/process",
        data="some text",
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
    mock_client_instance.post.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError):
        async with GatewayClient(base_url="http://test-gateway.com") as client:
            await client.submit_task(model_name="my_model", task="process")


@pytest.mark.asyncio
async def test_submit_task_wait_for_completion_and_return_result(mock_httpx_async_client, mocker):
    """Test submit_task with wait_for_completion and return_result."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"uuid": "task-123", "status": "pending"}
    mock_client_instance.post.return_value = mock_response

    mock_wait_for_task = mocker.patch(
        "client.cogstack_model_gateway_client.client.GatewayClient.wait_for_task",
        new=AsyncMock(return_value={"uuid": "task-123", "status": "succeeded"}),
    )
    mock_get_task_result = mocker.patch(
        "client.cogstack_model_gateway_client.client.GatewayClient.get_task_result",
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
async def test_redact_method(mocker):
    """Test the redact method."""
    async with GatewayClient(base_url="http://test-gateway.com") as client:
        mock_submit_task = mocker.patch.object(client, "submit_task", new=AsyncMock())
        await client.redact(text="sensitive text", model_name="deid_model")
        mock_submit_task.assert_awaited_once_with(
            model_name="deid_model",
            task="redact",
            data="sensitive text",
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
    mock_client_instance.get.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        task_info = await client.get_task("task-123")

    assert task_info == {"uuid": "task-123", "status": "succeeded"}
    mock_client_instance.get.assert_awaited_once_with(
        "http://test-gateway.com/tasks/task-123", params={"detail": True, "download": False}
    )
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_get_task_result_json(mock_httpx_async_client):
    """Test get_task_result with JSON content."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.content = b'{"key": "value"}'
    mock_response.raise_for_status.return_value = mock_response
    mock_client_instance.get.return_value = mock_response

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
    mock_client_instance.get.return_value = mock_response

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
    mock_client_instance.get.return_value = mock_response

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
    mock_client_instance.get.return_value = mock_response

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
    mock_client_instance.get.return_value = mock_response

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
    mocker.patch(
        "client.cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task
    )
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
    mocker.patch(
        "client.cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task
    )
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        client.timeout = 0.05
        client.polling_interval = 0.01

        with pytest.raises(
            TimeoutError, match="Timed out waiting for task 'task-polling' to complete"
        ):
            await client.wait_for_task("task-polling")

        assert mock_get_task.await_count >= (client.timeout / client.polling_interval)


@pytest.mark.asyncio
async def test_wait_for_task_failed_raise_on_error(mock_httpx_async_client, mocker):
    """Test wait_for_task raises RuntimeError on task failure with raise_on_error."""
    mock_get_task = AsyncMock()
    mock_get_task.side_effect = [
        {"uuid": "task-polling", "status": "pending"},
        {"uuid": "task-polling", "status": "failed", "error_message": "Something went wrong"},
    ]
    mocker.patch(
        "client.cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task
    )
    mocker.patch("asyncio.sleep", new=AsyncMock())

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        with pytest.raises(RuntimeError, match="Task 'task-polling' failed: Something went wrong"):
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
    mocker.patch(
        "client.cogstack_model_gateway_client.client.GatewayClient.get_task", new=mock_get_task
    )
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
    mock_client_instance.get.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        models = await client.get_models()
    assert models == ["model_a", "model_b"]
    mock_client_instance.get.assert_awaited_once_with(
        "http://test-gateway.com/models/", params={"verbose": False}
    )


@pytest.mark.asyncio
async def test_get_model_success(mock_httpx_async_client):
    """Test get_model for successful retrieval."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "my_model", "status": "deployed"}
    mock_client_instance.get.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        model_info = await client.get_model(model_name="my_model")
    assert model_info == {"name": "my_model", "status": "deployed"}
    mock_client_instance.get.assert_awaited_once_with(
        "http://test-gateway.com/models/my_model/info"
    )


@pytest.mark.asyncio
async def test_get_model_with_default_model(mock_httpx_async_client):
    """Test get_model using the default model."""
    _, mock_client_instance = mock_httpx_async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "default_model", "status": "deployed"}
    mock_client_instance.get.return_value = mock_response

    async with GatewayClient(
        base_url="http://test-gateway.com", default_model="default_model"
    ) as client:
        model_info = await client.get_model()
    assert model_info == {"name": "default_model", "status": "deployed"}
    mock_client_instance.get.assert_awaited_once_with(
        "http://test-gateway.com/models/default_model/info"
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
    mock_client_instance.post.return_value = mock_response

    async with GatewayClient(base_url="http://test-gateway.com") as client:
        deploy_info = await client.deploy_model(
            model_name="new_model",
            model_uri="mlflow-artifacts:/1/runidabcd1234/artifacts/new_model",
            ttl=3600,
        )

    assert deploy_info == {"name": "new_model", "status": "deploying"}
    mock_client_instance.post.assert_awaited_once_with(
        "http://test-gateway.com/models/new_model",
        json={
            "tracking_id": None,
            "model_uri": "mlflow-artifacts:/1/runidabcd1234/artifacts/new_model",
            "ttl": 3600,
        },
    )
