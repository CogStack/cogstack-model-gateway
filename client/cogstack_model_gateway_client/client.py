import asyncio
import json
from functools import wraps

import httpx


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        default_model: str = None,
        polling_interval: float = 2.0,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._client.aclose()
        self._client = None

    @staticmethod
    def require_client(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if self._client is None:
                raise RuntimeError(
                    "GatewayClient must be used as an async context manager. "
                    "Use: 'async with GatewayClient(...) as client:'"
                )
            return await func(self, *args, **kwargs)

        return wrapper

    @require_client
    async def submit_task(
        self,
        model_name: str = None,
        task: str = None,
        data=None,
        files=None,
        params=None,
        headers=None,
        wait_for_completion: bool = False,
        return_result: bool = True,
    ):
        """Submit a task to the Gateway and return the task info."""
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("Please provide a model name or set a default model for the client.")
        url = f"{self.base_url}/models/{model_name}/tasks/{task}"
        resp = await self._client.post(url, data=data, files=files, params=params, headers=headers)
        resp.raise_for_status()
        task_info = resp.json()
        if wait_for_completion:
            task_uuid = task_info["uuid"]
            task_info = await self.wait_for_task(task_uuid)
            if return_result:
                return await self.get_task_result(task_uuid)
        return task_info

    async def process(
        self,
        text: str,
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Generate annotations for the the provided text."""
        return await self.submit_task(
            model_name=model_name,
            task="process",
            data=text,
            headers={"Content-Type": "text/plain"},
            wait_for_completion=wait_for_completion,
            return_result=return_result,
        )

    async def redact(
        self,
        text: str,
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Redact sensitive information from the provided text."""
        return await self.submit_task(
            model_name=model_name,
            task="redact",
            data=text,
            headers={"Content-Type": "text/plain"},
            wait_for_completion=wait_for_completion,
            return_result=return_result,
        )

    @require_client
    async def _get_task(self, task_uuid: str, detail: bool = True, download: bool = False):
        """Get a Gateway task."""
        url = f"{self.base_url}/tasks/{task_uuid}"
        params = {"detail": detail, "download": download}
        resp = await self._client.get(url, params=params)
        return resp.raise_for_status()

    @require_client
    async def get_task(self, task_uuid: str, detail: bool = True):
        """Get a Gateway task details by its UUID."""
        resp = await self._get_task(task_uuid, detail=detail)
        return resp.json()

    @require_client
    async def get_task_result(self, task_uuid: str, parse: bool = True):
        """Get the result of a Gateway task by its UUID.

        If parse is True, try to infer and parse the result as JSON, JSONL, or text.
        Otherwise, return raw bytes.
        """
        resp = await self._get_task(task_uuid, detail=False, download=True)
        result = resp.content

        if not parse or not result:
            return result

        result_str = None
        try:
            result_str = result.decode("utf-8")
        except UnicodeDecodeError:
            return result

        try:
            return json.loads(result_str)
        except Exception:
            pass

        try:
            if jsonl := [json.loads(line) for line in result_str.split("\n") if line]:
                return jsonl
        except Exception:
            pass

        return result_str

    @require_client
    async def wait_for_task(
        self, task_uuid: str, detail: bool = True, raise_on_error: bool = False
    ):
        """Poll Gateway until the task reaches a final state."""
        start = asyncio.get_event_loop().time()
        while True:
            task = await self.get_task(task_uuid, detail=detail)
            if (status := task.get("status")) in ("succeeded", "failed"):
                if status == "failed" and raise_on_error:
                    error_message = task.get("error_message", "Unknown error")
                    raise RuntimeError(f"Task '{task_uuid}' failed: {error_message}")
                return task
            if asyncio.get_event_loop().time() - start > self.timeout:
                raise TimeoutError(f"Timed out waiting for task '{task_uuid}' to complete")
            await asyncio.sleep(self.polling_interval)

    @require_client
    async def get_models(self, verbose: bool = False):
        """Get the list of available models from the Gateway."""
        url = f"{self.base_url}/models/"
        resp = await self._client.get(url, params={"verbose": verbose})
        resp.raise_for_status()
        return resp.json()

    @require_client
    async def get_model(self, model_name: str = None):
        """Get details of a specific model."""
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("Please provide a model name or set a default model for the client.")
        url = f"{self.base_url}/models/{model_name}/info"
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    @require_client
    async def deploy_model(
        self,
        model_name: str = None,
        tracking_id: str = None,
        model_uri: str = None,
        ttl: int = None,
    ):
        """Deploy a CogStack Model Serve model through the Gateway."""
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("Please provide a model name or set a default model for the client.")
        url = f"{self.base_url}/models/{model_name}"
        data = {"tracking_id": tracking_id, "model_uri": model_uri, "ttl": ttl}
        resp = await self._client.post(url, json=data)
        resp.raise_for_status()
        return resp.json()


class GatewayClientSync:
    def __init__(self, *args, **kwargs):
        self._client = GatewayClient(*args, **kwargs)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._client.__aenter__())

    def __del__(self):
        try:
            if hasattr(self, "_client") and self._client and self._client._client is not None:
                self._loop.run_until_complete(self._client.__aexit__(None, None, None))
                self._loop.close()
        except Exception:
            pass

    def submit_task(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.submit_task(*args, **kwargs))

    def process(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.process(*args, **kwargs))

    def redact(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.redact(*args, **kwargs))

    def get_task(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.get_task(*args, **kwargs))

    def get_task_result(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.get_task_result(*args, **kwargs))

    def wait_for_task(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.wait_for_task(*args, **kwargs))

    def get_models(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.get_models(*args, **kwargs))

    def get_model(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.get_model(*args, **kwargs))

    def deploy_model(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.deploy_model(*args, **kwargs))
