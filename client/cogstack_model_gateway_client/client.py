import asyncio
import json
from collections.abc import Iterable
from functools import wraps

import httpx

from cogstack_model_gateway_client.exceptions import retry_if_network_error


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        default_model: str = None,
        polling_interval: float = 2.0,
        timeout: float = 300.0,
    ):
        """Initialize the GatewayClient with the base Gateway URL and optional parameters.

        Args:
            base_url (str): The base URL of the Gateway service.
            default_model (str, optional): The default model to use for tasks. Defaults to None.
            polling_interval (float, optional): The interval in seconds to poll for task completion.
                Defaults to 2.0 seconds, with a minimum of 0.5 and maximum of 3.0 seconds.
            timeout (float, optional): The client polling timeout while waiting for task completion.
                Defaults to 300.0 seconds. A TimeoutError in this case should rarely indicate that
                something went wrong, but rather that the task is taking longer than expected (e.g.
                common with long running tasks like training).
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._client = None

    @property
    def polling_interval(self):
        return self._polling_interval

    @polling_interval.setter
    def polling_interval(self, value: float):
        self._polling_interval = max(0.5, min(value, 3.0))

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
    @retry_if_network_error
    async def _request(
        self,
        method: str,
        url: str,
        *,
        data=None,
        json=None,
        files=None,
        params=None,
        headers=None,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP requests with retry logic."""
        resp = await self._client.request(
            method=method,
            url=url,
            data=data,
            json=json,
            files=files,
            params=params,
            headers=headers,
            **kwargs,
        )
        return resp.raise_for_status()

    @require_client
    async def submit_task(
        self,
        model_name: str = None,
        task: str = None,
        data=None,
        json=None,
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

        resp = await self._request(
            "POST", url, data=data, json=json, files=files, params=params, headers=headers
        )
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
        """Generate annotations for the provided text."""
        return await self.submit_task(
            model_name=model_name,
            task="process",
            data=text,
            headers={"Content-Type": "text/plain"},
            wait_for_completion=wait_for_completion,
            return_result=return_result,
        )

    async def process_bulk(
        self,
        texts: list[str],
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Generate annotations for a list of texts."""
        return await self.submit_task(
            model_name=model_name,
            task="process_bulk",
            json=texts,
            headers={"Content-Type": "application/json"},
            wait_for_completion=wait_for_completion,
            return_result=return_result,
        )

    async def redact(
        self,
        text: str,
        concepts_to_keep: Iterable[str] = None,
        warn_on_no_redaction: bool = None,
        mask: str = None,
        hash: bool = None,
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Redact sensitive information from the provided text."""
        params = {
            k: v
            for k, v in {
                "concepts_to_keep": concepts_to_keep,
                "warn_on_no_redaction": warn_on_no_redaction,
                "mask": mask,
                "hash": hash,
            }.items()
            if v is not None
        } or None

        return await self.submit_task(
            model_name=model_name,
            task="redact",
            data=text,
            params=params,
            headers={"Content-Type": "text/plain"},
            wait_for_completion=wait_for_completion,
            return_result=return_result,
        )

    @require_client
    async def _get_task(self, task_uuid: str, detail: bool = True, download: bool = False):
        """Get a Gateway task."""
        url = f"{self.base_url}/tasks/{task_uuid}"
        params = {"detail": detail, "download": download}
        return await self._request("GET", url, params=params)

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
            jsonl = [json.loads(line) for line in result_str.split("\n") if line]
            if jsonl:
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
            status = task.get("status")
            if status in ("succeeded", "failed"):
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
        resp = await self._request("GET", url, params={"verbose": verbose})
        return resp.json()

    @require_client
    async def get_model(self, model_name: str = None):
        """Get details of a specific model."""
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("Please provide a model name or set a default model for the client.")
        url = f"{self.base_url}/models/{model_name}/info"
        resp = await self._request("GET", url)
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
        resp = await self._request("POST", url, json=data)
        return resp.json()


class GatewayClientSync:
    def __init__(self, *args, **kwargs):
        self._runner = asyncio.Runner()
        self._client = self._runner.run(GatewayClient(*args, **kwargs).__aenter__())

    def __del__(self):
        if getattr(self, "_client", None):
            self._runner.run(self._client.__aexit__(None, None, None))
        if getattr(self, "_runner", None):
            self._runner.close()

    def _run_async(self, coro):
        """Run a coroutine inside private Runner."""
        return self._runner.run(coro)

    @property
    def base_url(self):
        return self._client.base_url

    @property
    def default_model(self):
        return self._client.default_model

    @default_model.setter
    def default_model(self, value: str):
        self._client.default_model = value

    @property
    def polling_interval(self):
        return self._client.polling_interval

    @polling_interval.setter
    def polling_interval(self, value: float):
        self._client.polling_interval = value

    @property
    def timeout(self):
        return self._client.timeout

    @timeout.setter
    def timeout(self, value: float):
        self._client.timeout = value

    # def _setup_event_loop(self):
    #     """Set up event loop handling for sync operations."""
    #     try:
    #         existing_loop = asyncio.get_running_loop()
    #         self._loop = existing_loop
    #         self._own_loop = False
    #         self._background_loop = None
    #         self._setup_background_loop()
    #     except RuntimeError:
    #         self._loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(self._loop)
    #         self._own_loop = True
    #         self._background_loop = None
    #         self._loop.run_until_complete(self._client.__aenter__())

    # def _setup_background_loop(self):
    #     """Set up a background thread with persistent event loop."""
    #     self._client_initialized = threading.Event()
    #     self._setup_error = None

    #     def run_background_loop():
    #         try:
    #             loop = asyncio.new_event_loop()
    #             self._background_loop = loop

    #             async def init_client():
    #                 try:
    #                     await self._client.__aenter__()
    #                 except Exception as e:
    #                     self._setup_error = e
    #                     raise
    #                 finally:
    #                     self._client_initialized.set()

    #             # Create task and keep reference to prevent it from being garbage collected
    #             self._init_task = loop.create_task(init_client())

    #             # Set a callback to handle completion or error
    #             def on_init_complete(task):
    #                 if task.exception():
    #                     if not self._setup_error:
    #                         self._setup_error = task.exception()

    #             self._init_task.add_done_callback(on_init_complete)

    #             loop.run_forever()

    #         except Exception as e:
    #             if not self._setup_error:
    #                 self._setup_error = e
    #             self._client_initialized.set()

    #     self._thread = threading.Thread(target=run_background_loop, daemon=True)
    #     self._thread.start()

    #     # Wait for client to be ready
    #     if not self._client_initialized.wait(timeout=5.0):
    #         raise RuntimeError("Failed to initialize client in background loop within timeout")

    #     if self._setup_error:
    #         raise RuntimeError(f"Failed to setup background client: {self._setup_error}")

    # def _run_async(self, coro):
    #     """Run an async coroutine, handling different event loop scenarios."""
    #     if self._own_loop:
    #         return self._loop.run_until_complete(coro)
    #     else:
    #         if not hasattr(self, "_background_loop") or not self._background_loop:
    #             raise RuntimeError("Background loop not initialized")

    #         future = asyncio.run_coroutine_threadsafe(coro, self._background_loop)
    #         return future.result(timeout=60)

    # def __del__(self):
    #     try:
    #         if not hasattr(self, "_client") or not self._client:
    #             return

    #         if self._own_loop and self._client._client is not None:
    #             try:
    #                 self._loop.run_until_complete(self._client.__aexit__(None, None, None))
    #                 self._loop.close()
    #             except Exception:
    #                 pass
    #         elif hasattr(self, "_background_loop") and hasattr(self, "_thread"):
    #             if self._background_loop and not self._background_loop.is_closed():
    #                 try:
    #                     # Cancel the init task if it's still pending
    #                     if (
    #                         hasattr(self, "_init_task")
    #                         and self._init_task
    #                         and not self._init_task.done()
    #                     ):
    #                         self._init_task.cancel()

    #                     # Only call __aexit__ if the client was successfully initialized
    #                     # (i.e., if the init task completed successfully)
    #                     if (
    #                         hasattr(self, "_init_task")
    #                         and self._init_task
    #                         and self._init_task.done()
    #                         and not self._init_task.exception()
    #                         and hasattr(self._client, "_client")
    #                         and self._client._client is not None
    #                     ):
    #                         future = asyncio.run_coroutine_threadsafe(
    #                             self._client.__aexit__(None, None, None), self._background_loop
    #                         )
    #                         future.result(timeout=0.5)
    #                 except Exception:
    #                     pass
    #                 try:
    #                     self._background_loop.call_soon_threadsafe(self._background_loop.stop)
    #                 except Exception:
    #                     pass
    #             if self._thread and self._thread.is_alive():
    #                 self._thread.join(timeout=0.5)
    #     except Exception:
    #         pass

    def submit_task(
        self,
        model_name: str = None,
        task: str = None,
        data=None,
        json=None,
        files=None,
        params=None,
        headers=None,
        wait_for_completion: bool = False,
        return_result: bool = True,
    ):
        """Submit a task to the Gateway and return the task info."""
        return self._run_async(
            self._client.submit_task(
                model_name=model_name,
                task=task,
                data=data,
                json=json,
                files=files,
                params=params,
                headers=headers,
                wait_for_completion=wait_for_completion,
                return_result=return_result,
            )
        )

    def process(
        self,
        text: str,
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Generate annotations for the provided text."""
        return self._run_async(
            self._client.process(
                text=text,
                model_name=model_name,
                wait_for_completion=wait_for_completion,
                return_result=return_result,
            )
        )

    def process_bulk(
        self,
        texts: list[str],
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Generate annotations for a list of texts."""
        return self._run_async(
            self._client.process_bulk(
                texts=texts,
                model_name=model_name,
                wait_for_completion=wait_for_completion,
                return_result=return_result,
            )
        )

    def redact(
        self,
        text: str,
        concepts_to_keep: Iterable[str] = None,
        warn_on_no_redaction: bool = None,
        mask: str = None,
        hash: bool = None,
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Redact sensitive information from the provided text."""
        return self._run_async(
            self._client.redact(
                text=text,
                concepts_to_keep=concepts_to_keep,
                warn_on_no_redaction=warn_on_no_redaction,
                mask=mask,
                hash=hash,
                model_name=model_name,
                wait_for_completion=wait_for_completion,
                return_result=return_result,
            )
        )

    def get_task(self, task_uuid: str, detail: bool = True):
        """Get a Gateway task details by its UUID."""
        return self._run_async(self._client.get_task(task_uuid=task_uuid, detail=detail))

    def get_task_result(self, task_uuid: str, parse: bool = True):
        """Get the result of a Gateway task by its UUID.

        If parse is True, try to infer and parse the result as JSON, JSONL, or text.
        Otherwise, return raw bytes.
        """
        return self._run_async(self._client.get_task_result(task_uuid=task_uuid, parse=parse))

    def wait_for_task(self, task_uuid: str, detail: bool = True, raise_on_error: bool = False):
        """Poll Gateway until the task reaches a final state."""
        return self._run_async(
            self._client.wait_for_task(
                task_uuid=task_uuid, detail=detail, raise_on_error=raise_on_error
            )
        )

    def get_models(self, verbose: bool = False):
        """Get the list of available models from the Gateway."""
        return self._run_async(self._client.get_models(verbose=verbose))

    def get_model(self, model_name: str = None):
        """Get details of a specific model."""
        return self._run_async(self._client.get_model(model_name=model_name))

    def deploy_model(
        self,
        model_name: str = None,
        tracking_id: str = None,
        model_uri: str = None,
        ttl: int = None,
    ):
        """Deploy a CogStack Model Serve model through the Gateway."""
        return self._run_async(
            self._client.deploy_model(
                model_name=model_name,
                tracking_id=tracking_id,
                model_uri=model_uri,
                ttl=ttl,
            )
        )
