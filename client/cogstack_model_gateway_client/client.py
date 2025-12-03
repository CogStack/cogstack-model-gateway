import asyncio
import json
import warnings
from collections.abc import Iterable
from functools import wraps

import httpx

from cogstack_model_gateway_client.exceptions import TaskFailedError, retry_if_network_error


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        default_model: str = None,
        request_timeout: float = 300.0,
        polling_interval: float = 2.0,
        polling_timeout: float | None = None,
    ):
        """Initialize the GatewayClient with the base Gateway URL and optional parameters.

        Args:
            base_url (str): The base URL of the Gateway service.
            default_model (str, optional): The default model to use for tasks. Defaults to None.
            request_timeout (float, optional): The HTTP request timeout in seconds for individual
                requests to the Gateway. Defaults to 300.0 seconds to accommodate slower requests,
                e.g. the ones triggering model auto-deployment, but should be adjusted as needed.
            polling_interval (float, optional): The interval in seconds to poll for task completion.
                Defaults to 2.0 seconds, with a minimum of 0.5 and maximum of 3.0 seconds.
            polling_timeout (float, optional): The client polling timeout while waiting for task
                completion. Defaults to None (no timeout). When set to a float value, a TimeoutError
                will be raised if the task takes longer than the specified number of seconds. When
                None, the client will wait indefinitely for task completion.
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.request_timeout = request_timeout
        self.polling_interval = polling_interval
        self.polling_timeout = polling_timeout
        self._client = None

    @property
    def polling_interval(self):
        return self._polling_interval

    @polling_interval.setter
    def polling_interval(self, value: float):
        self._polling_interval = max(0.5, min(value, 3.0))

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.request_timeout)
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
        """Submit a task to the Gateway and return the task info.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
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
                if task_info.get("status") == "succeeded":
                    return await self.get_task_result(task_uuid)
                else:
                    error_message = task_info.get("error_message", "Unknown error")
                    raise TaskFailedError(task_uuid, error_message, task_info)
        return task_info

    async def process(
        self,
        text: str,
        model_name: str = None,
        wait_for_completion: bool = True,
        return_result: bool = True,
    ):
        """Generate annotations for the provided text.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
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
        """Generate annotations for a list of texts.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
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
        """Redact sensitive information from the provided text.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
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
        """Poll Gateway until the task reaches a final state.

        Raises:
            TaskFailedError: If raise_on_error=True and the task fails.
            TimeoutError: If timeout is reached before task completion.
        """
        start = asyncio.get_event_loop().time()
        while True:
            task = await self.get_task(task_uuid, detail=detail)
            status = task.get("status")
            if status in ("succeeded", "failed"):
                if status == "failed" and raise_on_error:
                    error_message = task.get("error_message", "Unknown error")
                    raise TaskFailedError(task_uuid, error_message, task)
                return task
            if (
                self.polling_timeout is not None
                and asyncio.get_event_loop().time() - start > self.polling_timeout
            ):
                raise TimeoutError(f"Timed out waiting for task '{task_uuid}' to complete")
            await asyncio.sleep(self.polling_interval)

    @require_client
    async def get_models(self, verbose: bool = False):
        """Get the list of available models from the Gateway.

        Returns a dict with 'running' and 'on_demand' lists.
        When verbose=False: each model includes name, uri, is_running.
        When verbose=True: additionally includes description, model_type, deployment_type,
                           idle_ttl, resources, tracking, and runtime (for running models).
        """
        url = f"{self.base_url}/models/"
        resp = await self._request("GET", url, params={"verbose": verbose})
        return resp.json()

    @require_client
    async def get_model(self, model_name: str = None, verbose: bool = False):
        """Get details of a specific model.

        Args:
            model_name: Name of the model. Uses default_model if not provided.
            verbose: Include tracking metadata and CMS info (for running models).

        Returns:
            When verbose=False: dict with name, uri, is_running.
            When verbose=True: additionally includes description, model_type, deployment_type,
                               idle_ttl, resources, tracking, and runtime (for running models).
        """
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("Please provide a model name or set a default model for the client.")
        url = f"{self.base_url}/models/{model_name}"
        resp = await self._request("GET", url, params={"verbose": verbose})
        return resp.json()

    @require_client
    async def get_model_info(self, model_name: str = None):
        """Get CMS /info endpoint response for a running model.

        This mirrors the CMS /info endpoint and may trigger auto-deployment
        for on-demand models.

        Args:
            model_name: Name of the model. Uses default_model if not provided.
        """
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

    @require_client
    async def health_check(self):
        """Check if the Gateway and its components are healthy and responsive.

        Returns:
            dict: Health status information with 'status' and 'components' fields.
                  Status will be 'healthy' or 'unhealthy'.

        Raises:
            httpx.HTTPStatusError: For HTTP errors other than 503 Service Unavailable.
            Other exceptions: For network errors, timeouts, etc.
        """
        url = f"{self.base_url}/health"
        try:
            resp = await self._request("GET", url)
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                try:
                    return e.response.json()
                except Exception:
                    return {"status": "unhealthy", "error": "Service unavailable"}
            raise

    @require_client
    async def is_healthy(self):
        """Check if the Gateway and its components are healthy.

        Returns:
            bool: True if overall status is 'healthy', False otherwise.
        """
        try:
            health_data = await self.health_check()
            return health_data.get("status") == "healthy"
        except Exception:
            return False


class GatewayClientSync:
    """A simplified synchronous wrapper around the async GatewayClient.

    This client is intended for simple scripts or environments where async context management is not
    feasible. It uses asyncio.run() to manage the event loop for each operation, which may not be
    suitable for all use cases (e.g. when an event loop is running in the same thread).
    """

    def __init__(self, *args, **kwargs):
        async_ctx_err = RuntimeError(
            "GatewayClientSync can't be used inside an async context: use GatewayClient instead."
        )
        try:
            asyncio.get_running_loop()
            raise async_ctx_err
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise

        is_ipython, _ = self._is_running_in_ipython_or_jupyter()
        if is_ipython:
            self._warn_about_ipython_usage()

        self._client = GatewayClient(*args, **kwargs)
        self._initialized = False

        try:
            asyncio.run(self._client.__aenter__())
            self._initialized = True
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                raise async_ctx_err
            else:
                raise

    def _is_running_in_ipython_or_jupyter(self):
        """Detect if client is running inside an IPython or Jupyter environment."""
        try:
            import IPython

            ipython_instance = IPython.get_ipython()
            if ipython_instance is not None:
                if hasattr(ipython_instance, "kernel"):
                    return True, "Jupyter"
                else:
                    return True, "IPython"
        except ImportError:
            pass

        try:
            import sys

            if "ipykernel" in sys.modules:
                return True, "Jupyter"
        except ImportError:
            pass

        return False, None

    def _warn_about_ipython_usage(self):
        """Issue a warning about using GatewayClientSync in IPython/Jupyter environments."""
        warnings.warn(
            "You are using GatewayClientSync in an IPython/Jupyter environment."
            " This may cause 'RuntimeError: Event loop is closed' on subsequent calls."
            " Consider using the async GatewayClient with 'await' syntax instead:\n\n"
            "  async with GatewayClient(...) as client:\n"
            "      result = await client.process(text)\n\n"
            "Or use nest_asyncio to allow nested event loops:\n\n"
            "  import nest_asyncio\n"
            "  nest_asyncio.apply()\n"
            "  client = GatewayClientSync(...)\n",
            UserWarning,
            stacklevel=3,
        )

    def __del__(self):
        if hasattr(self, "_client") and self._client and getattr(self, "_initialized", False):
            try:
                asyncio.run(self._client.__aexit__(None, None, None))
            except Exception:
                pass  # Ignore cleanup errors

    @property
    def base_url(self):
        return self._client.base_url

    @base_url.setter
    def base_url(self, value: str):
        self._client.base_url = value

    @property
    def default_model(self):
        return self._client.default_model

    @default_model.setter
    def default_model(self, value: str):
        self._client.default_model = value

    @property
    def request_timeout(self):
        return self._client.request_timeout

    @request_timeout.setter
    def request_timeout(self, value: float | None):
        self._client.request_timeout = value

    @property
    def polling_interval(self):
        return self._client.polling_interval

    @polling_interval.setter
    def polling_interval(self, value: float):
        self._client.polling_interval = value

    @property
    def polling_timeout(self):
        return self._client.polling_timeout

    @polling_timeout.setter
    def polling_timeout(self, value: float | None):
        self._client.polling_timeout = value

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
        """Submit a task to the Gateway and return the task info.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
        return asyncio.run(
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
        """Generate annotations for the provided text.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
        return asyncio.run(
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
        """Generate annotations for a list of texts.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
        return asyncio.run(
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
        """Redact sensitive information from the provided text.

        Raises:
            TaskFailedError: If the task fails and wait_for_completion=True, return_result=True.
        """
        return asyncio.run(
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
        return asyncio.run(self._client.get_task(task_uuid=task_uuid, detail=detail))

    def get_task_result(self, task_uuid: str, parse: bool = True):
        """Get the result of a Gateway task by its UUID.

        If parse is True, try to infer and parse the result as JSON, JSONL, or text.
        Otherwise, return raw bytes.
        """
        return asyncio.run(self._client.get_task_result(task_uuid=task_uuid, parse=parse))

    def wait_for_task(self, task_uuid: str, detail: bool = True, raise_on_error: bool = False):
        """Poll Gateway until the task reaches a final state.

        Raises:
            TaskFailedError: If raise_on_error=True and the task fails.
            TimeoutError: If timeout is reached before task completion.
        """
        return asyncio.run(
            self._client.wait_for_task(
                task_uuid=task_uuid, detail=detail, raise_on_error=raise_on_error
            )
        )

    def get_models(self, verbose: bool = False):
        """Get the list of available models from the Gateway.

        Returns a dict with 'running' and 'on_demand' lists.
        When verbose=False: each model includes name, uri, is_running.
        When verbose=True: additionally includes description, model_type, deployment_type,
                           idle_ttl, resources, tracking, and runtime (for running models).
        """
        return asyncio.run(self._client.get_models(verbose=verbose))

    def get_model(self, model_name: str = None, verbose: bool = False):
        """Get details of a specific model.

        Args:
            model_name: Name of the model. Uses default_model if not provided.
            verbose: Include tracking metadata and CMS info (for running models).

        Returns:
            When verbose=False: dict with name, uri, is_running.
            When verbose=True: additionally includes description, model_type, deployment_type,
                               idle_ttl, resources, tracking, and runtime (for running models).
        """
        return asyncio.run(self._client.get_model(model_name=model_name, verbose=verbose))

    def get_model_info(self, model_name: str = None):
        """Get CMS /info endpoint response for a running model.

        This mirrors the CMS /info endpoint and may trigger auto-deployment
        for on-demand models.

        Args:
            model_name: Name of the model. Uses default_model if not provided.
        """
        return asyncio.run(self._client.get_model_info(model_name=model_name))

    def deploy_model(
        self,
        model_name: str = None,
        tracking_id: str = None,
        model_uri: str = None,
        ttl: int = None,
    ):
        """Deploy a CogStack Model Serve model through the Gateway."""
        return asyncio.run(
            self._client.deploy_model(
                model_name=model_name,
                tracking_id=tracking_id,
                model_uri=model_uri,
                ttl=ttl,
            )
        )

    def health_check(self):
        """Check if the Gateway and its components are healthy and responsive.

        Returns:
            dict: Health status information with 'status' and 'components' fields.
                  Status will be 'healthy' or 'unhealthy'.

        Raises:
            httpx.HTTPStatusError: For HTTP errors other than 503 Service Unavailable.
            Other exceptions: For network errors, timeouts, etc.
        """
        return asyncio.run(self._client.health_check())

    def is_healthy(self):
        """Check if the Gateway and its components are healthy.

        Returns:
            bool: True if overall status is 'healthy', False otherwise.
        """
        return asyncio.run(self._client.is_healthy())
