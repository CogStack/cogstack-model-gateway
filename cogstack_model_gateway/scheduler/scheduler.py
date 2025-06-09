import logging
import time

from requests import Response, request

from cogstack_model_gateway.common.exceptions import (
    retry_if_connection_error,
    retry_if_rate_limited,
    retry_if_timeout_error,
)
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, Task, TaskManager, UnexpectedStatusError
from cogstack_model_gateway.common.tracking import TrackingClient
from cogstack_model_gateway.common.utils import parse_content_type_header

log = logging.getLogger("cmg.scheduler")


class Scheduler:
    def __init__(
        self,
        task_object_store_manager: ObjectStoreManager,
        results_object_store_manager: ObjectStoreManager,
        queue_manager: QueueManager,
        task_manager: TaskManager,
    ):
        self.tracking_client = TrackingClient()
        self.task_object_store_manager = task_object_store_manager
        self.results_object_store_manager = results_object_store_manager
        self.queue_manager = queue_manager
        self.task_manager = task_manager

    def run(self):
        """Run the scheduler by starting to consume tasks from the queue."""
        self.queue_manager.consume(self.process_task)

    def process_task(self, task: dict, ack: callable, nack: callable) -> None:
        """Process a task by forwarding it to the model server and updating its status.

        Every task picked up from the queue for the first time is expected to be in the PENDING
        state. The task is then updated to SCHEDULED before being forwarded to the model server.
        """
        task_uuid = task["uuid"]
        log.info(f"Processing task '{task_uuid}'")

        try:
            self.task_manager.update_task(
                task_uuid, status=Status.SCHEDULED, expected_status=Status.PENDING
            )
        except UnexpectedStatusError as e:
            # Log a warning if a completed task is being reprocessed (we should never land here)
            # and make sure it's removed from the queue
            if e.status in {Status.SUCCEEDED, Status.FAILED}:
                log.warning(f"Task '{task_uuid}' is already completed with status '{e.status}'")
                ack()
                return
            else:
                log.error(f"Skipping task '{task_uuid}' (expected PENDING state): {e}")
                nack(requeue=False)
                return

        try:
            res, err_msg = self.route_task(task)
            task_obj = self.handle_server_response(task_uuid, res, err_msg, ack, nack)
            self.send_notification(task_obj)
        except Exception as e:
            err_msg = f"Unexpected error while processing task '{task_uuid}': {e}"
            log.error(err_msg)
            nack(requeue=False)
            self.task_manager.update_task(task_uuid, status=Status.FAILED, error_message=err_msg)
            return

    @retry_if_rate_limited
    @retry_if_connection_error
    @retry_if_timeout_error
    def _send_request_with_retries(self, req: dict) -> Response:
        """Send a request to the model server with retries for potentially transient errors."""
        log.debug(f"Request: {req}")
        # FIXME: Enable SSL verification when certificates are properly set up
        response = request(
            method=req["method"],
            url=req["url"],
            headers=req["headers"],
            params=req["params"],
            data=req["data"],
            files=req["files"],
            verify=False,
        )
        log.debug(f"Response: {response.text}")
        response.raise_for_status()
        return response

    def route_task(self, task: dict) -> tuple[Response, str]:
        """Route a task to the correct model server and return the response and error message."""
        log.info(f"Routing task '{task['uuid']}' to model server at {task['url']}")
        req = self._prepare_request(task)
        response = None
        try:
            response = self._send_request_with_retries(req)
            log.info(f"Task '{task['uuid']}' forwarded successfully to {task['url']}")
            return response, None
        except Exception as e:
            err_msg = f"Failed to forward task '{task['uuid']}': {e}"
            log.error(err_msg)
            return response, err_msg

    def handle_server_response(
        self,
        task_uuid: str,
        response: Response,
        err_msg: str,
        ack: callable,
        nack: callable,
    ) -> Task:
        """Handle the model server response on success or failure."""
        if response is None or response.status_code >= 400:
            return self._handle_task_failure(task_uuid, response, err_msg, nack)
        else:
            return self._handle_task_success(task_uuid, response, ack)

    def poll_task_status(self, task_uuid: str, tracking_id: str = None) -> dict:
        """Poll tracking server for the status of a task and return the results once finalized."""
        while True:
            tracking_id = tracking_id or self.task_manager.get_task(task_uuid).tracking_id
            task = self.tracking_client.get_task(tracking_id)
            if task is None:
                raise ValueError(f"Task '{task_uuid}' not found in tracking server")
            res = {"url": task.url, "error": task.get_exceptions()}
            if task.is_finished:
                return {"status": Status.SUCCEEDED, **res}
            elif task.is_failed or task.is_killed:
                return {"status": Status.FAILED, **res}
            else:
                # Task is scheduled or still running
                time.sleep(5)

    def send_notification(self, task: Task):
        """Send a notification to the user once a task is completed."""
        # FIXME: notify user if task is completed
        if task.status.is_final():
            log.info(f"Task '{task.uuid}' {task.status.value}: {task.result or task.error_message}")

    def _get_payload_from_refs(self, refs: list) -> str:
        """Extract a request payload from the task references."""
        if len(refs) > 1:
            raise ValueError(f"Payload references can't contain more than 1 object: {refs}")
        elif len(refs) == 0:
            return None

        ref = refs.pop()
        return self.task_object_store_manager.get_object(ref["key"]).decode()

    def _get_multipart_data_from_refs(self, refs: list) -> tuple:
        """Extract multipart data and files from the task references."""
        multipart_data, files = {}, []
        for ref in refs:
            if "part=file" in ref["content_type"]:
                file_content = self.task_object_store_manager.get_object(ref["key"])
                files.append((ref["field"], (ref["filename"], file_content)))
            else:
                multipart_data[ref["field"]] = ref["value"]
        return multipart_data, files

    def _prepare_request(self, task: dict) -> dict:
        """Prepare a request object from a task dictionary based on its content type."""
        payload, files = None, None
        content_type, _ = parse_content_type_header(task["content_type"])
        if content_type in ("text/plain", "application/x-ndjson", "application/json"):
            payload = self._get_payload_from_refs(task["refs"])
        elif content_type == "multipart/form-data":
            payload, files = self._get_multipart_data_from_refs(task["refs"])
        else:
            raise ValueError(f"Unsupported content type: {task['content_type']}")

        # Allow requests to set the content type header with the correct boundary for multipart data
        headers = {"Content-Type": task["content_type"]} if not files else None

        return {
            "method": task["method"],
            "url": task["url"],
            "params": task["params"],
            "data": payload,
            "files": files,
            "headers": headers,
        }

    def _handle_task_failure(
        self, task_uuid: str, response: Response, err_msg: str, nack: callable
    ) -> Task:
        """Handle empty or failed responses from the model server.

        If the response is empty, the request most likely never reached the server, so it's marked
        as failed and removed from the queue. If the response is not empty but the status code is
        503, we can deduce that a training task is already running on the model server, therefore
        the task is requeued for processing and its status is reset to PENDING. In any other case,
        the task is marked as failed and removed from the queue.
        """
        # FIXME: Add fine-grained error handling for different status codes
        if not response:
            nack(requeue=False)
            return self.task_manager.update_task(
                task_uuid, status=Status.FAILED, error_message=err_msg or "Failed to process task"
            )
        elif (
            response.status_code == 503
            and (experiment_id := response.json().get("experiment_id"))
            and (run_id := response.json().get("run_id"))
        ):
            warn_msg = (
                f"Task '{task_uuid}' wasn't accepted for processing: a training run is already in"
                f" progress (experiment_id={experiment_id}, run_id={run_id}). Requeuing task..."
            )
            log.warning(warn_msg)
            nack()
            return self.task_manager.update_task(
                task_uuid, status=Status.PENDING, error_message=warn_msg
            )
        else:
            log.error(f"Task '{task_uuid}' failed with unexpected error: {response.text}")
            nack(requeue=False)
            return self.task_manager.update_task(
                task_uuid, status=Status.FAILED, error_message=response.text
            )

    def _handle_task_success(self, task_uuid: str, response: Response, ack: callable) -> Task:
        """Handle successful responses from the model server.

        Successful HTTP responses are either 202 (Accepted) for long-running tasks (e.g. training)
        or 200 (OK) for short-running tasks (e.g. redaction). In the case of the latter, the
        response body already contains the results, which are uploaded to the object store before
        the task is marked as succeeded. For the former, the task is marked as running and the
        tracking server is polled for the task status until it's completed. Once the task is
        finalized, its results are uploaded to the object store and the task is marked as succeeded.
        """
        ack()
        if response.status_code == 202:
            log.info(f"Task '{task_uuid}' accepted for processing, waiting for results")
            tracking_id = response.json().get("run_id") if response.json() else None
            self.task_manager.update_task(
                task_uuid,
                status=Status.RUNNING,
                expected_status=Status.SCHEDULED,
                tracking_id=tracking_id,
            )

            results = self.poll_task_status(task_uuid, tracking_id)
            if results["status"] == Status.FAILED:
                log.error(f"Task '{task_uuid}' failed: {results['error']}")
                return self.task_manager.update_task(
                    task_uuid, status=Status.FAILED, error_message=str(results["error"])
                )
            else:
                log.info(f"Task '{task_uuid}' completed, writing results to object store")
                object_key = self.results_object_store_manager.upload_object(
                    results["url"].encode(), "results.url", prefix=task_uuid
                )
                return self.task_manager.update_task(
                    task_uuid, status=Status.SUCCEEDED, result=object_key
                )
        else:
            log.info(f"Task '{task_uuid}' completed, writing results to object store")
            object_key = self.results_object_store_manager.upload_object(
                response.content, "results.json", prefix=task_uuid
            )
            return self.task_manager.update_task(
                task_uuid, status=Status.SUCCEEDED, result=object_key
            )
