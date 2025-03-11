import logging
import time

from requests import Response, request

from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, Task, TaskManager
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
        self.queue_manager.consume(self.process_task)

    def process_task(self, task: dict, ack: callable, nack: callable) -> None:
        task_uuid = task["uuid"]
        log.info(f"Processing task '{task_uuid}'")

        self.task_manager.update_task(
            task_uuid, status=Status.SCHEDULED, expected_status=Status.PENDING
        )
        res, err_msg = self.route_task(task)
        task_obj = self.handle_server_response(task_uuid, res, err_msg, ack, nack)
        self.send_notification(task_obj)

    def route_task(self, task: dict) -> tuple[Response, str]:
        log.info(f"Routing task '{task['uuid']}' to model server at {task['url']}")
        req = self._prepare_request(task)
        response = None
        try:
            log.debug(f"Request: {req}")
            response = request(
                method=req["method"],
                url=req["url"],
                headers=req["headers"],
                params=req["params"],
                data=req["data"],
                files=req["files"],
            )
            log.debug(f"Response: {response.text}")
            response.raise_for_status()
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
        if response is None or response.status_code >= 400:
            return self._handle_task_failure(task_uuid, response, err_msg, nack)
        else:
            return self._handle_task_success(task_uuid, response, ack)

    def poll_task_status(self, task_uuid: str, tracking_id: str = None) -> dict:
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
        # FIXME: notify user if task is completed
        if task.status.is_final():
            log.info(f"Task '{task.uuid}' {task.status.value}: {task.result or task.error_message}")

    def _get_payload_from_refs(self, refs: list) -> str:
        if len(refs) > 1:
            raise ValueError(f"Payload references can't contain more than 1 object: {refs}")
        elif len(refs) == 0:
            return None

        ref = refs.pop()
        return self.task_object_store_manager.get_object(ref["key"]).decode()

    def _get_multipart_data_from_refs(self, refs: list) -> tuple:
        multipart_data, files = {}, []
        for ref in refs:
            if "part=file" in ref["content_type"]:
                file_content = self.task_object_store_manager.get_object(ref["key"])
                files.append((ref["field"], (ref["filename"], file_content)))
            else:
                multipart_data[ref["field"]] = ref["value"]
        return multipart_data, files

    def _prepare_request(self, task: dict) -> dict:
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
