import io
import json
import logging
import time

import requests

from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, Task, TaskManager

log = logging.getLogger("cmg.scheduler")


class Scheduler:
    def __init__(
        self,
        object_store_manager: ObjectStoreManager,
        queue_manager: QueueManager,
        task_manager: TaskManager,
    ):
        self.object_store_manager = object_store_manager
        self.queue_manager = queue_manager
        self.task_manager = task_manager

    def process_task(self, task: dict, ack: callable, nack: callable) -> None:
        # FIXME: Handle ACK and NACK appropriately
        task_uuid = task["uuid"]
        log.info(f"Processing task '{task_uuid}'")

        self.task_manager.update_task(
            task_uuid, status=Status.RUNNING, expected_status=Status.PENDING
        )
        self.route_task(task)
        ack()
        self.poll_task_status(task_uuid)

    def poll_task_status(self, task_uuid: str) -> None:
        while True:
            task = self.task_manager.get_task(task_uuid)
            if task.status in (Status.SUCCEEDED, Status.FAILED):
                self.send_notification(task)
                return
            time.sleep(5)

    def _get_payload_from_refs(self, refs: list) -> str:
        if len(refs) > 1:
            raise ValueError(f"Payload references can't contain more than 1 object: {refs}")
        elif len(refs) == 0:
            return None

        ref = refs.pop()
        payload = self.object_store_manager.get_object(ref["key"])
        return json.dumps(payload) if ref["content_type"] == "application/json" else str(payload)

    def _get_multipart_data_from_refs(self, refs: list) -> tuple:
        multipart_data, files = {}, []
        for ref in refs:
            if "file" in ref["content_type"]:
                file_content = self.object_store_manager.get_object(ref["key"])
                files.append((ref["field"], (ref["filename"], io.BytesIO(file_content))))
            else:
                multipart_data[ref["field"]] = ref["value"]
        return multipart_data, files

    def _prepare_request(self, task: dict) -> dict:
        payload = None
        files = None
        if task["content_type"] in ("text/plain", "application/x-ndjson", "application/json"):
            payload = self._get_payload_from_refs(task["refs"])
        elif task["content_type"] == "multipart/form-data":
            payload, files = self._get_multipart_data_from_refs(task["refs"])
        else:
            raise ValueError(f"Unsupported content type: {task['content_type']}")

        return {
            "method": task["method"],
            "url": task["url"],
            "params": task["params"],
            "data": payload,
            "files": files,
            "headers": {"Content-Type": task["content_type"]},
        }

    def route_task(self, task: dict) -> None:
        log.info(f"Routing task '{task['uuid']}' to model server at {task['url']}")
        request = self._prepare_request(task)

        try:
            log.info(f"Request: {request}")
            response = requests.request(
                method=request["method"],
                url=request["url"],
                headers=request["headers"],
                params=request["params"],
                data=request["data"],
                files=request["files"],
            )
            response.raise_for_status()
            log.info(f"Response: {response.text}")
            log.info(f"Task '{task['uuid']}' forwarded successfully to {task['url']}")
        except Exception as e:
            log.error(f"Failed to process task '{task['uuid']}']: {e}")

    def send_notification(self, task: Task):
        # FIXME: notify user
        log.info(f"Task '{task.uuid}' {task.status.value}: {task.result}")

    def run(self):
        self.queue_manager.consume(self.process_task)
