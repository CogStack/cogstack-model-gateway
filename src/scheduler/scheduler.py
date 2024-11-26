import time

import requests

from common.db import DatabaseManager
from common.queue import QueueManager
from common.tasks import Status, Task, TaskManager


class Scheduler:
    def __init__(
        self, db_manager: DatabaseManager, queue_manager: QueueManager, task_manager: TaskManager
    ):
        self.db_manager = db_manager
        self.queue_manager = queue_manager
        self.task_manager = task_manager

    def process_task(self, task: dict):
        task_uuid = task["uuid"]
        model_server_url = task["model_server_url"]

        self.task_manager.update_task(
            task_uuid, status=Status.RUNNING, expected_status=Status.PENDING
        )

        response = requests.post(model_server_url, json=task)
        if response.status_code != 200:
            self.task_manager.update_task(
                task_uuid, status=Status.FAILED, error_message=response.text
            )
            return

        self.poll_task_status(task_uuid)

    def poll_task_status(self, task_uuid: str) -> None:
        while True:
            task = self.task_manager.get_task(task_uuid)
            if task.status in (Status.SUCCEEDED, Status.FAILED):
                self.send_notification(task)
                return
            time.sleep(5)

    def send_notification(self, task: Task):
        # FIXME: notify user
        print(f"Task '{task.uuid}' {task.status}: {task.result}")

    def run(self):
        self.queue_manager.consume(self.process_task)
