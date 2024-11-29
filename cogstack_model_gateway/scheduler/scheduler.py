import logging
import time

from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, Task, TaskManager

log = logging.getLogger("cmg.scheduler")


class Scheduler:
    def __init__(
        self, db_manager: DatabaseManager, queue_manager: QueueManager, task_manager: TaskManager
    ):
        self.db_manager = db_manager
        self.queue_manager = queue_manager
        self.task_manager = task_manager

    def process_task(self, task: dict):
        task_uuid = task["uuid"]
        log.info(f"Processing task '{task_uuid}'")

        self.task_manager.update_task(
            task_uuid, status=Status.RUNNING, expected_status=Status.PENDING
        )
        self.route_task(task)
        self.poll_task_status(task_uuid)

    def poll_task_status(self, task_uuid: str) -> None:
        while True:
            task = self.task_manager.get_task(task_uuid)
            if task.status in (Status.SUCCEEDED, Status.FAILED):
                self.send_notification(task)
                return
            time.sleep(5)

    def route_task(self, task: dict) -> None:
        # FIXME: Route task to a model server
        log.info(f"Routing task '{task['uuid']}' to model server at {task["model_server_url"]}")

    def send_notification(self, task: Task):
        # FIXME: notify user
        log.info(f"Task '{task.uuid}' {task.status.value}: {task.result}")

    def run(self):
        self.queue_manager.consume(self.process_task)
