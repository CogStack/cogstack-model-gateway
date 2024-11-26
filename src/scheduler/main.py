import sys

from common.config import config
from common.db import DatabaseManager
from common.queue import QueueManager
from common.tasks import TaskManager
from scheduler.scheduler import Scheduler


def initialize_connections() -> tuple[DatabaseManager, QueueManager, TaskManager]:
    dbm = DatabaseManager(database_url=config.db.url)
    dbm.init_db()

    qm = QueueManager(queue_name=config.rabbitmq.queue, url=config.rabbitmq.url)
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    return dbm, qm, tm


def main():
    db_manager, queue_manager, task_manager = initialize_connections()

    scheduler = Scheduler(
        db_manager=db_manager, queue_manager=queue_manager, task_manager=task_manager
    )
    scheduler.run()


if __name__ == "__main__":
    sys.exit(main())
