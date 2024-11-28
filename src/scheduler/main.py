import logging
import sys

from common.config import config
from common.db import DatabaseManager
from common.queue import QueueManager
from common.tasks import TaskManager
from scheduler.scheduler import Scheduler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cmg.scheduler")


def initialize_connections() -> tuple[DatabaseManager, QueueManager, TaskManager]:
    log.info("Initializing database and queue connections")
    dbm = DatabaseManager(
        user=config.env.db_user,
        password=config.env.db_password,
        host=config.env.db_host,
        port=config.env.db_port,
        db_name=config.env.db_name,
    )
    dbm.init_db()

    qm = QueueManager(
        user=config.env.queue_user,
        password=config.env.queue_password,
        host=config.env.queue_host,
        port=config.env.queue_port,
        queue_name=config.env.queue_name,
    )
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
