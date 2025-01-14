import logging
import sys

from cogstack_model_gateway.common.config import load_config
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import TaskManager
from cogstack_model_gateway.scheduler.scheduler import Scheduler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cmg.scheduler")


def initialize_connections() -> (
    tuple[DatabaseManager, ObjectStoreManager, QueueManager, TaskManager]
):
    log.info("Initializing database and queue connections")
    config = load_config()
    dbm = DatabaseManager(
        user=config.env.db_user,
        password=config.env.db_password,
        host=config.env.db_host,
        port=config.env.db_port,
        db_name=config.env.db_name,
    )
    dbm.init_db()

    task_osm = ObjectStoreManager(
        host=config.env.object_store_host,
        port=config.env.object_store_port,
        access_key=config.env.object_store_access_key,
        secret_key=config.env.object_store_secret_key,
        default_bucket=config.env.object_store_bucket_tasks,
    )

    results_osm = ObjectStoreManager(
        host=config.env.object_store_host,
        port=config.env.object_store_port,
        access_key=config.env.object_store_access_key,
        secret_key=config.env.object_store_secret_key,
        default_bucket=config.env.object_store_bucket_results,
    )

    qm = QueueManager(
        user=config.env.queue_user,
        password=config.env.queue_password,
        host=config.env.queue_host,
        port=config.env.queue_port,
        queue_name=config.env.queue_name,
        max_concurrent_tasks=int(config.env.scheduler_max_concurrent_tasks),
    )
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    return dbm, task_osm, results_osm, qm, tm


def main():
    connections = initialize_connections()

    scheduler = Scheduler(
        task_object_store_manager=connections[1],
        results_object_store_manager=connections[2],
        queue_manager=connections[3],
        task_manager=connections[4],
    )
    scheduler.run()


if __name__ == "__main__":
    sys.exit(main())
