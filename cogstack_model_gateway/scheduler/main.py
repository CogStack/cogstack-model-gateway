import logging
import sys

from cogstack_model_gateway.common.config import load_config
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.logging import configure_logging
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import TaskManager
from cogstack_model_gateway.scheduler.scheduler import Scheduler

log = logging.getLogger("cmg.scheduler")


def initialize_connections() -> tuple[
    DatabaseManager, ObjectStoreManager, QueueManager, TaskManager
]:
    """Initialize database, object store, queue, and task manager connections for the scheduler."""
    log.info("Initializing database and queue connections")
    config = load_config()
    dbm = DatabaseManager(
        user=config.cmg.db_user,
        password=config.cmg.db_password,
        host=config.cmg.db_host,
        port=config.cmg.db_port,
        db_name=config.cmg.db_name,
    )
    dbm.init_db()

    task_osm = ObjectStoreManager(
        host=config.cmg.object_store_host,
        port=config.cmg.object_store_port,
        access_key=config.cmg.object_store_access_key,
        secret_key=config.cmg.object_store_secret_key,
        default_bucket=config.cmg.object_store_bucket_tasks,
    )

    results_osm = ObjectStoreManager(
        host=config.cmg.object_store_host,
        port=config.cmg.object_store_port,
        access_key=config.cmg.object_store_access_key,
        secret_key=config.cmg.object_store_secret_key,
        default_bucket=config.cmg.object_store_bucket_results,
    )

    qm = QueueManager(
        user=config.cmg.queue_user,
        password=config.cmg.queue_password,
        host=config.cmg.queue_host,
        port=config.cmg.queue_port,
        queue_name=config.cmg.queue_name,
        max_concurrent_tasks=int(config.cmg.scheduler_max_concurrent_tasks),
    )
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    return dbm, task_osm, results_osm, qm, tm


def main():
    """Run the scheduler service."""
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    configure_logging()
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
