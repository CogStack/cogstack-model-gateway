import logging
import sys

from prometheus_client import start_http_server

from cogstack_model_gateway.common.config import Config, load_config
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.logging import configure_logging
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import TaskManager
from cogstack_model_gateway.common.tracking import TrackingClient
from cogstack_model_gateway.scheduler.scheduler import Scheduler

log = logging.getLogger("cmg.scheduler")


def initialize_connections(
    config: Config,
) -> tuple[DatabaseManager, ObjectStoreManager, QueueManager, TaskManager]:
    """Initialize database, object store, queue, and task manager connections for the scheduler."""
    log.info("Initializing database and queue connections")
    dbm = DatabaseManager(
        user=config.db.user,
        password=config.db.password,
        host=config.db.host,
        port=config.db.port,
        db_name=config.db.name,
    )

    task_osm = ObjectStoreManager(
        host=config.object_store.host,
        port=config.object_store.port,
        access_key=config.object_store.access_key,
        secret_key=config.object_store.secret_key,
        default_bucket=config.object_store.bucket_tasks,
    )

    results_osm = ObjectStoreManager(
        host=config.object_store.host,
        port=config.object_store.port,
        access_key=config.object_store.access_key,
        secret_key=config.object_store.secret_key,
        default_bucket=config.object_store.bucket_results,
    )

    qm = QueueManager(
        user=config.queue.user,
        password=config.queue.password,
        host=config.queue.host,
        port=config.queue.port,
        queue_name=config.queue.name,
        max_concurrent_tasks=config.scheduler.max_concurrent_tasks,
    )
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    tc = TrackingClient(
        tracking_uri=config.tracking.uri,
        username=config.tracking.username,
        password=config.tracking.password,
        s3_endpoint_url=config.tracking.s3.endpoint_url,
        s3_access_key_id=config.tracking.s3.access_key_id,
        s3_secret_access_key=config.tracking.s3.secret_access_key,
    )

    return dbm, task_osm, results_osm, qm, tm, tc


def main():
    """Run the scheduler service."""
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    configure_logging()
    config = load_config()
    connections = initialize_connections(config)

    start_http_server(config.scheduler.metrics_port)

    scheduler = Scheduler(
        task_object_store_manager=connections[1],
        results_object_store_manager=connections[2],
        queue_manager=connections[3],
        task_manager=connections[4],
        tracking_client=connections[5],
    )
    scheduler.run()


if __name__ == "__main__":
    sys.exit(main())
