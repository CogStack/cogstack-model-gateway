import logging
from contextlib import asynccontextmanager

import urllib3
from fastapi import FastAPI

from cogstack_model_gateway.common.config import load_config
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.logging import configure_logging
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import TaskManager
from cogstack_model_gateway.gateway.routers import models, tasks

log = logging.getLogger("cmg.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup gateway and initialize database, object store, queue, and task manager connections."""
    configure_logging()
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
    )
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    config.set("database_manager", dbm)
    config.set("task_object_store_manager", task_osm)
    config.set("results_object_store_manager", results_osm)
    config.set("queue_manager", qm)
    config.set("task_manager", tm)

    yield


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(lifespan=lifespan)
app.include_router(models.router)
app.include_router(tasks.router)


@app.get("/")
async def root():
    """Root endpoint for the gateway API."""
    return {"message": "Enter the cult... I mean, the API."}
