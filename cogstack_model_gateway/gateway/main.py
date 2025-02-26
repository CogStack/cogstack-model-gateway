import logging
from contextlib import asynccontextmanager

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
    configure_logging()
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
    )
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    config.set("database_manager", dbm)
    config.set("task_object_store_manager", task_osm)
    config.set("results_object_store_manager", results_osm)
    config.set("queue_manager", qm)
    config.set("task_manager", tm)

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(models.router)
app.include_router(tasks.router)


@app.get("/")
async def root():
    return {"message": "Enter the cult... I mean, the API."}
