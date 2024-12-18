import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from cogstack_model_gateway.common.config import Config, config, get_config
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, TaskManager
from cogstack_model_gateway.gateway.core.priority import calculate_task_priority
from cogstack_model_gateway.gateway.routers import models, tasks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cmg.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Initializing database and queue connections")

    global config
    dbm = DatabaseManager(
        user=config.env.db_user,
        password=config.env.db_password,
        host=config.env.db_host,
        port=config.env.db_port,
        db_name=config.env.db_name,
    )
    dbm.init_db()

    osm = ObjectStoreManager(
        host=config.env.object_store_host,
        port=config.env.object_store_port,
        access_key=config.env.object_store_access_key,
        secret_key=config.env.object_store_secret_key,
        default_bucket=config.env.object_store_bucket_tasks,
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
    config.set("object_store_manager", osm)
    config.set("queue_manager", qm)
    config.set("task_manager", tm)

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(models.router)
app.include_router(tasks.router)


@app.get("/")
async def root():
    return {"message": "Enter the cult... I mean, the API."}


@app.post("/")
async def submit_task(request: Request, config: Annotated[Config, Depends(get_config)]):
    data = await request.json()

    priority = calculate_task_priority(data, config)

    tm: TaskManager = config.task_manager
    task_uuid = tm.create_task(Status.PENDING)
    task = {"uuid": task_uuid, **data}

    qm: QueueManager = config.queue_manager
    qm.publish(task, priority)

    return {"uuid": task_uuid, "status": "Task submitted successfully"}
