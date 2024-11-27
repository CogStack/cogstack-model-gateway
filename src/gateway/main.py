from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from common.config import Config, config, get_config
from common.db import DatabaseManager
from common.queue import QueueManager
from common.tasks import Status, TaskManager
from gateway.core.priority import calculate_task_priority
from gateway.routers import tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config
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

    config.set("database_manager", dbm)
    config.set("queue_manager", qm)
    config.set("task_manager", tm)

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(tasks.router)


@app.get("/")
async def root():
    return {"message": "Enter the cult... I mean, the API."}


@app.post("/")
async def submit_task(request: Request, config: Annotated[Config, Depends(get_config)]):
    data = await request.json()

    priority = calculate_task_priority(data, config)

    tm: TaskManager = config.task_manager
    task_uuid = tm.create_task(Status.PENDING, priority)
    task = {"uuid": task_uuid, **data}

    qm: QueueManager = config.queue_manager
    qm.publish(task, priority)

    return {"uuid": task_uuid, "status": "Task submitted successfully"}
