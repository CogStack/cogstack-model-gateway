from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from common.config import Config, get_config
from common.tasks import TaskManager

router = APIRouter()


@router.get("/tasks/", tags=["tasks"])
async def get_tasks():
    # FIXME: Implement authn/authz
    raise HTTPException(status_code=403, detail="Only admins can list tasks")


@router.get("/tasks/{task_uuid}", tags=["tasks"])
async def get_task_by_uuid(
    task_uuid: str, config: Annotated[Config, Depends(get_config)], detail: bool = Query(False)
):
    tm: TaskManager = config.task_manager
    task = tm.get_task(task_uuid)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return task if detail else {"uuid": task.task_uuid, "status": task.status}
