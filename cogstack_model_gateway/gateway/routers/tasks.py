import io
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from cogstack_model_gateway.common.config import Config, get_config
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.tasks import TaskManager

router = APIRouter()


@router.get("/tasks/", tags=["tasks"])
async def get_tasks():
    # FIXME: Implement authn/authz
    raise HTTPException(status_code=403, detail="Only admins can list tasks")


@router.get("/tasks/{task_uuid}", tags=["tasks"])
async def get_task_by_uuid(
    task_uuid: str,
    config: Annotated[Config, Depends(get_config)],
    detail: bool = Query(False),
    download: bool = Query(False),
):
    tm: TaskManager = config.task_manager
    osm: ObjectStoreManager = config.results_object_store_manager
    task = tm.get_task(task_uuid)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_uuid}' not found")

    if not detail:
        return {"uuid": task.uuid, "status": task.status}

    if download and task.result:
        return StreamingResponse(
            io.BytesIO(osm.get_object(task.result)),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={task.result}"},
        )

    return task
