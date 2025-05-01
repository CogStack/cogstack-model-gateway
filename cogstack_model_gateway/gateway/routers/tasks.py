import io
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from cogstack_model_gateway.common.config import Config, get_config
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.tasks import TaskManager

router = APIRouter()


@router.get(
    "/tasks/",
    tags=["tasks"],
    name="List all tasks created through the CogStack Model Gateway",
)
async def get_tasks():
    """List all tasks (not implemented)."""
    # FIXME: Implement authn/authz
    raise HTTPException(status_code=403, detail="Only admins can list tasks")


@router.get(
    "/tasks/{task_uuid}",
    tags=["tasks"],
    name="Get a task created through the CogStack Model Gateway by its UUID",
)
async def get_task_by_uuid(
    task_uuid: str,
    config: Annotated[Config, Depends(get_config)],
    detail: bool = Query(False),
    download: bool = Query(False),
):
    """Get a task by its UUID.

    This endpoint retrieves a task created through the CogStack Model Gateway by its UUID. If
    `download` is True, the result of the task is returned as a downloadable file. If not, the task
    status with optional details is returned in JSON format: if `detail` is False, only the task
    UUID and status are returned. Otherwise, the full details are included in the response (e.g.
    tracking ID, result reference, error message).
    """
    tm: TaskManager = config.task_manager
    osm: ObjectStoreManager = config.results_object_store_manager
    task = tm.get_task(task_uuid)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_uuid}' not found")

    if download and task.result:
        return StreamingResponse(
            io.BytesIO(osm.get_object(task.result)),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={task.result}",
                "X-Task-UUID": task.uuid,
                "X-Task-Status": task.status,
            },
        )

    return task if detail else {"uuid": task.uuid, "status": task.status}
