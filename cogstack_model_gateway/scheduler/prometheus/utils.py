from cogstack_model_gateway.common.tasks import Task


def get_task_labels(task: Task) -> dict:
    """Get label values for Prometheus metrics from a Task."""
    return (
        {
            "model": task.model or "N/A",
            "type": task.type or "N/A",
            "source": task.source or "N/A",
            "created_at": task.created_at or "N/A",
            "started_at": task.started_at or "N/A",
            "finished_at": task.finished_at or "N/A",
        }
        if task is not None
        else {
            "model": "N/A",
            "type": "N/A",
            "source": "N/A",
            "created_at": "N/A",
            "started_at": "N/A",
            "finished_at": "N/A",
        }
    )
