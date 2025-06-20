from prometheus_client import Counter

TASKS_COMMON_LABELS = ["model", "type", "source", "created_at", "started_at", "finished_at"]


tasks_scheduled_total = Counter(
    "scheduler_tasks_scheduled_total",
    "Total number of tasks scheduled by the scheduler",
    [*TASKS_COMMON_LABELS],
)

tasks_completed_total = Counter(
    "scheduler_tasks_completed_total",
    "Total number of tasks completed by the scheduler",
    [*TASKS_COMMON_LABELS, "status"],
)

tasks_requeued_total = Counter(
    "scheduler_tasks_requeued_total",
    "Total number of tasks requeued by the scheduler",
    [*TASKS_COMMON_LABELS],
)
