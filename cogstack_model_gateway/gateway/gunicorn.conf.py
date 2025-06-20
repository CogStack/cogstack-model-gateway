from prometheus_client import multiprocess

bind = "0.0.0.0:8000"

workers = 4

worker_class = "uvicorn.workers.UvicornWorker"


def child_exit(server, worker):
    """Mark the Prometheus metrics for this worker as dead before a worker exits.

    This function is called by Gunicorn when a worker process exits. It marks the process as dead in
    the Prometheus multiprocess registry, allowing MultiProcessCollector to ignore its old files.
    """
    multiprocess.mark_process_dead(worker.pid)
