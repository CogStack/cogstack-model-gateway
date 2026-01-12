import os
import shutil

from prometheus_client import multiprocess

from cogstack_model_gateway.gateway.prometheus.metrics import PROMETHEUS_MULTIPROC_DIR

bind = "0.0.0.0:8000"

workers = 4
keepalive = 5
worker_class = "uvicorn.workers.UvicornWorker"


def on_starting(server):
    """Setup multiprocess directory before starting workers."""
    if os.path.exists(PROMETHEUS_MULTIPROC_DIR):
        shutil.rmtree(PROMETHEUS_MULTIPROC_DIR, ignore_errors=True)
    os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)


def child_exit(server, worker):
    """Mark the Prometheus metrics for this worker as dead before a worker exits.

    This function is called by Gunicorn when a worker process exits. It marks the process as dead in
    the Prometheus multiprocess registry, allowing MultiProcessCollector to ignore its old files.
    """
    multiprocess.mark_process_dead(worker.pid)
