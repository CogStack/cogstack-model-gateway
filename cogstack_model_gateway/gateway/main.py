import logging
import os
from contextlib import asynccontextmanager

import urllib3
from fastapi import FastAPI, HTTPException
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

from cogstack_model_gateway.common.config import get_config, load_config
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.logging import configure_logging
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import TaskManager
from cogstack_model_gateway.gateway.prometheus.metrics import gateway_requests_total
from cogstack_model_gateway.gateway.routers import models, tasks

log = logging.getLogger("cmg.gateway")


def make_metrics_app():
    """Create a registry for each process and aggregate metrics with MultiProcessCollector."""
    prometheus_multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus")
    os.makedirs(prometheus_multiproc_dir, exist_ok=True)
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=prometheus_multiproc_dir)
    return make_asgi_app(registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup gateway and initialize database, object store, queue, and task manager connections."""
    configure_logging()
    log.info("Initializing database and queue connections")

    config = load_config()
    dbm = DatabaseManager(
        user=config.cmg.db_user,
        password=config.cmg.db_password,
        host=config.cmg.db_host,
        port=config.cmg.db_port,
        db_name=config.cmg.db_name,
    )
    dbm.init_db()

    task_osm = ObjectStoreManager(
        host=config.cmg.object_store_host,
        port=config.cmg.object_store_port,
        access_key=config.cmg.object_store_access_key,
        secret_key=config.cmg.object_store_secret_key,
        default_bucket=config.cmg.object_store_bucket_tasks,
    )

    results_osm = ObjectStoreManager(
        host=config.cmg.object_store_host,
        port=config.cmg.object_store_port,
        access_key=config.cmg.object_store_access_key,
        secret_key=config.cmg.object_store_secret_key,
        default_bucket=config.cmg.object_store_bucket_results,
    )

    qm = QueueManager(
        user=config.cmg.queue_user,
        password=config.cmg.queue_password,
        host=config.cmg.queue_host,
        port=config.cmg.queue_port,
        queue_name=config.cmg.queue_name,
    )
    qm.init_queue()

    tm = TaskManager(db_manager=dbm)

    config.set("database_manager", dbm)
    config.set("task_object_store_manager", task_osm)
    config.set("results_object_store_manager", results_osm)
    config.set("queue_manager", qm)
    config.set("task_manager", tm)

    yield


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(lifespan=lifespan)
app.include_router(models.router)
app.include_router(tasks.router)

app.mount("/metrics", make_metrics_app())


@app.middleware("http")
async def prometheus_request_counter(request, call_next):
    response = await call_next(request)
    gateway_requests_total.labels(method=request.method, endpoint=request.url.path).inc()
    return response


@app.get("/")
async def root():
    """Root endpoint for the gateway API."""
    return {"message": "Enter the cult... I mean, the API."}


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies the status of critical components."""
    try:
        config = get_config()

        components_to_check = {
            "database": config.database_manager,
            "task_object_store": config.task_object_store_manager,
            "results_object_store": config.results_object_store_manager,
            "queue": config.queue_manager,
        }

        component_status = {
            name: "healthy" if manager.health_check() else "unhealthy"
            for name, manager in components_to_check.items()
        }

        overall_status = (
            "healthy"
            if all(status == "healthy" for status in component_status.values())
            else "unhealthy"
        )

        health_status = {"status": overall_status, "components": component_status}

        if overall_status == "unhealthy":
            raise HTTPException(status_code=503, detail=health_status)

        return health_status

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "error": f"Failed to perform health check: {str(e)}"},
        )
