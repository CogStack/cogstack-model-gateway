import pytest
from fastapi.testclient import TestClient
from testcontainers.minio import MinioContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

from cogstack_model_gateway.gateway.main import app
from tests.integration.utils import (
    clone_cogstack_model_serve,
    configure_environment,
    remove_cogstack_model_serve,
    remove_testcontainers,
    start_cogstack_model_serve,
    start_scheduler,
    start_testcontainers,
    stop_cogstack_model_serve,
    stop_scheduler,
)

POSTGRES_IMAGE = "postgres:17.2"
RABBITMQ_IMAGE = "rabbitmq:4.0.4-management-alpine"
MINIO_IMAGE = "minio/minio:RELEASE.2024-11-07T00-52-20Z"


@pytest.fixture(scope="module", autouse=True)
def setup(request):
    postgres = PostgresContainer(POSTGRES_IMAGE)
    rabbitmq = RabbitMqContainer(RABBITMQ_IMAGE)
    minio = MinioContainer(MINIO_IMAGE)

    containers = [postgres, rabbitmq, minio]
    request.addfinalizer(lambda: remove_testcontainers(containers))

    start_testcontainers(containers)

    configure_environment(postgres, rabbitmq, minio)

    scheduler_process = start_scheduler()
    request.addfinalizer(lambda: stop_scheduler(scheduler_process))

    clone_cogstack_model_serve()
    request.addfinalizer(remove_cogstack_model_serve)

    cms_compose_envs = start_cogstack_model_serve()
    request.addfinalizer(lambda: stop_cogstack_model_serve(cms_compose_envs))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Enter the cult... I mean, the API."}
