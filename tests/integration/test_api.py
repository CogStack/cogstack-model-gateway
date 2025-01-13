import os

import pytest
from fastapi.testclient import TestClient
from testcontainers.minio import MinioContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

from cogstack_model_gateway.gateway.main import app


@pytest.fixture(scope="module", autouse=True)
def setup(request):
    print("Setting up test containers")
    postgres = PostgresContainer("postgres:17.2")
    rabbitmq = RabbitMqContainer("rabbitmq:4.0.4-management-alpine")
    minio = MinioContainer("minio/minio:RELEASE.2024-11-07T00-52-20Z")

    def remove_containers():
        for testcontainer in [postgres, rabbitmq, minio]:
            testcontainer.stop()

    request.addfinalizer(remove_containers)

    for testcontainer in [postgres, rabbitmq, minio]:
        testcontainer.start()

    queue_connection_params = rabbitmq.get_connection_params()
    minio_host, minio_port = minio.get_config()["endpoint"].split(":")
    env = {
        "CMG_DB_USER": postgres.username,
        "CMG_DB_PASSWORD": postgres.password,
        "CMG_DB_HOST": postgres.get_container_host_ip(),
        "CMG_DB_PORT": postgres.get_exposed_port(postgres.port),
        "CMG_DB_NAME": "test",
        "CMG_QUEUE_USER": rabbitmq.username,
        "CMG_QUEUE_PASSWORD": rabbitmq.password,
        "CMG_QUEUE_HOST": queue_connection_params.host,
        "CMG_QUEUE_PORT": str(queue_connection_params.port),
        "CMG_QUEUE_NAME": "test",
        "CMG_OBJECT_STORE_HOST": minio_host,
        "CMG_OBJECT_STORE_PORT": minio_port,
        "CMG_OBJECT_STORE_ACCESS_KEY": minio.access_key,
        "CMG_OBJECT_STORE_SECRET_KEY": minio.secret_key,
        "CMG_OBJECT_STORE_BUCKET_TASKS": "test-tasks",
        "CMG_OBJECT_STORE_BUCKET_RESULTS": "test-results",
        "CMG_SCHEDULER_MAX_CONCURRENT_TASKS": "1",
    }

    os.environ.update(env)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Enter the cult... I mean, the API."}
