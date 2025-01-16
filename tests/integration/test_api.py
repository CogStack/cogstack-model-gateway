import json

import pytest
import requests
from fastapi.testclient import TestClient

from cogstack_model_gateway.common.config import Config, load_config
from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, TaskManager
from cogstack_model_gateway.gateway.main import app
from tests.integration.utils import (
    TEST_MODEL_SERVICE,
    configure_environment,
    setup_cms,
    setup_scheduler,
    setup_testcontainers,
)


@pytest.fixture(scope="module", autouse=True)
def setup(request: pytest.FixtureRequest, cleanup_cms: bool):
    postgres, rabbitmq, minio = setup_testcontainers(request)

    svc_addr_map = setup_cms(request, cleanup_cms)
    request.config.cache.set("TEST_MODEL_SERVICE_IP", svc_addr_map[TEST_MODEL_SERVICE]["address"])

    mlflow_addr = svc_addr_map["mlflow-ui"]["address"]
    mlflow_port = svc_addr_map["mlflow-ui"]["port"]
    env = {
        "MLFLOW_TRACKING_URI": f"http://{mlflow_addr}:{mlflow_port}",
    }
    configure_environment(postgres, rabbitmq, minio, extras=env)

    setup_scheduler(request)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def config(client: TestClient) -> Config:
    return load_config()


@pytest.fixture(scope="module")
def test_model_service_ip(request: pytest.FixtureRequest) -> str:
    return request.config.cache.get("TEST_MODEL_SERVICE_IP", None)


def test_config_loaded(config: Config):
    assert config
    assert all(
        key in config
        for key in [
            "database_manager",
            "task_object_store_manager",
            "results_object_store_manager",
            "queue_manager",
            "task_manager",
        ]
    )


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Enter the cult... I mean, the API."}


def test_get_tasks(client: TestClient):
    response = client.get("/tasks/")
    assert response.status_code == 403
    assert response.json() == {"detail": "Only admins can list tasks"}


def test_get_task_by_uuid(client: TestClient, config: Config):
    task_uuid = "nonexistent-uuid"
    response = client.get(f"/tasks/{task_uuid}")
    assert response.status_code == 404
    assert response.json() == {"detail": f"Task '{task_uuid}' not found"}

    tm: TaskManager = config.task_manager
    task_uuid = tm.create_task(status="pending")
    response = client.get(f"/tasks/{task_uuid}")
    assert response.status_code == 200
    assert response.json() == {"uuid": task_uuid, "status": "pending"}

    tm.update_task(task_uuid, status="succeeded", result="result.txt", error_message=None)
    response = client.get(f"/tasks/{task_uuid}", params={"detail": True})
    assert response.status_code == 200
    assert response.json() == {
        "uuid": task_uuid,
        "status": "succeeded",
        "result": "result.txt",
        "error_message": None,
        "tracking_id": None,
    }


def test_get_models(client: TestClient):
    response = client.get("/models/")
    assert response.status_code == 200

    response_json = response.json()
    assert isinstance(response_json, list)
    assert len(response_json) == 1
    assert all(key in response_json[0] for key in ["name", "uri"])
    assert response_json[0]["name"] == TEST_MODEL_SERVICE


def test_get_model_info(client: TestClient, test_model_service_ip: str):
    response = client.get(f"/models/{test_model_service_ip}/info")
    assert response.status_code == 200
    assert all(
        key in response.json()
        for key in ["api_version", "model_type", "model_description", "model_card"]
    )


def test_unsupported_task(client: TestClient, test_model_service_ip: str):
    response = client.post(
        f"/models/{test_model_service_ip}/unsupported-task",
        headers={"Content-Type": "dummy"},
    )
    assert response.status_code == 404
    assert "Task 'unsupported-task' not found. Supported tasks are:" in response.json()["detail"]


def test_process(client: TestClient, config: Config, test_model_service_ip: str):
    response = client.post(
        f"/models/{test_model_service_ip}/process",
        data="Spinal stenosis",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert all(key in response_json for key in ["uuid", "status"])

    task_uuid = response_json["uuid"]
    tm: TaskManager = config.task_manager
    assert tm.get_task(task_uuid), "Failed to submit task: not found in the database"

    # Wait for the task to complete
    while (task := tm.get_task(task_uuid)).status != Status.SUCCEEDED:
        pass

    # Verify that the task payload was stored in the object store
    task_payload_key = f"{task_uuid}_payload.txt"
    tom: ObjectStoreManager = config.task_object_store_manager
    payload = tom.get_object(task_payload_key)
    assert payload == b"Spinal stenosis"

    # Verify that the queue is empty after the task is processed
    qm: QueueManager = config.queue_manager
    assert qm.is_queue_empty()

    # Verify task results
    assert task.error_message is None, f"Task failed unexpectedly: {task.error_message}"
    assert task.result is not None, "Task results are missing"

    rom: ObjectStoreManager = config.results_object_store_manager
    result = rom.get_object(task.result)

    try:
        result_json = json.loads(result.decode("utf-8"))
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse the result as JSON: {result}, {e}")

    assert result_json["text"] == "Spinal stenosis"
    assert len(result_json["annotations"]) == 1

    annotation = result_json["annotations"][0]
    assert all(
        key in annotation
        for key in [
            "start",
            "end",
            "label_name",
            "label_id",
            "categories",
            "accuracy",
            "meta_anns",
            "athena_ids",
        ]
    )
    assert annotation["label_name"] == "Spinal Stenosis"

    # Verify that the above match the information exposed through the user-facing API
    get_response = client.get(f"/tasks/{task_uuid}", params={"detail": True, "download_url": True})
    assert get_response.status_code == 200

    get_response_json = get_response.json()
    assert get_response_json["uuid"] == task.uuid
    assert get_response_json["status"] == task.status
    assert get_response_json["error_message"] is None
    assert get_response_json["tracking_id"] is None

    # Download results and verify they match the ones read from the object store
    download_results = requests.get(get_response_json["result"])
    assert download_results.status_code == 200
    assert download_results.content == result
