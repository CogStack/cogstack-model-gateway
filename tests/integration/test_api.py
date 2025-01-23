import pytest
from fastapi.testclient import TestClient

from cogstack_model_gateway.common.config import Config, load_config
from cogstack_model_gateway.common.tasks import Status, TaskManager
from cogstack_model_gateway.gateway.main import app
from tests.integration.utils import (
    ANNOTATION_FIELDS_JSON,
    ANNOTATION_FIELDS_JSONL,
    TEST_MODEL_SERVICE,
    configure_environment,
    download_result_object,
    setup_cms,
    setup_scheduler,
    setup_testcontainers,
    validate_api_response,
    verify_annotation_contains_keys,
    verify_queue_is_empty,
    verify_results_match_api_info,
    verify_task_payload_in_object_store,
    verify_task_submitted_successfully,
    wait_for_task_completion,
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
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_payload.txt"
    expected_payload = b"Patient diagnosed with kidney failure"
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager)

    assert "text" in parsed
    assert parsed["text"] == "Patient diagnosed with kidney failure"
    assert "annotations" in parsed
    assert len(parsed["annotations"]) == 1

    annotation = parsed["annotations"][0]
    verify_annotation_contains_keys(annotation, ANNOTATION_FIELDS_JSON)
    assert annotation["label_name"] == "Loss Of Kidney Function"

    verify_results_match_api_info(client, task, res)


def test_process_jsonl(client: TestClient, config: Config, test_model_service_ip: str):
    response = client.post(
        f"/models/{test_model_service_ip}/process_jsonl",
        data=(
            '{"name": "doc1", "text": "Patient diagnosed with kidney failure"}\n'
            '{"name": "doc2", "text": "Patient diagnosed with kidney failure again, what a week"}'
        ),
        headers={"Content-Type": "application/x-ndjson"},
    )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_payload.ndjson"
    expected_payload = (
        b'{"name": "doc1", "text": "Patient diagnosed with kidney failure"}\n'
        b'{"name": "doc2", "text": "Patient diagnosed with kidney failure again, what a week"}'
    )
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "jsonl")
    assert len(parsed) == 2

    for annotation in parsed:
        verify_annotation_contains_keys(annotation, ANNOTATION_FIELDS_JSONL)
        assert annotation["label_name"] == "Loss Of Kidney Function"

    verify_results_match_api_info(client, task, res)


def test_process_bulk(client: TestClient, config: Config, test_model_service_ip: str):
    response = client.post(
        f"/models/{test_model_service_ip}/process_bulk",
        json=[
            "Patient diagnosed with kidney failure",
            "Patient diagnosed with kidney failure again, what a week",
        ],
        headers={"Content-Type": "application/json"},
    )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_payload.json"
    expected_payload = (
        b'["Patient diagnosed with kidney failure",'
        b' "Patient diagnosed with kidney failure again, what a week"]'
    )
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager)

    assert len(parsed) == 2

    for doc in parsed:
        assert "text" in doc
        assert "annotations" in doc
        assert len(doc["annotations"]) == 1

        annotation = doc["annotations"][0]
        verify_annotation_contains_keys(annotation, ANNOTATION_FIELDS_JSON)
        assert annotation["label_name"] == "Loss Of Kidney Function"

    verify_results_match_api_info(client, task, res)
