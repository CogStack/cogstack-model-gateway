import concurrent.futures
import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cogstack_model_gateway.common.config import Config, load_config
from cogstack_model_gateway.common.models import ModelDeploymentType
from cogstack_model_gateway.common.tasks import Status, TaskManager
from cogstack_model_gateway.common.tracking import TrackingClient
from cogstack_model_gateway.gateway.main import app
from tests.integration.utils import (
    ANNOTATION_FIELDS_BASE,
    ANNOTATION_FIELDS_JSONL,
    TEST_CONFIG_FILE,
    TEST_MODEL_SERVICE,
    cleanup_deployed_model_containers,
    configure_environment,
    count_deployed_model_containers,
    download_result_object,
    get_deployed_model_container,
    parse_mlflow_url,
    setup_cms,
    setup_scheduler,
    setup_testcontainers,
    validate_api_response,
    verify_annotation_contains_keys,
    verify_container_labels,
    verify_queue_is_empty,
    verify_results_match_api_info,
    verify_task_payload_in_object_store,
    verify_task_submitted_successfully,
    wait_for_task_completion,
)

TEST_ASSETS_DIR = Path("tests/integration/assets")
MULTI_TEXT_FILE_PATH = TEST_ASSETS_DIR / "multi_text_file.json"
PUBLIC_KEY_PEM_PATH = TEST_ASSETS_DIR / "public_key.pem"
TRAINER_EXPORT_PATH = TEST_ASSETS_DIR / "trainer_export.json"
ANOTHER_TRAINER_EXPORT_PATH = TEST_ASSETS_DIR / "another_trainer_export.json"
CONCATENATED_TRAINER_EXPORTS_PATH = TEST_ASSETS_DIR / "concatenated_trainer_exports.json"
ANNOTATION_STATS_CSV_PATH = TEST_ASSETS_DIR / "annotation_stats.csv"


@pytest.fixture(scope="function", autouse=True)
def cleanup_test_deployments():
    """Clean up any CMG-managed model containers after each test."""
    yield
    cleanup_deployed_model_containers()


@pytest.fixture(scope="module", autouse=True)
def setup(request: pytest.FixtureRequest, cleanup_cms: bool):
    os.environ["CONFIG_FILE"] = str(TEST_CONFIG_FILE.absolute())

    postgres, rabbitmq, minio = setup_testcontainers(request)

    svc_addr_map = setup_cms(request, cleanup_cms)

    mlflow_addr = svc_addr_map["mlflow-ui"]["address"]
    mlflow_port = svc_addr_map["mlflow-ui"]["port"]
    mlflow_tracking_uri = f"http://{mlflow_addr}:{mlflow_port}"

    minio_addr = svc_addr_map["minio"]["address"]
    minio_port = svc_addr_map["minio"]["port"]
    mlflow_s3_endpoint_url = f"http://{minio_addr}:{minio_port}"

    enable_cmg_logging = request.config.getoption("--enable-cmg-logging")
    configure_environment(
        postgres,
        rabbitmq,
        minio,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        enable_logs=enable_cmg_logging,
    )

    setup_scheduler(request)


@pytest.fixture(scope="module")
def client(setup):
    # Depends on setup to ensure CONFIG_FILE is set before TestClient creates the app
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def config(client: TestClient) -> Config:
    config = load_config(str(TEST_CONFIG_FILE.absolute()))
    config.database_manager.init_db()  # init DB schema to avoid setting up the migrations service
    return config


@pytest.fixture(scope="module")
def trained_model(client: TestClient, config: Config) -> tuple[str, str]:
    """Train a model once and return its tracking_id and model_uri for use in tests.

    Returns:
        tuple of (tracking_id (MLflow run ID), model_uri)
    """
    payload = str.encode('["Patient diagnosed with kidney failure"]')
    with tempfile.NamedTemporaryFile("r+b") as f:
        f.write(payload)
        f.seek(0)
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/train_unsupervised",
            files=[("training_data", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    _, parsed = download_result_object(task.result, config.results_object_store_manager, "text")
    _, _, run_id = parse_mlflow_url(parsed)

    tc: TrackingClient = config.tracking_client
    model_uri = tc.get_model_uri(run_id)

    return run_id, model_uri


def test_config_loaded(config: Config):
    assert config
    assert all(
        hasattr(config, key)
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


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200

    health_data = response.json()
    assert "status" in health_data
    assert "components" in health_data

    expected_components = ["database", "task_object_store", "results_object_store", "queue"]
    for component in expected_components:
        assert component in health_data["components"]

    # In a properly configured test environment, all components should be healthy
    assert health_data["status"] == "healthy"
    for component in expected_components:
        assert health_data["components"][component] == "healthy"


def test_get_tasks(client: TestClient, config: Config):
    response = client.get("/tasks/")
    assert response.status_code == 403
    assert response.json() == {"detail": "Only admins can list tasks"}


def test_get_task_by_uuid(client: TestClient, config: Config):
    task_uuid = "nonexistent-uuid"
    response = client.get(f"/tasks/{task_uuid}")
    assert response.status_code == 404
    assert response.json() == {"detail": f"Task '{task_uuid}' not found"}

    tm: TaskManager = config.task_manager
    task = tm.create_task()
    response = client.get(f"/tasks/{task.uuid}")
    assert response.status_code == 200
    assert response.json() == {"uuid": task.uuid, "status": "pending"}

    tm.update_task(task.uuid, status=Status.SUCCEEDED, result="result.txt", error_message=None)
    response = client.get(f"/tasks/{task.uuid}", params={"detail": True})
    assert response.status_code == 200
    res = response.json()
    assert res["uuid"] == task.uuid
    assert res["status"] == Status.SUCCEEDED.value
    assert res["model"] is None
    assert res["type"] is None
    assert res["source"] is None
    assert res["created_at"] is not None
    assert res["started_at"] is None
    assert res["finished_at"] is not None
    assert res["result"] == "result.txt"
    assert res["error_message"] is None
    assert res["tracking_id"] is None


def test_get_models(client: TestClient, config: Config):
    response = client.get("/models/")
    assert response.status_code == 200

    response_json = response.json()
    assert isinstance(response_json, dict)
    assert "running" in response_json
    assert "on_demand" in response_json
    assert isinstance(response_json["running"], list)
    assert isinstance(response_json["on_demand"], list)
    assert len(response_json["running"]) == 1
    assert all(key in response_json["running"][0] for key in ["name", "is_running"])
    assert response_json["running"][0]["name"] == TEST_MODEL_SERVICE
    assert response_json["running"][0]["is_running"] is True

    response = client.get("/models/", params={"verbose": True})
    assert response.status_code == 200

    response_json = response.json()
    assert isinstance(response_json, dict)
    assert "running" in response_json
    assert "on_demand" in response_json
    assert len(response_json["running"]) == 1

    model = response_json["running"][0]
    assert all(
        key in model for key in ["name", "is_running", "deployment_type", "model_type", "runtime"]
    )
    assert model["name"] == TEST_MODEL_SERVICE
    assert model["is_running"] is True
    assert model["deployment_type"] == ModelDeploymentType.STATIC.value
    assert model["model_type"] is not None
    assert "api_version" in model["runtime"]


def test_get_model(client: TestClient, config: Config):
    response = client.get(f"/models/{TEST_MODEL_SERVICE}")
    assert response.status_code == 200

    response_json = response.json()
    assert all(key in response_json for key in ["name", "is_running"])
    assert response_json["name"] == TEST_MODEL_SERVICE
    assert response_json["is_running"] is True

    response = client.get(f"/models/{TEST_MODEL_SERVICE}", params={"verbose": True})
    assert response.status_code == 200

    response_json = response.json()
    assert all(
        key in response_json
        for key in ["name", "is_running", "deployment_type", "model_type", "runtime"]
    )
    assert response_json["name"] == TEST_MODEL_SERVICE
    assert response_json["is_running"] is True
    assert response_json["deployment_type"] == ModelDeploymentType.STATIC.value
    assert response_json["model_type"] is not None
    assert "api_version" in response_json["runtime"]


def test_get_model_info(client: TestClient, config: Config):
    response = client.get(f"/models/{TEST_MODEL_SERVICE}/info")
    assert response.status_code == 200
    assert all(
        key in response.json()
        for key in ["api_version", "model_type", "model_description", "model_card"]
    )


def test_unsupported_task(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/unsupported-task",
        headers={"Content-Type": "dummy"},
    )
    assert response.status_code == 404
    assert "Task 'unsupported-task' not found. Supported tasks are:" in response.json()["detail"]


def test_process(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/process",
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
    verify_annotation_contains_keys(annotation, ANNOTATION_FIELDS_BASE)
    assert annotation["label_name"] == "Loss Of Kidney Function"

    verify_results_match_api_info(client, task, res)


def test_process_jsonl(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/process_jsonl",
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


def test_process_bulk(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/process_bulk",
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
        verify_annotation_contains_keys(annotation, ANNOTATION_FIELDS_BASE)
        assert annotation["label_name"] == "Loss Of Kidney Function"

    verify_results_match_api_info(client, task, res)


def test_process_bulk_file(client: TestClient, config: Config):
    with open(MULTI_TEXT_FILE_PATH, "rb") as f:
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/process_bulk_file",
            files=[("multi_text_file", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_{MULTI_TEXT_FILE_PATH.name}"
    with open(MULTI_TEXT_FILE_PATH, "rb") as f:
        expected_payload = f.read()
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager)

    assert len(parsed) == 2

    for doc in parsed:
        assert "text" in doc
        assert "annotations" in doc
        assert len(doc["annotations"]) == 1

        annotation = doc["annotations"][0]
        verify_annotation_contains_keys(annotation, ANNOTATION_FIELDS_BASE)
        assert annotation["label_name"] == "Loss Of Kidney Function"

    verify_results_match_api_info(client, task, res)


def test_redact(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/redact",
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

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    assert parsed == "Patient diagnosed with [Loss Of Kidney Function]"

    verify_results_match_api_info(client, task, res)


def test_redact_with_encryption(client: TestClient, config: Config):
    with open(PUBLIC_KEY_PEM_PATH) as f:
        payload = {
            "text": "Patient diagnosed with kidney failure",
            "public_key_pem": f.read(),
        }
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/redact_with_encryption",
        json=payload,
    )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_payload.json"
    expected_payload = json.dumps(payload).encode()
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager)

    encrypted_label = "[REDACTED_0]"
    assert "redacted_text" in parsed
    assert parsed["redacted_text"] == f"Patient diagnosed with {encrypted_label}"
    assert "encryptions" in parsed
    assert len(parsed["encryptions"]) == 1

    assert "label" in parsed["encryptions"][0]
    assert parsed["encryptions"][0]["label"] == encrypted_label
    assert "encryption" in parsed["encryptions"][0]
    assert isinstance(parsed["encryptions"][0]["encryption"], str)
    assert len(parsed["encryptions"][0]["encryption"]) > 0

    verify_results_match_api_info(client, task, res)


def test_preview(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/preview",
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

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    assert parsed.startswith("<div") and parsed.endswith("</div>")
    assert "Patient diagnosed with" in parsed
    assert "kidney failure" in parsed
    assert "Loss Of Kidney Function" in parsed

    verify_results_match_api_info(client, task, res)


def test_preview_trainer_export(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post(
                f"/models/{TEST_MODEL_SERVICE}/tasks/preview_trainer_export",
                files=[("trainer_export", f1), ("trainer_export", f2)],
            )

    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key1 = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        expected_payload1 = f1.read()
    verify_task_payload_in_object_store(key1, expected_payload1, config.task_object_store_manager)

    key2 = f"{task.uuid}_{ANOTHER_TRAINER_EXPORT_PATH.name}"
    with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
        expected_payload2 = f2.read()
    verify_task_payload_in_object_store(key2, expected_payload2, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    assert len(parsed.split("<br/>")) == 12

    verify_results_match_api_info(client, task, res)


@pytest.mark.flaky(reruns=6)  # MedCAT training fails ~40% of the time due to small dataset
def test_train_supervised(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/train_supervised",
            files=[("trainer_export", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        expected_payload = f.read()
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    _, _, run_id = parse_mlflow_url(parsed)
    assert run_id == task.tracking_id

    verify_results_match_api_info(client, task, res)


def test_train_unsupervised(client: TestClient, config: Config):
    payload = str.encode('["Patient diagnosed with kidney failure"]')
    with tempfile.NamedTemporaryFile("r+b") as f:
        payload_file = Path(f.name).name
        f.write(payload)
        f.seek(0)
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/train_unsupervised",
            files=[("training_data", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)
    key = f"{task.uuid}_{payload_file}"
    verify_task_payload_in_object_store(key, payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    _, _, run_id = parse_mlflow_url(parsed)
    assert run_id == task.tracking_id

    verify_results_match_api_info(client, task, res)


def test_train_unsupervised_with_hf_hub_dataset(client: TestClient, config: Config):
    response = client.post(
        f"/models/{TEST_MODEL_SERVICE}/tasks/train_unsupervised_with_hf_hub_dataset",
        params={"hf_dataset_repo_id": "imdb"},
        headers={"Content-Type": "text/plain"},
    )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    _, _, run_id = parse_mlflow_url(parsed)
    assert run_id == task.tracking_id

    verify_results_match_api_info(client, task, res)


@pytest.mark.skip(reason="MetaCAT training is currently disabled on the CMS side")
def test_train_metacat(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/train_metacat",
            files=[("trainer_export", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        expected_payload = f.read()
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    _, _, run_id = parse_mlflow_url(parsed)
    assert run_id == task.tracking_id

    verify_results_match_api_info(client, task, res)


def test_evaluate(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/evaluate",
            files=[("trainer_export", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        expected_payload = f.read()
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    _, _, run_id = parse_mlflow_url(parsed)
    assert run_id == task.tracking_id

    verify_results_match_api_info(client, task, res)


def test_sanity_check(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(
            f"/models/{TEST_MODEL_SERVICE}/tasks/sanity-check",
            files=[("trainer_export", f)],
        )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        expected_payload = f.read()
    verify_task_payload_in_object_store(key, expected_payload, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    assert parsed == "concept,name,precision,recall,f1\n"

    verify_results_match_api_info(client, task, res)


def test_iaa_scores(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post(
                f"/models/{TEST_MODEL_SERVICE}/tasks/iaa-scores",
                params={
                    "annotator_a_project_id": 14,
                    "annotator_b_project_id": 15,
                    "scope": "per_concept",
                },
                files=[("trainer_export", f1), ("trainer_export", f2)],
            )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key1 = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        expected_payload1 = f1.read()
    verify_task_payload_in_object_store(key1, expected_payload1, config.task_object_store_manager)

    key2 = f"{task.uuid}_{ANOTHER_TRAINER_EXPORT_PATH.name}"
    with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
        expected_payload2 = f2.read()
    verify_task_payload_in_object_store(key2, expected_payload2, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    assert parsed == "concept,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta\n"

    verify_results_match_api_info(client, task, res)


def test_concat_trainer_exports(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post(
                f"/models/{TEST_MODEL_SERVICE}/tasks/concat_trainer_exports",
                files=[("trainer_export", f1), ("trainer_export", f2)],
            )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key1 = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        expected_payload1 = f1.read()
    verify_task_payload_in_object_store(key1, expected_payload1, config.task_object_store_manager)

    key2 = f"{task.uuid}_{ANOTHER_TRAINER_EXPORT_PATH.name}"
    with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
        expected_payload2 = f2.read()
    verify_task_payload_in_object_store(key2, expected_payload2, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    with open(CONCATENATED_TRAINER_EXPORTS_PATH) as f:
        assert json.loads(parsed) == json.load(f)

    verify_results_match_api_info(client, task, res)


def test_annotation_stats(client: TestClient, config: Config):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post(
                f"/models/{TEST_MODEL_SERVICE}/tasks/annotation-stats",
                files=[("trainer_export", f1), ("trainer_export", f2)],
            )
    response_json = validate_api_response(response, expected_status_code=200, return_json=True)

    tm: TaskManager = config.task_manager
    verify_task_submitted_successfully(response_json["uuid"], tm)

    task = wait_for_task_completion(response_json["uuid"], tm, expected_status=Status.SUCCEEDED)

    key1 = f"{task.uuid}_{TRAINER_EXPORT_PATH.name}"
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        expected_payload1 = f1.read()
    verify_task_payload_in_object_store(key1, expected_payload1, config.task_object_store_manager)

    key2 = f"{task.uuid}_{ANOTHER_TRAINER_EXPORT_PATH.name}"
    with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
        expected_payload2 = f2.read()
    verify_task_payload_in_object_store(key2, expected_payload2, config.task_object_store_manager)

    verify_queue_is_empty(config.queue_manager)

    res, parsed = download_result_object(task.result, config.results_object_store_manager, "text")

    with open(ANNOTATION_STATS_CSV_PATH) as f:
        assert parsed == f.read()

    verify_results_match_api_info(client, task, res)


def test_deploy_model(client: TestClient, config: Config, trained_model: tuple[str, str]):
    """Test manual model deployment with tracking_id and model_uri."""
    run_id, model_uri = trained_model

    # Deploy with tracking_id
    model_name_1 = "test-model-tracking-id"
    response = client.post(f"/models/{model_name_1}", json={"tracking_id": run_id})
    assert response.status_code == 200
    deploy_info = response.json()
    assert deploy_info["model_uri"] == model_uri
    assert deploy_info["container_name"] == model_name_1
    assert all(k in deploy_info for k in ["ttl", "container_id"])

    response = client.get(f"/models/{model_name_1}")
    assert response.status_code == 200
    assert response.json()["is_running"] is True

    # Try to deploy same model again (should fail with conflict)
    response = client.post(f"/models/{model_name_1}", json={"tracking_id": run_id})
    assert response.status_code == 409
    assert "already running" in response.json()["detail"].lower()

    # Clean up first deployment
    response = client.delete(f"/models/{model_name_1}")
    assert response.status_code == 204

    # Deploy with model_uri directly
    model_name_2 = "test-model-uri"
    response = client.post(f"/models/{model_name_2}", json={"model_uri": model_uri})
    assert response.status_code == 200
    deploy_info = response.json()
    assert deploy_info["model_uri"] == model_uri
    assert deploy_info["container_name"] == model_name_2
    assert all(k in deploy_info for k in ["ttl", "container_id"])

    # Clean up second deployment
    response = client.delete(f"/models/{model_name_2}")
    assert response.status_code == 204

    # Deploy with both tracking_id and model_uri (model_uri takes precedence)
    model_name_3 = "test-model-both"
    response = client.post(
        f"/models/{model_name_3}", json={"tracking_id": run_id, "model_uri": model_uri}
    )
    assert response.status_code == 200
    deploy_info = response.json()
    assert deploy_info["model_uri"] == model_uri
    assert deploy_info["container_name"] == model_name_3

    # Clean up third deployment
    response = client.delete(f"/models/{model_name_3}")
    assert response.status_code == 204

    # Deploy with invalid model URI (should fail due to require_model_uri_validation=true in config)
    model_name_4 = "test-model-invalid"
    response = client.post(f"/models/{model_name_4}", json={"model_uri": "invalid://uri"})
    assert response.status_code == 404


def test_remove_model(client: TestClient, config: Config, trained_model: tuple[str, str]):
    """Test model removal endpoint."""
    run_id, _ = trained_model

    # Deploy a model
    model_name = "test-model-for-removal"
    response = client.post(f"/models/{model_name}", json={"tracking_id": run_id, "ttl": 3600})
    assert response.status_code == 200

    response = client.get(f"/models/{model_name}")
    assert response.status_code == 200
    assert response.json()["is_running"] is True

    # Remove the deployed model
    response = client.delete(f"/models/{model_name}")
    assert response.status_code == 204

    response = client.get(f"/models/{model_name}")
    assert response.status_code == 404

    # Try to remove non-existent model (should fail)
    response = client.delete(f"/models/{model_name}")
    assert response.status_code == 404

    # Try to remove non-existent model with force flag (should succeed)
    response = client.delete("/models/nonexistent-model", params={"force": True})
    assert response.status_code == 204

    # Deploy another model and remove it by force
    model_name_2 = "test-model-for-force-removal"
    response = client.post(f"/models/{model_name_2}", json={"tracking_id": run_id})
    assert response.status_code == 200

    response = client.delete(f"/models/{model_name_2}", params={"force": True})
    assert response.status_code == 204


def test_admin_list_on_demand_configs(client: TestClient, config: Config):
    """Test listing on-demand model configurations (GET /admin/on-demand)."""
    response = client.get("/admin/on-demand")
    assert response.status_code == 200
    data = response.json()
    assert "configs" in data
    assert "total" in data
    assert isinstance(data["configs"], list)
    assert data["total"] == 0

    response = client.post(
        "/admin/on-demand", json={"model_name": "test-list-model", "model_uri": "mlflow://test/uri"}
    )
    assert response.status_code == 201

    response = client.get("/admin/on-demand")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["configs"]) == 1
    assert data["configs"][0]["model_name"] == "test-list-model"
    assert data["configs"][0]["model_uri"] == "mlflow://test/uri"
    assert data["configs"][0]["enabled"] is True

    response = client.delete("/admin/on-demand/test-list-model")
    assert response.status_code == 204

    # List without include_disabled should be empty
    response = client.get("/admin/on-demand")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0

    # List with include_disabled should show the disabled config
    response = client.get("/admin/on-demand", params={"include_disabled": True})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["configs"]) == 1
    assert data["configs"][0]["model_name"] == "test-list-model"
    assert data["configs"][0]["enabled"] is False


def test_admin_create_on_demand_config(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test creating on-demand model configurations (POST /admin/on-demand)."""
    run_id, model_uri = trained_model

    # Create with model_uri only
    model_name_1 = "test-create-with-uri"
    response = client.post(
        "/admin/on-demand",
        json={
            "model_name": model_name_1,
            "model_uri": model_uri,
            "description": "Test model created with URI",
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert created["model_name"] == model_name_1
    assert created["model_uri"] == model_uri
    assert created["description"] == "Test model created with URI"
    assert created["idle_ttl"] == config.get_default_idle_ttl()
    assert created["enabled"] is True
    assert all(k in created for k in ["id", "created_at", "updated_at"])

    # Try to create another config with same model_name (should fail with 409)
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name_1, "model_uri": model_uri, "replace_enabled": False},
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()

    # Create with replace_enabled=true (should succeed and disable previous)
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name_1, "model_uri": model_uri, "description": "Updated model"},
    )
    assert response.status_code == 201
    created = response.json()
    assert created["model_name"] == model_name_1
    assert created["description"] == "Updated model"
    assert created["enabled"] is True

    response = client.get(f"/admin/on-demand/{model_name_1}/history")
    assert response.status_code == 200
    history = response.json()
    assert history["total"] == 2
    assert sum(1 for c in history["configs"] if c["enabled"]) == 1

    # Inherit model_uri from previous config (new version with updated idle_ttl)
    response = client.post("/admin/on-demand", json={"model_name": model_name_1, "idle_ttl": 1800})
    assert response.status_code == 201
    created = response.json()
    assert created["model_name"] == model_name_1
    assert created["model_uri"] == model_uri
    assert created["idle_ttl"] == 1800

    client.delete(f"/admin/on-demand/{model_name_1}")

    # Attempt to inherit with missing model_uri from non-existent config (should fail with 400)
    response = client.post("/admin/on-demand", json={"model_name": model_name_1, "idle_ttl": 7200})
    assert response.status_code == 400
    assert "model_uri is required" in response.json()["detail"].lower()

    # Create with tracking_id (should resolve to model_uri)
    model_name_2 = "test-create-with-tracking-id"
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name_2, "tracking_id": run_id, "idle_ttl": 7200},
    )
    assert response.status_code == 201
    created = response.json()
    assert created["model_name"] == model_name_2
    assert created["model_uri"] == model_uri
    assert created["tracking_id"] == run_id
    assert created["idle_ttl"] == 7200

    client.delete(f"/admin/on-demand/{model_name_2}")

    # Create with full spec including deploy config
    model_name_3 = "test-create-full-spec"
    response = client.post(
        "/admin/on-demand",
        json={
            "model_name": model_name_3,
            "model_uri": model_uri,
            "tracking_id": run_id,
            "idle_ttl": 3600,
            "description": "Full spec test",
            "deploy": {
                "resources": {
                    "limits": {"cpus": "2", "memory": "4g"},
                    "reservations": {"cpus": "1", "memory": "2g"},
                }
            },
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert created["model_name"] == model_name_3
    assert created["idle_ttl"] == 3600
    assert created["description"] == "Full spec test"
    assert created["deploy"] is not None
    assert created["deploy"]["resources"]["limits"] == {"cpus": "2", "memory": "4g"}
    assert created["deploy"]["resources"]["reservations"] == {"cpus": "1", "memory": "2g"}

    client.delete(f"/admin/on-demand/{model_name_3}")

    # Create with inherit_config (first config for this model, so no inheritance)
    model_name_4 = "test-inherit-no-parent"
    response = client.post(
        "/admin/on-demand",
        json={
            "model_name": model_name_4,
            "model_uri": model_uri,
            "inherit_config": True,
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert created["model_name"] == model_name_4

    client.delete(f"/admin/on-demand/{model_name_4}")


def test_admin_get_on_demand_config(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test getting a specific on-demand config (GET /admin/on-demand/{model_name})."""
    _, model_uri = trained_model

    # Try to get non-existent config
    response = client.get("/admin/on-demand/nonexistent-model")
    assert response.status_code == 404

    model_name = "test-get-config"
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri, "description": "Test get config"},
    )
    assert response.status_code == 201
    created = response.json()

    response = client.get(f"/admin/on-demand/{model_name}")
    assert response.status_code == 200
    fetched = response.json()
    assert fetched["id"] == created["id"]
    assert fetched["model_name"] == model_name
    assert fetched["model_uri"] == model_uri
    assert fetched["description"] == "Test get config"
    assert fetched["enabled"] is True

    response = client.delete(f"/admin/on-demand/{model_name}")
    assert response.status_code == 204

    # Try to get disabled config (should return 404)
    response = client.get(f"/admin/on-demand/{model_name}")
    assert response.status_code == 404


def test_admin_update_on_demand_config(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test updating an on-demand config (PUT /admin/on-demand/{model_name})."""
    run_id, model_uri = trained_model

    # Try to update non-existent config
    response = client.put("/admin/on-demand/nonexistent-model", json={"description": "Not there"})
    assert response.status_code == 404

    model_name = "test-update-config"
    response = client.post(
        "/admin/on-demand",
        json={
            "model_name": model_name,
            "model_uri": model_uri,
            "description": "Original description",
            "idle_ttl": 3600,
        },
    )
    assert response.status_code == 201

    # Update description only
    response = client.put(
        f"/admin/on-demand/{model_name}", json={"description": "Updated description"}
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["description"] == "Updated description"
    assert updated["model_uri"] == model_uri
    assert updated["idle_ttl"] == 3600

    # Update idle_ttl
    response = client.put(f"/admin/on-demand/{model_name}", json={"idle_ttl": 7200})
    assert response.status_code == 200
    updated = response.json()
    assert updated["idle_ttl"] == 7200
    assert updated["description"] == "Updated description"

    # Update model_uri directly
    new_uri = "mlflow://test/new/uri"
    response = client.put(f"/admin/on-demand/{model_name}", json={"model_uri": new_uri})
    assert response.status_code == 200
    updated = response.json()
    assert updated["model_uri"] == new_uri

    # Update tracking_id (should also update model_uri)
    response = client.put(f"/admin/on-demand/{model_name}", json={"tracking_id": run_id})
    assert response.status_code == 200
    updated = response.json()
    assert updated["tracking_id"] == run_id
    assert updated["model_uri"] == model_uri

    # Update deploy spec
    response = client.put(
        f"/admin/on-demand/{model_name}",
        json={
            "deploy": {
                "resources": {
                    "limits": {"cpus": "1", "memory": "2g"},
                    "reservations": {"cpus": "0.5", "memory": "1g"},
                },
            }
        },
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["deploy"] is not None
    assert updated["deploy"]["resources"]["limits"] == {"cpus": "1", "memory": "2g"}
    assert updated["deploy"]["resources"]["reservations"] == {"cpus": "0.5", "memory": "1g"}

    # Clear optional fields using clear_* flags
    response = client.put(
        f"/admin/on-demand/{model_name}",
        json={
            "clear_description": True,
            "clear_idle_ttl": True,
            "clear_tracking_id": True,
            "clear_deploy": True,
        },
    )
    assert response.status_code == 200
    updated = response.json()
    # None fields are excluded from response JSON
    assert all(k not in updated for k in ["description", "tracking_id", "deploy"])
    assert updated["idle_ttl"] == config.get_default_idle_ttl()
    assert updated["model_uri"] == model_uri

    client.delete(f"/admin/on-demand/{model_name}")

    # Verify that only one config version exists in history
    response = client.get(f"/admin/on-demand/{model_name}/history")
    assert response.status_code == 200
    history = response.json()
    assert history["total"] == 1


def test_admin_delete_on_demand_config(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test soft-deleting an on-demand config (DELETE /admin/on-demand/{model_name})."""
    _, model_uri = trained_model

    # Try to delete non-existent config
    response = client.delete("/admin/on-demand/nonexistent-model")
    assert response.status_code == 404

    model_name = "test-delete-config"
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri},
    )
    assert response.status_code == 201
    config_id = response.json()["id"]

    response = client.get(f"/admin/on-demand/{model_name}")
    assert response.status_code == 200
    assert response.json()["enabled"] is True

    # Delete (soft-delete) a config
    response = client.delete(f"/admin/on-demand/{model_name}")
    assert response.status_code == 204

    # Verify it's no longer returned by default
    response = client.get(f"/admin/on-demand/{model_name}")
    assert response.status_code == 404

    # But it should appear in history
    response = client.get(f"/admin/on-demand/{model_name}/history")
    assert response.status_code == 200
    history = response.json()
    assert history["total"] == 1
    assert history["configs"][0]["enabled"] is False

    # Try to delete again (should fail)
    response = client.delete(f"/admin/on-demand/{model_name}")
    assert response.status_code == 404

    # Re-enable and delete again
    response = client.post(f"/admin/on-demand/{config_id}/enable")
    assert response.status_code == 200

    response = client.delete(f"/admin/on-demand/{model_name}")
    assert response.status_code == 204


def test_admin_get_on_demand_config_history(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test getting config history (GET /admin/on-demand/{model_name}/history)."""
    _, model_uri = trained_model

    # Try to get history for non-existent model
    response = client.get("/admin/on-demand/nonexistent-model/history")
    assert response.status_code == 404

    model_name = "test-history-config"
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri, "description": "Version 1"},
    )
    assert response.status_code == 201

    # Get history (should have one entry)
    response = client.get(f"/admin/on-demand/{model_name}/history")
    assert response.status_code == 200
    history = response.json()
    assert history["total"] == 1
    assert len(history["configs"]) == 1
    assert history["configs"][0]["description"] == "Version 1"
    assert history["configs"][0]["enabled"] is True

    # Create version 2 with replace
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri, "description": "Version 2"},
    )
    assert response.status_code == 201

    # Get history (should have two entries)
    response = client.get(f"/admin/on-demand/{model_name}/history")
    assert response.status_code == 200
    history = response.json()
    assert history["total"] == 2
    assert len(history["configs"]) == 2

    # Verify only one is enabled (newest)
    enabled_configs = [c for c in history["configs"] if c["enabled"]]
    assert len(enabled_configs) == 1
    assert enabled_configs[0]["description"] == "Version 2"

    # Create version 3
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri, "description": "Version 3"},
    )
    assert response.status_code == 201

    # Get history (should have three entries)
    response = client.get(f"/admin/on-demand/{model_name}/history")
    assert response.status_code == 200
    history = response.json()
    assert history["total"] == 3
    assert len(history["configs"]) == 3

    # Verify ordering (newest first)
    descriptions = [c["description"] for c in history["configs"]]
    assert descriptions[0] == "Version 3"

    client.delete(f"/admin/on-demand/{model_name}")


def test_admin_enable_on_demand_config(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test enabling a previously disabled config (POST /admin/on-demand/{config_id}/enable)."""
    _, model_uri = trained_model

    # Try to enable non-existent config
    response = client.post("/admin/on-demand/999999/enable")
    assert response.status_code == 404

    model_name = "test-enable-config"
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri, "description": "Version 1"},
    )
    assert response.status_code == 201
    config_id_1 = response.json()["id"]

    # Create version 2 (disables version 1)
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name, "model_uri": model_uri, "description": "Version 2"},
    )
    assert response.status_code == 201
    config_id_2 = response.json()["id"]

    # Verify version 2 is enabled
    response = client.get(f"/admin/on-demand/{model_name}")
    assert response.status_code == 200
    assert response.json()["description"] == "Version 2"
    assert response.json()["id"] == config_id_2

    # Re-enable version 1 (should disable version 2)
    response = client.post(f"/admin/on-demand/{config_id_1}/enable")
    assert response.status_code == 200
    enabled = response.json()
    assert enabled["id"] == config_id_1
    assert enabled["description"] == "Version 1"
    assert enabled["enabled"] is True

    # Verify version 1 is now the enabled config
    response = client.get(f"/admin/on-demand/{model_name}")
    assert response.status_code == 200
    assert response.json()["id"] == config_id_1
    assert response.json()["description"] == "Version 1"

    # Verify history shows version 2 as disabled
    response = client.get(f"/admin/on-demand/{model_name}/history")
    assert response.status_code == 200
    history = response.json()
    v2_configs = [c for c in history["configs"] if c["id"] == config_id_2]
    assert len(v2_configs) == 1
    assert v2_configs[0]["enabled"] is False

    client.delete(f"/admin/on-demand/{model_name}")


def test_auto_deploy_basic_workflows(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test basic auto-deployment workflows.

    Scenarios:
    1. Create on-demand config → Submit task → Verify model auto-deploys → Verify task succeeds
    2. Submit task to already-running on-demand model to verify no new deployment is triggered
    3. Verify config with tracking_id (not model_uri) resolves and deploys correctly
    """
    run_id, model_uri = trained_model

    model_name_1 = "test-auto-deploy-basic"
    response = client.post(
        "/admin/on-demand",
        json={
            "model_name": model_name_1,
            "model_uri": model_uri,
            "idle_ttl": 9999,
            "description": "Test auto-deploy",
        },
    )
    assert response.status_code == 201
    created_config = response.json()
    assert created_config["model_name"] == model_name_1

    # Verify model appears in on_demand list but not running
    response = client.get("/models/")
    assert response.status_code == 200
    models_data = response.json()
    on_demand_names = [m["name"] for m in models_data["on_demand"]]
    running_names = [m["name"] for m in models_data["running"]]
    assert model_name_1 in on_demand_names
    assert model_name_1 not in running_names

    # Submit task to trigger auto-deployment
    response = client.post(
        f"/models/{model_name_1}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    task_data = response.json()
    task_uuid = task_data["uuid"]

    # Model should auto-deploy and process task
    tm: TaskManager = config.task_manager
    task = wait_for_task_completion(task_uuid, tm, expected_status=Status.SUCCEEDED)
    assert task.error_message is None

    response = client.get("/models/")
    assert response.status_code == 200
    models_data = response.json()
    running_names = [m["name"] for m in models_data["running"]]
    assert model_name_1 in running_names

    container = get_deployed_model_container(model_name_1)
    assert container is not None
    verify_container_labels(
        container,
        {
            "org.cogstack.model-gateway.deployment-type": "auto",
            "org.cogstack.model-gateway.managed-by": "cmg",
            "org.cogstack.model-gateway.ttl": "9999",
            "org.cogstack.model-serve": model_name_1,
        },
    )

    response = client.get(f"/models/{model_name_1}", params={"verbose": True})
    assert response.status_code == 200
    model_info = response.json()
    assert model_info["deployment_type"] == ModelDeploymentType.AUTO.value
    assert model_info["is_running"] is True

    # Submit task to already-running on-demand model (no re-deployment)
    initial_count = count_deployed_model_containers()
    response = client.post(
        f"/models/{model_name_1}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    task_uuid = response.json()["uuid"]

    task = wait_for_task_completion(task_uuid, tm, expected_status=Status.SUCCEEDED)
    assert task.error_message is None

    final_count = count_deployed_model_containers()
    assert final_count == initial_count

    response = client.delete(f"/models/{model_name_1}")
    assert response.status_code == 204
    client.delete(f"/admin/on-demand/{model_name_1}")

    # Deploy based on config with tracking_id and verify correct resolution
    model_name_2 = "test-auto-deploy-tracking-id"
    response = client.post(
        "/admin/on-demand",
        json={"model_name": model_name_2, "tracking_id": run_id, "idle_ttl": 1800},
    )
    assert response.status_code == 201
    created = response.json()
    assert created["tracking_id"] == run_id
    assert created["model_uri"] == model_uri

    # Submit task to trigger auto-deployment
    response = client.post(
        f"/models/{model_name_2}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    task_uuid = response.json()["uuid"]

    task = wait_for_task_completion(task_uuid, tm, expected_status=Status.SUCCEEDED)
    assert task.error_message is None

    container = get_deployed_model_container(model_name_2)
    assert container is not None

    client.delete(f"/models/{model_name_2}")
    client.delete(f"/admin/on-demand/{model_name_2}")


def test_auto_deploy_advanced_scenarios(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test advanced auto-deployment scenarios.

    Scenarios:
    1. Verify system waits for model to become healthy before processing
    2. Verify that multiple concurrent requests don't cause duplicate deployments
    """
    _, model_uri = trained_model
    tm: TaskManager = config.task_manager

    # Verify system waits for model to become healthy before processing
    model_name_1 = "test-auto-deploy-health-check"
    client.post("/admin/on-demand", json={"model_name": model_name_1, "model_uri": model_uri})

    start_time = time.time()
    response = client.post(
        f"/models/{model_name_1}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    task_uuid = response.json()["uuid"]

    # Completion should take at least a few seconds for container startup
    task = wait_for_task_completion(task_uuid, tm, expected_status=Status.SUCCEEDED)
    elapsed = time.time() - start_time
    assert elapsed > 5, f"Task completed too quickly ({elapsed}s), expected container startup time"

    model = config.model_manager.get_model(model_name_1)
    assert model is not None
    assert model.ready is True

    client.delete(f"/models/{model_name_1}")
    client.delete(f"/admin/on-demand/{model_name_1}")

    # Verify that multiple concurrent requests don't cause duplicate deployments
    model_name_2 = "test-auto-deploy-concurrent"
    client.post("/admin/on-demand", json={"model_name": model_name_2, "model_uri": model_uri})

    initial_container_count = count_deployed_model_containers()

    def submit_task():
        response = client.post(
            f"/models/{model_name_2}/tasks/process",
            data="Patient diagnosed with kidney failure",
            headers={"Content-Type": "text/plain"},
        )
        return response.json()["uuid"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(submit_task) for _ in range(3)]
        task_uuids = [f.result() for f in concurrent.futures.as_completed(futures)]
    assert len(task_uuids) == 3

    for task_uuid in task_uuids:
        task = wait_for_task_completion(task_uuid, tm, expected_status=Status.SUCCEEDED)
        assert task.error_message is None

    final_container_count = count_deployed_model_containers()
    assert final_container_count == initial_container_count + 1

    model = config.model_manager.get_model(model_name_2)
    assert model is not None

    client.delete(f"/models/{model_name_2}")
    client.delete(f"/admin/on-demand/{model_name_2}")


def test_auto_deploy_error_handling(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test auto-deployment error handling.

    Scenarios:
    1. Verify request to unconfigured model fails gracefully
    2. Verify config with invalid model_uri fails deployment gracefully
    """
    _, model_uri = trained_model
    tm: TaskManager = config.task_manager

    # Verify request to non-existent model fails gracefully
    nonexistent_model = "nonexistent-auto-deploy-model"
    initial_count = count_deployed_model_containers()
    response = client.post(
        f"/models/{nonexistent_model}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 503
    assert all(
        k in response.json()["detail"].lower()
        for k in ["not available", "running models", "on-demand models"]
    )

    assert count_deployed_model_containers() == initial_count
    assert config.model_manager.get_model(nonexistent_model) is None

    # Verify deployment with invalid model_uri fails gracefully (uri validation off by default)
    model_name_2 = "test-auto-deploy-invalid-uri"
    client.post(
        "/admin/on-demand",
        json={
            "model_name": model_name_2,
            "model_uri": "mlflow://invalid/experiment/run/artifacts/model",
        },
    )

    initial_count = count_deployed_model_containers()
    response = client.post(
        f"/models/{model_name_2}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    if response.status_code == 200:
        task_uuid = response.json()["uuid"]
        task = wait_for_task_completion(task_uuid, tm, expected_status=Status.FAILED)
        assert task.error_message is not None
    else:
        assert response.status_code == 503

    container = get_deployed_model_container(model_name_2)
    assert container is None or container.status != "running"

    if model := config.model_manager.get_model(model_name_2):
        assert model.ready is False or model is None

    client.delete(f"/models/{model_name_2}", params={"force": True})
    client.delete(f"/admin/on-demand/{model_name_2}")


def test_auto_deploy_models_listing(
    client: TestClient, config: Config, trained_model: tuple[str, str]
):
    """Test models listing with auto-deployment integration.

    Scenarios:
    1. GET /models correctly shows running and on-demand models at each state
    2. GET /models/{name} for on-demand model shows correct info when not running vs running
    """
    _, model_uri = trained_model
    tm: TaskManager = config.task_manager

    response = client.get("/models/")
    assert response.status_code == 200
    initial_models = response.json()
    initial_running_count = len(initial_models["running"])
    initial_on_demand_count = len(initial_models["on_demand"])

    model_a = "test-listing-model-a"
    model_b = "test-listing-model-b"

    client.post("/admin/on-demand", json={"model_name": model_a, "model_uri": model_uri})
    client.post("/admin/on-demand", json={"model_name": model_b, "model_uri": model_uri})

    response = client.get("/models/")
    assert response.status_code == 200
    models_data = response.json()

    on_demand_names = {m["name"] for m in models_data["on_demand"]}
    running_names = {m["name"] for m in models_data["running"]}

    assert model_a in on_demand_names
    assert model_b in on_demand_names
    assert model_a not in running_names
    assert model_b not in running_names
    assert len(models_data["on_demand"]) == initial_on_demand_count + 2
    assert len(models_data["running"]) == initial_running_count

    response = client.get(f"/models/{model_a}")
    assert response.status_code == 200
    model_info = response.json()
    assert model_info["name"] == model_a
    assert model_info["is_running"] is False

    response = client.get(f"/models/{model_a}", params={"verbose": True})
    assert response.status_code == 200
    model_info = response.json()
    assert model_info["deployment_type"] == ModelDeploymentType.AUTO.value
    assert model_info["is_running"] is False

    response = client.post(
        f"/models/{model_a}/tasks/process",
        data="Patient diagnosed with kidney failure",
        headers={"Content-Type": "text/plain"},
    )
    task_uuid = response.json()["uuid"]
    wait_for_task_completion(task_uuid, tm, expected_status=Status.SUCCEEDED)

    response = client.get("/models/")
    assert response.status_code == 200
    models_data = response.json()

    on_demand_names = {m["name"] for m in models_data["on_demand"]}
    running_names = {m["name"] for m in models_data["running"]}

    assert model_a not in on_demand_names
    assert model_b in on_demand_names
    assert model_a in running_names
    assert model_b not in running_names
    assert len(models_data["running"]) == initial_running_count + 1

    response = client.get(f"/models/{model_a}", params={"verbose": True})
    assert response.status_code == 200
    model_info = response.json()
    assert model_info["deployment_type"] == ModelDeploymentType.AUTO.value
    assert model_info["is_running"] is True
    assert "runtime" in model_info

    client.delete(f"/admin/on-demand/{model_b}")

    response = client.get("/models/")
    assert response.status_code == 200
    models_data = response.json()

    on_demand_names = {m["name"] for m in models_data["on_demand"]}
    running_names = {m["name"] for m in models_data["running"]}

    assert model_a not in on_demand_names
    assert model_b not in on_demand_names
    assert model_a in running_names
    assert len(models_data["on_demand"]) == initial_on_demand_count

    client.delete(f"/models/{model_a}")
    client.delete(f"/admin/on-demand/{model_a}")
