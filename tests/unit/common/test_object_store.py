import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from cogstack_model_gateway.common.object_store import (
    DEFAULT_ACCESS_KEY,
    DEFAULT_MINIO_BUCKET,
    DEFAULT_MINIO_ENDPOINT,
    DEFAULT_SECRET_KEY,
    ObjectStoreManager,
)


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_default_initialization(mock_minio_client: MagicMock) -> None:
    object_store_manager = ObjectStoreManager()

    assert object_store_manager.endpoint == DEFAULT_MINIO_ENDPOINT
    assert object_store_manager.access_key == DEFAULT_ACCESS_KEY
    assert object_store_manager.secret_key == DEFAULT_SECRET_KEY
    assert object_store_manager.default_bucket == DEFAULT_MINIO_BUCKET

    mock_minio_client.assert_called_once_with(
        DEFAULT_MINIO_ENDPOINT,
        access_key=DEFAULT_ACCESS_KEY,
        secret_key=DEFAULT_SECRET_KEY,
        secure=False,
    )


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_custom_initialization(mock_minio_client: MagicMock) -> None:
    custom_host = "custom.minio.com"
    custom_port = 9001
    custom_access_key = "custom_access"
    custom_secret_key = "custom_secret"
    custom_bucket = "custom-bucket"

    manager: ObjectStoreManager = ObjectStoreManager(
        host=custom_host,
        port=custom_port,
        access_key=custom_access_key,
        secret_key=custom_secret_key,
        secure=True,
        default_bucket=custom_bucket,
    )

    assert manager.endpoint == f"{custom_host}:{custom_port}"
    assert manager.access_key == custom_access_key
    assert manager.secret_key == custom_secret_key
    assert manager.default_bucket == custom_bucket
    assert manager.secure is True


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_create_bucket(mock_minio_client: MagicMock) -> None:
    mock_minio = mock_minio_client.return_value

    mock_minio.make_bucket = MagicMock()
    mock_minio.bucket_exists = MagicMock()
    mock_minio.bucket_exists.return_value = False

    object_store_manager = ObjectStoreManager()

    test_bucket = "test-bucket"
    object_store_manager.create_bucket(test_bucket)

    mock_minio.bucket_exists.assert_has_calls(
        [call(DEFAULT_MINIO_BUCKET), call(test_bucket)], any_order=False
    )
    mock_minio.make_bucket.assert_has_calls(
        [call(DEFAULT_MINIO_BUCKET), call(test_bucket)], any_order=False
    )


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_upload_object(mock_minio_client: MagicMock) -> None:
    mock_minio = mock_minio_client.return_value
    mock_minio.put_object = MagicMock()

    object_store_manager = ObjectStoreManager()
    test_file_data = b"test file content"
    test_filename = "test_file.txt"

    test_uuid = "12345678-1234-5678-1234-567812345678"
    with patch("uuid.uuid4", return_value=uuid.UUID(test_uuid)):
        _ = object_store_manager.upload_object(test_file_data, test_filename)

    expected_object_key = f"{test_uuid}_{test_filename}"
    mock_minio.put_object.assert_called_once()

    put_object_call = mock_minio.put_object.call_args
    assert put_object_call[0][0] == DEFAULT_MINIO_BUCKET
    assert put_object_call[0][1] == expected_object_key


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_upload_object_custom_bucket(mock_minio_client: MagicMock) -> None:
    mock_minio = mock_minio_client.return_value
    mock_minio.put_object = MagicMock()
    mock_minio.make_bucket = MagicMock()
    mock_minio.bucket_exists = MagicMock()
    mock_minio.bucket_exists.return_value = False

    object_store_manager = ObjectStoreManager()
    test_file_data = b"test file content"
    test_filename = "test_file.txt"
    custom_bucket = "custom-bucket"
    test_prefix = "test_prefix"

    object_store_manager.upload_object(
        test_file_data, test_filename, bucket_name=custom_bucket, prefix=test_prefix
    )

    mock_minio.bucket_exists.assert_has_calls(
        [call(DEFAULT_MINIO_BUCKET), call(custom_bucket)], any_order=False
    )
    mock_minio.make_bucket.assert_has_calls(
        [call(DEFAULT_MINIO_BUCKET), call(custom_bucket)], any_order=False
    )
    mock_minio.put_object.assert_called_once()

    put_object_call = mock_minio.put_object.call_args
    assert put_object_call[0][0] == custom_bucket
    assert put_object_call[0][1] == f"{test_prefix}_{test_filename}"
    assert put_object_call[1]["data"].read() == test_file_data
    assert put_object_call[1]["length"] == len(test_file_data)


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_get_object(mock_minio_client: MagicMock) -> None:
    mock_minio = mock_minio_client.return_value
    mock_minio.get_object = MagicMock()
    mock_minio.get_object.return_value.read.return_value = b"mocked object content"

    object_store_manager = ObjectStoreManager()
    test_object_key = "test_object_key"

    retrieved_content: bytes = object_store_manager.get_object(test_object_key)

    mock_minio.get_object.assert_called_once_with(DEFAULT_MINIO_BUCKET, test_object_key)

    assert retrieved_content == b"mocked object content"


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_get_object_custom_bucket(mock_minio_client: MagicMock) -> None:
    mock_minio = mock_minio_client.return_value
    mock_minio.get_object = MagicMock()
    mock_minio.get_object.return_value.read.return_value = b"mocked object content"

    object_store_manager = ObjectStoreManager()
    test_object_key = "test_object_key"
    custom_bucket = "custom-bucket"

    retrieved_content: bytes = object_store_manager.get_object(
        test_object_key, bucket_name=custom_bucket
    )

    mock_minio.get_object.assert_called_once_with(custom_bucket, test_object_key)

    assert retrieved_content == b"mocked object content"


@patch("cogstack_model_gateway.common.object_store.Minio")
def test_create_bucket_error(mock_minio_client: MagicMock) -> None:
    mock_minio = mock_minio_client.return_value
    mock_minio.bucket_exists = MagicMock()
    mock_minio.bucket_exists.side_effect = Exception("Test error")

    with pytest.raises(Exception, match="Test error"):
        _ = ObjectStoreManager()
