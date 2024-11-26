import json
from unittest.mock import MagicMock, patch

import pika
import pytest

from cogstack_model_gateway.common.queue import (
    DEFAULT_QUEUE_NAME,
    DEFAULT_RABBITMQ_URL,
    QueueManager,
)


@pytest.fixture
def queue_manager() -> QueueManager:
    """Create a QueueManager instance for testing."""
    return QueueManager()


def test_queue_manager_init_with_params():
    queue_manager = QueueManager(
        user="test_user",
        password="test_password",
        host="test_host",
        port=5672,
        queue_name="test_queue",
    )
    expected_url = "amqp://test_user:test_password@test_host:5672/"
    assert queue_manager.connection_url == expected_url
    assert queue_manager.queue_name == "test_queue"


def test_queue_manager_init_with_connection_url():
    connection_url = "amqp://test_user:test_password@test_host:5672/"
    queue_manager = QueueManager(connection_url=connection_url)
    assert queue_manager.connection_url == connection_url
    assert queue_manager.queue_name == DEFAULT_QUEUE_NAME


def test_queue_manager_init_with_default_url():
    queue_manager = QueueManager()
    assert queue_manager.connection_url == DEFAULT_RABBITMQ_URL
    assert queue_manager.queue_name == DEFAULT_QUEUE_NAME


@patch("pika.BlockingConnection")
def test_queue_manager_connect(mock_blocking_connection: MagicMock, queue_manager: QueueManager):
    queue_manager.connect()
    mock_blocking_connection.assert_called_once_with(pika.URLParameters(DEFAULT_RABBITMQ_URL))
    assert queue_manager.connection is not None
    assert queue_manager.channel is not None


@patch("pika.BlockingConnection")
def test_queue_manager_close_connection(
    mock_blocking_connection: MagicMock, queue_manager: QueueManager
):
    queue_manager.connect()
    queue_manager.close_connection()
    queue_manager.connection.close.assert_called_once()


@patch("pika.BlockingConnection")
def test_queue_manager_init_queue(mock_blocking_connection: MagicMock, queue_manager: QueueManager):
    queue_manager.channel = MagicMock()
    queue_manager.channel.queue_declare = MagicMock()

    queue_manager.init_queue()
    queue_manager.channel.queue_declare.assert_called_once_with(
        queue=DEFAULT_QUEUE_NAME, durable=True, arguments={"x-max-priority": 10}
    )
    queue_manager.channel.basic_qos.assert_called_once_with(prefetch_count=1)


@patch("pika.BlockingConnection")
def test_queue_manager_publish(mock_blocking_connection: MagicMock, queue_manager: QueueManager):
    queue_manager.channel = MagicMock()
    queue_manager.channel.basic_publish = MagicMock()

    task = {"uuid": "test_uuid"}
    queue_manager.publish(task, priority=5)

    queue_manager.channel.basic_publish.assert_called_once_with(
        exchange="",
        routing_key=DEFAULT_QUEUE_NAME,
        body=json.dumps(task),
        properties=pika.BasicProperties(
            delivery_mode=2,
            priority=5,
        ),
    )


@patch("pika.BlockingConnection")
def test_queue_manager_consume(
    mock_blocking_connection: MagicMock,
    queue_manager: QueueManager,
):
    queue_manager.channel = MagicMock()
    queue_manager.channel.basic_consume = MagicMock()
    queue_manager.channel.start_consuming = MagicMock()

    def process_fn(task, ack, nack):
        pass

    queue_manager.consume(process_fn)
    queue_manager.channel.basic_consume.assert_called_once()
    queue_manager.channel.start_consuming.assert_called_once()
