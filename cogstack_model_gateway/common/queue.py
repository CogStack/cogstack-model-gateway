import json
import logging
from functools import partial, wraps

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("cmg")

DEFAULT_RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
DEFAULT_QUEUE_NAME = "cmg_tasks"


class QueueManager:
    def __init__(
        self,
        user: str = None,
        password: str = None,
        host: str = None,
        port: int = None,
        queue_name: str = None,
        connection_url: str = None,
    ):
        if user and password and host and port:
            self.connection_url = f"amqp://{user}:{password}@{host}:{port}/"
        elif connection_url:
            self.connection_url = connection_url
        else:
            self.connection_url = DEFAULT_RABBITMQ_URL

        self.queue_name = queue_name if queue_name else DEFAULT_QUEUE_NAME
        self.connection = None
        self.channel = None

    def connect(self):
        log.debug("Connecting to queue '%s'", self.queue_name)
        self.connection = pika.BlockingConnection(pika.URLParameters(self.connection_url))
        self.channel = self.connection.channel()

    def close_connection(self):
        log.debug("Closing connection to queue '%s'", self.queue_name)
        if self.connection:
            self.connection.close()

    @staticmethod
    def with_connection(func):
        @wraps(func)
        def wrapper(self: "QueueManager", *args, **kwargs):
            self.connect()
            try:
                result = func(self, *args, **kwargs)
            finally:
                self.close_connection()
            return result

        return wrapper

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def init_queue(self):
        log.info("Initializing queue '%s'", self.queue_name)
        self.channel.queue_declare(
            queue=self.queue_name, durable=True, arguments={"x-max-priority": 10}
        )
        log.info("Queue '%s' initialized", self.queue_name)

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def publish(self, task: dict, priority: int):
        log.info("Publishing task %s with priority %s", task["uuid"], priority)
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=json.dumps(task),
            properties=pika.BasicProperties(
                delivery_mode=2,
                priority=priority,
            ),
        )
        log.info("Task '%s' published", task["uuid"])

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def consume(self, process_fn: callable):
        def callback(
            ch: BlockingChannel, method: Basic.Deliver, properties: BasicProperties, body: bytes
        ):
            try:
                task = json.loads(body)
            except json.JSONDecodeError:
                log.error("Invalid task received: %s", body)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            log.info("Received task '%s'", task)
            try:
                ack = partial(ch.basic_ack, delivery_tag=method.delivery_tag)
                nack = partial(ch.basic_nack, delivery_tag=method.delivery_tag, requeue=True)
                process_fn(task, ack, nack)
            except Exception as e:
                log.error("Error processing task '%s': %s", task["uuid"], e)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        try:
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback)
            self.channel.start_consuming()
        except Exception:
            log.exception("Error consuming tasks from queue '%s'", self.queue_name)
