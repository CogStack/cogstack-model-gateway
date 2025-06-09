import json
import logging
import threading
from functools import partial, wraps

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("cmg.common")

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
        max_concurrent_tasks: int = 1,
    ):
        logging.getLogger("pika").setLevel(logging.WARNING)

        if user and password and host and port:
            self.connection_url = f"amqp://{user}:{password}@{host}:{port}/"
        elif connection_url:
            self.connection_url = connection_url
        else:
            self.connection_url = DEFAULT_RABBITMQ_URL

        self.queue_name = queue_name if queue_name else DEFAULT_QUEUE_NAME
        self.connection = None
        self.channel = None
        self.semaphore = threading.Semaphore(max_concurrent_tasks)
        self._max_concurrent_tasks = max_concurrent_tasks

    def connect(self):
        """Connect to the RabbitMQ server and initialize a channel."""
        log.debug("Connecting to queue '%s'", self.queue_name)
        self.connection = pika.BlockingConnection(pika.URLParameters(self.connection_url))
        self.channel = self.connection.channel()

    def close_connection(self):
        """Close the connection to the RabbitMQ server."""
        log.debug("Closing connection to queue '%s'", self.queue_name)
        if self.connection:
            self.connection.close()

    @staticmethod
    def with_connection(func):
        """Decorator to connect to the RabbitMQ server before calling the decorated function."""

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
    def init_queue(self, prefetch_count: int = None):
        """Initialize a queue and set the prefetch count."""
        log.info("Initializing queue '%s'", self.queue_name)
        self.channel.queue_declare(
            queue=self.queue_name, durable=True, arguments={"x-max-priority": 10}
        )
        self.channel.basic_qos(prefetch_count=prefetch_count or self._max_concurrent_tasks)
        log.info("Queue '%s' initialized", self.queue_name)

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def publish(self, task: dict, priority: int):
        """Publish a task to the queue with the specified priority."""
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

    @staticmethod
    def _ack_message(ch: BlockingChannel, delivery_tag: int):
        """Set a threadsafe acknowledgement callback."""
        cb = partial(ch.basic_ack, delivery_tag=delivery_tag)
        ch.connection.add_callback_threadsafe(cb)

    @staticmethod
    def _nack_message(ch: BlockingChannel, delivery_tag: int, requeue: bool = True):
        """Set a threadsafe negative acknowledgement callback."""
        cb = partial(ch.basic_nack, delivery_tag=delivery_tag, requeue=requeue)
        ch.connection.add_callback_threadsafe(cb)

    @staticmethod
    def _on_message(
        ch: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
        process_fn: callable,
        semaphore: threading.Semaphore,
        threads: list,
    ):
        """Process a message event from the queue by starting a new thread to handle it.

        It is important to process messages in separate threads to avoid blocking the main thread,
        which should always remain responsive, ensuring that healthchecks and other critical
        operations can be performed.
        """

        def _process_threadsafe():
            """Process a message."""
            ack = partial(QueueManager._ack_message, ch, method.delivery_tag)
            nack = partial(QueueManager._nack_message, ch, method.delivery_tag)

            with semaphore:
                try:
                    task = json.loads(body)
                except json.JSONDecodeError:
                    log.error("Invalid task received: %s", body)
                    nack(requeue=False)
                    return

                log.info("Received task '%s'", task)
                try:
                    process_fn(task, ack, nack)
                except Exception as e:
                    log.error("Error processing task '%s': %s", task["uuid"], e)
                    nack(requeue=False)

        t = threading.Thread(target=_process_threadsafe)
        t.start()
        threads.append(t)

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def consume(self, process_fn: callable):
        """Consume tasks from the queue and process them using the specified function."""
        threads = []
        try:
            on_message = partial(
                QueueManager._on_message,
                process_fn=process_fn,
                semaphore=self.semaphore,
                threads=threads,
            )
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=on_message)
            self.channel.start_consuming()
        except Exception:
            log.exception("Error consuming tasks from queue '%s'", self.queue_name)
            raise
        finally:
            for t in threads:
                t.join()
            self.channel.stop_consuming()
            log.info("Stopped consuming tasks from queue '%s'", self.queue_name)

    @with_connection
    def is_queue_empty(self):
        """Check if the queue is empty."""
        try:
            queue = self.channel.queue_declare(queue=self.queue_name, passive=True)
            return queue.method.message_count == 0
        except pika.exceptions.ChannelClosed:
            return True
