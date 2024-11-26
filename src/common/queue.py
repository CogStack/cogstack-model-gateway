import json
from functools import wraps

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
from tenacity import retry, stop_after_attempt, wait_exponential

RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
QUEUE_NAME = "tasks"


class QueueManager:
    def __init__(self, queue_name: str = QUEUE_NAME, rabbitmq_url: str = RABBITMQ_URL):
        self.queue_name = queue_name
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None

    def connect(self):
        self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
        self.channel = self.connection.channel()

    def close_connection(self):
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
        self.channel.queue_declare(
            queue=self.queue_name, durable=True, arguments={"x-max-priority": 10}
        )
        print(f"Queue '{self.queue_name}' initialized.")

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def publish(self, task: dict, priority: int):
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=json.dumps(task),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                priority=priority,
            ),
        )
        print(f"Task {task['uuid']} published with priority {priority}")

    @with_connection
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def consume(self, process_fn: callable):
        def callback(
            ch: BlockingChannel, method: Basic.Deliver, properties: BasicProperties, body: bytes
        ):
            task = json.loads(body)
            try:
                process_fn(task)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                print(f"Error processing task {task['uuid']}: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback)
        self.channel.start_consuming()
