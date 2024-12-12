import io
import logging
import uuid

from minio import Minio

log = logging.getLogger("cmg")

DEFAULT_MINIO_ENDPOINT = "localhost:9000"
DEFAULT_ACCESS_KEY = "admin"
DEFAULT_SECRET_KEY = "admin"
DEFAULT_MINIO_BUCKET = "cmg-tasks"


class ObjectStoreManager:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        secure: bool = False,
        default_bucket: str = None,
    ):
        if host and port:
            self.endpoint = f"{host}:{port}"
        elif endpoint:
            self.endpoint = endpoint
        else:
            self.endpoint = DEFAULT_MINIO_ENDPOINT

        self.access_key = access_key if access_key else DEFAULT_ACCESS_KEY
        self.secret_key = secret_key if secret_key else DEFAULT_SECRET_KEY
        self.secure = secure

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

        self.default_bucket = default_bucket if default_bucket else DEFAULT_MINIO_BUCKET
        self.create_bucket(self.default_bucket)

    def create_bucket(self, bucket_name: str) -> None:
        log.info("Creating bucket '%s'", bucket_name)
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                log.info("Bucket '%s' created", bucket_name)
            else:
                log.info("Bucket '%s' already exists", bucket_name)
        except Exception as e:
            log.error("Error creating bucket '%s': %s", bucket_name, e)
            raise

    def upload_object(
        self, file_data: bytes, filename: str, bucket_name: str = None, prefix: str = None
    ) -> str:
        if bucket_name and bucket_name != self.default_bucket:
            self.create_bucket(bucket_name)
        else:
            bucket_name = self.default_bucket

        log.info("Uploading file '%s' to bucket '%s'", filename, bucket_name)
        object_key = f"{prefix if prefix else uuid.uuid4()}_{filename}"
        self.client.put_object(
            bucket_name, object_key, data=io.BytesIO(file_data), length=len(file_data)
        )
        log.info("File '%s' stored with object key '%s'", filename, object_key)

        return object_key

    def get_object(self, object_key: str, bucket_name: str = None) -> bytes:
        bucket_name = bucket_name if bucket_name else self.default_bucket
        log.info("Fetching object '%s' from bucket '%s'", object_key, bucket_name)
        return self.client.get_object(bucket_name, object_key).read()
