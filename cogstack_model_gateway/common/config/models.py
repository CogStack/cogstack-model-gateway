import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ResourceLimits(BaseModel):
    """Resource limits for Docker containers.

    Follows Docker Compose resource specification format.
    """

    memory: str | None = Field(
        None,
        description="Memory limit (e.g., '4g', '512m')",
        examples=["4g", "512m", "2048m"],
    )
    cpus: str | None = Field(
        None,
        description="CPU limit as string (e.g., '2.0', '0.5')",
        examples=["2.0", "1.5", "0.5"],
    )

    @field_validator("memory")
    @classmethod
    def validate_memory_format(cls, v: str | None) -> str | None:
        """Validate memory format matches Docker specification."""
        if v is None:
            return v
        if not re.match(r"^\d+(\.\d+)?[kmgKMG]?$", v):
            raise ValueError(
                f"Invalid memory format: {v}. Expected format like '4g', '512m', '2048m'"
            )
        return v


class DeployResources(BaseModel):
    """Resource constraints for container deployment.

    Mirrors Docker Compose deploy.resources specification.
    """

    limits: ResourceLimits | None = Field(
        None, description="Maximum resources the container can use"
    )
    reservations: ResourceLimits | None = Field(
        None, description="Minimum resources guaranteed for the container"
    )


class DeploySpec(BaseModel):
    """Deployment specification for model containers.

    Mirrors Docker Compose deploy specification.
    """

    resources: DeployResources | None = Field(None, description="Resource limits and reservations")


class TrackingMetadata(BaseModel):
    """Model metadata from MLflow tracking server.

    Dict representation of mlflow.models.ModelInfo, returned by TrackingClient.get_model_metadata().
    """

    uuid: str = Field(..., description="Model UUID")
    run_id: str = Field(..., description="MLflow run ID that produced the model")
    artifact_path: str = Field(..., description="Path to model artifact within the run")
    signature: dict[str, Any] = Field(..., description="Model signature (inputs/outputs/params)")
    flavors: dict[str, Any] = Field(..., description="Model flavors (e.g. python_function)")
    utc_time_created: str = Field(..., description="UTC timestamp when model was created")
    mlflow_version: str = Field(..., description="MLflow version used to log the model")


class AutoDeployment(BaseModel):
    """Auto-deployment configuration for on-demand models."""

    health_check_timeout: int = Field(
        300,
        description="Max time in seconds to wait for a model to become healthy after deployment",
        gt=0,
        examples=[300, 600],
    )
    default_idle_ttl: int = Field(
        3600,
        description="Default idle TTL in seconds (if not specified per-model)",
        gt=0,
        examples=[3600, 7200, 86400],
    )
    max_concurrent_deployments: int | None = Field(
        None,
        description="Maximum number of models that can be deployed concurrently (None = unlimited)",
        gt=0,
        examples=[3, 5, 10],
    )
    deployment_retry_attempts: int = Field(
        2,
        description="Number of times to retry a failed deployment",
        ge=0,
        examples=[0, 1, 2, 3],
    )
    require_model_uri_validation: bool = Field(
        False,
        description="Whether to validate that the model URI exists before creating configs",
    )


class ManualDeployment(BaseModel):
    """Configuration for manual model deployments via POST /models API."""

    default_ttl: int = Field(
        86400,
        description="Default TTL in seconds for manually deployed models (1 day)",
        gt=0,
        examples=[3600, 86400, 604800],
    )
    allow_ttl_override: bool = Field(
        True,
        description="Whether users can override the default TTL when deploying models",
    )
    max_ttl: int | None = Field(
        None,
        description="Maximum allowed TTL in seconds (None = unlimited)",
        gt=0,
        examples=[604800, 2592000],  # 1 week, 30 days
    )
    require_model_uri_validation: bool = Field(
        False,
        description="Whether to validate that the model URI exists before deployment",
    )


class StaticDeployment(BaseModel):
    """Configuration for static/always-on models from CMS stack."""

    pass


class ModelsDeployment(BaseModel):
    """Complete deployment configuration for all model types."""

    auto: AutoDeployment = Field(
        default_factory=AutoDeployment,
        description="Auto-deployment configuration for on-demand models",
    )
    manual: ManualDeployment = Field(
        default_factory=ManualDeployment,
        description="Configuration for manual model deployments",
    )
    static: StaticDeployment = Field(
        default_factory=StaticDeployment,
        description="Configuration for static model management",
    )
    use_ip_addresses: bool = Field(
        False,
        description=(
            "Use IP addresses instead of container names when attempting to connect to CogStack"
            " Model Serve instances. Set to true when components (scheduler, tests) run outside"
            " Docker network."
        ),
    )


class ModelsConfig(BaseModel):
    """Model-related configuration."""

    deployment: ModelsDeployment = Field(
        default_factory=ModelsDeployment,
        description="Deployment configuration for all model types",
    )


class HealthCheckConfig(BaseModel):
    """Health check configuration for model containers."""

    interval: int = Field(90, description="Health check interval in seconds", gt=0)
    timeout: int = Field(10, description="Health check timeout in seconds", gt=0)
    retries: int = Field(3, description="Health check retries", ge=0)
    start_period: int = Field(60, description="Health check start period in seconds", gt=0)


class S3Config(BaseModel):
    """S3/MinIO configuration for artifact storage."""

    access_key_id: str | None = Field(None, description="S3/MinIO access key")
    secret_access_key: str | None = Field(None, description="S3/MinIO secret access key")
    endpoint_url: str = Field("http://minio:9000", description="S3/MinIO endpoint URL")


class AuthConfig(BaseModel):
    """Authentication configuration for model containers."""

    user_enabled: bool = Field(False, description="Enable user authentication")
    jwt_secret: str | None = Field(None, description="JWT secret for token signing")
    access_token_expire_seconds: int = Field(3600, description="Access token expiration time", gt=0)
    database_url: str = Field(
        "sqlite+aiosqlite:///./cms-users.db", description="Authentication database URL"
    )


class ProxyConfig(BaseModel):
    """Proxy configuration for model containers."""

    http_proxy: str | None = Field(None, description="HTTP proxy URL")
    https_proxy: str | None = Field(None, description="HTTPS proxy URL")
    no_proxy: str = Field(
        "mlflow-ui,minio,graylog,auth-db,localhost",
        description="Comma-separated list of hosts to exclude from proxying",
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    user: str = Field("admin", description="Database username")
    password: str = Field("admin", description="Database password")
    name: str = Field("cmg_tasks", description="Database name")
    host: str = Field("db", description="Database host")
    port: int = Field(5432, description="Database port")


class ObjectStoreConfig(BaseModel):
    """Object store configuration."""

    host: str = Field("object-store", description="Object store host")
    port: int = Field(9000, description="Object store port")
    access_key: str = Field("admin", description="Object store access key")
    secret_key: str = Field("admin123", description="Object store secret key")
    bucket_tasks: str = Field("cmg-tasks", description="Bucket for task payloads")
    bucket_results: str = Field("cmg-results", description="Bucket for task results")


class QueueConfig(BaseModel):
    """Message queue configuration."""

    user: str = Field("admin", description="Queue username")
    password: str = Field("admin", description="Queue password")
    name: str = Field("cmg_tasks", description="Queue name")
    host: str = Field("queue", description="Queue host")
    port: int = Field(5672, description="Queue port")


class SchedulerConfig(BaseModel):
    """Scheduler service configuration."""

    max_concurrent_tasks: int = Field(1, description="Max concurrent tasks")
    metrics_port: int = Field(8001, description="Prometheus metrics port")


class RipperConfig(BaseModel):
    """Ripper service configuration."""

    interval: int = Field(60, description="Ripper interval in seconds")
    metrics_port: int = Field(8002, description="Prometheus metrics port")


class TrackingConfig(BaseModel):
    """Tracking server configuration.

    Used by:
    - CMG services (gateway/scheduler) to connect to tracking server for fetching model metadata
    - CMS containers as environment variables for logging training runs and metrics
    """

    uri: str = Field("http://mlflow-ui:5000", description="Tracking server URI")
    username: str = Field("admin", description="Tracking server username")
    password: str = Field("password", description="Tracking server password")
    s3: S3Config = Field(default_factory=S3Config, description="Tracking server artifact store")
    enable_system_metrics_logging: bool = Field(
        True, description="Enable system metrics logging (for CMS containers)"
    )


class LabelsConfig(BaseModel):
    """Docker labels used by CogStack Model Gateway.

    These labels are used for:
    - Identifying CogStack ModelServe containers (CMS labels)
    - Marking containers as managed by the gateway (CMG labels)
    - Tracking TTL and other metadata
    """

    # CMS-related labels (for discovering static models)
    cms_model_label: str = Field(
        "org.cogstack.model-serve",
        description="Label to identify CogStack ModelServe containers",
    )
    cms_model_uri_label: str = Field(
        "org.cogstack.model-serve.uri",
        description="Label storing the model URI on CMS containers",
    )

    # CMG-related labels (for managing models deployed through the gateway)
    deployment_type_label: str = Field(
        "org.cogstack.model-gateway.deployment-type",
        description="Label storing deployment type (auto/manual/static)",
    )
    managed_by_label: str = Field(
        "org.cogstack.model-gateway.managed-by",
        description="Label to identify which system manages a container",
    )
    managed_by_value: str = Field(
        "cmg",
        description="Value for managed_by_label for CMG-managed containers",
    )
    ttl_label: str = Field(
        "org.cogstack.model-gateway.ttl",
        description="Label storing TTL (time-to-live) in seconds for auto-deployed models",
    )


class CMSConfig(BaseModel):
    """CogStack ModelServe related configuration."""

    host_url: str = Field("https://proxy/cms", description="CMS host URL")
    project_name: str = Field("cms", description="CMS Docker Compose project name")
    server_port: int = Field(8000, description="CMS server port")
    network: str = Field("cogstack-model-serve_cms", description="CMS Docker network")
    image: str = Field(
        "cogstacksystems/cogstack-modelserve:latest",
        description="Docker image for CogStack ModelServe",
    )
    volumes: dict[str, str] = Field(
        default_factory=lambda: {"retrained-models": "/app/model/retrained"},
        description="Volume mappings for model containers (volume_name: container_path)",
    )
    gelf_input_uri: str = Field("http://graylog:12201", description="GELF input URI for logging")
    enable_evaluation_apis: bool = Field(True, description="Enable CMS evaluation APIs")
    enable_previews_apis: bool = Field(True, description="Enable CMS preview APIs")
    enable_training_apis: bool = Field(True, description="Enable CMS training APIs")
    tracking: TrackingConfig = Field(
        default_factory=TrackingConfig, description="Model tracking configuration"
    )
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication configuration")
    proxy: ProxyConfig = Field(default_factory=ProxyConfig, description="Proxy configuration")
    health_check: HealthCheckConfig = Field(
        default_factory=HealthCheckConfig, description="Health check configuration"
    )


class Config(BaseModel):
    """Root configuration schema for CogStack Model Gateway."""

    cms: CMSConfig = Field(default_factory=CMSConfig, description="CogStack ModelServe config")

    db: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    object_store: ObjectStoreConfig = Field(
        default_factory=ObjectStoreConfig, description="Object store configuration"
    )
    queue: QueueConfig = Field(
        default_factory=QueueConfig, description="Message queue configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig, description="Scheduler configuration"
    )
    ripper: RipperConfig = Field(default_factory=RipperConfig, description="Ripper configuration")
    tracking: TrackingConfig = Field(
        default_factory=TrackingConfig, description="Model tracking server configuration"
    )

    models: ModelsConfig = Field(
        default_factory=ModelsConfig, description="Model deployment and discovery configuration"
    )
    labels: LabelsConfig = Field(
        default_factory=LabelsConfig, description="Docker labels configuration"
    )

    # Runtime managers (populated after initialization)
    database_manager: object | None = Field(
        None, description="Database manager instance", exclude=True
    )
    task_object_store_manager: object | None = Field(
        None, description="Task object store manager instance", exclude=True
    )
    results_object_store_manager: object | None = Field(
        None, description="Results object store manager instance", exclude=True
    )
    queue_manager: object | None = Field(None, description="Queue manager instance", exclude=True)
    task_manager: object | None = Field(None, description="Task manager instance", exclude=True)
    model_manager: object | None = Field(None, description="Model manager instance", exclude=True)
    tracking_client: object | None = Field(
        None, description="Model tracking client instance", exclude=True
    )

    _was_tracking_explicit: bool = False

    @model_validator(mode="before")
    @classmethod
    def mark_explicit_tracking(cls, data: dict) -> dict:
        """Mark if tracking config was explicitly provided before validation."""
        if isinstance(data, dict):
            if "tracking" in data:
                data["_was_tracking_explicit"] = True
        return data

    @model_validator(mode="after")
    def default_tracking_from_cms(self) -> "Config":
        """Use cms.tracking as default tracking config if not explicitly provided.

        This allows a single tracking server config (cms.tracking) to be used for both
        CMS containers and CMG services when they point to the same server.
        """
        if not self._was_tracking_explicit:
            self.tracking = self.cms.tracking.model_copy(deep=True)
        return self

    def get_default_idle_ttl(self) -> int:
        """Get the default idle TTL for auto-deployed on-demand models."""
        return self.models.deployment.auto.default_idle_ttl

    def get_auto_deployment_config(self) -> AutoDeployment:
        """Get auto-deployment behaviour configuration."""
        return self.models.deployment.auto

    def get_manual_deployment_config(self) -> ManualDeployment:
        """Get manual deployment configuration."""
        return self.models.deployment.manual

    def get_static_deployment_config(self) -> StaticDeployment:
        """Get static model discovery configuration."""
        return self.models.deployment.static
