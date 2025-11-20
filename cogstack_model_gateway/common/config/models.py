import re

from pydantic import BaseModel, Field, field_validator


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
        gt=0,
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

    Currently supports resource constraints, but can be extended to include
    other Docker Compose deploy options like restart_policy, labels, placement, etc.
    """

    resources: DeployResources | None = Field(None, description="Resource limits and reservations")


class OnDemandModel(BaseModel):
    """Configuration for an on-demand model that can be auto-deployed."""

    service_name: str = Field(
        ...,
        description="Docker service/container name for the model",
        examples=["medcat-snomed-large", "medcat-umls-small"],
    )
    model_uri: str = Field(
        ...,
        description="URI pointing to the model artifact (e.g., MLflow model URI)",
        examples=[
            "s3://models/medcat/snomed_large_v1.0",
            "models:/medcat-snomed/Production",
            "runs:/abc123/model",
        ],
    )
    idle_ttl: int | None = Field(
        None,
        description="Time in seconds after which an idle model is removed (overrides default)",
        gt=0,
        examples=[3600, 7200, 86400],
    )
    description: str | None = Field(
        None,
        description="Human-readable description of the model",
        examples=["Large SNOMED CT model for clinical NLP"],
    )
    deploy: DeploySpec = Field(
        default_factory=DeploySpec,
        description="Deployment specification including resource constraints",
    )

    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, v: str) -> str:
        """Validate service name follows Docker naming constraints.

        Docker container names must:
        - Start with alphanumeric character
        - Contain only alphanumeric, underscore, period, or hyphen
        """
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", v):
            raise ValueError(
                f"Invalid service name: {v}. Must start with alphanumeric and contain "
                "only alphanumeric, underscore, period, or hyphen characters"
            )
        if len(v) > 255:
            raise ValueError(f"Service name too long: {v}. Maximum length is 255 characters")
        return v

    @field_validator("model_uri")
    @classmethod
    def validate_model_uri(cls, v: str) -> str:
        """Validate model URI format."""
        if not v or not v.strip():
            raise ValueError("Model URI cannot be empty")
        # Basic validation - just ensure it's not empty
        # More specific validation (s3://, models:/, runs:/) can be added if needed
        return v.strip()


class AutoDeploymentConfig(BaseModel):
    """Configuration for automatic model deployment behaviour."""

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


class AutoDeployment(BaseModel):
    """Auto-deployment configuration including behaviour and on-demand models."""

    config: AutoDeploymentConfig = Field(
        default_factory=AutoDeploymentConfig,
        description="Auto-deployment behaviour configuration",
    )
    on_demand: list[OnDemandModel] = Field(
        default_factory=list,
        description="List of models available for on-demand deployment",
    )

    @field_validator("on_demand")
    @classmethod
    def validate_unique_service_names(cls, v: list[OnDemandModel]) -> list[OnDemandModel]:
        """Ensure all service names are unique."""
        service_names = [model.service_name for model in v]
        duplicates = [name for name in service_names if service_names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate service names found in on_demand models: {set(duplicates)}"
            )
        return v

    @model_validator(mode="after")
    def apply_default_idle_ttl(self) -> "AutoDeployment":
        """Apply default_idle_ttl to on-demand models that don't have an explicit idle_ttl."""
        for model in self.on_demand:
            if model.idle_ttl is None:
                model.idle_ttl = self.config.default_idle_ttl
        return self


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


class MLflowS3Config(BaseModel):
    """MLflow S3 configuration for model containers."""

    access_key_id: str | None = Field(None, description="AWS access key ID")
    secret_access_key: str | None = Field(None, description="AWS secret access key")
    endpoint_url: str = Field("http://minio:9000", description="MLflow artifact storage endpoint")


class MLflowConfig(BaseModel):
    """MLflow configuration for model containers."""

    s3: MLflowS3Config = Field(default_factory=MLflowS3Config, description="MLflow S3 settings")
    tracking_uri: str = Field("http://mlflow-ui:5000", description="MLflow tracking server URI")
    tracking_username: str = Field("admin", description="MLflow tracking server username")
    tracking_password: str = Field("password", description="MLflow tracking server password")
    enable_system_metrics_logging: bool = Field(True, description="Enable system metrics logging")


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
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig, description="MLflow configuration")
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication configuration")
    proxy: ProxyConfig = Field(default_factory=ProxyConfig, description="Proxy configuration")
    health_check: HealthCheckConfig = Field(
        default_factory=HealthCheckConfig, description="Health check configuration"
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    user: str = Field("admin", description="Database username")
    password: str = Field("admin", description="Database password")
    name: str = Field("cmg_tasks", description="Database name")
    host: str = Field("postgres", description="Database host")
    port: int = Field(5432, description="Database port")


class ObjectStoreConfig(BaseModel):
    """Object store configuration."""

    host: str = Field("minio", description="Object store host")
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
    host: str = Field("rabbitmq", description="Queue host")
    port: int = Field(5672, description="Queue port")


class SchedulerConfig(BaseModel):
    """Scheduler service configuration."""

    max_concurrent_tasks: int = Field(1, description="Max concurrent tasks")
    metrics_port: int = Field(8001, description="Prometheus metrics port")


class RipperConfig(BaseModel):
    """Ripper service configuration."""

    interval: int = Field(60, description="Ripper interval in seconds")
    metrics_port: int = Field(8002, description="Prometheus metrics port")


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

    def get_on_demand_model(self, service_name: str) -> OnDemandModel | None:
        """Get configuration for a specific on-demand model by service name."""
        for model in self.models.deployment.auto.on_demand:
            if model.service_name == service_name:
                return model
        return None

    def list_on_demand_models(self) -> list[OnDemandModel]:
        """Get list of all configured on-demand models."""
        return self.models.deployment.auto.on_demand

    def get_auto_deployment_config(self) -> AutoDeploymentConfig:
        """Get auto-deployment behaviour configuration."""
        return self.models.deployment.auto.config

    def get_manual_deployment_config(self) -> ManualDeployment:
        """Get manual deployment configuration."""
        return self.models.deployment.manual

    def get_static_deployment_config(self) -> StaticDeployment:
        """Get static model discovery configuration."""
        return self.models.deployment.static
