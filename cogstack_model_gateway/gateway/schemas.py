from typing import Any

from pydantic import BaseModel, Field

from cogstack_model_gateway.common.config.models import DeployResources


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


class ModelResponse(BaseModel):
    """Unified model response schema.

    Non-verbose response (default) includes only:
    - name, uri, is_running

    Verbose response (verbose=true) additionally includes:
    - description, model_type, deployment_type, idle_ttl, resources, runtime, tracking
    """

    # Non-verbose fields
    name: str = Field(..., description="Model service/container name")
    uri: str | None = Field(None, description="Tracking server model URI")
    is_running: bool = Field(..., description="Whether the model server is currently running")

    # Verbose-only fields
    description: str | None = Field(None, description="Human-readable model description")
    model_type: str | None = Field(None, description="Model type (e.g. 'medcat_deid')")
    deployment_type: str | None = Field(
        None, description="Deployment type: 'auto', 'manual', or 'static'"
    )
    idle_ttl: int | None = Field(None, description="Idle TTL in seconds (for 'auto' deployments)")
    resources: DeployResources | None = Field(None, description="Deployment resource specification")
    runtime: dict[str, Any] | None = Field(
        None, description="CMS /info response (verbose only, running models only)"
    )
    tracking: TrackingMetadata | None = Field(
        None, description="Model metadata from the configured tracking server (verbose only)"
    )


class ModelsListResponse(BaseModel):
    """Response schema for GET /models/ endpoint."""

    running: list[ModelResponse] = Field(..., description="Currently running model containers")
    on_demand: list[ModelResponse] = Field(..., description="Models available on-demand")
