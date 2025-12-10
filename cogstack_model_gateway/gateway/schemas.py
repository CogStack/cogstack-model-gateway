from typing import Any

from pydantic import BaseModel, Field

from cogstack_model_gateway.common.config.models import DeploySpec, TrackingMetadata
from cogstack_model_gateway.common.models import OnDemandModelConfig


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
    deploy: DeploySpec | None = Field(None, description="Deployment specification")
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


class OnDemandModelCreate(BaseModel):
    """Schema for creating a new on-demand model configuration."""

    model_name: str = Field(
        ...,
        description="Docker service/container name for the model",
        examples=["medcat-snomed-large", "medcat-umls-small"],
    )
    model_uri: str | None = Field(
        None,
        description=(
            "URI pointing to the model artifact (e.g. MLflow model URI)."
            " Either `tracking_id` or `model_uri` must be provided (unless `inherit_config=true`)."
            " If both provided, `model_uri` takes precedence."
        ),
        examples=["models:/medcat-snomed/Production", "s3://models/medcat/v1.0.zip"],
    )
    tracking_id: str | None = Field(
        None,
        description=(
            "Tracking server run ID (e.g. MLflow run ID) to resolve model URI."
            " Either `tracking_id` or `model_uri` must be provided (unless `inherit_config=true`)."
        ),
        examples=["a1b2c3d4e5f6", "mlflow-run-123"],
    )
    idle_ttl: int | None = Field(
        None,
        description=(
            "Idle TTL in seconds (uses system default if omitted or inherits from previous config))"
        ),
        gt=0,
        examples=[3600, 7200],
    )
    description: str | None = Field(
        None,
        description="Human-readable description of the model",
        examples=["Large SNOMED CT model for clinical NLP"],
    )
    deploy: DeploySpec | None = Field(
        None, description="Deployment specification (resources, placement, etc)"
    )
    replace_enabled: bool = Field(
        True,
        description=(
            "If true and an enabled config exists, disable it and create a new one,"
            " replacing it as the enabled one."
        ),
    )
    inherit_config: bool = Field(
        True,
        description=(
            "If true, inherit settings from the currently enabled config for this model name."
            " Only explicitly provided fields will override inherited values."
        ),
    )


class OnDemandModelUpdate(BaseModel):
    """Schema for updating an on-demand model configuration.

    All fields are optional; only provided fields will be updated. Use clear_* flags to explicitly
    unset optional fields.
    """

    model_uri: str | None = Field(None, description="New URI pointing to the model artifact")
    tracking_id: str | None = Field(
        None,
        description=(
            "Update tracking ID. If `model_uri` is not provided, the tracking ID is used to"
            " resolve the model URI."
        ),
    )
    idle_ttl: int | None = Field(None, description="New idle TTL in seconds", gt=0)
    description: str | None = Field(None, description="New description")
    deploy: DeploySpec | None = Field(None, description="New deployment specification")

    clear_tracking_id: bool = Field(False, description="Clear `tracking_id` reference")
    clear_idle_ttl: bool = Field(False, description="Set `idle_ttl` to null (use system default)")
    clear_description: bool = Field(False, description="Set `description` to null")
    clear_deploy: bool = Field(False, description="Set `deploy` specification to null")


class OnDemandModelListResponse(BaseModel):
    """Response schema for listing on-demand model configurations."""

    configs: list[OnDemandModelConfig] = Field(
        ..., description="List of on-demand model configurations"
    )
    total: int = Field(..., description="Total number of configurations returned")
