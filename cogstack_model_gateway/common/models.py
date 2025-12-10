import json
import logging
import re
from datetime import UTC, datetime
from enum import Enum
from functools import wraps

from dateutil import parser
from pydantic import ConfigDict, ValidationError, computed_field, field_validator
from sqlalchemy import Index, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, select

from cogstack_model_gateway.common.config import get_config
from cogstack_model_gateway.common.config.models import DeploySpec
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.exceptions import ConfigConflictError, ConfigValidationError

log = logging.getLogger("cmg.common")


class ModelDeploymentType(Enum):
    AUTO = "auto"
    MANUAL = "manual"
    STATIC = "static"


class Model(SQLModel, table=True):
    """Deployed model server record for lifecycle and usage management."""

    model_name: str = Field(primary_key=True, description="Unique model service name")
    deployment_type: ModelDeploymentType = Field(
        description="Type of deployment: 'auto', 'manual', or 'static'"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="When the model record was created (UTC, ISO format)",
    )
    ready: bool = Field(
        default=False,
        description="Whether the model deployment is complete and ready to serve requests",
    )
    last_used_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Last time a request for this model was received (UTC, ISO format)",
    )
    idle_ttl: int | None = Field(default=None, description="Idle TTL in seconds")

    __table_args__ = (
        Index("ix_model_usage_last_used_at", "last_used_at"),
        Index("ix_model_usage_deployment_type", "deployment_type"),
    )


class OnDemandModelConfig(SQLModel, table=True):
    """Persistent configuration for on-demand models with version history.

    Supports soft delete via the `enabled` field, allowing multiple historical versions of a
    configuration for the same `model_name`. A partial unique index ensures only one enabled
    configuration exists per `model_name` at any time.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(default=None, primary_key=True)
    model_name: str = Field(index=True, description="Docker service/container name for the model")
    model_uri: str = Field(description="URI pointing to the model artifact (e.g. MLflow model URI)")
    tracking_id: str | None = Field(
        default=None,
        description="Tracking server run ID (e.g. MLflow run ID) that produced this model",
    )
    idle_ttl: int = Field(
        description="Idle TTL in seconds (time after which an idle model is removed)", gt=0
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the model"
    )
    deploy_spec_json: str | None = Field(
        default=None,
        description="JSON-encoded deployment specification (resources, placement, etc.)",
        exclude=True,  # Don't serialize this internal field
    )
    enabled: bool = Field(default=True, description="Whether this config is active")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="When the record was created (UTC, ISO format)",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="When the record was last updated (UTC, ISO format)",
    )

    __table_args__ = (
        # Partial unique index: only one enabled config per model_name
        Index(
            "ix_on_demand_unique_enabled",
            "model_name",
            unique=True,
            postgresql_where=text("enabled = true"),
            sqlite_where=text("enabled = 1"),
        ),
    )

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name follows Docker naming constraints.

        The provided model name will be used as the Docker container/service name.

        Docker container names must:
        - Start with alphanumeric character
        - Contain only alphanumeric, underscore, period, or hyphen
        """
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", v):
            raise ValueError(
                f"Invalid model name: {v}. Must start with alphanumeric and contain "
                "only alphanumeric, underscore, period, or hyphen characters"
            )
        if len(v) > 255:
            raise ValueError(f"Service name too long: {v}. Maximum length is 255 characters")
        return v

    @field_validator("model_uri", mode="before")
    @classmethod
    def validate_model_uri(cls, v: str) -> str:
        """Validate model URI format."""
        if not v or not v.strip():
            raise ValueError("Model URI cannot be empty")
        return v.strip()

    @computed_field
    @property
    def deploy(self) -> DeploySpec | None:
        """Computed field that deserializes deploy_spec_json for API responses."""
        if not self.deploy_spec_json:
            return None
        return DeploySpec.model_validate(json.loads(self.deploy_spec_json))


class ModelManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    @staticmethod
    def with_db_session(func):
        """Decorator to provide a database session to a method."""

        @wraps(func)
        def wrapper(self: "ModelManager", *args, **kwargs):
            with self.db_manager.get_session() as session:
                return func(self, session, *args, **kwargs)

        return wrapper

    @with_db_session
    def create_model(
        self,
        session: Session,
        model_name: str,
        deployment_type: ModelDeploymentType,
        idle_ttl: int | None = None,
        ready: bool = False,
        last_used_at: str | None = None,
    ) -> Model:
        """Create a new model record.

        Args:
            session: Database session
            model_name: Unique model/container name
            deployment_type: Either 'auto', 'manual', or 'static'
            idle_ttl: Idle TTL in seconds
            ready: Whether the model deployment is complete and ready to serve requests
            last_used_at: Last used timestamp in ISO format (UTC)

        Returns:
            Created Model instance

        Raises:
            ValueError: If a model with this name already exists in the database
        """
        model = Model(
            model_name=model_name,
            deployment_type=deployment_type,
            ready=ready,
            last_used_at=last_used_at,
            idle_ttl=idle_ttl,
        )
        try:
            session.add(model)
            session.commit()
            session.refresh(model)
            log.debug(
                "Created model record: %s (type=%s, idle_ttl=%s)",
                model_name,
                deployment_type.value,
                idle_ttl,
            )
            return model
        except IntegrityError as e:
            session.rollback()
            # Check if it's a primary key/unique constraint violation on model_name
            error_msg = str(e.orig).lower() if hasattr(e, "orig") else str(e).lower()
            if "model_name" in error_msg or "model_pkey" in error_msg:
                log.warning("Model '%s' already exists in database", model_name)
                raise ValueError(f"Model '{model_name}' already exists") from e

            log.error("Unexpected IntegrityError creating model '%s': %s", model_name, e)
            raise

    @with_db_session
    def get_model(self, session: Session, model_name: str) -> Model | None:
        """Get a model record by name returning None if not found."""
        model = session.get(Model, model_name)
        if not model:
            log.debug("Model record not found: %s", model_name)
            return None
        return model

    @with_db_session
    def get_model_deployment_type(
        self, session: Session, model_name: str
    ) -> ModelDeploymentType | None:
        """Get a model's deployment type by name returning None if not found."""
        model = session.get(Model, model_name)
        if not model:
            log.debug("Model record not found: %s", model_name)
            return None
        return model.deployment_type

    @with_db_session
    def mark_model_ready(self, session: Session, model_name: str) -> Model | None:
        """Mark a model as ready returning None if not found."""
        model = session.get(Model, model_name)
        if not model:
            log.warning("Model record not found: %s", model_name)
            return None

        model.ready = True
        log.debug("Marked model as ready: %s", model_name)

        session.add(model)
        session.commit()
        session.refresh(model)
        return model

    @with_db_session
    def record_model_usage(self, session: Session, model_name: str) -> Model | None:
        """Update model's last used timestamp to now returning None if the model is not found."""
        model = session.get(Model, model_name)
        if not model:
            log.warning("Model record not found: %s", model_name)
            return None

        model.last_used_at = datetime.now(UTC).isoformat()
        log.debug("Updated last_used_at for model: %s", model_name)

        session.add(model)
        session.commit()
        session.refresh(model)
        return model

    @with_db_session
    def delete_model(self, session: Session, model_name: str) -> bool:
        """Delete model record, returning True if deleted and False if not found."""
        model = session.get(Model, model_name)
        if not model:
            log.warning("Model record not found for deletion: %s", model_name)
            return False

        session.delete(model)
        session.commit()
        log.debug("Deleted model record: %s", model_name)
        return True

    @with_db_session
    def is_model_idle(self, session: Session, model_name: str) -> tuple[bool, float]:
        """Check if a model has exceeded its idle TTL.

        Returns:
            Tuple of (is_idle: bool, idle_seconds: float)
        """
        model = session.get(Model, model_name)
        if not model:
            log.warning("Model record not found: %s", model_name)
            return False, 0.0

        if model.idle_ttl is None:
            return False, 0.0

        idle_seconds = (datetime.now(UTC) - parser.isoparse(model.last_used_at)).total_seconds()
        is_idle = idle_seconds >= model.idle_ttl

        log.debug(
            "Model %s idle check: %s (idle_time=%.0fs, ttl=%ss)",
            model_name,
            is_idle,
            idle_seconds,
            model.idle_ttl,
        )
        return is_idle, idle_seconds

    @with_db_session
    def create_on_demand_config(
        self,
        session: Session,
        model_name: str,
        model_uri: str | None = None,
        tracking_id: str | None = None,
        idle_ttl: int | None = None,
        description: str | None = None,
        deploy_spec: dict | None = None,
        replace_enabled: bool = True,
        inherit_config: bool = True,
    ) -> OnDemandModelConfig:
        """Create a new on-demand model configuration.

        Args:
            session: Database session
            model_name: Unique service/container name for the model
            model_uri: URI pointing to the model artifact (optional if `inherit_config=True`)
            tracking_id: Tracking server run ID (optional, for reference/audit)
            idle_ttl: Idle TTL in seconds (optional, inherits or uses default)
            description: Human-readable description (optional)
            deploy_spec: Deployment specification dict (optional)
            replace_enabled: If True and an enabled config exists, disable it first (preserving it
                in history) and create the new one atomically to replace the original as the enabled
                config. If False and an enabled config exists, raises `ConfigConflictError`.
            inherit_config: If True, inherit settings from the currently enabled config for this
                `model_name`, if one exists. If no config with the same name exists, attempt to
                create a new one, as long as the mandatory fields are provided (i.e. `model_uri` and
                `idle_ttl`). Explicitly provided values override inherited ones.

        Returns:
            Created OnDemandModelConfig instance

        Raises:
            ConfigValidationError: If required fields are missing or invalid
            ConfigConflictError: If an enabled config already exists and `replace_enabled=False`
        """
        existing = session.exec(
            select(OnDemandModelConfig).where(
                OnDemandModelConfig.model_name == model_name,
                OnDemandModelConfig.enabled == True,  # noqa: E712
            )
        ).first()

        if inherit_config and existing:
            model_uri = existing.model_uri if model_uri is None else model_uri
            tracking_id = existing.tracking_id if tracking_id is None else tracking_id
            idle_ttl = existing.idle_ttl if idle_ttl is None else idle_ttl
            description = existing.description if description is None else description
            deploy_spec = (
                json.loads(existing.deploy_spec_json)
                if deploy_spec is None and existing.deploy_spec_json
                else deploy_spec
            )

        if model_uri is None:
            raise ConfigValidationError(
                "model_uri is required: either provide it explicitly or use inherit_config=True to"
                " copy from the existing configuration (only possible when an enabled one exists)"
            )

        # Use default idle TTL if not provided and not inherited (e.g. no existing config found)
        idle_ttl = get_config().get_default_idle_ttl() if idle_ttl is None else idle_ttl

        if existing:
            if not replace_enabled:
                raise ConfigConflictError(
                    f"An enabled on-demand config for '{model_name}' already exists"
                )

            existing.enabled = False
            existing.updated_at = datetime.now(UTC).isoformat()
            session.add(existing)
            log.info(
                "Disabled existing on-demand config for replacement: %s (id=%d)",
                model_name,
                existing.id,
            )

        try:
            config = OnDemandModelConfig.model_validate(
                {
                    "model_name": model_name,
                    "model_uri": model_uri,
                    "tracking_id": tracking_id,
                    "idle_ttl": idle_ttl,
                    "description": description,
                    "deploy_spec_json": json.dumps(deploy_spec) if deploy_spec else None,
                    "enabled": True,
                }
            )
        except ValidationError as e:
            raise ConfigValidationError(f"Invalid on-demand model config: {e}") from e

        session.add(config)
        session.commit()
        session.refresh(config)
        log.info(
            "Created on-demand config: %s (uri=%s, tracking_id=%s, idle_ttl=%s)",
            model_name,
            model_uri,
            tracking_id,
            idle_ttl,
        )
        return config

    @with_db_session
    def get_on_demand_config(self, session: Session, model_name: str) -> OnDemandModelConfig | None:
        """Get the enabled on-demand config for a model name.

        Returns None if no enabled config exists for this model.
        """
        config = session.exec(
            select(OnDemandModelConfig).where(
                OnDemandModelConfig.model_name == model_name,
                OnDemandModelConfig.enabled == True,  # noqa: E712
            )
        ).first()
        if not config:
            log.debug("No enabled on-demand config found for: %s", model_name)
            return None
        return config

    @with_db_session
    def get_on_demand_config_by_id(
        self, session: Session, config_id: int
    ) -> OnDemandModelConfig | None:
        """Get an on-demand config by its ID (regardless of enabled status)."""
        return session.get(OnDemandModelConfig, config_id)

    @with_db_session
    def list_on_demand_configs(
        self, session: Session, include_disabled: bool = False
    ) -> list[OnDemandModelConfig]:
        """List on-demand model configurations.

        Args:
            session: Database session
            include_disabled: If True, include disabled (soft-deleted) configs

        Returns:
            List of OnDemandModelConfig instances
        """
        statement = select(OnDemandModelConfig)
        if not include_disabled:
            statement = statement.where(OnDemandModelConfig.enabled == True)  # noqa: E712
        statement = statement.order_by(OnDemandModelConfig.model_name)
        return list(session.exec(statement).all())

    @with_db_session
    def get_on_demand_config_history(
        self, session: Session, model_name: str
    ) -> list[OnDemandModelConfig]:
        """Get all versions (enabled and disabled) for a model name.

        Returns configs ordered by created_at descending (newest first).
        """
        statement = (
            select(OnDemandModelConfig)
            .where(OnDemandModelConfig.model_name == model_name)
            .order_by(OnDemandModelConfig.created_at.desc())
        )
        return list(session.exec(statement).all())

    @with_db_session
    def update_on_demand_config(
        self,
        session: Session,
        model_name: str,
        model_uri: str | None = None,
        tracking_id: str | None = None,
        idle_ttl: int | None = None,
        description: str | None = None,
        deploy_spec: dict | None = None,
        clear_tracking_id: bool = False,
        clear_idle_ttl: bool = False,
        clear_description: bool = False,
        clear_deploy_spec: bool = False,
    ) -> OnDemandModelConfig | None:
        """Update an existing on-demand model configuration.

        Only updates fields that are explicitly provided. Use clear_* flags to set
        optional fields to None.

        Args:
            session: Database session
            model_name: Model name of the config to update
            model_uri: New model URI
            tracking_id: New tracking ID
            idle_ttl: New idle TTL
            description: New description
            deploy_spec: New deployment specification dict
            clear_*: Set corresponding field to None

        Returns:
            Updated OnDemandModelConfig or None if not found
        """
        config = session.exec(
            select(OnDemandModelConfig).where(
                OnDemandModelConfig.model_name == model_name,
                OnDemandModelConfig.enabled == True,  # noqa: E712
            )
        ).first()

        if not config:
            log.warning("No enabled on-demand config found for update: %s", model_name)
            return None

        config.model_uri = model_uri if model_uri is not None else config.model_uri
        config.tracking_id = tracking_id if tracking_id is not None else config.tracking_id
        config.idle_ttl = idle_ttl if idle_ttl is not None else config.idle_ttl
        config.description = description if description is not None else config.description
        config.deploy_spec_json = (
            json.dumps(deploy_spec) if deploy_spec is not None else config.deploy_spec_json
        )

        config.tracking_id = None if clear_tracking_id else config.tracking_id
        config.idle_ttl = get_config().get_default_idle_ttl() if clear_idle_ttl else config.idle_ttl
        config.description = None if clear_description else config.description
        config.deploy_spec_json = None if clear_deploy_spec else config.deploy_spec_json

        config.updated_at = datetime.now(UTC).isoformat()

        session.add(config)
        session.commit()
        session.refresh(config)
        log.info("Updated on-demand config: %s", model_name)
        return config

    @with_db_session
    def disable_on_demand_config(self, session: Session, model_name: str) -> bool:
        """Soft-delete an on-demand config by setting enabled=False.

        Returns True if disabled, False if no enabled config was found.
        """
        config = session.exec(
            select(OnDemandModelConfig).where(
                OnDemandModelConfig.model_name == model_name,
                OnDemandModelConfig.enabled == True,  # noqa: E712
            )
        ).first()

        if not config:
            log.warning("No enabled on-demand config found for disable: %s", model_name)
            return False

        config.enabled = False
        config.updated_at = datetime.now(UTC).isoformat()

        session.add(config)
        session.commit()
        log.info("Disabled on-demand config: %s (id=%d)", model_name, config.id)
        return True

    @with_db_session
    def enable_on_demand_config(
        self, session: Session, config_id: int
    ) -> OnDemandModelConfig | None:
        """Enable an on-demand configuration by ID.

        This will first disable any currently enabled config for the same `model_name`, before
        enabling the specified config.

        Returns the enabled config, or None if the `config_id` was not found.
        """
        config = session.get(OnDemandModelConfig, config_id)
        if not config:
            log.warning("On-demand config not found for enable: id=%d", config_id)
            return None

        if config.enabled:
            log.debug("On-demand config already enabled: id=%d", config_id)
            return config

        # Disable any currently enabled config for this model_name FIRST
        # (commit separately to avoid unique constraint issues with partial index)
        current_enabled = session.exec(
            select(OnDemandModelConfig).where(
                OnDemandModelConfig.model_name == config.model_name,
                OnDemandModelConfig.enabled == True,  # noqa: E712
            )
        ).first()

        if current_enabled:
            current_enabled.enabled = False
            current_enabled.updated_at = datetime.now(UTC).isoformat()
            session.add(current_enabled)
            session.commit()
            log.info(
                "Disabled previous on-demand config: %s (id=%d)",
                config.model_name,
                current_enabled.id,
            )

        config.enabled = True
        config.updated_at = datetime.now(UTC).isoformat()
        session.add(config)
        session.commit()
        session.refresh(config)
        log.info("Enabled on-demand config: %s (id=%d)", config.model_name, config_id)
        return config
