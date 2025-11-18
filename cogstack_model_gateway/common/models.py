import logging
from datetime import UTC, datetime
from enum import Enum
from functools import wraps

from dateutil import parser
from sqlalchemy import Index
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel

from cogstack_model_gateway.common.db import DatabaseManager

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
        ready: bool = False,
        last_used_at: str | None = None,
        idle_ttl: int | None = None,
    ) -> Model:
        """Create a new model record.

        Args:
            session: Database session
            model_name: Unique model/container name
            deployment_type: Either 'auto', 'manual', or 'static'
            ready: Whether the model deployment is complete and ready to serve requests
            last_used_at: Last used timestamp in ISO format (UTC)
            idle_ttl: Idle TTL in seconds

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
