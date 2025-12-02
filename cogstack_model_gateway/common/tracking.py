import logging
import os

import mlflow
import mlflow.models
from mlflow import MlflowClient, MlflowException
from mlflow.entities import Run, RunStatus

MODEL_URI_TAG = "training.output.model_uri"
MODEL_TYPE_TAG = "model_type"

log = logging.getLogger("cmg.common")


class TrackingTask:
    def __init__(self, run: Run, url: str):
        self._run = run
        self._status = RunStatus.from_string(self._run.info.status)
        self._url = url

    @property
    def uuid(self):
        return self._run.info.run_id

    @property
    def name(self):
        return self._run.info.run_name

    @property
    def data(self):
        return self._run.data

    @property
    def info(self):
        return self._run.info

    @property
    def status(self):
        return self._run.info.status

    @property
    def is_finished(self) -> bool:
        return self._status == RunStatus.FINISHED

    @property
    def is_failed(self) -> bool:
        return self._status == RunStatus.FAILED

    @property
    def is_killed(self) -> bool:
        return self._status == RunStatus.KILLED

    @property
    def is_running(self) -> bool:
        return self._status == RunStatus.RUNNING

    @property
    def is_scheduled(self) -> bool:
        return self._status == RunStatus.SCHEDULED

    @property
    def url(self):
        return self._url

    def get_exceptions(self):
        """Get exceptions logged as part of a task.

        The tracking client logs single exceptions as "exception" in the run's tags. Lists of
        exceptions are logged in multiple indexed tags, e.g. "exception_0", "exception_1", etc.
        """
        return (
            [self.data.tags["exception"]]
            if "exception" in self.data.tags
            else [value for key, value in self.data.tags.items() if key.startswith("exception_")]
        )


class TrackingClient:
    def __init__(
        self,
        tracking_uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        s3_endpoint_url: str | None = None,
        s3_access_key_id: str | None = None,
        s3_secret_access_key: str | None = None,
    ):
        """Initialize tracking client.

        Args:
            tracking_uri: Tracking server URI.
                Defaults to MLFLOW_TRACKING_URI env var.
            username: Tracking server username.
                Defaults to MLFLOW_TRACKING_USERNAME env var.
            password: Tracking server password.
                Defaults to MLFLOW_TRACKING_PASSWORD env var.
            s3_endpoint_url: S3 endpoint URL for artifact storage (e.g. MinIO).
                Defaults to MLFLOW_S3_ENDPOINT_URL env var.
            s3_access_key_id: S3 access key ID for downloading artifacts.
                Defaults to AWS_ACCESS_KEY_ID env var.
            s3_secret_access_key: S3 secret access key for downloading artifacts.
                Defaults to AWS_SECRET_ACCESS_KEY env var.
        """
        self.tracking_uri = tracking_uri or mlflow.get_tracking_uri()

        # Set credentials in environment for MLflow and boto3
        # Note: Always set these even if None/empty, as boto3 checks for their presence
        if username is not None:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password is not None:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        if s3_endpoint_url is not None:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url
        if s3_access_key_id is not None:
            os.environ["AWS_ACCESS_KEY_ID"] = s3_access_key_id
        if s3_secret_access_key is not None:
            os.environ["AWS_SECRET_ACCESS_KEY"] = s3_secret_access_key

        self._mlflow_client = MlflowClient(self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

    def get_task(self, tracking_id: str) -> TrackingTask:
        """Get a task by its tracking ID."""
        try:
            run = self._mlflow_client.get_run(tracking_id)
            experiment_id = run.info.experiment_id
            url = f"{self.tracking_uri}/#/experiments/{experiment_id}/runs/{tracking_id}"
            return TrackingTask(run, url)
        except MlflowException as e:
            log.error(f"Failed to get task with tracking ID '{tracking_id}': {e}")
            return None

    def _find_unique_logged_model(self, tracking_id: str) -> str:
        """Find a unique model artifact for a given tracking ID.

        This method employs heuristics in an attempt to find a model artifact. It is not guaranteed
        to return the correct model artifact, and should only be used as a fallback. It is
        recommended that model artifact URIs are added as tags to the runs that generated them.
        """
        artifacts = self._mlflow_client.list_artifacts(tracking_id)

        # check if an artifact called "model" exists
        if any(artifact.path == "model" and artifact.is_dir for artifact in artifacts):
            return "model"

        # fall back to artifacts that contain "model" in their path
        model_candidates = [
            artifact.path for artifact in artifacts if artifact.is_dir and "model" in artifact.path
        ]

        # ensure exactly one model was found
        if len(model_candidates) == 1:
            return f"runs:/{tracking_id}/{model_candidates[0]}"
        elif len(model_candidates) > 1:
            raise ValueError(f"Multiple model artifacts found: {model_candidates}.")
        else:
            raise ValueError("No model artifacts found.")

    def get_model_uri(self, tracking_id: str) -> str:
        """Get the model URI for a given tracking ID."""
        try:
            run = self._mlflow_client.get_run(tracking_id)
            model_uri = run.data.tags.get(MODEL_URI_TAG)
            if not model_uri:
                model_uri = self._find_unique_logged_model(tracking_id)
            return model_uri
        except Exception as e:
            log.error(f"Failed to get model URI for task with tracking ID '{tracking_id}': {e}")
            return None

    def get_model_metadata(self, model_uri: str) -> dict:
        """Get metadata for a model URI."""
        try:
            model_info = mlflow.models.get_model_info(model_uri)
            return {
                "uuid": model_info.model_uuid,
                "signature": model_info.signature.to_dict() if model_info.signature else {},
                "flavors": model_info.flavors,
                "run_id": model_info.run_id,
                "artifact_path": model_info.artifact_path,
                "utc_time_created": model_info.utc_time_created,
                "mlflow_version": model_info.mlflow_version,
            }
        except MlflowException as e:
            log.error(f"Failed to get model metadata for model URI '{model_uri}': {e}")
            return None

    def get_model_type(self, model_uri: str) -> str | None:
        """Get model type from tracking server.

        CMS sets the 'model_type' tag on registered model versions. This method retrieves that tag
        from the model registry, handling various URI formats uniformly by finding the corresponding
        run first.

        Args:
            model_uri: MLflow model URI in any of these formats:
                - 'models:/model-name/1' (version number)
                - 'models:/model-name/Production' (stage name)
                - 'runs:/run-id/artifact-path' (direct run reference)
                - 's3://bucket/path' (direct artifact path)

        Returns:
            Model type string (e.g. 'medcat_deid') or None
        """
        try:
            model_info = mlflow.models.get_model_info(model_uri)

            if not model_info.run_id:
                return None

            model_versions = self._mlflow_client.search_model_versions(
                filter_string=f"run_id='{model_info.run_id}'"
            )

            # Return model_type from the first matching model version
            # (typically there's only one registered model per run)
            if model_versions:
                return model_versions[0].tags.get(MODEL_TYPE_TAG)

            return None

        except MlflowException as e:
            log.warning(f"Failed to get model type for model URI '{model_uri}': {e}")
            return None
