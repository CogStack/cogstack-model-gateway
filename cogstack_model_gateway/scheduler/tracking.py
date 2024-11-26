import logging

import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities import Run, RunStatus

log = logging.getLogger("cmg.scheduler")


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
    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or mlflow.get_tracking_uri()
        self._mlflow_client = MlflowClient(self.tracking_uri)

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
