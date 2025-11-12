import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

import docker
from dateutil import parser
from docker.models.containers import Container
from prometheus_client import start_http_server

from cogstack_model_gateway.common.config import load_config
from cogstack_model_gateway.common.logging import configure_logging
from cogstack_model_gateway.ripper.prometheus.metrics import containers_purged_total

log = logging.getLogger("cmg.ripper")


def stop_and_remove_container(container: Container):
    """Stop and remove a Docker container."""
    log.info(f"Stopping and removing expired container: {container.name}")
    container.stop()
    container.remove()
    containers_purged_total.inc()


def purge_expired_containers(config):
    """Run periodically and purge Docker containers that have exceeded their TTL.

    List Docker containers and fetch the ones managed by the CogStack Model Gateway that correspond
    to model servers according to their labels. For each container, check if it has a TTL label set
    and if the current time exceeds the expiration time; if so, stop and remove the container
    (containers without a TTL label or with a TTL value of -1 are skipped). Finally, sleep for the
    specified interval before repeating the process.
    """
    client = docker.from_env()

    while True:
        now = datetime.now(UTC)

        containers = client.containers.list(
            filters={
                "label": [
                    f"{config.labels.managed_by_label}={config.labels.managed_by_value}",
                    config.labels.cms_model_label,
                ]
            },
        )

        with ThreadPoolExecutor() as executor:
            futures = []
            for container in containers:
                container: Container
                ttl = int(container.labels.get(config.labels.ttl_label, -1))

                if ttl == -1:
                    continue  # Skip containers with TTL set to -1

                created_at = parser.isoparse(container.attrs["Created"])
                expiration_time = created_at + timedelta(seconds=ttl)

                if now >= expiration_time:
                    futures.append(executor.submit(stop_and_remove_container, container))

            for future in as_completed(futures):
                future.result()

        time.sleep(config.ripper.interval)


def main():
    """Run the ripper service."""
    configure_logging()
    config = load_config()

    start_http_server(config.ripper.metrics_port)

    log.info(f"Starting ripper with interval={config.ripper.interval}s")
    purge_expired_containers(config)


if __name__ == "__main__":
    sys.exit(main())
