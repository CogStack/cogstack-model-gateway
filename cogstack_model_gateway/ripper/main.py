import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

import docker
from dateutil import parser
from docker.models.containers import Container

from cogstack_model_gateway.common.containers import (
    IS_MODEL_LABEL,
    MANAGED_BY_LABEL,
    MANAGED_BY_LABEL_VALUE,
    TTL_LABEL,
)
from cogstack_model_gateway.common.logging import configure_logging

PURGE_INTERVAL = int(os.getenv("CMG_RIPPER_INTERVAL") or 60)

log = logging.getLogger("cmg.ripper")


def stop_and_remove_container(container: Container):
    """Stop and remove a Docker container."""
    log.info(f"Stopping and removing expired container: {container.name}")
    container.stop()
    container.remove()


def purge_expired_containers():
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
            filters={"label": [f"{MANAGED_BY_LABEL}={MANAGED_BY_LABEL_VALUE}", IS_MODEL_LABEL]},
        )

        with ThreadPoolExecutor() as executor:
            futures = []
            for container in containers:
                container: Container
                ttl = int(container.labels.get(TTL_LABEL, -1))

                if ttl == -1:
                    continue  # Skip containers with TTL set to -1

                created_at = parser.isoparse(container.attrs["Created"])
                expiration_time = created_at + timedelta(seconds=ttl)

                if now >= expiration_time:
                    futures.append(executor.submit(stop_and_remove_container, container))

            for future in as_completed(futures):
                future.result()

        time.sleep(PURGE_INTERVAL)


def main():
    """Run the ripper service."""
    configure_logging()
    purge_expired_containers()


if __name__ == "__main__":
    sys.exit(main())
