import asyncio
import logging
import time
from datetime import UTC, datetime

import httpx
from dateutil import parser
from docker.models.containers import Container

from cogstack_model_gateway.common.config.models import Config, OnDemandModel
from cogstack_model_gateway.common.models import ModelDeploymentType, ModelManager
from cogstack_model_gateway.gateway.core.models import get_running_models, run_model_container
from cogstack_model_gateway.gateway.routers.utils import get_cms_url

log = logging.getLogger("cmg.gateway.auto_deploy")

# Stale lock threshold: deployment locks older than this are considered stale
STALE_DEPLOYMENT_LOCK_SECONDS = 300  # 5 minutes


def is_model_running(model_name: str) -> bool:
    """Check if a model container is currently running."""
    return model_name in {m["service_name"] for m in get_running_models()}


async def wait_for_model_health(model_name: str, timeout: int, check_interval: float = 1.0) -> bool:
    """Poll model's /readyz endpoint until healthy or timeout.

    Args:
        model_name: Name of the model
        timeout: Maximum seconds to wait
        check_interval: Seconds between health checks (exponential backoff)

    Returns:
        True if model became healthy, False if timeout
    """
    url = get_cms_url(model_name, "readyz")
    start_time = time.time()
    interval = check_interval

    log.info("Waiting for model '%s' to become ready (timeout: %ds)", model_name, timeout)

    async with httpx.AsyncClient(verify=False) as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    log.info("Model '%s' is ready (took %.1fs)", model_name, elapsed)
                    return True
                log.debug(
                    "Model '%s' not ready yet (status: %d), retrying in %.1fs",
                    model_name,
                    response.status_code,
                    interval,
                )
            except (httpx.RequestError, httpx.TimeoutException) as e:
                log.debug(
                    "Health check failed for '%s': %s, retrying in %.1fs", model_name, e, interval
                )

            await asyncio.sleep(interval)
            # Exponential backoff, max 8 seconds
            interval = min(interval * 2, 8.0)

    elapsed = time.time() - start_time
    log.warning(
        "Model '%s' did not become ready within %ds (elapsed: %.1fs)", model_name, timeout, elapsed
    )
    return False


def deploy_on_demand_model(model_config: OnDemandModel, model_manager: ModelManager) -> Container:
    """Deploy an on-demand model container.

    Creates database entry with ready=False, deploys container, then marks ready=True.

    Args:
        model_config: Configuration for the on-demand model
        model_manager: Model manager for database operations

    Returns:
        Deployed container

    Raises:
        ValueError: If model already exists in database (another worker is deploying)
        Exception: If container deployment fails
    """
    model_name = model_config.service_name

    log.info("Starting deployment of on-demand model: %s", model_name)

    # This will raise ValueError if model already exists (another worker deploying)
    model_manager.create_model(
        model_name=model_name,
        deployment_type=ModelDeploymentType.AUTO,
        idle_ttl=model_config.idle_ttl,
    )

    try:
        container = run_model_container(
            model_name=model_name,
            model_uri=model_config.model_uri,
            # FIXME: add model type
            model_type="medcat_umls",
            deployment_type=ModelDeploymentType.AUTO,
            resources=model_config.deploy.resources if model_config.deploy else None,
        )

        log.info("Successfully deployed container for model: %s", model_name)
        return container

    except Exception as e:
        log.error("Failed to deploy model '%s': %s", model_name, e)
        model_manager.delete_model(model_name)
        raise


async def ensure_model_available(
    model_name: str,
    config: Config,
    model_manager: ModelManager,
) -> bool:
    """Ensure a model is available, deploying it if necessary.

    This is the main entry point for ensuring ANY model (STATIC, MANUAL, or AUTO)
    is available before processing a request. Flow:
    1. Check if model is running
    2. Check if model is healthy
    3. If both pass, ensure model is tracked in database (auto-create STATIC entry if needed)
    4. If not available, attempt auto-deployment (only for AUTO-configured models)

    Note: This function does NOT record usage - that should be done AFTER the actual
    operation succeeds (e.g., after HTTP request completes successfully).

    Args:
        model_name: Name of the model to ensure is available
        config: Global configuration
        model_manager: Model manager for database operations

    Returns:
        True if model is available and ready, False otherwise
    """
    # Step 1: Check if model is currently running
    if is_model_running(model_name):
        log.debug("Model '%s' container is running, checking health", model_name)

        # Step 2: Check if model is healthy
        url = get_cms_url(model_name, "readyz")
        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    log.debug("Model '%s' is running and healthy", model_name)

                    # Step 3: Ensure model is tracked (auto-create STATIC entry if needed)
                    existing_model = model_manager.get_model(model_name)
                    if not existing_model:
                        log.info(
                            "Model '%s' is running but not tracked, creating STATIC entry",
                            model_name,
                        )
                        try:
                            model_manager.create_model(
                                model_name=model_name,
                                deployment_type=ModelDeploymentType.STATIC,
                                ready=True,
                            )
                        except ValueError:
                            # Another worker just created it, that's fine
                            log.debug("Model '%s' was created by another worker", model_name)

                    return True
                log.info(
                    "Model '%s' is running but not healthy (status: %d)",
                    model_name,
                    response.status_code,
                )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            log.info("Model '%s' is running but health check failed: %s", model_name, e)
    else:
        log.info("Model '%s' is not currently running", model_name)

    # Step 4: Model not running/healthy - check if we can auto-deploy
    model_config = config.get_on_demand_model(model_name)
    if not model_config:
        log.warning(
            "Model '%s' is not available and not configured for auto-deployment",
            model_name,
        )
        return False

    log.info("Model '%s' not available, initiating auto-deployment", model_name)

    # Step 5: Check if another worker is currently deploying this model
    existing_model = model_manager.get_model(model_name)
    if existing_model and not existing_model.ready:
        created_at = parser.isoparse(existing_model.created_at)
        age_seconds = (datetime.now(UTC) - created_at).total_seconds()

        if age_seconds < STALE_DEPLOYMENT_LOCK_SECONDS:
            # Fresh deployment in progress - wait for it
            log.info(
                "Model '%s' is being deployed by another worker (age: %.1fs), waiting...",
                model_name,
                age_seconds,
            )
            # Wait for the other worker's deployment to complete
            is_healthy = await wait_for_model_health(
                model_name,
                config.models.deployment.auto.config.health_check_timeout,
            )
            if is_healthy:
                # Mark as ready (the other worker might have crashed after deploying)
                model_manager.mark_model_ready(model_name)
                return True
            log.error("Model '%s' deployment by another worker failed or timed out", model_name)
            return False

        # Stale lock - take over deployment
        log.warning(
            "Model '%s' has stale deployment lock (age: %.1fs), breaking lock",
            model_name,
            age_seconds,
        )
        model_manager.delete_model(model_name)

    # Step 6: Deploy the model
    try:
        container = deploy_on_demand_model(model_config, model_manager)
    except ValueError as e:
        # Another worker just created the entry (race condition)
        log.info("Another worker started deploying '%s', waiting for completion: %s", model_name, e)
        is_healthy = await wait_for_model_health(
            model_name,
            config.models.deployment.auto.config.health_check_timeout,
        )
        if is_healthy:
            model_manager.mark_model_ready(model_name)
            return True
        return False
    except Exception as e:
        log.error("Failed to deploy model '%s': %s", model_name, e)
        return False

    # Step 7: Wait for model to become healthy
    is_healthy = await wait_for_model_health(
        model_name,
        config.models.deployment.auto.config.health_check_timeout,
    )

    if is_healthy:
        # Mark model as ready in database
        model_manager.mark_model_ready(model_name)
        log.info("Model '%s' is now available and ready", model_name)
        return True

    log.error("Model '%s' failed health check, cleaning up", model_name)
    try:
        container.stop()
        container.remove()
    except Exception as e:
        log.warning("Failed to clean up container for '%s': %s", model_name, e)
    model_manager.delete_model(model_name)
    return False
