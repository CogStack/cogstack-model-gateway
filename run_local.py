import os
import subprocess
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

DOCKER_NETWORK_IP_ADDRESS_FILTER = "{{.NetworkSettings.Networks.cmg_gateway.IPAddress}}"

CMG_SRC_DIR = "cogstack_model_gateway"
CMG_GATEWAY_DIR = f"{CMG_SRC_DIR}/gateway"
CMG_GATEWAY_ENTRYPOINT = f"{CMG_GATEWAY_DIR}/main.py"
CMG_SCHEDULER_DIR = f"{CMG_SRC_DIR}/scheduler"
CMG_SCHEDULER_ENTRYPOINT = f"{CMG_SCHEDULER_DIR}/main.py"
CMG_RIPPER_DIR = f"{CMG_SRC_DIR}/ripper"
CMG_RIPPER_ENTRYPOINT = f"{CMG_RIPPER_DIR}/main.py"
CMG_MIGRATIONS_DIR = f"{CMG_SRC_DIR}/migrations"

CMG_DOCKER_SERVICES = ["object-store", "pgadmin", "db", "queue"]


class ServiceManager:
    def __init__(self):
        self.processes: list[subprocess.Popen] = []

    def get_container_ip(self, container_name):
        """Get the IP address of a Docker container by inspecting its network settings."""
        result = subprocess.run(
            ["docker", "inspect", "-f", DOCKER_NETWORK_IP_ADDRESS_FILTER, container_name],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def start_services(self):
        """Start the gateway, scheduler, and ripper services as subprocesses."""
        self.stop_services()
        print("Starting Gateway...")
        gateway = subprocess.Popen(
            ["poetry", "run", "fastapi", "run", "--workers", "1", CMG_GATEWAY_ENTRYPOINT]
        )
        self.processes.append(gateway)

        print("Starting Scheduler...")
        scheduler = subprocess.Popen(["poetry", "run", "python3", CMG_SCHEDULER_ENTRYPOINT])
        self.processes.append(scheduler)

        print("Starting Ripper...")
        ripper = subprocess.Popen(["poetry", "run", "python3", CMG_RIPPER_ENTRYPOINT])
        self.processes.append(ripper)

        print("Running migrations...")
        subprocess.run(["poetry", "run", "alembic", "upgrade", "head"], cwd=CMG_MIGRATIONS_DIR)

    def stop_services(self):
        """Terminate all running services."""
        for process in self.processes:
            process.terminate()
        self.processes = []

    def cleanup(self):
        """Stop all running services and remove Docker containers."""
        print("Stopping all services...")
        self.stop_services()
        subprocess.run(["docker", "compose", "-f", "docker-compose.yaml", "down"])
        exit(0)


class EventHandler(FileSystemEventHandler):
    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager

    def on_modified(self, event):
        """Restart services when a Python file is modified."""
        if event.src_path.endswith(".py"):
            print("File change detected. Restarting services...")
            self.service_manager.start_services()


if __name__ == "__main__":
    service_manager = ServiceManager()

    print("Installing dependencies...")
    subprocess.run(["poetry", "install", "--with", "dev", "--with", "migrations"])

    print("Starting Docker services...")
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yaml", "up", *CMG_DOCKER_SERVICES, "--wait"]
    )

    os.environ["CMG_DB_HOST"] = service_manager.get_container_ip("cmg-db-1")
    os.environ["CMG_OBJECT_STORE_HOST"] = service_manager.get_container_ip("cmg-object-store-1")
    os.environ["CMG_QUEUE_HOST"] = service_manager.get_container_ip("cmg-queue-1")

    service_manager.start_services()

    event_handler = EventHandler(service_manager)

    observer = Observer()
    observer.schedule(event_handler, path=CMG_SRC_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        service_manager.cleanup()
    finally:
        observer.join()
