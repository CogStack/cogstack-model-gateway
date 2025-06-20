from prometheus_client import Counter

containers_purged_total = Counter(
    "ripper_containers_purged_total",
    "Total number of Docker containers purged by the Ripper",
)
