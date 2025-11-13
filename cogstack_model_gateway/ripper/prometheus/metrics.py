from prometheus_client import Counter, Histogram

containers_purged_total = Counter(
    "ripper_containers_purged_total",
    "Total number of Docker containers purged by the Ripper",
    ["deployment_type", "reason"],
)

containers_checked_total = Counter(
    "ripper_containers_checked_total",
    "Total number of Docker containers checked by the Ripper",
    ["deployment_type"],
)

model_idle_time_seconds = Histogram(
    "ripper_model_idle_time_seconds",
    "Time in seconds that a model has been idle before removal",
    ["model"],
    buckets=[
        60,  # 1 minute
        300,  # 5 minutes
        600,  # 10 minutes
        1800,  # 30 minutes
        3600,  # 1 hour
        7200,  # 2 hours
        14400,  # 4 hours
        28800,  # 8 hours
        86400,  # 1 day
        172800,  # 2 days
        432000,  # 5 days
        604800,  # 1 week
        1209600,  # 2 weeks
        2592000,  # 1 month
        15552000,  # 6 months
        31536000,  # 1 year
    ],
)
