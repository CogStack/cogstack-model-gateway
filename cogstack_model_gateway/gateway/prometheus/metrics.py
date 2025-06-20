from prometheus_client import Counter

gateway_requests_total = Counter(
    "gateway_requests_total",
    "Total number of HTTP requests to the CogStack Model Gateway",
    ["method", "endpoint"],
)

gateway_models_deployed_total = Counter(
    "gateway_models_deployed_total",
    "Total number of models deployed via the CogStack Model Gateway",
    ["model", "model_uri"],
)

gateway_tasks_processed_total = Counter(
    "gateway_tasks_processed_total",
    "Total number of tasks processed through the CogStack Model Gateway",
    ["model", "task"],
)
