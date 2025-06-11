# CogStack Model Gateway Client

A Python client for interacting with the CogStack Model Gateway. The default and recommended client
implementation is asynchronous, but a blocking synchronous one is also provided for easier
experimentation.

## Installation

```bash
pip install cogstack-model-gateway-client
```

## Usage

The project is under active development and so is the documentation. Seems like looking at the
source code is the only way to figure out what the client does.

### Async Client

```python
import asyncio

from cogstack_model_gateway_client import GatewayClient


async def main():
    async with GatewayClient(base_url="http://your-gateway-url", default_model="your-model") as client:
        # Generate annotations for a given text
        result = await client.process("Sample input text")

    print(result)


asyncio.run(main())
```

### Synchronous Client

```python
from cogstack_model_gateway_client import GatewayClientSync

client = GatewayClientSync(base_url="http://your-gateway-url", default_model="your-model")
result = client.process("Sample input text")
print(result)
```
