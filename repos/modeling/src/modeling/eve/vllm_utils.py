from __future__ import annotations

import asyncio

from synapse.utils.logging import configure_logging

logger = configure_logging(__name__)


async def wait_for_servers_ready(server_urls: list[str]):
    """Wait for all vLLM servers to be ready by polling their health endpoints."""
    import aiohttp

    async def check_server_ready(url: str, index: int) -> bool:
        """Check if a single vLLM server is ready."""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{url}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response,
            ):
                if response.status == 200:
                    logger.info(f"Server {index} ({url}) is ready")
                    return True
        except Exception:
            # Server not ready yet, continue polling
            pass
        return False

    ready_servers = set()
    max_attempts = 300  # 5 minutes with 1 second intervals
    attempt = 0

    while len(ready_servers) < len(server_urls) and attempt < max_attempts:
        attempt += 1

        # Check all servers that aren't ready yet
        for i, url in enumerate(server_urls):
            if i in ready_servers:
                continue

            if await check_server_ready(url, i):
                ready_servers.add(i)

        if len(ready_servers) < len(server_urls):
            await asyncio.sleep(1)

    if len(ready_servers) < len(server_urls):
        not_ready = set(range(len(server_urls))) - ready_servers
        logger.warning(
            f"Timeout waiting for servers {[server_urls[i] for i in not_ready]} to be ready"
        )

    return len(ready_servers) == len(server_urls)
