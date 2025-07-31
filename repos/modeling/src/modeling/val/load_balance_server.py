# proxy_min.py
from __future__ import annotations

import asyncio
import itertools
from datetime import timedelta
from typing import Annotated

import httpx
import typer
from fastapi import FastAPI, HTTPException, Request
from synapse.utils.async_typer import AsyncTyper

from modeling.utils.max_timeout import max_timeout

# Global state
prefix_cache: dict[str, str] = {}  # prefix -> backend
backend_cycle: itertools.cycle[str] | None = None
lock = asyncio.Lock()

app = FastAPI()
cli_app = AsyncTyper()


# --- helpers ----------------------------------------------------------
async def forward(backend: str, payload: dict):
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{backend}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


def extract_prefix(payload: dict) -> str | None:
    try:
        return payload["messages"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None


# --- main endpoint ----------------------------------------------------
@app.post("/v1/chat/completions")
async def generate(req: Request):
    payload = await req.json()
    prefix = extract_prefix(payload)
    if prefix is None:
        print("[CACHE-MISS] No prefix found in payload")

    # pick a backend (prefix-aware cache  ->  round-robin fallback)
    async with lock:
        backend = prefix_cache.get(prefix) if prefix else None
        if backend:
            assert prefix is not None
            print(f"[CACHE-HIT] {backend} <- {prefix[:60]!r}")
        else:
            if backend_cycle is None:
                raise HTTPException(status_code=500, detail="No backends configured")
            backend = next(backend_cycle)  # NEW  deterministic order
            if prefix:  # only cache non-empty prefixes
                prefix_cache[prefix] = backend
            print(f"[CACHE-MISS] {backend} selected via round-robin")

    try:
        result = await forward(backend, payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {"backend": backend, **result}


@app.get("/health")
async def health():
    """Health check endpoint for the load balancer."""
    if backend_cycle is None:
        raise HTTPException(status_code=503, detail="Backends not configured")
    return {"status": "healthy", "message": "Load balancer is ready"}


async def setup_backends(backends: list[str]):
    """Initialize the backend cycle with the provided backend URLs."""
    from modeling.val.vllm_utils import wait_for_servers_ready

    print(f"Verifying {len(backends)} backends are healthy: {backends}")

    # Wait for all backends to be ready before setting up the cycle
    all_ready = await wait_for_servers_ready(backends)

    if not all_ready:
        raise RuntimeError("Not all backends are healthy")

    global backend_cycle
    backend_cycle = itertools.cycle(backends)
    print(f"All backends are healthy! Configured {len(backends)} backends")


@cli_app.async_command(name="serve")
async def serve(
    backends: Annotated[
        list[str], typer.Option("--backend", "-b", help="Backend vLLM server URLs")
    ],
    port: Annotated[int, typer.Option("--port", "-p", help="Port to serve on")] = 8080,
    setup_max_timeout: Annotated[
        int,
        typer.Option(
            "--setup-max-timeout", help="Maximum timeout in seconds for backend setup"
        ),
    ] = 300,
):
    """Start the load balancer FastAPI server."""
    import uvicorn

    if not backends:
        raise typer.BadParameter("At least one backend must be specified")

    await max_timeout(
        setup_backends(backends),
        timedelta(seconds=setup_max_timeout),
        "Timeout waiting for backends to be ready",
    )

    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="info", timeout_keep_alive=300
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    cli_app()
