# proxy_min.py
import asyncio
import itertools  # NEW

import httpx
from fastapi import FastAPI, HTTPException, Request

# --- config -----------------------------------------------------------
BACKENDS = [f"http://127.0.0.1:{8000 + i}" for i in range(8)]

# in-memory state
prefix_cache: dict[str, str] = {}  # prefix -> backend
backend_cycle = itertools.cycle(BACKENDS)  # NEW  round-robin iterator
lock = asyncio.Lock()

app = FastAPI()


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
        backend = prefix_cache.get(prefix)
        if backend:
            print(f"[CACHE-HIT] {backend} <- {prefix[:60]!r}")
        else:
            backend = next(backend_cycle)  # NEW  deterministic order
            if prefix:  # only cache non-empty prefixes
                prefix_cache[prefix] = backend
            print(f"[CACHE-MISS] {backend} selected via round-robin")

    try:
        result = await forward(backend, payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {"backend": backend, **result}
