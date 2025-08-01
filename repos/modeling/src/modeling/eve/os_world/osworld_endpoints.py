from typing import Any

import aiohttp

# Refactored client functions to accept an existing aiohttp.ClientSession


async def start_env(
    session: aiohttp.ClientSession, base_url: str = "http://localhost:8000"
) -> dict[str, Any]:
    """
    Start a new environment session.
    """
    async with session.post(f"{base_url}/start_env") as resp:
        resp.raise_for_status()
        return await resp.json()


async def reset(
    session: aiohttp.ClientSession,
    env_id: str,
    task_config: dict[str, Any] | None = None,
    base_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """
    Reset the environment with the given ID and optional task configuration.
    """
    payload = {"env_id": env_id, "task_config": task_config}
    async with session.post(f"{base_url}/reset", json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def evaluate(
    session: aiohttp.ClientSession, env_id: str, base_url: str = "http://localhost:8000"
) -> dict[str, Any]:
    """
    Evaluate the environment with the given ID.
    """
    async with session.post(f"{base_url}/evaluate", json={"env_id": env_id}) as resp:
        resp.raise_for_status()
        return await resp.json()


async def start_action_record(
    session: aiohttp.ClientSession,
    env_id: str,
    username: str,
    output_dir: str,
    output_bucket: str,
    base_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """
    Begin recording actions for the specified environment and user.
    """
    payload = {
        "env_id": env_id,
        "username": username,
        "output_dir": output_dir,
        "output_bucket": output_bucket,
    }
    async with session.post(f"{base_url}/start_action_record", json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def end_action_record(
    session: aiohttp.ClientSession, env_id: str, base_url: str = "http://localhost:8000"
) -> dict[str, Any]:
    """
    End recording actions for the specified environment.
    """
    async with session.post(
        f"{base_url}/end_action_record", json={"env_id": env_id}
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def step_env(
    session: aiohttp.ClientSession,
    env_id: str,
    action: str,
    base_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """
    Step the environment with a given action.
    """
    payload = {"env_id": env_id, "action": action}
    async with session.post(f"{base_url}/step_env", json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def stop_env(
    session: aiohttp.ClientSession, env_id: str, base_url: str = "http://localhost:8000"
) -> dict[str, Any]:
    """
    Stop the environment with the given ID.
    """
    async with session.post(f"{base_url}/stop_env", json={"env_id": env_id}) as resp:
        resp.raise_for_status()
        return await resp.json()


async def kill_all_envs(
    session: aiohttp.ClientSession, base_url: str = "http://localhost:8000"
) -> dict[str, Any]:
    """
    Kill all running environments.
    """
    async with session.post(f"{base_url}/kill_all_envs") as resp:
        resp.raise_for_status()
        return await resp.json()


async def meta_get_vm(meta_endpoint: str):
    async with (
        aiohttp.ClientSession() as s,
        s.post(
            f"{meta_endpoint}/backend/vm", json={"email": "evals@inductionlabs.com"}
        ) as r,
    ):
        return await r.json()


async def meta_return_vm(machine: dict, meta_endpoint: str):
    # {"instance_name": str, "internal_ip": str}
    async with (
        aiohttp.ClientSession() as s,
        s.post(f"{meta_endpoint}/backend/return_vm", json=machine) as r,
    ):
        return await r.json()
