from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import typer
from smart_open import open as smart_open
from synapse.utils.async_typer import AsyncTyper
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm

from modeling.checkpoints.save import upload_to_gcs
from modeling.eve.os_world.agents.uitars15 import UITarsAgent
from modeling.eve.vllm_utils import wait_for_servers_ready
from modeling.utils.cloud_path import CloudPath
from modeling.utils.max_timeout import max_timeout

from .osworld_endpoints import (
    end_action_record,
    evaluate,
    reset,
    start_action_record,
    start_env,
    step_env,
    stop_env,
)

# whitelist ip from
# https://ipinfo.io/ip in gcp vm console for annotation head central us
# us_central_annotation_head = "34.136.17.148"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("aiohttp").setLevel(logging.DEBUG)

# Connect with Tailscale

app = AsyncTyper()


def create_endpoint(meta_endpoint: str, internal_ip: str, port: int = 8000) -> str:
    return f"{meta_endpoint}/{internal_ip}/{port}"


def load_tasks_file(tasks_file: str) -> list:
    """Load tasks from a local file or gs:// URL using smart_open."""
    with smart_open(tasks_file, "r") as f:
        return json.load(f)


def setup_output_folder(output_folder: str) -> tuple[str, CloudPath | None]:
    """
    Setup output folder handling for local or cloud paths.

    Returns:
        tuple: (local_output_path, cloud_path_or_none)
    """
    cloud_path = CloudPath.from_str(output_folder)

    if cloud_path.cloud != CloudPath.Cloud.FILE:
        if cloud_path.cloud == CloudPath.Cloud.S3:
            raise NotImplementedError("S3 paths not supported yet")

        # Create a temporary directory for cloud outputs
        temp_dir = tempfile.mkdtemp(prefix="osworld_eval_")
        return temp_dir, cloud_path
    else:
        # Use local path directly
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder, None


class AsyncRoundRobinLegacy:
    def __init__(self, items):
        self._items = list(items)  # keep a copy
        self._i = 0  # current index
        self._lock = asyncio.Lock()  # protects _i

    async def next(self):
        async with self._lock:  # one caller at a time
            item = self._items[self._i]
            self._i = (self._i + 1) % len(self._items)
            return item


class Language(str, Enum):
    ZH = "zh"
    EN = "en"


class Platform(str, Enum):
    AZURE = "azure"
    GCP = "gcp"


@dataclass
class EvalOptions:
    use_thinking: bool = True
    language: Language = Language.ZH
    temperature: float = 0.3
    top_p: float = 0.9
    max_trajectory_length: int = 50


# Default values for command options
DEFAULT_TASKS_FILE = "gs://induction-labs/jonathan/osworld/osworld_subset_solved_by_annotators_induction_mirror.json"
DEFAULT_MODEL_ENDPOINT = "http://localhost:8080/v1/chat/completions"
DEFAULT_META_ENDPOINT = "http://100.110.93.44"
# azure meta endpoint = "http://100.118.139.110"
DEFAULT_LANGUAGE = Language.EN
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TRAJECTORY_LENGTH = 100
DEFAULT_GPU_COUNT = 8
DEFAULT_PARALLEL_REQUESTS_PER_GPU = 12
DEFAULT_PARALLEL_VMS = 3
DEFAULT_TASK_REPEATS = 10
DEFAULT_OUTPUT_FOLDER = "gs://induction-labs/evals/osworld-evals-testing/"


def get_vm_pool(cloud: Platform) -> Any:
    if cloud == Platform.AZURE:
        from .create_osworld_vms_azure import create_vm_pool
    elif cloud == Platform.GCP:
        from .create_osworld_vms import create_vm_pool

    return create_vm_pool


async def evaluate_task(
    agent: UITarsAgent,
    task: dict,
    log_dir: str,
    base_url: str,
    session: aiohttp.ClientSession,
    max_steps: int = 5,
):
    metadata = []
    start_result = await start_env(session, base_url=base_url)
    env_id = start_result["env_id"]
    await asyncio.sleep(1)  # wait for the environment to be ready

    logger.debug(f"setting up env {env_id}")
    try:
        reset_res = await reset(session, env_id, task_config=task, base_url=base_url)
        await start_action_record(
            session,
            env_id=env_id,
            username="evals",
            output_dir=log_dir,
            output_bucket="induction-labs-data-ext-mumbai",
            base_url=base_url,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("Error during environment setup: " + str(e))
        return "ERROR", metadata

    logger.debug(f"finished setting up env {env_id}")

    obs = reset_res["obs"]
    reward = 0.0

    outputted_action = None
    model_text = None
    try:
        for step in range(max_steps):
            model_text, outputted_action = await agent.predict(task["instruction"], obs)
            metadata.append(
                {
                    "step": step,
                    "image": obs["screenshot"],
                    "action": outputted_action[0],
                    "text": model_text,
                }
            )
            bad_states = ["INTERNAL_FAIL", "PARSE_FAIL"]
            if outputted_action[0] in bad_states:
                logger.error(
                    f"Agent returned bad state {outputted_action[0]} at step {step}, stopping evaluation."
                )
                reward = outputted_action[0]
                break

            logger.debug(f"step {step} | output {model_text}\n---")
            logger.debug(f"outputted action: {outputted_action[0]}")
            if outputted_action[0] in ["DONE", "FAIL"]:
                await end_action_record(session, env_id, base_url=base_url)

            step_res = await step_env(
                session, env_id, outputted_action[0], base_url=base_url
            )

            obs = step_res["obs"]
            reward = step_res["reward"]
            finished = step_res["done"]

            if finished:
                break
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("Error during evaluation: " + str(e))

    try:
        await end_action_record(session, env_id, base_url=base_url)
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("Error ending action record: " + str(e))

    if not isinstance(reward, str):
        try:
            eval_result = await evaluate(session, env_id, base_url=base_url)
            reward = eval_result["evaluation"]
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error("Error during evaluation: " + str(e))
            reward = "ERROR"

    metadata.append(
        {
            "step": metadata[-1]["step"] + 1 if metadata else 0,
            "image": obs["screenshot"],
            "action": outputted_action[0] if outputted_action else None,
            "text": model_text,
            "reward": reward,
        }
    )

    await stop_env(session, env_id, base_url=base_url)

    return reward, metadata


async def eval_task_with_semaphore(
    semaphore,
    eval_options: EvalOptions,
    output_folder: str,
    recording_output_folder: str,
    file_lock: asyncio.Lock,
    vms: Any,
    task: dict,
    meta_endpoint: str,
    task_index: int = 0,
    model_endpoint: str = "http://localhost:8080/v1/chat/completions",
):
    await asyncio.sleep(task_index * 0.01)
    async with semaphore:
        agent = UITarsAgent(
            model_endpoint=model_endpoint,
            language=eval_options.language.value,
            use_thinking=True,
            temperature=eval_options.temperature,
            use_vllm=True,
        )
        attempt_id = uuid.uuid4().hex
        dump_folder = recording_output_folder + "/" + attempt_id
        base_url = create_endpoint(meta_endpoint, (await vms.next())["internal_ip"])
        timeout = aiohttp.ClientTimeout(total=60 * 5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            reward, metadata = await evaluate_task(
                agent,
                task,
                dump_folder,
                base_url,
                session,
                eval_options.max_trajectory_length,
            )
        await agent.aiohttp_client.close()

        eval_output_row = {
            "eval_task_id": task["id"],
            "attempt_id": attempt_id,
            "instruction": task["instruction"],
            "reward": reward,
            # "metadata": metadata,
            "output_folder": f"gs://induction-labs-data-ext-mumbai/evals/{dump_folder}",
            "trajectory_length": len(metadata),
        }

    async with file_lock:
        with open(output_folder + "/samples.jsonl", "a") as f:
            f.write(json.dumps(eval_output_row))
            f.write("\n")

    if not os.path.exists(f"{output_folder}/metadata"):
        os.makedirs(f"{output_folder}/metadata")

    with open(f"{output_folder}/metadata/{attempt_id}.json", "w") as f:
        json.dump(metadata, f)


async def evaluate_tasks_parallel(
    tasks: list,
    cloud: Platform,
    eval_options: EvalOptions,
    output_folder: str,
    recording_output_folder: str,
    max_concurrent: int,
    num_vms: int,
    meta_endpoint: str,
    model_endpoint: str = "http://localhost:8080/v1/chat/completions",
):
    create_vm_pool = get_vm_pool(cloud)
    vms = await create_vm_pool(
        num_vms=num_vms,
    )

    print(f"got {num_vms} servers, waiting for them to be ready...")
    await asyncio.sleep(10)
    print("starting evaluation...")

    try:
        semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent tasks
        file_lock = asyncio.Lock()  # Ensure thread-safe file operations
        tasks = [
            eval_task_with_semaphore(
                semaphore,
                eval_options,
                output_folder,
                recording_output_folder,
                file_lock,
                vms,
                task,
                task_index=i,
                meta_endpoint=meta_endpoint,
                model_endpoint=model_endpoint,
            )
            for i, task in enumerate(tasks)
        ]
        result = await tqdm_asyncio.gather(*tasks)
        return result
    except Exception:
        import traceback

        traceback.print_exc()

        print("error, closing vms")
    finally:
        await vms.cleanup()
        await cancel_all_tasks()


async def cancel_all_tasks():
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()  # don't cancel ourselves
    tasks = [t for t in asyncio.all_tasks(loop) if t is not current]

    for t in tasks:
        t.cancel()

    # Wait for every task to acknowledge the cancellation
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)


@app.async_command(name="run")
async def run_evaluation(
    meta_endpoint: Annotated[
        str, typer.Option("--meta-endpoint", help="Meta endpoint for VM management")
    ] = DEFAULT_META_ENDPOINT,
    tasks_file: Annotated[
        str, typer.Argument(help="Path to tasks JSON file")
    ] = DEFAULT_TASKS_FILE,
    cloud: Annotated[
        Platform, typer.Option("--cloud", help="Cloud provider")
    ] = Platform.GCP,
    model_endpoint: Annotated[
        str, typer.Option("--endpoint", help="Model endpoint URL")
    ] = DEFAULT_MODEL_ENDPOINT,
    language: Annotated[
        Language, typer.Option("--lang", help="Language for responses")
    ] = DEFAULT_LANGUAGE,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    top_p: Annotated[
        float, typer.Option("--top-p", help="Top-p for generation")
    ] = DEFAULT_TOP_P,
    max_trajectory_length: Annotated[
        int, typer.Option("--max-steps", help="Maximum steps per task")
    ] = DEFAULT_MAX_TRAJECTORY_LENGTH,
    gpu_count: Annotated[
        int, typer.Option("--gpus", help="Number of GPUs available")
    ] = DEFAULT_GPU_COUNT,
    parallel_requests_per_gpu: Annotated[
        int, typer.Option("--requests-per-gpu", help="Parallel requests per GPU")
    ] = DEFAULT_PARALLEL_REQUESTS_PER_GPU,
    parallel_vms: Annotated[
        int, typer.Option("--parallel-vms", help="Parallel VMs per request group")
    ] = DEFAULT_PARALLEL_VMS,
    task_repeats: Annotated[
        int, typer.Option("--repeats", help="Number of times to repeat each task")
    ] = DEFAULT_TASK_REPEATS,
    output_folder: Annotated[
        str, typer.Option("--output", help="Output folder for results")
    ] = DEFAULT_OUTPUT_FOLDER,
    max_tasks: Annotated[
        int | None,
        typer.Option("--max-tasks", help="Maximum number of unique tasks to evaluate"),
    ] = None,
    print_cmd: Annotated[
        bool, typer.Option("--print-cmd", help="Print command in k8s format and exit")
    ] = False,
):
    """Run OSWorld evaluation with specified parameters."""

    # Handle print-cmd option
    if print_cmd:
        cmd_parts = ["eve", "osworld", "run"]

        # meta_endpoint is required, so always add it
        cmd_parts.extend(["--meta-endpoint", meta_endpoint])

        # Add positional argument if different from default
        if tasks_file != DEFAULT_TASKS_FILE:
            cmd_parts.append(tasks_file)
        else:
            # Add the default tasks file as positional argument
            cmd_parts.append(tasks_file)

        # Add all options that differ from defaults
        if model_endpoint != DEFAULT_MODEL_ENDPOINT:
            cmd_parts.extend(["--endpoint", model_endpoint])
        if language != DEFAULT_LANGUAGE:
            cmd_parts.extend(["--lang", language.value])
        if temperature != DEFAULT_TEMPERATURE:
            cmd_parts.extend(["--temperature", str(temperature)])
        if top_p != DEFAULT_TOP_P:
            cmd_parts.extend(["--top-p", str(top_p)])
        if max_trajectory_length != DEFAULT_MAX_TRAJECTORY_LENGTH:
            cmd_parts.extend(["--max-steps", str(max_trajectory_length)])
        if gpu_count != DEFAULT_GPU_COUNT:
            cmd_parts.extend(["--gpus", str(gpu_count)])
        if parallel_requests_per_gpu != DEFAULT_PARALLEL_REQUESTS_PER_GPU:
            cmd_parts.extend(["--requests-per-gpu", str(parallel_requests_per_gpu)])
        if parallel_vms != DEFAULT_PARALLEL_VMS:
            cmd_parts.extend(["--parallel-vms", str(parallel_vms)])
        if task_repeats != DEFAULT_TASK_REPEATS:
            cmd_parts.extend(["--repeats", str(task_repeats)])
        if output_folder != DEFAULT_OUTPUT_FOLDER:
            cmd_parts.extend(["--output", output_folder])
        if max_tasks is not None:
            cmd_parts.extend(["--max-tasks", str(max_tasks)])
        if cloud != Platform.GCP:
            cmd_parts.extend(["--cloud", cloud.value])

        print(str(cmd_parts))
        return str(cmd_parts)

    import datetime

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    tasks = load_tasks_file(tasks_file)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    tasks = tasks * task_repeats  # repeat each task

    with logging_redirect_tqdm():
        concurrent_requests = gpu_count * parallel_requests_per_gpu
        number_of_osworld_vms = gpu_count * parallel_requests_per_gpu // parallel_vms
        print(
            f"starting evaluation with {concurrent_requests} concurrent requests and {number_of_osworld_vms} vms"
        )

        # Wait for vLLM server to be ready
        print(f"Waiting for vLLM server at {model_endpoint} to be ready...")

        await max_timeout(
            wait_for_servers_ready(
                [model_endpoint.replace("/v1/chat/completions", "")]
            ),
            timedelta(minutes=10),
            "Timeout waiting for vLLM server to be ready",
        )

        # Setup output folder handling
        local_output_folder, cloud_output_path = setup_output_folder(output_folder)

        try:
            await evaluate_tasks_parallel(
                tasks=tasks,
                cloud=cloud,
                eval_options=EvalOptions(
                    use_thinking=True,
                    language=language,
                    temperature=temperature,
                    top_p=top_p,
                    max_trajectory_length=max_trajectory_length,
                ),
                output_folder=local_output_folder,
                recording_output_folder=f"testing-task-{date}",
                max_concurrent=concurrent_requests,
                num_vms=number_of_osworld_vms,
                meta_endpoint=meta_endpoint,
                model_endpoint=model_endpoint,
            )

            # Upload to GCS if needed
            if cloud_output_path:
                print(f"Uploading results to {cloud_output_path.to_str()}...")
                bucket_name, gcs_path = cloud_output_path.bucket_and_path
                await asyncio.to_thread(
                    upload_to_gcs,
                    local_dir=Path(local_output_folder),
                    gcs_bucket=bucket_name,
                    gcs_prefix=gcs_path,
                )
                print("Upload completed!")

        finally:
            # Clean up temporary directory if used
            if cloud_output_path and os.path.exists(local_output_folder):
                import shutil

                shutil.rmtree(local_output_folder)


# eve osworld run --output gs://induction-labs/evals/osworld-evals-testing/ --gpus 1 --requests-per-gpu 12 --parallel-vms 3 --repeats 1 --max-tasks 20

if __name__ == "__main__":
    app()
