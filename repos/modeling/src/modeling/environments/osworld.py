import asyncio
import json
import logging
import contextlib
import os
import uuid
from dataclasses import dataclass
from typing import Literal

import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm

from modeling.environments.agents.uitars15 import UITarsAgent
from modeling.environments.create_osworld_vms import AsyncVMRoundRobin, create_vm_pool
from modeling.environments.osworld_endpoints import (
    end_action_record,
    evaluate,
    meta_get_vm,
    meta_return_vm,
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
USE_VLLM = True


META_ENDPOINT = "http://34.136.17.148"


def create_endpoint(internal_ip: str, port: int = 8000) -> str:
    return f"{META_ENDPOINT}/{internal_ip}/{port}"


MODEL_ENDPOINT = (
    "http://localhost:8080/generate"
    if not USE_VLLM
    else "http://localhost:8080/v1/chat/completions"
)


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


@dataclass
class EvalOptions:
    model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B"
    use_thinking: bool = True
    language: Literal["zh", "en"] = "zh"
    temperature: float = 0.2
    top_p: float = 0.9
    max_trajectory_length: int = 50


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
    vms: AsyncVMRoundRobin,
    task: dict,
    task_index: int = 0,
):
    await asyncio.sleep(task_index * 0.01)
    async with semaphore:
        agent = UITarsAgent(
            model_endpoint=MODEL_ENDPOINT,
            language=eval_options.language,
            use_thinking=True,
            use_vllm=USE_VLLM,
        )
        attempt_id = uuid.uuid4().hex
        dump_folder = recording_output_folder + "/" + attempt_id
        base_url = create_endpoint((await vms.next())["internal_ip"])
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

        eval_output_row = {
            "eval_task_id": task["id"],
            "attempt_id": attempt_id,
            "instruction": task["instruction"],
            "reward": reward,
            # "metadata": metadata,
            "output_folder": f"gs://induction-labs-data-ext-mumbai/evals/{dump_folder}",
        }

        async with file_lock:
            with open(output_folder + "/samples.jsonl", "a") as f:
                f.write(json.dumps(eval_output_row))
                f.write("\n")

        if not os.path.exists(f"{output_folder}/metadata"):
            os.makedirs(f"{output_folder}/metadata")

        with open(f"{output_folder}/metadata/{attempt_id}.json", "w") as f:
            json.dump(metadata, f)

        await agent.aiohttp_client.close()


async def evaluate_tasks_parallel(
    tasks: list,
    eval_options: EvalOptions,
    output_folder: str,
    recording_output_folder: str,
    max_concurrent: int,
    num_vms: int,
):
    if not USE_VLLM:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                "http://localhost:8000/change_model",
                json={
                    "model_name": eval_options.model_name,
                    "temperature": eval_options.temperature,
                    "top_p": eval_options.top_p,
                },
            ) as response,
        ):
            if response.status != 200:
                logger.error(f"Failed to change model: {response.status}")
            else:
                logger.info("Model changed successfully")

    vms = await create_vm_pool(
        num_vms=num_vms,
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
    current = asyncio.current_task()              # don’t cancel ourselves
    tasks = [t for t in asyncio.all_tasks(loop) if t is not current]

    for t in tasks:
        t.cancel()

    # Wait for every task to acknowledge the cancellation
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    import datetime

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open("/home/ubuntu/induction-labs/repos/modeling/solveable_tasks.json") as f:
        tasks = json.load(f)
        tasks = tasks * 10  # run each task 5 times

    with logging_redirect_tqdm():
        gpu_count = 8
        parallel_requests_per_gpu = 12
        parallel_vms = 3
        # gpu_count = 1
        # parallel_requests_per_gpu = 1
        # parallel_vms = 1

        concurrent_requests = gpu_count * parallel_requests_per_gpu
        number_of_osworld_vms = gpu_count * parallel_requests_per_gpu // parallel_vms
        print(
            f"starting evaluation with {concurrent_requests} concurrent requests and {number_of_osworld_vms} vms"
        )

        asyncio.run(
            evaluate_tasks_parallel(
                tasks=tasks,
                eval_options=EvalOptions(
                    model_name="ByteDance-Seed/UI-TARS-1.5-7B",
                    use_thinking=True,
                    language="en",
                    temperature=0.2,
                    top_p=0.9,
                    max_trajectory_length=100,
                ),
                output_folder="osworld_uitars_testing",
                recording_output_folder=f"testing-task-{date}",
                # 4 requests per gpu - i think we can do more
                max_concurrent=concurrent_requests,
                num_vms=number_of_osworld_vms,
            )
        )
