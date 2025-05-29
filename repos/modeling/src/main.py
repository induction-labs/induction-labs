from __future__ import annotations

import os

os.environ["VLLM_USE_V1"] = "1"
from vllm import envs as vllm_envs

assert vllm_envs.VLLM_USE_V1, f"vllm_envs.VLLM_USE_V1={vllm_envs.VLLM_USE_V1}"
import asyncio  # noqa: E402

import ray  # noqa: E402
import vllm  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

# print("MEMES")


# from modeling
@ray.remote
class VllmApp:
    def __init__(self):
        import vllm.envs as vllm_envs

        assert vllm_envs.VLLM_USE_V1, f"vllm_envs.VLLM_USE_V1={vllm_envs.VLLM_USE_V1}"
        print(f"vllm_envs.VLLM_USE_V1={vllm_envs.VLLM_USE_V1}")
        import vllm

        engine_args = vllm.AsyncEngineArgs(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            distributed_executor_backend="ray",
            disable_log_requests=True,
        )
        self.llm = AsyncLLM.from_engine_args(engine_args)

    async def run(self):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=32,
            min_tokens=32,
        )
        stream = self.llm.generate(
            prompt="The capital of France is",
            sampling_params=sampling_params,
            request_id="1",
        )
        async for output in stream:
            if output.finished:
                print(output.outputs[0].text)


async def main():
    engine_args = vllm.AsyncEngineArgs(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        # distributed_executor_backend="ray",
        disable_log_requests=True,
    )
    llm = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        min_tokens=32,
    )
    stream = llm.generate(
        prompt="The capital of France is",
        sampling_params=sampling_params,
        request_id="1",
    )
    async for output in stream:
        if output.finished:
            print(output.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(main())

    # pg_resources = [
    #     {"CPU": 1.0},
    #     {"CPU": 1.0, "GPU": 1.0},
    # ]
    # pg = ray.util.placement_group(pg_resources)
    # app = VllmApp.options(placement_group=pg).remote()
    # result = app.run.remote()
    # ray.get(result)


def hello() -> str:
    return "Hello from numbers!"
