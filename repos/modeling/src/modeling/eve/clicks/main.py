from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from PIL import Image, ImageOps

# Import the evaluate_csv function from the showdown clicks package
from pydantic import BaseModel, computed_field
from synapse.utils.async_typer import AsyncTyper
from synapse.utils.logging import configure_logging, logging

import wandb
from modeling.checkpoints.save import upload_to_gcs
from modeling.eve.clicks.api_client import ClickModelClient, ClickModelClientResponse
from modeling.eve.clicks.model_template import (
    MODEL_TEMPLATES,
    BaseClickModelTemplate,
    ModelTemplateChoice,
)
from modeling.eve.clicks.mp import run_mp
from modeling.eve.os_world.agents.uitars15 import (
    COMPUTER_USE_15,
    COMPUTER_USE_15_ONLY_CLICKS,
    THOUGHT_BRIEF,
    THOUGHT_LONG,
    THOUGHT_LONG_REPEAT,
)
from modeling.eve.vllm_utils import wait_for_servers_ready
from modeling.utils.cloud_path import CloudPath
from modeling.utils.max_timeout import max_timeout

logger = configure_logging(__name__, level=logging.INFO)

app = AsyncTyper()

k = [THOUGHT_LONG, THOUGHT_BRIEF, THOUGHT_LONG_REPEAT]

PROMPT_TEMPLATE = """Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2].'""".strip()


class PromptTemplates(str, Enum):
    uitars15 = "computer_use_15"
    only_clicks = "only_clicks"


prompt_templates: Mapping[PromptTemplates, str] = {
    PromptTemplates.uitars15: COMPUTER_USE_15,
    PromptTemplates.only_clicks: COMPUTER_USE_15_ONLY_CLICKS,
}


class ClickDatasets(str, Enum):
    click = (
        "gs://click-eval/generalagents-showdown-clicks/showdown-clicks-dev/data.jsonl"
    )


# Default values for command options
DEFAULT_API_URL = "http://127.0.0.1:8080"
DEFAULT_UI_TARS_MODEL = ""
DEFAULT_DATASET = ClickDatasets.click
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_OUTPUT_FOLDER = "gs://induction-labs/evals/clicks-evals/"
DEFAULT_MODEL_TEMPLATE = ModelTemplateChoice.uitars


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
        temp_dir = tempfile.mkdtemp(prefix="clicks_eval_")
        return temp_dir, cloud_path
    else:
        # Use local path directly
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder, None


class ClickInput(BaseModel):
    id: str
    image_url: str
    instruction: str
    id: str
    width: int
    height: int
    x1: float
    y1: float
    x2: float
    y2: float


class AugmentedEvaluationResult(BaseModel):
    input: ClickInput
    response: ClickModelClientResponse
    prompt_text: str
    prediction_point: tuple[float, float] | None = None

    @computed_field
    @property
    def center_coords(self) -> tuple[float, float] | None:
        center_x = (self.input.x1 + self.input.x2) / 2.0
        center_y = (self.input.y1 + self.input.y2) / 2.0
        return (center_x, center_y)

    @computed_field
    @property
    def x_error(self) -> float | None:
        if self.center_coords and self.prediction_point is not None:
            return self.prediction_point[0] - self.center_coords[0]
        return None

    @computed_field
    @property
    def y_error(self) -> float | None:
        if self.center_coords and self.prediction_point is not None:
            return self.prediction_point[1] - self.center_coords[1]
        return None

    @computed_field
    @property
    def pixel_distance(self) -> float | None:
        if self.x_error is not None and self.y_error is not None:
            return ((self.x_error) ** 2 + (self.y_error) ** 2) ** 0.5
        return None

    @computed_field
    @property
    def is_in_bbox(self) -> bool:
        if self.prediction_point is not None:
            return (
                self.input.x1 <= self.prediction_point[0] <= self.input.x2
                and self.input.y1 <= self.prediction_point[1] <= self.input.y2
            )
        return False


def get_base64_from_image_path(
    image_path: str, image_dimensions: tuple[int, int] | None = None
) -> str:
    """
    Load an image from a URL, filesystem path, or base64 string (optionally a data URL),
    convert it to PNG, optionally assert its dimensions, and return a data URL:
    'data:image/png;base64,<...>'.

    Supports PNG/JPEG/WebP inputs.
    Prefers assertion to silent failure when dimensions are provided.
    """
    data_url_prefix = "data:image/png;base64,"

    def _read_bytes_from_input(s: str) -> bytes:
        s = s.strip()

        # Case 1: data URL like "data:image/xxx;base64,<b64>"
        m = re.match(r"^data:image/[^;]+;base64,(?P<payload>[A-Za-z0-9+/=\n\r]+)$", s)
        if m:
            return base64.b64decode(m.group("payload"), validate=True)

        # Case 2: looks like raw base64 (no filesystem path, no scheme)
        looks_like_path = ("://" in s) or os.path.exists(s)
        if not looks_like_path:
            try:
                return base64.b64decode(s, validate=True)
            except Exception:
                pass  # fall through to other options

        # Case 3: URL (http/https)
        if s.startswith(("http://", "https://")):
            with urllib.request.urlopen(s, timeout=30) as resp:
                return resp.read()

        # Case 4: filesystem path
        if os.path.exists(s):
            with open(s, "rb") as f:
                return f.read()

        raise ValueError("Input is not a valid URL, file path, or base64 image.")

    raw_bytes = _read_bytes_from_input(image_path)

    # Open with Pillow, apply EXIF-aware orientation, then (optionally) assert size.
    with Image.open(io.BytesIO(raw_bytes)) as im:
        im = ImageOps.exif_transpose(im)
        if image_dimensions is not None:
            assert im.size == image_dimensions, (
                f"Image dimensions mismatch: expected {image_dimensions}, got {im.size}"
            )

        # Convert to PNG bytes
        # Use RGBA for broad compatibility; Pillow will drop alpha if not needed.
        converted = im.convert("RGBA")
        buf = io.BytesIO()
        converted.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"{data_url_prefix}{b64}"


def process_single_item(
    item: ClickInput,
    click_client: ClickModelClient,
    model_template: BaseClickModelTemplate,
) -> AugmentedEvaluationResult:
    base64_image = get_base64_from_image_path(item.image_url)
    prompt_text = model_template.instruction_text(item.instruction)
    response: ClickModelClientResponse = click_client.call_model(
        base64_image=base64_image,
        prompt_text=prompt_text,
    )
    response_point = model_template.extract_coordinates(
        response.content, (item.width, item.height)
    )
    return AugmentedEvaluationResult(
        input=item,
        response=response,
        prompt_text=prompt_text,
        prediction_point=response_point,
    )


#
# return process_single_item


@app.async_command(name="run")
async def run_clicks_evaluation(
    api_url: Annotated[
        str, typer.Option("--api-url", help="API endpoint for the UI-TARS model")
    ] = DEFAULT_API_URL,
    checkpoint_dir: Annotated[
        str, typer.Option("--checkpoint-dir", help="UI-TARS model name")
    ] = DEFAULT_UI_TARS_MODEL,
    dataset: Annotated[
        ClickDatasets, typer.Option("--dataset", help="Dataset to evaluate on")
    ] = DEFAULT_DATASET,
    model_template: Annotated[
        ModelTemplateChoice,
        typer.Option(
            "--model-template",
            help="Model template to use for prompt formatting and response parsing",
        ),
    ] = DEFAULT_MODEL_TEMPLATE,
    num_workers: Annotated[
        int, typer.Option("--num-workers", help="Number of concurrent workers")
    ] = DEFAULT_NUM_WORKERS,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Maximum tokens for model response")
    ] = DEFAULT_MAX_TOKENS,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Temperature for sampling")
    ] = DEFAULT_TEMPERATURE,
    frequency_penalty: Annotated[
        float, typer.Option("--frequency-penalty", help="Frequency penalty parameter")
    ] = DEFAULT_FREQUENCY_PENALTY,
    sample_size: Annotated[
        int | None,
        typer.Option(
            "--sample-size", help="Number of samples to evaluate (for testing)"
        ),
    ] = None,
    output_folder: Annotated[
        str, typer.Option("--output", help="Output folder for results")
    ] = DEFAULT_OUTPUT_FOLDER,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Custom run ID")
    ] = None,
    print_cmd: Annotated[
        bool, typer.Option("--print-cmd", help="Print command in k8s format and exit")
    ] = False,
):
    """Run clicks evaluation with specified parameters."""

    # Handle print-cmd option
    if print_cmd:
        cmd_parts = ["eve", "clicks", "run"]
        # Add all options that differ from defaults
        if api_url != DEFAULT_API_URL:
            cmd_parts.extend(["--api-url", api_url])
        if checkpoint_dir != DEFAULT_UI_TARS_MODEL:
            cmd_parts.extend(["--checkpoint-dir", checkpoint_dir])
        if dataset != DEFAULT_DATASET:
            cmd_parts.extend(["--dataset", dataset])
        if num_workers != DEFAULT_NUM_WORKERS:
            cmd_parts.extend(["--num-workers", str(num_workers)])
        if max_tokens != DEFAULT_MAX_TOKENS:
            cmd_parts.extend(["--max-tokens", str(max_tokens)])
        if temperature != DEFAULT_TEMPERATURE:
            cmd_parts.extend(["--temperature", str(temperature)])
        if frequency_penalty != DEFAULT_FREQUENCY_PENALTY:
            cmd_parts.extend(["--frequency-penalty", str(frequency_penalty)])
        if sample_size is not None:
            cmd_parts.extend(["--sample-size", str(sample_size)])
        if model_template != DEFAULT_MODEL_TEMPLATE:
            cmd_parts.extend(["--model-template", model_template.value])
        if output_folder != DEFAULT_OUTPUT_FOLDER:
            cmd_parts.extend(["--output", output_folder])
        if run_id is not None:
            cmd_parts.extend(["--run-id", run_id])

        print(str(cmd_parts))
        return str(cmd_parts)

    # Generate run ID if not provided
    if run_id is None:
        run_id = f"clicks-eval-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    print(f"Starting clicks evaluation with run ID: {run_id}")

    # Wait for vLLM server to be ready
    print(f"Waiting for vLLM server at {api_url} to be ready...")
    await max_timeout(
        wait_for_servers_ready([api_url]),
        timedelta(minutes=5),
        "Timeout waiting for vLLM server to be ready",
    )
    local_output_folder, cloud_output_path = setup_output_folder(output_folder)

    wandb.init(
        project="clicks-eval",
        name=run_id,
        config={
            "api_url": api_url,
            "checkpoint_dir": checkpoint_dir,
            "dataset": dataset,
            "num_workers": num_workers,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "sample_size": sample_size,
            "output_folder": output_folder,
            "cloud_output_path": cloud_output_path.uri if cloud_output_path else None,
            "model_template": model_template.value,
        },
        tags=["clicks", "evaluation", "ui-tars"],
    )

    model_template_instance: BaseClickModelTemplate = MODEL_TEMPLATES[model_template]
    click_client = ClickModelClient(
        api_url=api_url,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        model_name="",
    )
    process_func = partial(
        process_single_item,
        click_client=click_client,
        model_template=model_template_instance,
    )
    data_df = pd.read_json(dataset, lines=True)
    if sample_size is not None:
        data_df = data_df.sample(n=sample_size, random_state=42)
        print(f"limited dataset to {len(data_df)} samples")
    data_inputs = [
        ClickInput.model_validate(row) for row in data_df.to_dict(orient="records")
    ]

    try:
        results = await asyncio.to_thread(
            run_mp,
            items=data_inputs,
            process_func=process_func,
            output_cls=AugmentedEvaluationResult,
            num_workers=num_workers,
        )

        # Run the evaluation using the evaluate_csv function from showdown

        if results:
            print("\nEvaluation completed successfully!")
            print(f"Results: {len(results)} items processed")

            # Calculate accuracy
            total_processed = len(results)
            total_in_bbox = sum(1 for result in results if result.is_in_bbox)
            accuracy = (
                (total_in_bbox / total_processed) * 100 if total_processed > 0 else 0
            )
            err_x = (
                sum(result.x_error for result in results if result.x_error is not None)
                / total_processed
            )
            err_y = (
                sum(result.y_error for result in results if result.y_error is not None)
                / total_processed
            )
            avg_dist = (
                sum(
                    result.pixel_distance
                    for result in results
                    if result.pixel_distance is not None
                )
                / total_processed
                if total_processed > 0
                else 0
            )
            avg_elapsed = (
                sum(result.response.latency_seconds for result in results)
                / total_processed
                if total_processed > 0
                else 0
            )

            print(f"Accuracy: {accuracy:.2f}% ({total_in_bbox}/{total_processed})")

            # Save summary metrics
            wandb_metrics = {
                "accuracy": accuracy,
                "total_processed": total_processed,
                "total_in_bbox": total_in_bbox,
                "accuracy_percentage": accuracy,
                "success_rate": accuracy / 100.0,
                "avg_x_error": err_x,
                "avg_y_error": err_y,
                "avg_pixel_distance": avg_dist,
                "avg_latency_seconds": avg_elapsed,
            }

            metrics = {
                "run_id": run_id,
                "model": "ui-tars",
                "dataset": dataset,
                "api_url": api_url,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "num_workers": num_workers,
                "sample_size": sample_size,
                **wandb_metrics,
            }

            # Log metrics to wandb
            wandb.log(wandb_metrics)

            # Log the full results as a table for detailed analysis
            results_dict = [result.model_dump() for result in results]
            table_columns = list(results_dict[0].keys()) if results else []
            results_table = wandb.Table(
                columns=table_columns,
                data=[
                    [result[col] for col in table_columns] for result in results_dict
                ],
            )
            wandb.log({"results_table": results_table})
            results_df = pd.DataFrame(results_dict)
            results_df.to_json(
                os.path.join(local_output_folder, "results.jsonl"),
                orient="records",
                lines=True,
            )

            import json

            with open(os.path.join(local_output_folder, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        else:
            print("Evaluation failed: No results returned.")
            return

        # Upload to GCS if needed
        if cloud_output_path:
            print(f"Uploading results to {cloud_output_path.uri}...")
            bucket_name, gcs_path = cloud_output_path.bucket_and_path
            await asyncio.to_thread(
                upload_to_gcs,
                local_dir=Path(local_output_folder),
                gcs_bucket=bucket_name,
                gcs_prefix=gcs_path,
            )
            print("Upload completed!")

    finally:
        # Finish wandb run
        wandb.finish()

        # Clean up output temporary directory if used
        if cloud_output_path and os.path.exists(local_output_folder):
            shutil.rmtree(local_output_folder)


if __name__ == "__main__":
    app()
