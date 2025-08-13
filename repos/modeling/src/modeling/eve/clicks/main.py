from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

# Import the evaluate_csv function from the showdown clicks package
from clicks.eval import EvaluationResult, evaluate_csv
from clicks.third_party import get_ui_tars_api_client
from pydantic import computed_field
from synapse.utils.async_typer import AsyncTyper
from synapse.utils.logging import configure_logging, logging

import wandb
from modeling.checkpoints.load import download_gcs_folder
from modeling.checkpoints.save import upload_to_gcs
from modeling.eve.os_world.agents.uitars15 import (
    COMPUTER_USE_15,
    COMPUTER_USE_15_ONLY_CLICKS,
    THOUGHT_LONG,
)
from modeling.eve.vllm_utils import wait_for_servers_ready
from modeling.utils.cloud_path import CloudPath
from modeling.utils.max_timeout import max_timeout

logger = configure_logging(__name__, level=logging.INFO)

app = AsyncTyper()


class PromptTemplates(str, Enum):
    uitars15 = "computer_use_15"
    only_clicks = "only_clicks"


prompt_templates: Mapping[PromptTemplates, str] = {
    PromptTemplates.uitars15: COMPUTER_USE_15,
    PromptTemplates.only_clicks: COMPUTER_USE_15_ONLY_CLICKS,
}
# Default values for command options
DEFAULT_API_URL = "http://127.0.0.1:8080"
DEFAULT_UI_TARS_MODEL = ""
DEFAULT_DATASET = "dev"
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_OUTPUT_FOLDER = "gs://induction-labs/evals/clicks-evals/"
DEFAULT_PROMPT_TEMPLATE = PromptTemplates.only_clicks


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


class AugmentedEvaluationResult(EvaluationResult):
    @computed_field
    @property
    def center_coords(self) -> tuple[float, float] | None:
        if self.pred_x is not None and self.pred_y is not None:
            return (self.pred_x, self.pred_y)
        return None

    @computed_field
    @property
    def x_error(self) -> float | None:
        if self.center_coords and self.gt_x1 is not None:
            return self.gt_x1 - self.center_coords[0]
        return None

    @computed_field
    @property
    def y_error(self) -> float | None:
        if self.center_coords and self.gt_y1 is not None:
            return self.gt_y1 - self.center_coords[1]
        return None

    @computed_field
    @property
    def pixel_distance(self) -> float | None:
        if self.center_coords and self.gt_x1 is not None and self.gt_y1 is not None:
            return (
                (self.gt_x1 - self.center_coords[0]) ** 2
                + (self.gt_y1 - self.center_coords[1]) ** 2
            ) ** 0.5
        return None


@app.async_command(name="run")
async def run_clicks_evaluation(
    api_url: Annotated[
        str, typer.Option("--api-url", help="API endpoint for the UI-TARS model")
    ] = DEFAULT_API_URL,
    checkpoint_dir: Annotated[
        str, typer.Option("--checkpoint-dir", help="UI-TARS model name")
    ] = DEFAULT_UI_TARS_MODEL,
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset to evaluate on")
    ] = DEFAULT_DATASET,
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
    prompt_template: Annotated[
        PromptTemplates,
        typer.Option(
            "--prompt-template",
            help="Prompt template to use for evaluation",
            case_sensitive=False,
            show_choices=True,
            show_default=True,
        ),
    ] = DEFAULT_PROMPT_TEMPLATE,
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
        if prompt_template != DEFAULT_PROMPT_TEMPLATE:
            cmd_parts.extend(["--prompt-template", prompt_template.value])
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

    # Setup output folder handling
    local_output_folder, cloud_output_path = setup_output_folder(output_folder)
    prompt_template_str = prompt_templates[prompt_template].format(
        language="en",
        thought_mode=THOUGHT_LONG,
        instruction="{instruction}",
    )

    # Initialize wandb
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
            "prompt_template_str": prompt_template_str,
            "prompt_template": prompt_template.value,
        },
        tags=["clicks", "evaluation", "ui-tars"],
    )

    # Download the dataset from GCS
    print("Downloading clicks dataset from GCS...")
    dataset_tmpdir = tempfile.mkdtemp(prefix="clicks_dataset_")

    try:
        # Create API client
        api_client = get_ui_tars_api_client(
            api_url=api_url,
            prompt_template=prompt_template_str,
            api_key="super-secret-key",  # Default key for vLLM
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            # Model name is always blank because we set blank model name on vllm.
            model_name="",
        )

        await asyncio.to_thread(
            download_gcs_folder,
            bucket_name="click-eval",
            prefix="generalagents-showdown-clicks",
            local_dir=Path(dataset_tmpdir),
        )
        data_dir = dataset_tmpdir

        if dataset == "dev":
            csv_file = os.path.join(data_dir, "showdown-clicks-dev/data.csv")
            frames_dir = os.path.join(data_dir, "showdown-clicks-dev")
        else:
            raise ValueError("Only 'dev' dataset is currently supported")

        # Set output file path
        output_file = os.path.join(local_output_folder, f"clicks_results_{dataset}.csv")

        print("Running evaluation:")
        print(f"  Dataset: {dataset}")
        print(f"  CSV file: {csv_file}")
        print(f"  Frames directory: {frames_dir}")
        print(f"  API URL: {api_url}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Workers: {num_workers}")
        print(f"  Output file: {output_file}")
        if sample_size:
            print(f"  Sample size: {sample_size}")

        # Run the evaluation using the evaluate_csv function from showdown
        results = await asyncio.to_thread(
            evaluate_csv,
            csv_file=csv_file,
            frames_dir=frames_dir,
            api_client=api_client,
            output_file=output_file,
            sample_size=sample_size,
            num_workers=num_workers,
            run_id=run_id,
        )
        results = [
            AugmentedEvaluationResult.model_validate(result.model_dump())
            for result in results
            if isinstance(result, EvaluationResult)
        ]

        if results:
            print("\nEvaluation completed successfully!")
            print(f"Results: {len(results)} items processed")
            print(f"Results saved to: {output_file}")

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
                sum(result.latency_seconds for result in results) / total_processed
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
                "prompt_template_str": prompt_template_str,
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

        # Clean up dataset temporary directory
        if os.path.exists(dataset_tmpdir):
            shutil.rmtree(dataset_tmpdir)

        # Clean up output temporary directory if used
        if cloud_output_path and os.path.exists(local_output_folder):
            shutil.rmtree(local_output_folder)


if __name__ == "__main__":
    app()
