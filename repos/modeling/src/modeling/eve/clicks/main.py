from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

# Import the evaluate_csv function from the showdown clicks package
from clicks.eval import evaluate_csv
from clicks.third_party import get_ui_tars_api_client
from synapse.utils.async_typer import AsyncTyper

from modeling.checkpoints.save import upload_to_gcs
from modeling.eve.vllm_utils import wait_for_servers_ready
from modeling.utils.cloud_path import CloudPath
from modeling.utils.max_timeout import max_timeout

app = AsyncTyper()


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


@app.async_command(name="run")
async def run_clicks_evaluation(
    api_url: Annotated[
        str, typer.Option("--api-url", help="API endpoint for the UI-TARS model")
    ] = "http://127.0.0.1:8080",
    ui_tars_model: Annotated[
        str, typer.Option("--ui-tars-model", help="UI-TARS model name")
    ] = "",
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset to evaluate on")
    ] = "dev",
    num_workers: Annotated[
        int, typer.Option("--num-workers", help="Number of concurrent workers")
    ] = 1,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Maximum tokens for model response")
    ] = 128,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Temperature for sampling")
    ] = 0.0,
    frequency_penalty: Annotated[
        float, typer.Option("--frequency-penalty", help="Frequency penalty parameter")
    ] = 1.0,
    sample_size: Annotated[
        int | None,
        typer.Option(
            "--sample-size", help="Number of samples to evaluate (for testing)"
        ),
    ] = None,
    output_folder: Annotated[
        str, typer.Option("--output", help="Output folder for results")
    ] = "gs://induction-labs/evals/clicks-evals/",
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Custom run ID")
    ] = None,
):
    """Run clicks evaluation with specified parameters."""

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
    run_output_folder = os.path.join(local_output_folder, run_id)
    os.makedirs(run_output_folder, exist_ok=True)

    try:
        # Create API client
        api_client = get_ui_tars_api_client(
            api_url=api_url,
            api_key="super-secret-key",  # Default key for vLLM
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            model_name=ui_tars_model,
        )

        # Import necessary modules for dataset loading
        from huggingface_hub import snapshot_download

        # Download the dataset
        print("Downloading clicks dataset...")
        data_dir = snapshot_download(
            repo_id="generalagents/showdown-clicks",
            repo_type="dataset",
        )

        if dataset == "dev":
            csv_file = os.path.join(data_dir, "showdown-clicks-dev/data.csv")
            frames_dir = os.path.join(data_dir, "showdown-clicks-dev")
        else:
            raise ValueError("Only 'dev' dataset is currently supported")

        # Set output file path
        output_file = os.path.join(run_output_folder, f"clicks_results_{dataset}.csv")

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

        if results:
            print("\nEvaluation completed successfully!")
            print(f"Results: {len(results)} items processed")
            print(f"Results saved to: {output_file}")

            # Calculate accuracy
            total_processed = len(results)
            total_in_bbox = sum(
                1 for result in results if result.get("is_in_bbox", False)
            )
            accuracy = (
                (total_in_bbox / total_processed) * 100 if total_processed > 0 else 0
            )

            print(f"Accuracy: {accuracy:.2f}% ({total_in_bbox}/{total_processed})")

            # Save summary metrics
            metrics = {
                "run_id": run_id,
                "model": "ui-tars",
                "dataset": dataset,
                "accuracy": accuracy,
                "total_processed": total_processed,
                "total_in_bbox": total_in_bbox,
                "api_url": api_url,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "num_workers": num_workers,
                "sample_size": sample_size,
            }

            import json

            with open(os.path.join(run_output_folder, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        else:
            print("Evaluation failed: No results returned.")
            return

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
            shutil.rmtree(local_output_folder)


if __name__ == "__main__":
    app()
