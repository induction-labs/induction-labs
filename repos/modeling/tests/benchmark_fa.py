from __future__ import annotations

import argparse
import gc
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=512,
        help="",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="",
    )
    return parser


model_id = "meta-llama/Meta-Llama-3-8B"


def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def reset_memory_stats():
    """Reset memory statistics and clear cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


@torch.no_grad()
def warmup_and_benchmark(
    model,
    tokenizer,
    max_seq_len,
    num_batches,
    max_new_tokens,
):
    inputs = tokenizer("Hi" * max_seq_len, return_tensors="pt").to("cuda")

    # Reset memory stats before benchmarking
    reset_memory_stats()

    # warmup
    _ = model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # Clear cache and reset memory stats after warmup
    reset_memory_stats()

    # Record baseline memory
    baseline_memory = get_memory_usage()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    stream = torch.cuda.default_stream()

    with torch.no_grad():
        start_event.record(stream)
        for _ in range(num_batches):
            _ = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        end_event.record(stream)
        torch.cuda.synchronize()

    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    # current_memory = get_memory_usage()
    memory_used = peak_memory - baseline_memory

    forward_timing = (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches

    return forward_timing, memory_used, peak_memory


if __name__ == "__main__":
    # Record script start time
    script_start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()
    num_batches = args.num_batches
    max_seq_len = args.max_seqlen
    max_new_tokens = args.max_new_tokens

    print("Starting benchmark with parameters:")
    print(f"  Model: {model_id}")
    print(f"  Num batches: {num_batches}")
    print(f"  Max new tokens: {max_new_tokens}")
    print("  Sequence lengths to test: [256, 1024, 2048, 4096, 8192]")
    print("-" * 60)

    # Load models
    print("Loading tokenizer and models...")
    model_load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")

    model_fa = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    model_load_time = time.time() - model_load_start
    print(f"Models loaded in {model_load_time:.2f} seconds")
    print("-" * 60)

    # Initialize dictionaries for results
    native_total_time_dict = {}
    fa2_total_time_dict = {}
    forward_speedups = {}
    native_memory_dict = {}
    fa2_memory_dict = {}
    memory_savings = {}

    # Create list to store detailed results for table
    results_data = []

    for max_seq_len in [256, 1024, 2048, 4096, 8192]:
        print(f"Running for sequence length {max_seq_len}")
        seq_start_time = time.time()

        # Benchmark native model
        print("  Benchmarking native model...")
        native_timing, native_memory, native_peak = warmup_and_benchmark(
            model,
            tokenizer,
            max_seq_len,
            num_batches,
            max_new_tokens,
        )
        native_total_time_dict[f"{max_seq_len}"] = native_timing
        native_memory_dict[f"{max_seq_len}"] = native_memory

        # Benchmark FA2 model
        print("  Benchmarking Flash Attention-2 model...")
        fa2_timing, fa2_memory, fa2_peak = warmup_and_benchmark(
            model_fa,
            tokenizer,
            max_seq_len,
            num_batches,
            max_new_tokens,
        )
        fa2_total_time_dict[f"{max_seq_len}"] = fa2_timing
        fa2_memory_dict[f"{max_seq_len}"] = fa2_memory

        # Calculate speedup and memory savings
        speedup = native_timing / fa2_timing
        forward_speedups[f"{max_seq_len}"] = speedup

        memory_saving = (
            ((native_memory - fa2_memory) / native_memory * 100)
            if native_memory > 0
            else 0
        )
        memory_savings[f"{max_seq_len}"] = memory_saving

        seq_total_time = time.time() - seq_start_time

        # Store results for table
        results_data.append(
            {
                "Sequence Length": max_seq_len,
                "Native Time (s)": f"{native_timing:.4f}",
                "FA2 Time (s)": f"{fa2_timing:.4f}",
                "Speedup": f"{speedup:.2f}x",
                "Native Memory (MB)": f"{native_memory:.1f}",
                "FA2 Memory (MB)": f"{fa2_memory:.1f}",
                "Memory Savings (%)": f"{memory_saving:.1f}%",
                "Total Time (s)": f"{seq_total_time:.2f}",
            }
        )

        print(
            f"  Native: {native_timing:.4f}s ({native_memory:.1f}MB) | FA2: {fa2_timing:.4f}s ({fa2_memory:.1f}MB)"
        )
        print(f"  Speedup: {speedup:.2f}x | Memory Savings: {memory_saving:.1f}%")
        print(f"  Completed in {seq_total_time:.2f} seconds")
        print()

    # Create output directory
    dir_name = f"flash-attn-2-benchmarks/{model_id}/seq_len_{args.max_seqlen}/"
    os.makedirs(dir_name, exist_ok=True)

    # Create and save plots
    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Convert keys to integers for proper plotting
    seq_lengths = [int(k) for k in native_total_time_dict]
    native_times = [native_total_time_dict[str(k)] for k in seq_lengths]
    fa2_times = [fa2_total_time_dict[str(k)] for k in seq_lengths]
    native_memories = [native_memory_dict[str(k)] for k in seq_lengths]
    fa2_memories = [fa2_memory_dict[str(k)] for k in seq_lengths]

    # Plot 1: Timing comparison
    ax1.plot(
        seq_lengths,
        native_times,
        "b-o",
        label=f"{model_id}-native",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        seq_lengths,
        fa2_times,
        "orange",
        marker="o",
        label=f"{model_id}-FA2",
        linewidth=2,
        markersize=6,
    )
    ax1.set_ylabel("Average inference time (s)")
    ax1.set_xlabel("Sequence Length")
    ax1.set_title("Timing Comparison: Native vs Flash Attention-2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory usage comparison
    ax2.plot(
        seq_lengths,
        native_memories,
        "b-o",
        label=f"{model_id}-native",
        linewidth=2,
        markersize=6,
    )
    ax2.plot(
        seq_lengths,
        fa2_memories,
        "orange",
        marker="o",
        label=f"{model_id}-FA2",
        linewidth=2,
        markersize=6,
    )
    ax2.set_ylabel("Peak Memory Usage (MB)")
    ax2.set_xlabel("Sequence Length")
    ax2.set_title("Memory Usage Comparison: Native vs Flash Attention-2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(dir_name, "timing_and_memory_plot.jpg")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to: {plot_path}")

    # Calculate total script time
    total_script_time = time.time() - script_start_time

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Create and display results table
    df = pd.DataFrame(results_data)
    table = tabulate(df, headers="keys", tablefmt="grid", showindex=False)
    print(table)

    print("\nOverall Statistics:")
    print(f"  Total script runtime: {total_script_time:.2f} seconds")
    print(f"  Model loading time: {model_load_time:.2f} seconds")
    print(
        f"  Average speedup: {sum(forward_speedups.values()) / len(forward_speedups):.2f}x"
    )
    print(
        f"  Max speedup: {max(forward_speedups.values()):.2f}x (at seq_len {max(forward_speedups, key=forward_speedups.get)})"
    )
    print(
        f"  Min speedup: {min(forward_speedups.values()):.2f}x (at seq_len {min(forward_speedups, key=forward_speedups.get)})"
    )

    # Memory statistics
    avg_memory_saving = sum(memory_savings.values()) / len(memory_savings)
    max_memory_saving = max(memory_savings.values())
    min_memory_saving = min(memory_savings.values())
    max_mem_key = max(memory_savings, key=memory_savings.get)
    min_mem_key = min(memory_savings, key=memory_savings.get)

    print("\nMemory Usage Statistics:")
    print(f"  Average memory savings: {avg_memory_saving:.1f}%")
    print(f"  Max memory savings: {max_memory_saving:.1f}% (at seq_len {max_mem_key})")
    print(f"  Min memory savings: {min_memory_saving:.1f}% (at seq_len {min_mem_key})")

    # Show peak memory usage
    max_native_mem = max(native_memory_dict.values())
    max_fa2_mem = max(fa2_memory_dict.values())
    print(f"  Peak native memory usage: {max_native_mem:.1f} MB")
    print(f"  Peak FA2 memory usage: {max_fa2_mem:.1f} MB")

    # Save results to CSV
    csv_path = os.path.join(dir_name, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("=" * 80)
