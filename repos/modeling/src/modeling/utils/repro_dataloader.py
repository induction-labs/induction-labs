import torch
import typer


def get_dataloader_indices_for_step(
    generator: torch.Generator, batch_size: int, dataset_length: int, step: int
) -> list[int]:
    """
    Get the indices that would be used by a DataLoader at a specific step.

    Args:
        generator: PyTorch generator with the desired seed
        batch_size: Size of each batch
        dataset_length: Total length of the dataset
        step: The step number (0-indexed)

    Returns:
        List of indices that would be used for the given step
    """
    # Create a copy of the generator to avoid modifying the original
    gen_copy = torch.Generator()
    gen_copy.set_state(generator.get_state())

    # Generate the full permutation for the epoch
    indices = torch.randperm(dataset_length, generator=gen_copy).tolist()

    # Calculate the start and end indices for the requested step
    start_idx = step * batch_size
    end_idx = min(start_idx + batch_size, dataset_length)

    # Return the indices for this step
    return indices[start_idx:end_idx]


def simulate_dataloader_steps(
    generator: torch.Generator,
    batch_size: int,
    dataset_length: int,
    num_steps: int | None = None,
) -> list[list[int]]:
    """
    Simulate multiple steps of a DataLoader and return all batch indices.

    Args:
        generator: PyTorch generator with the desired seed
        batch_size: Size of each batch
        dataset_length: Total length of the dataset
        num_steps: Number of steps to simulate (if None, simulates full epoch)

    Returns:
        List of batch indices for each step
    """
    if num_steps is None:
        num_steps = (dataset_length + batch_size - 1) // batch_size  # Ceiling division

    # Create a copy of the generator to avoid modifying the original
    gen_copy = torch.Generator()
    gen_copy.set_state(generator.get_state())

    # Generate the full permutation for the epoch
    indices = torch.randperm(dataset_length, generator=gen_copy).tolist()

    # Split into batches
    batches = []
    for step in range(num_steps):
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, dataset_length)
        if start_idx < dataset_length:
            batches.append(indices[start_idx:end_idx])
        else:
            break

    return batches


def main(
    seed: int = typer.Argument(..., help="Random seed for the generator"),
    batch_size: int = typer.Argument(..., help="Batch size"),
    dataset_length: int = typer.Argument(..., help="Total dataset length"),
    target_step: int = typer.Argument(..., help="Target step number (0-indexed)"),
):
    """
    Get the indices that would be used by a DataLoader at a specific step.
    """
    # Create generator with the specified seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Get indices for the target step
    indices = get_dataloader_indices_for_step(
        generator, batch_size, dataset_length, target_step
    )

    # Output the indices
    print(f"Step {target_step} indices: {indices}")
    return indices


if __name__ == "__main__":
    typer.run(main)
