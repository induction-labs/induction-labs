import torch
import typer
from torch.utils.data import DataLoader, Dataset


class ReproDataset(Dataset[int]):
    def __init__(self, length: int) -> None:
        """
        Initialize the dataset with a specified length.
        """
        self.length = length

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> int:
        """
        Return the item at the specified index.
        """
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds")
        return index


def get_dataloader_indices_for_step(
    seed: int, batch_size: int, dataset_length: int, step: int
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
    gen_copy.manual_seed(seed)

    # Generate the full permutation for the epoch
    dataset = ReproDataset(dataset_length)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=gen_copy,
        # TODO: Figure out optimal prefetch factor + num_workers
        # prefetch_factor=full_config.run.dataloader_prefetch_factor,
        # Need num_workers!=0 so that this runs in MP mode, so that
        # num_workers=full_config.run.dataloader_num_workers,
        # persistent_workers=True,
        collate_fn=torch.utils.data.default_collate,
    )

    # Now just iterator through the DataLoader to get the indices for the specified step
    for current_step, batch in enumerate(data_loader):
        if current_step == step:
            return batch.tolist()


def main(
    seed: int = typer.Argument(..., help="Random seed for the generator"),
    batch_size: int = typer.Argument(..., help="Batch size"),
    dataset_length: int = typer.Argument(..., help="Total dataset length"),
    target_step: int = typer.Argument(..., help="Target step number (0-indexed)"),
):
    """
    Get the indices that would be used by a DataLoader at a specific step.
    """

    # Get indices for the target step
    indices = get_dataloader_indices_for_step(
        seed, batch_size, dataset_length, target_step
    )

    # Output the indices
    print(f"Step {target_step} indices: {indices}")
    return indices


if __name__ == "__main__":
    typer.run(main)


# seed 93208839
# batch size 16
# dataset length 27160
# target step 0
# uv run python src/modeling/utils/repro_dataloader.py 93208839 16 27160 0

# Step 47 indices: [13465, 5268, 24642, 6433, 363, 5182, 10119, 1745, 7241, 3610, 16782, 2672, 19830, 22479, 6338, 25899]


# seed 93208
# batch size 16
# dataset length 13580
# target step 370
# uv run python src/modeling/utils/repro_dataloader.py 93208 16 13580 370
