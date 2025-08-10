from __future__ import annotations

import datetime
from pathlib import Path


def exp_module_path(module_path: str, file_extension=".toml"):
    """
    Transform a module path to a file path with timestamp.

    Args:
        module_path: String like 'modeling.experiments.text_pretrain.default'
        file_extension: File extension for the output file (default: '.toml')

    Returns:
        Transformed path like 'modeling/exp_configs/text_pretrain/default/{timestamp}.toml'

    Raises:
        ValueError: If module path doesn't contain 'experiments' after 'modeling'
    """
    # Split the module path into parts
    parts = module_path.split(".")

    # Find the pattern: modeling.experiments

    assert parts[0] == "modeling" and parts[1] == "experiments", (
        f"Module path '{module_path}' should start with 'modeling.experiments'"
    )

    # Build new path:
    # everything before experiments + exp_configs + everything after experiments
    suffix_parts = parts[2:]  # everything after 'experiments'

    # Replace 'experiments' with 'exp_configs'
    new_parts = ["exp_configs", *suffix_parts]

    # Convert to file path (dots become slashes)
    base_path = Path("/".join(new_parts))

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Final path: base_path/{timestamp}.toml
    final_path = base_path / f"{timestamp}{file_extension}"

    return final_path
