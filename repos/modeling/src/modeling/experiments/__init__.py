from __future__ import annotations

import datetime
from pathlib import Path

from modeling.config import (
    ExperimentConfig,
    serialize_experiment_config,
)


def transform_module_path(module_path: str, file_extension=".toml"):
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

    # Build new path: everything before experiments + exp_configs + everything after experiments
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


def transform_module_path_with_write(module_path, content, file_extension=".toml"):
    """
    Transform module path and write content to the resulting file.
    Creates all necessary parent directories.
    """

    # Get the transformed path
    path = transform_module_path(module_path, file_extension)

    # Create all parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write content to file
    path.write_text(content)

    return path


# Example usage:
if __name__ == "__main__":
    # Test the transformation
    module_path = "modeling.experiments.text_pretrain.default"
    result = transform_module_path(module_path)
    print(f"Module: {module_path}")
    print(f"Transformed: {result}")

    # Test with writing
    config_content = """
[model]
name = "text_pretrain"
type = "transformer"

[training]
batch_size = 32
learning_rate = 0.001
"""

    created_file = transform_module_path_with_write(module_path, config_content.strip())
    print(f"Created file: {created_file}")

    # Test other examples
    test_cases = [
        "modeling.experiments.vision.resnet",
        "some.other.modeling.experiments.nlp.bert",
        "modeling.experiments.audio.wav2vec",
    ]

    for test_case in test_cases:
        try:
            result = transform_module_path(test_case)
            print(f"{test_case} -> {result}")
        except ValueError as e:
            print(f"Error with {test_case}: {e}")


def transform_experiment_path(path_str):
    """
    Transform an experiment path to a config path with timestamp.

    Args:
        path_str: String path like '.../modeling/src/modeling/experiments/text_pretrain/default.py'

    Returns:
        Transformed path like '.../modeling/src/modeling/exp_configs/text_pretrain/default/{timestamp}.toml'

    Raises:
        ValueError: If path is not in the expected experiments folder structure
    """
    path = Path(path_str)

    # Find if this pattern exists anywhere in the path
    path_parts = path.parts
    experiments_index = None

    for i in range(len(path_parts) - 3):  # Need at least 4 parts for the pattern
        if (
            path_parts[i] == "modeling"
            and path_parts[i + 1] == "src"
            and path_parts[i + 2] == "modeling"
            and path_parts[i + 3] == "experiments"
        ):
            experiments_index = i + 3
            break

    if experiments_index is None:
        raise ValueError(
            f"Path '{path_str}' is not in modeling/src/modeling/experiments folder"
        )

    # Get the parts before and after experiments
    prefix_parts = path_parts[:experiments_index]
    suffix_parts = path_parts[experiments_index + 1 :]  # Skip "experiments"

    # Replace experiments with exp_configs
    new_parts = [*list(prefix_parts), "exp_configs", *list(suffix_parts)]

    # Remove .py extension from the last part and create new path
    if new_parts[-1].endswith(".py"):
        new_parts[-1] = new_parts[-1][:-3]  # Remove .py

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build final path: original_path_without_py/{timestamp}.toml
    final_path = Path(*new_parts) / f"{timestamp}.toml"

    return final_path


# Example usage:
if __name__ == "__main__":
    test_path = "../vllm-project/repos/modeling/src/modeling/experiments/text_pretrain/default.py"
    result = transform_experiment_path(test_path)
    print(f"Original: {test_path}")
    print(f"Transformed: {result}")


def export_experiment(experiment_config: ExperimentConfig, file_path: str):
    output_path = transform_experiment_path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_experiment_config(experiment_config, output_path)
    # Big Print to command Line
    print("##################################")
    print(f"Experiment configuration exported to: {output_path}")
    print("##################################")
