from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path
from tomllib import TOMLDecodeError
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from modeling.config import (
    ExperimentConfig,
)
from modeling.config.serde import (
    build_experiment_config,
    serialize_experiment_config,
)
from modeling.experiments.text_pretrain.default import TextPretrainExperimentConfig
from modeling.modules.text_pretrain.default import TextPretrainLIT


# from modeling.utils.
@pytest.fixture
def temp_toml_file():
    """Create a temporary TOML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def basic_experiment_setup(temp_toml_file: Path):
    """Create a basic experiment TOML file."""

    mock_experiment_config = TextPretrainExperimentConfig
    serialize_experiment_config(
        mock_experiment_config,
        output_path=temp_toml_file,
    )

    return (temp_toml_file, mock_experiment_config)


@pytest.fixture
def invalid_toml_file(temp_toml_file: Path):
    """Create an invalid TOML file for testing error cases."""
    toml_content = textwrap.dedent("""
    [metadata]
    output_dir = "/tmp/experiments/test_run"
    # Missing wandb config - should cause validation error
    
    [distributed]
    
    [module_config]
    config_path = "nonexistent.module.Config"
    """)

    temp_toml_file.write_text(toml_content)
    return temp_toml_file


def assert_config_equality(config1: BaseModel, config2: BaseModel, ignore_fields=None):
    """
    Assert that two configurations are equal, ignoring specified fields.

    Args:
        config1: First configuration object
        config2: Second configuration object
        ignore_fields: List of field names to ignore in comparison
    """
    ignore_fields = ignore_fields or []

    dict1 = config1.model_dump()
    dict2 = config2.model_dump()

    for field in ignore_fields:
        dict1.pop(field, None)
        dict2.pop(field, None)

    assert dict1 == dict2


class TestBuildExperimentConfig:
    """Test cases for build_experiment_config function."""

    def test_build_experiment_config_basic(self, basic_experiment_setup):
        """Test basic functionality of build_experiment_config."""
        basic_experiment_toml, basic_experiment_config = basic_experiment_setup

        config = build_experiment_config(basic_experiment_toml)
        assert isinstance(config, ExperimentConfig)
        assert_config_equality(
            config,
            basic_experiment_config,
        )

        module = config.module_config.create_module()
        assert isinstance(module, TextPretrainLIT)

    def test_build_experiment_config_file_not_found(self):
        """Test error handling when TOML file doesn't exist."""
        non_existent_path = Path("/nonexistent/path/config.toml")

        with pytest.raises(FileNotFoundError):
            build_experiment_config(non_existent_path)

    def test_build_experiment_config_invalid_toml(self, temp_toml_file):
        """Test error handling with malformed TOML."""
        # Write invalid TOML syntax
        temp_toml_file.write_text("invalid toml content [[[")

        with pytest.raises(TOMLDecodeError):  # Should raise tomllib parsing error
            build_experiment_config(temp_toml_file)

    def test_build_experiment_config_missing_required_fields(self, invalid_toml_file):
        """Test validation error with missing required fields."""
        with pytest.raises(ValidationError):  # Should raise pydantic validation error
            build_experiment_config(invalid_toml_file)

    def test_build_experiment_config_import_error(self, basic_experiment_setup):
        """Test error handling when module import fails."""
        basic_experiment_toml, _ = basic_experiment_setup
        with patch("modeling.utils.dynamic_import.import_from_string") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            with pytest.raises(ImportError):
                build_experiment_config(basic_experiment_toml)

    def test_build_experiment_config_invalid_module_class(self, basic_experiment_setup):
        """Test error handling when imported class is not a ModuleConfig subclass."""
        basic_experiment_toml, _ = basic_experiment_setup
        with patch("modeling.utils.dynamic_import.import_from_string") as mock_import:
            # Return a class that doesn't inherit from ModuleConfig
            mock_import.return_value = str
            with pytest.raises(AssertionError):
                build_experiment_config(basic_experiment_toml)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
