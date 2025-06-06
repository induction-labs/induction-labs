from __future__ import annotations

from modeling.config import DistributedConfig, ExperimentConfig, ExperimentMetadata
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_pretrain import TextPretrainLITConfig

TextPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata.mock_data(),
    distributed=DistributedConfig.mock_data(),
    module_config=TextPretrainLITConfig(),
    datapack_config=TextPretrainDatapackConfig(),
    sequence_length=1024,  # Default sequence length
    batch_size=32,  # Default batch size
)

# mdl export modeling.experiments.text_pretrain.default.TextPretrainExperimentConfig
