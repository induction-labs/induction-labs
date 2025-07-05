from __future__ import annotations

from typing import Any, cast

from modeling.checkpoints.save import Path
from modeling.config import DatapackConfig, RunConfig
from modeling.data.video_action import ActionDataSample, ActionDatapackConfig
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig
from .qwen_25o_actions import (
    Qwen2_5OmniThinkerForActionModelling,
    Qwen2_5OmniThinkerActionConfig,
)
from synapse.actions.mouse_movements import (
    Cubic,
    cubics_to_points,
    generate_image_from_segments,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
import wandb

from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
import torch

MODEL_TYPE = Qwen2_5OmniThinkerForActionModelling


class Qwen25OActionLIT(
    BaseLITModule[MODEL_TYPE, ActionDataSample, "Qwen25OActionLITConfig"]
):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    def init_model_meta(
        self,
    ):
        module_config = self.module_config
        config = Qwen2_5OmniThinkerActionConfig.from_pretrained(
            module_config.model_name,
            freeze_network=module_config.freeze_network,
            freeze_vision=module_config.freeze_vision,
            freeze_action_head=module_config.freeze_action_head,
            freeze_action_embedding=module_config.freeze_action_embedding,
        )

        model = MODEL_TYPE.from_pretrained(
            module_config.model_name,
            config=config,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).train()
        assert isinstance(model, MODEL_TYPE), (
            f"Expected model to be of type Qwen2_5OmniThinkerForActionModelling, "
            f"got {type(model)}"
        )
        return model

    def load_weights(self, tmpdir: Path) -> Qwen2_5OmniThinkerForActionModelling:
        """
        Load the model weights from the specified checkpoint directory.
        This method should handle the loading of pre-trained weights or checkpoint files.
        """
        # logger.debug(f"Loading model weights from {tmpdir}")
        # Load the model weights and dispatch them to the appropriate devices
        self.model.to_empty(device=torch.cuda.current_device())
        loaded_model = MODEL_TYPE.from_pretrained(
            self.module_config.model_name,
            config=self.model.config,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).train()
        return cast(MODEL_TYPE, loaded_model)

    def run_training_step(self, inputs: ActionDataSample):
        # Forward pass through the model
        outputs = self.model(
            cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )

        return outputs.loss

    def run_validation_step(self, inputs: ActionDataSample):
        # Forward pass through the model
        outputs = self.model(
            cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )

        # print(outputs.logits)
        # print(outputs.logits.shape)
        # print("---------")
        # print(inputs.cursor_path[inputs.action_tokens])
        # print(inputs.cursor_path[inputs.action_tokens].shape)
        # print("---------")
        # print(outputs.logits[inputs.action_tokens])
        # print(outputs.logits[inputs.action_tokens].shape)

        predicted_image = self.visualize_action(outputs.logits[inputs.action_tokens])
        real_image = self.visualize_action(
            inputs.cursor_path[inputs.action_tokens].reshape(-1, 6)
        )

        wandb.log(
            {
                "validation/real_image": [wandb.Image(real_image)],
                "validation/predicted_image": [wandb.Image(predicted_image)],
            }
        )

        return outputs.loss, {}

    def visualize_action(
        self,
        actions: torch.Tensor,  # [n, 6]
        resolution: tuple[int, int] = (854, 480),
    ):
        predicted_cubics_x = []
        predicted_cubics_y = []
        print(actions.shape)
        for value in actions:
            value = value.to(dtype=torch.float32).cpu().numpy()
            predicted_cubics_x.append(Cubic(m=value[0], n=value[1], a=value[2]))
            predicted_cubics_y.append(Cubic(m=value[3], n=value[4], a=value[5]))

        SCREEN_SIZE = (854, 480)
        ts, all_poly_x, all_poly_y = cubics_to_points(
            0.5, 0.5, predicted_cubics_x, predicted_cubics_y
        )
        base_image = generate_image_from_segments(
            ts, all_poly_x, all_poly_y, SCREEN_SIZE
        )

        return base_image

    def shard_model(
        self,
        *,
        mp_policy: MixedPrecisionPolicy,
        device_mesh: DeviceMesh,
    ):
        # if self.model.device.type != "meta":
        #     return  # already configured

        dp_mesh = device_mesh["data_parallel"]  # provided by ModelParallelStrategy
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        fully_shard(self.model.visual, **fsdp_config)

        for layer_id, transformer_block in enumerate(self.model.model.layers):
            # Apply activation checkpointing

            # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928
            #             from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            #     checkpoint_wrapper,
            # )
            # transformer_block = checkpoint_wrapper(transformer_block)

            reshard_after_forward = int(layer_id) < len(self.model.model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            self.model.model.layers[layer_id] = transformer_block
        return fully_shard(self.model, **fsdp_config)


class Qwen25OActionLITConfig(BaseModuleConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = (
        "modeling.modules.action_instruct.qwen_25o.Qwen25OActionLITConfig"
    )
    model_name: str = "Qwen/Qwen2.5-Omni-3B"
    tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"

    freeze_network: bool = True
    freeze_vision: bool = True
    freeze_action_head: bool = False
    freeze_action_embedding: bool = False

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> ActionDatapackConfig:
        assert isinstance(datapack_config, ActionDatapackConfig), (
            f"Expected {datapack_config=} to be of type ActionDatapackConfig"
        )
        return datapack_config

    def create_module(self, run_config: RunConfig, tmp_dir: Path) -> Qwen25OActionLIT:
        return Qwen25OActionLIT(self, run_config, tmp_dir)
