from __future__ import annotations

from typing import Any, cast

from modeling.checkpoints.save import Path
from modeling.config import (
    DatapackConfig,
    RunConfig,
    RuntimeConfig,
    InstanceConfig,
)
from modeling.config.distributed import MeshAxis
from modeling.data.video_action import ActionDataSample, ActionDatapackConfig
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig
from modeling.utils.class_property import class_property
from .qwen_25o_actions import (
    Qwen2_5OmniThinkerForActionModelling,
    Qwen2_5OmniThinkerActionConfig,
    Qwen2_5OmniActionCausalLMOutputWithPast,
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
from synapse.utils.logging import configure_logging, logging
import numpy as np
import time

from torch import nn

logger = configure_logging(__name__, level=logging.DEBUG)

MODEL_TYPE = Qwen2_5OmniThinkerForActionModelling


def __call__(self, x: float):
    """Evaluate the cubic at x (scalar or array)."""
    c3, c2, c1 = self.coeffs()
    return ((c3 * x + c2) * x + c1) * x  # Horner form


def cubics_to_points_torch(
    coeffs: torch.Tensor,  # [seq, 3]
    num_points: int = 100,
) -> torch.Tensor:
    x = torch.linspace(0, 1, num_points, device=coeffs.device).unsqueeze(
        0
    )  # [1, num_points]
    m, n, a = coeffs.unbind(dim=1)  # shape: [seq, 3]
    c3 = m + n - 2 * a
    c2 = 3 * a - 2 * m - n
    c1 = m

    c3 = c3.unsqueeze(1)  # shape: [seq, 1]
    c2 = c2.unsqueeze(1)  # shape: [seq, 1]
    c1 = c1.unsqueeze(1)  # shape: [seq, 1]
    y = ((c3 * x + c2) * x + c1) * x  # [seq, num_points]
    return y


SCREEN_SIZE = (854, 480)


def to_numpy_clean(tensor: torch.Tensor, dtype=torch.float32) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array, ensuring it is on CPU and detached.
    """
    return tensor.to(dtype=dtype, device="cpu").detach().numpy()


class Qwen25OActionLIT(
    BaseLITModule[MODEL_TYPE, ActionDataSample, "Qwen25OActionLITConfig"]
):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    model: MODEL_TYPE

    @class_property
    def model_cls(cls) -> type[MODEL_TYPE]:
        return MODEL_TYPE

    def init_model_meta(
        self,
    ):
        module_config = self.module_config
        delay = self.instance_config.device_rank * 0.5
        time.sleep(delay)
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
            # local_files_only=True,
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
        logger.debug(f"Loading model weights from {tmpdir} to device {self.device}")
        self.model.to_empty(device=self.device)
        # This is so stupid - but if you try to load the model on multiple devices fucking huggingface throws an error
        # Huggingface for sure is real company :/
        delay = self.instance_config.device_rank * 0.5
        time.sleep(delay)
        loaded_model = MODEL_TYPE.from_pretrained(
            self.module_config.model_name,
            config=self.model.config,
            torch_dtype=self.dtype,
            device_map={
                "": self.device  # Use the device index for the model
            },  # Ensure model is loaded on the correct device
            attn_implementation=self.attn_impl,
            local_files_only=True,
        ).train()

        assert isinstance(loaded_model, MODEL_TYPE), (
            f"Expected loaded_model to be of type Qwen2_5OmniThinkerForActionModelling, "
            f"got {type(loaded_model)}"
        )

        return cast(MODEL_TYPE, loaded_model)

    def run_training_step(self, inputs: ActionDataSample):
        # Forward pass through the model
        outputs = self.model(
            # cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )
        return self.compute_loss(outputs, inputs)[0]

    def compute_loss(
        self, outputs: Qwen2_5OmniActionCausalLMOutputWithPast, inputs: ActionDataSample
    ):
        """
        Compute the loss for the training step.
        This method is called by run_training_step after the forward pass.
        """
        output_actions = (
            outputs.action_outputs[inputs.action_tokens]
            .reshape(-1, 2, 3)
            .to(device=self.device, dtype=self.dtype)
        )
        cursor_path = (
            inputs.cursor_path[inputs.action_tokens]
            .reshape(-1, 2, 3)
            .to(device=self.device, dtype=self.dtype)
        )

        # l2_loss = self.model.l2_loss(output_actions, cursor_path)
        # loss = l2_loss(
        #     output_actions,
        # )  # [B, S, 6]
        # loss = torch.sum(loss, dim=-1)
        # loss *= action_tokens
        # loss = loss.sum() / action_tokens.sum().clamp(min=1.0)

        # print(f"{self.device=}")

        assert output_actions.shape == cursor_path.shape, (
            f"Expected output_actions shape {output_actions.shape} to match "
            f"cursor_path shape {cursor_path.shape}"
        )
        actual_xs, actual_ys = (
            cubics_to_points_torch(coeffs=cursor_path[:, 0, :]),
            cubics_to_points_torch(coeffs=cursor_path[:, 1, :]),
        )  # [k, num_points]

        predicted_xs, predicted_ys = (
            cubics_to_points_torch(coeffs=output_actions[:, 0, :]),
            cubics_to_points_torch(coeffs=output_actions[:, 1, :]),
        )

        # if inputs.cursor_path is not None:
        loss = self.model.l2_loss(actual_xs, predicted_xs) + self.model.l2_loss(
            actual_ys, predicted_ys
        )
        loss = loss.sum() / inputs.action_tokens.sum().clamp(min=1.0)
        assert isinstance(loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )

        return (
            loss,
            predicted_xs,
            predicted_ys,
            actual_xs,
            actual_ys,
            output_actions,
            cursor_path,
        )

    def run_validation_step(self, inputs: ActionDataSample, global_step: int):
        # Forward pass through the model
        outputs = self.model(
            cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )
        (
            loss,
            predicted_xs,
            predicted_ys,
            actual_xs,
            actual_ys,
            output_actions,
            cursor_path,
        ) = self.compute_loss(outputs, inputs)

        computed_predicted_image = generate_image_from_segments(
            to_numpy_clean(predicted_xs[0]) + 0.5,
            to_numpy_clean(predicted_ys[0]) + 0.5,
            SCREEN_SIZE,
        )

        computed_real_image = generate_image_from_segments(
            to_numpy_clean(actual_xs[0]) + 0.5,
            to_numpy_clean(actual_ys[0]) + 0.5,
            SCREEN_SIZE,
        )
        action_outputs = outputs.action_outputs[inputs.action_tokens]
        cursor_path = inputs.cursor_path[inputs.action_tokens].reshape(-1, 6)

        np_predicted_xs, np_predicted_ys = self.visualize_action(action_outputs)
        np_actual_xs, np_actual_ys = self.visualize_action(cursor_path)

        real_image = generate_image_from_segments(
            np_actual_xs, np_actual_ys, SCREEN_SIZE
        )
        predicted_image = generate_image_from_segments(
            np_predicted_xs, np_predicted_ys, SCREEN_SIZE
        )
        table = wandb.Table(
            columns=["actual", "predicted"],
            data=[
                [str(actual), str(predicted)]
                for actual, predicted in zip(
                    to_numpy_clean(cursor_path),
                    to_numpy_clean(action_outputs),
                )
            ],
        )

        metrics = {
            f"validation/cubics/{global_step}": table,
            "validation/real_image": [wandb.Image(real_image)],
            "validation/predicted_image": [wandb.Image(predicted_image)],
            "validation/computed_predicted_image": [
                wandb.Image(computed_predicted_image)
            ],
            "validation/computed_real_image": [wandb.Image(computed_real_image)],
        }
        return loss, metrics

    def visualize_action(
        self,
        actions: torch.Tensor,  # [n, 6]
    ):
        predicted_cubics_x = []
        predicted_cubics_y = []
        for value in actions:
            value = value.to(dtype=torch.float32).cpu().numpy()
            predicted_cubics_x.append(Cubic(m=value[0], n=value[1], a=value[2]))
            predicted_cubics_y.append(Cubic(m=value[3], n=value[4], a=value[5]))

        ts, all_poly_x, all_poly_y = cubics_to_points(
            0.5, 0.5, predicted_cubics_x, predicted_cubics_y
        )
        return all_poly_x, all_poly_y
        base_image = generate_image_from_segments(all_poly_x, all_poly_y, SCREEN_SIZE)

        return base_image

    def shard_model(
        self,
        *,
        mp_policy: MixedPrecisionPolicy,
        device_mesh: DeviceMesh,
    ):
        """
        Shard the model using Fully Sharded Data Parallel (FSDP).
        This method is called during the model configuration phase.
        """

        fsdp_config = {
            "mesh": device_mesh[MeshAxis.FSDP],
            "mp_policy": mp_policy,
        }
        fully_shard(self.model.visual, **fsdp_config)

        for layer_id, transformer_block in enumerate(self.model.model.layers):
            # Activation checkpointing kinda broken
            # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928

            reshard_after_forward = int(layer_id) < len(self.model.model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            self.model.model.layers[layer_id] = transformer_block

        ignored_params = cast(
            set[nn.Parameter], set(self.model.action_token_embedding.weight)
        )
        # for param in self.model.action_head.parameters():
        #     ignored_params.add(param)
        return fully_shard(
            self.model,
            ignored_params=ignored_params,
            **fsdp_config,
        )


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

    freeze_network: bool = False
    freeze_vision: bool = False
    freeze_action_head: bool = False
    freeze_action_embedding: bool = False

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> ActionDatapackConfig:
        assert isinstance(datapack_config, ActionDatapackConfig), (
            f"Expected {datapack_config=} to be of type ActionDatapackConfig"
        )
        return datapack_config

    def create_module(
        self,
        run_config: RunConfig,
        runtime_config: RuntimeConfig,
        instance_config: InstanceConfig,
    ) -> Qwen25OActionLIT:
        return Qwen25OActionLIT(self, run_config, runtime_config, instance_config)
