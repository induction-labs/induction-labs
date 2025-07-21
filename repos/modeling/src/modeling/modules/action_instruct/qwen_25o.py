from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import torch
import wandb
from synapse.actions.mouse_movements import (
    Cubic,
    cubics_to_points,
    generate_image_from_segments,
)
from synapse.utils.logging import configure_logging, logging
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from modeling.config import (
    DatapackConfig,
    InstanceConfig,
    RunConfig,
)
from modeling.config.distributed import MeshAxis
from modeling.data.video_action import ActionDatapackConfig, ActionDataSample
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig
from modeling.utils.check_nans import check_nans
from modeling.utils.class_property import class_property

from .qwen_25o_actions import (
    Qwen2_5OmniActionCausalLMOutputWithPast,
    Qwen2_5OmniActionModel,
    Qwen2_5OmniThinkerActionConfig,
)

logger = configure_logging(__name__, level=logging.DEBUG)

MODEL_TYPE = Qwen2_5OmniActionModel

T = TypeVar("T")


def not_none(value: T | None) -> T:
    """
    Ensure that the value is not None.
    Raises an AssertionError if the value is None.
    """
    assert value is not None, "Value cannot be None"
    return value


def analytical_distance(
    a: torch.Tensor,  # [seq, 3]
    b: torch.Tensor,  # [seq, 3]
) -> torch.Tensor:
    m1, n1, a1 = a.unbind(dim=1)  # shape: [seq, 3]
    m2, n2, a2 = b.unbind(dim=1)  # shape:
    c1 = torch.stack(
        [
            m1 + n1 - 2 * a1,
            3 * a1 - 2 * m1 - n1,
            m1,
        ],
        dim=1,
    )  # shape: [seq, 3]
    c2 = torch.stack(
        [
            m2 + n2 - 2 * a2,
            3 * a2 - 2 * m2 - n2,
            m2,
        ],
        dim=1,
    )  # shape: [seq, 3]

    d = c1 - c2
    d1, d2, d3 = d[..., 0], d[..., 1], d[..., 2]
    dist_sq = (
        d1**2 * (1 / 3)
        + d1 * d2 * (2 / 4)
        + d1 * d3 * (2 / 5)
        + d2**2 * (1 / 5)
        + d2 * d3 * (2 / 6)
        + d3**2 * (1 / 7)
    )
    # https://chatgpt.com/share/68733eb1-a248-8006-a5a3-9494aa3ed24a

    return dist_sq


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
        config = Qwen2_5OmniThinkerActionConfig.from_pretrained(
            module_config.model_name,
            freeze_network=module_config.freeze_network,
            freeze_vision=module_config.freeze_vision,
            freeze_action_head=module_config.freeze_action_head,
            freeze_action_embedding=module_config.freeze_action_embedding,
        )

        model = MODEL_TYPE(config)
        assert isinstance(model, MODEL_TYPE), (
            f"Expected model to be of type Qwen2_5OmniThinkerForActionModelling, "
            f"got {type(model)}"
        )
        return model

    def run_training_step(self, inputs: ActionDataSample):
        # Forward pass through the model
        outputs = self.model(
            # cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )
        check_nans(outputs.action_outputs, "outputs")
        return self.compute_loss(outputs, inputs)[0], {}

    def compute_loss(
        self, outputs: Qwen2_5OmniActionCausalLMOutputWithPast, inputs: ActionDataSample
    ):
        """
        Compute the loss for the training step.
        This method is called by run_training_step after the forward pass.
        """
        output_actions = (
            not_none(outputs.action_outputs)[inputs.action_tokens]
            .reshape(-1, 2, 3)
            .to(device=self.device, dtype=self.dtype)
        )
        cursor_path = (
            inputs.cursor_path[inputs.action_tokens]
            .reshape(-1, 2, 3)
            .to(device=self.device, dtype=self.dtype)
        )

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
        loss = analytical_distance(
            a=output_actions[:, 0, :],
            b=cursor_path[:, 0, :],
        ) + analytical_distance(
            a=output_actions[:, 1, :],
            b=cursor_path[:, 1, :],
        )  # [k, num_points]

        loss = loss.sum()
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

    @classmethod
    def validation_wandb_metrics(
        cls, all_metrics: list[dict[str, Any]], global_step: int
    ) -> dict[str, Any]:
        first_metrics = all_metrics[0]
        cursor_path = first_metrics.get("val/cursor_path", [])
        action_outputs = first_metrics.get("val/action_outputs", [])

        np_predicted_xs, np_predicted_ys = cls.visualize_action(action_outputs)
        np_actual_xs, np_actual_ys = cls.visualize_action(cursor_path)

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
                    strict=False,
                )
            ],
        )

        metrics = {
            f"val/cubics/{global_step}": table,
            "val/real_image": [wandb.Image(real_image)],
            "val/predicted_image": [wandb.Image(predicted_image)],
            "val/loss": first_metrics.get("val/loss", 0.0),
        }
        return metrics

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
        action_outputs = outputs.action_outputs[inputs.action_tokens]
        cursor_path = cursor_path.reshape(-1, 6)
        return loss, {
            "val/loss": loss,
            "val/predicted_xs": predicted_xs,
            "val/predicted_ys": predicted_ys,
            "val/actual_xs": actual_xs,
            "val/actual_ys": actual_ys,
            "val/output_actions": output_actions,
            "val/cursor_path": cursor_path,
            "val/action_tokens": inputs.action_tokens,
            "val/action_outputs": action_outputs,
        }

    @classmethod
    def visualize_action(
        cls,
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
        fully_shard(self.model.thinker.visual, **fsdp_config)

        for layer_id, transformer_block in enumerate(self.model.thinker.model.layers):
            # Activation checkpointing kinda broken
            # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928

            reshard_after_forward = (
                int(layer_id) < len(self.model.thinker.model.layers) - 1
            )
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            self.model.thinker.model.layers[layer_id] = transformer_block

        return fully_shard(
            self.model,
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
        instance_config: InstanceConfig,
    ) -> Qwen25OActionLIT:
        return Qwen25OActionLIT(self, run_config, instance_config)

    @classmethod
    def module_cls(cls) -> type[Qwen25OActionLIT]:
        return Qwen25OActionLIT
