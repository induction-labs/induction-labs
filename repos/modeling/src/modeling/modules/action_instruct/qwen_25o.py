from __future__ import annotations

from enum import Enum
from gc import freeze
from typing import Any, TypeVar
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from synapse.actions.mouse_movements import (
    Cubic,
    cubics_to_points,
    generate_image_from_segments,
)
from transformers.loss.loss_utils import ForCausalLMLoss
from synapse.actions.keyboard_tokenizer import Tokenizer
from synapse.utils.logging import configure_logging, logging
from torch import nn
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
l2_loss = nn.MSELoss(reduce=False)

T = TypeVar("T")

TOKENIZER = Tokenizer.load("gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json")


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


@dataclass
class CursorLossOutput:
    predicted_xs: torch.Tensor | None
    predicted_ys: torch.Tensor | None
    actual_xs: torch.Tensor | None
    actual_ys: torch.Tensor | None
    output_actions: torch.Tensor | None
    cursor_path: torch.Tensor | None

    analytical_loss: torch.Tensor | None
    l2_points_loss: torch.Tensor | None
    coefficients_loss: torch.Tensor | None


@dataclass
class LossOutput:
    loss: torch.Tensor
    cursor_aux: ActionLossOutput | None = None


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
            freeze_keyboard_embedding=module_config.freeze_keyboard_embedding,
            freeze_keyboard_head=module_config.freeze_keyboard_head,
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
            cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            keyboard_token_mask=inputs.keyboard_tokens_mask,
            **inputs.qwen_inputs.model_dump(),
        )
        if inputs.action_tokens is not None:
            check_nans(outputs.action_outputs, "outputs")
        # check_nans(outputs.action_outputs, "outputs")

        outputs = self.compute_loss(outputs, inputs)

        aux_metrics = {}

        if outputs.cursor_aux is not None:
            cursor_aux = outputs.cursor_aux
            aux_metrics.update(
                {
                    "train/analytical_loss": cursor_aux.analytical_loss,
                    "train/l2_points_loss": cursor_aux.l2_points_loss,
                    "train/coefficients_loss": cursor_aux.coefficients_loss,
                }
            )

        return outputs.loss, {
            "train/loss": outputs.loss,
            **aux_metrics,
        }

    def compute_loss(
        self, outputs: Qwen2_5OmniActionCausalLMOutputWithPast, inputs: ActionDataSample
    ) -> LossOutput:
        """
        Compute the loss for the training step.
        This method is called by run_training_step after the forward pass.
        """
        losses = []

        cursor_aux = None
        if inputs.cursor_path is not None and inputs.action_tokens is not None:
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
            num_actions = inputs.action_tokens.sum().item()

            # if inputs.cursor_path is not None:
            analytical_loss = (
                analytical_distance(
                    a=output_actions[:, 0, :],
                    b=cursor_path[:, 0, :],
                )
                + analytical_distance(
                    a=output_actions[:, 1, :],
                    b=cursor_path[:, 1, :],
                )
            ).sum() / min(num_actions, 1)

            l2_points_loss = (
                l2_loss(predicted_xs, actual_xs) + l2_loss(predicted_ys, actual_ys)
            ).sum() / min(num_actions, 1)
            coefficients_loss = (
                l2_loss(output_actions[:, 0, :], cursor_path[:, 0, :])
                + l2_loss(output_actions[:, 1, :], cursor_path[:, 1, :])
            ).sum() / min(num_actions, 1)

            loss = None
            match self.module_config.loss_type:
                case Qwen25OActionLITConfig.CursorPredictionLoss.ANALYTICAL_DISTANCE:
                    loss = analytical_loss.sum()
                case Qwen25OActionLITConfig.CursorPredictionLoss.L2_DISTANCE:
                    loss = l2_points_loss
                case Qwen25OActionLITConfig.CursorPredictionLoss.COEFFICIENTS_DISTANCE:
                    loss = coefficients_loss
                case _:
                    raise ValueError(
                        f"Unknown loss type: {self.module_config.loss_type}"
                    )

            assert isinstance(loss, torch.Tensor), (
                f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
            )

            losses.append(loss)

            cursor_aux = CursorLossOutput(
                predicted_xs=predicted_xs,
                predicted_ys=predicted_ys,
                actual_xs=actual_xs,
                actual_ys=actual_ys,
                output_actions=output_actions,
                cursor_path=cursor_path,
                analytical_loss=analytical_loss,
                l2_points_loss=l2_points_loss,
                coefficients_loss=coefficients_loss,
            )

        if inputs.keyboard_tokens_mask is not None:
            assert outputs.keyboard_outputs is not None, (
                "Expected outputs.keyboard_outputs to be not None when keyboard_tokens_mask is provided"
            )
            input_ids_with_non_keyboard_neg_100 = (
                inputs.qwen_inputs.input_ids.masked_fill(
                    ~inputs.keyboard_tokens_mask.bool(), -100
                )
            )  # [B, S]
            token_outputs = outputs.keyboard_outputs

            loss = ForCausalLMLoss(
                logits=token_outputs,
                labels=input_ids_with_non_keyboard_neg_100,
                vocab_size=token_outputs.shape[-1],
            )
            losses.append(loss)

        final_loss = torch.stack(losses).sum()

        return LossOutput(loss=final_loss, cursor_aux=cursor_aux)

    @classmethod
    def validation_wandb_metrics(
        cls, all_metrics: list[dict[str, Any]], global_step: int
    ) -> dict[str, Any]:
        # First split off "val/image"

        metrics = [
            {
                k: v
                for k, v in m.items()
                if not k.startswith("val/image") and not k.startswith("val/tokens")
            }
            for m in all_metrics
        ]

        aux_metrics = {}
        if all_metrics and "val/image" in all_metrics[0]:
            first_val_image = all_metrics[0]["val/image"]

            cursor_path = first_val_image["val/cursor_path"]
            action_outputs = first_val_image["val/action_outputs"]

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

            aux_metrics.update(
                {
                    f"val/images/{global_step}": table,
                    "val/real_image": [wandb.Image(real_image)],
                    "val/predicted_image": [wandb.Image(predicted_image)],
                }
            )

        def output_tokens_as_string(tokens: list[int]) -> str:
            """
            Convert a list of token IDs to a string representation.
            """
            return "\n".join(
                TOKENIZER.debug_reverse_mapping(token_id) for token_id in tokens
            )

        if all_metrics and "val/tokens" in all_metrics[0]:
            tokens_table = wandb.Table(
                columns=[
                    "predicted_tokens",
                    "predicted_tokens_k2",
                    "real_tokens",
                    "predicted_tokens_str",
                    "predicted_tokens_k2_str",
                    "real_tokens_str",
                ],
                data=[
                    [
                        str(m["val/tokens"]["val/predicted_tokens"]),
                        str(m["val/tokens"]["val/predicted_tokens_k2"]),
                        str(m["val/tokens"]["val/real_tokens"]),
                        output_tokens_as_string(
                            m["val/tokens"]["val/predicted_tokens"]
                        ),
                        output_tokens_as_string(
                            m["val/tokens"]["val/predicted_tokens_k2"]
                        ),
                        output_tokens_as_string(m["val/tokens"]["val/real_tokens"]),
                    ]
                    for m in all_metrics
                ],
            )

            aux_metrics.update(
                {
                    f"val/tokens/{global_step}": tokens_table,
                }
            )

        # Get mean over all metrics
        mean_metrics = {
            k: torch.tensor([m[k] for m in metrics]).mean().item()
            for k in metrics[0]
            if k not in {"val/image", "val/tokens"}
        }

        metrics = {
            "val/first_loss": metrics[0].get("val/loss", 0.0),
            **mean_metrics,
            **aux_metrics,
        }
        return metrics

    def run_validation_step(self, inputs: ActionDataSample, global_step: int):
        # Forward pass through the model
        outputs = self.model(
            cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            keyboard_token_mask=inputs.keyboard_tokens_mask,
            **inputs.qwen_inputs.model_dump(),
        )
        tokens_metrics = {}
        if inputs.keyboard_tokens_mask is not None:
            keyboard_logits = outputs.keyboard_outputs[inputs.keyboard_tokens_mask]
            top2_values, top2_indices = keyboard_logits.topk(2, dim=-1)

            # highest‑value tokens (what you were already doing):
            keyboard_tokens = top2_indices[:, 0]

            # second‑highest‑value tokens:
            second_most_likely_logits = top2_indices[:, 1]

            gold_logits = inputs.qwen_inputs.input_ids[inputs.keyboard_tokens_mask]
            # print(outputs.keyboard_outputs.shape, inputs.keyboard_tokens_mask.shape)

            predicted = to_numpy_clean(keyboard_tokens)
            real = to_numpy_clean(gold_logits)
            second_most_likely = to_numpy_clean(second_most_likely_logits)
            tokens_metrics = {
                "val/tokens": {
                    "val/predicted_tokens": predicted,
                    "val/predicted_tokens_k2": second_most_likely,
                    "val/real_tokens": real,
                }
            }

        # outputs = self.model.generate(
        #     # cursor_path=inputs.cursor_path,
        #     # action_tokens=inputs.action_tokens,
        #     keyboard_token_mask=inputs.keyboard_tokens_mask,
        #     **inputs.qwen_inputs.model_dump(),
        # )

        loss_output = self.compute_loss(outputs, inputs)

        cursor_metrics = {}
        if inputs.action_tokens is not None:
            action_outputs = outputs.action_outputs[inputs.action_tokens]
            cursor_aux = loss_output.cursor_aux
            assert cursor_aux is not None, (
                "Expected cursor_aux to be not None when action_tokens are provided"
            )
            cursor_path = cursor_aux.cursor_path.reshape(-1, 6)
            cursor_metrics = {
                "val/image": {
                    "val/predicted_xs": cursor_aux.predicted_xs,
                    "val/predicted_ys": cursor_aux.predicted_ys,
                    "val/actual_xs": cursor_aux.actual_xs,
                    "val/actual_ys": cursor_aux.actual_ys,
                    "val/output_actions": cursor_aux.output_actions,
                    "val/cursor_path": cursor_path,
                    "val/action_tokens": inputs.action_tokens,
                    "val/action_outputs": action_outputs,
                },
                "val/analytical_loss": cursor_aux.analytical_loss,
                "val/l2_points_loss": cursor_aux.l2_points_loss,
                "val/coefficients_loss": cursor_aux.coefficients_loss,
            }

        return loss_output.loss, {
            "val/loss": loss_output.loss,
            **tokens_metrics,
            **cursor_metrics,
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
    class CursorPredictionLoss(str, Enum):
        """
        Enum for cursor prediction loss types.
        """

        ANALYTICAL_DISTANCE = "analytical_distance"
        L2_DISTANCE = "l2_distance"
        COEFFICIENTS_DISTANCE = "coefficients_distance"

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
    freeze_keyboard_embedding: bool = False
    freeze_keyboard_head: bool = False
    loss_type: CursorPredictionLoss = CursorPredictionLoss.ANALYTICAL_DISTANCE

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
