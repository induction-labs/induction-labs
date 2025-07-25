from dataclasses import dataclass
from typing import Unpack, cast

import torch
from synapse.utils.logging import configure_logging, logging
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen2_5_vl import Qwen2_5_VLPreTrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,  # Same as Qwen2_5_VLMLP
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.utils.generic import TransformersKwargs
from transformers.utils.import_utils import is_torchdynamo_compiling

from modeling.types import AttentionImplementation

# from modeling.utils.check_nans import check_nans

logger = configure_logging(__name__, level=logging.INFO)


class Qwen2_5_VLActionConfig(Qwen2_5_VLConfig):
    def __init__(
        self,
        freeze_network=None,
        freeze_vision=None,
        freeze_action_head=None,
        freeze_action_embedding=None,
        freeze_mlps=None,
        use_fun_mask=False,
        **kwargs,
    ):
        self.freeze_network = freeze_network
        self.freeze_vision = freeze_vision
        self.freeze_action_head = freeze_action_head
        self.freeze_action_embedding = freeze_action_embedding
        self.freeze_mlps = freeze_mlps
        self.use_fun_mask = use_fun_mask

        super().__init__(**kwargs)


@dataclass
class Qwen2_5_VLModelOutputWithPastWithPosition(Qwen2_5_VLModelOutputWithPast):
    position_ids: torch.LongTensor | None = None


class Qwen2_5_VLActionModel(Qwen2_5_VLModel, GenerationMixin):
    def __call__(self, *args, **kwargs) -> Qwen2_5_VLCausalLMOutputWithPast:
        """
        Override the __call__ method to return the custom output type.
        """
        return super().__call__(*args, **kwargs)

    config_class = Qwen2_5_VLActionConfig

    def __init__(self, config: Qwen2_5_VLActionConfig):
        super().__init__(config)
        self.config = cast(Qwen2_5_VLActionConfig, config)

        self.action_token_embedding = nn.Embedding(1, config.text_config.hidden_size)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        *args,
        action_tokens: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5_VLModelOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing videos).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        second_per_grid_ts (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

        >>>  = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = .generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # check_nans(self.get_input_embeddings().weight, "input_embeddings_weight")
            inputs_embeds = self.get_input_embeddings()(input_ids)  # [B, S, D]
        # 2. Merge text , audios , image and video
        assert inputs_embeds is not None
        assert pixel_values is None

        if (
            input_ids is not None
            and input_ids.shape[1] != 1
            and pixel_values_videos is not None
        ):  # Prefill stage
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # We need to cat here because qwen 2.5 does a split
            video_embeds = torch.cat(video_embeds, dim=0)

            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            n_video_tokens = (input_ids == self.config.video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if (action_tokens is not None) and action_tokens.shape[
            1
        ] == inputs_embeds.shape[1]:
            # Otherwise this breaks torch.compile graph
            mask = action_tokens.unsqueeze(-1)  # [B, L, 1]
            action_vec = self.action_token_embedding.weight[0]  # [hidden_size]
            action_vec = action_vec.view(1, 1, -1).expand_as(inputs_embeds)  # [B, S, D]
            inputs_embeds = torch.where(mask, action_vec, inputs_embeds)
        # check_nans(inputs_embeds, "input_embeds")

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # # If we are in decode, we only multiply out on the first k tokens
        if self.config.use_fun_mask:
            assert (
                self.config._attn_implementation
                != AttentionImplementation.FLASH_ATTENTION_2
            ), (
                "Fun mask is not supported with Flash Attention 2, please use SDPA or Eager"
            )
            if action_tokens is not None:
                attention_mask[:, : action_tokens.shape[1]] = (
                    attention_mask[:, : action_tokens.shape[1]] * ~action_tokens
                )
            if cache_position is None:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            full_attention_mask = create_causal_mask(**mask_kwargs)
            # Allow action tokens to attend to themselves but not be attended to.
            if full_attention_mask.shape[2] == action_tokens.shape[1]:
                unmasked_value = (
                    True if full_attention_mask.dtype == torch.bool else 0.0
                )
                # This is because fucking mask dtype is different depending on attention impl
                # SPDA is bool, eager is float
                diag = full_attention_mask[:, 0, :, :].diagonal(dim1=-2, dim2=-1)
                assert diag[action_tokens].ne(unmasked_value).all(), (
                    "Action tokens should not be masked %s",
                    diag[action_tokens],
                )
                diag[action_tokens] = unmasked_value
            # Create the masks
            causal_mask_mapping = {
                "full_attention": full_attention_mask,
            }
            attention_mask = causal_mask_mapping

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            # attention_mask=causal_mask_mapping,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen2_5_VLModelOutputWithPastWithPosition(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            position_ids=position_ids,
        )


@dataclass
class Qwen2_5_VLActionCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    action_outputs: torch.FloatTensor | None = None
    keyboard_outputs: torch.FloatTensor | None = None
    position_ids: torch.LongTensor | None = None


class Qwen2_5_VLForActionModel(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLActionConfig

    def __init__(self, config: Qwen2_5_VLActionConfig):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.model = Qwen2_5_VLActionModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        hidden_size = config.text_config.hidden_size

        # `bias=True` here creates NaN gradients when fsdp is enabled + torch.use_deterministic_algorithms(True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, 6, bias=False),
        )

        if self.config.freeze_network is not None:
            for param in self.parameters():
                param.requires_grad = not self.config.freeze_network

        if self.config.freeze_action_head is not None:
            for param in self.action_head.parameters():
                param.requires_grad = not self.config.freeze_action_head

        if self.config.freeze_action_embedding is not None:
            for param in self.model.action_token_embedding.parameters():
                param.requires_grad = not self.config.freeze_action_embedding

        if self.config.freeze_vision is not None:
            for param in self.visual.parameters():
                param.requires_grad = not self.config.freeze_vision

        if self.config.freeze_mlps is not None:
            for name, param in self.named_parameters():
                if "mlp" in name:
                    param.requires_grad = not self.config.freeze_mlps

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        *args,
        action_tokens: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen2_5_VLActionCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            action_tokens=action_tokens,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)  # [B, S, V]
        action_outputs = self.action_head(hidden_states)  # [B, S, 6]

        return Qwen2_5_VLActionCausalLMOutputWithPast(
            loss=None,
            position_ids=outputs.position_ids,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            action_outputs=action_outputs,
            keyboard_outputs=None,  # Placeholder for keyboard outputs
        )


async def main():
    from transformers import AutoTokenizer
    from transformers.generation.configuration_utils import GenerationConfig

    from modeling.data.video_action import (
        ActionDataSample,
        VideoProcessorConfig,
        calc_min_num_tokens_for_n_actions,
        fetch_data,
        make_raw_prompt,
    )

    # default: Load the model on the available device(s)
    t = AutoTokenizer.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
    # config = Qwen2_5_VLConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-Omni-3B",
    #     config=config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    model = Qwen2_5_VLForActionModel.from_pretrained(
        "ByteDance-Seed/UI-TARS-1.5-7B",
    )
    model = model.to(torch.cuda.current_device())
    processor_config = VideoProcessorConfig.Qwen25VL(
        name="ByteDance-Seed/UI-TARS-1.5-7B"
    )
    # processor_config = VideoProcessorConfig.Qwen25O()

    # VideoProcessorConfig
    raw_prompt = make_raw_prompt(
        processor_config,
    )
    num_tokens = calc_min_num_tokens_for_n_actions(
        840 * 476, 2, raw_prompt, processor_config
    )

    data, _ = await fetch_data(
        num_actions=2,
        path="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_25.zarr",
        seq_length=num_tokens,
        config=processor_config,
        raw_prompt=raw_prompt,
        start=5,
    )

    data = ActionDataSample.combine_batch([data])
    data = data.to_device("cuda")
    inputs = data.qwen_inputs.model_dump()
    inputs["second_per_grid_ts"] = inputs.pop("video_second_per_grid")

    with torch.no_grad():
        # result = model.forward(**inputs, action_tokens=data.action_tokens)
        result2 = model.generate(
            **inputs,
            action_tokens=data.action_tokens,
            generation_config=GenerationConfig(max_new_tokens=400),
        )

    # print(result, result2)
    print(t.decode(result2[0], skip_special_tokens=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
