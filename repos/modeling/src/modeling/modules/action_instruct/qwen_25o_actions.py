from dataclasses import dataclass

import torch
from synapse.utils.logging import configure_logging, logging
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.qwen2_5_omni import Qwen2_5OmniPreTrainedModel
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,  # Same as Qwen2_5OmniMLP
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionEncoder,
)

# from modeling.utils.check_nans import check_nans

logger = configure_logging(__name__, level=logging.INFO)


class Qwen2_5OmniThinkerActionConfig(Qwen2_5OmniThinkerConfig):
    def __init__(
        self,
        freeze_network=False,
        freeze_vision=False,
        freeze_action_head=False,
        freeze_action_embedding=False,
        **kwargs,
    ):
        self.freeze_network = freeze_network
        self.freeze_vision = freeze_vision
        self.freeze_action_head = freeze_action_head
        self.freeze_action_embedding = freeze_action_embedding

        super().__init__(**kwargs)


@dataclass
class Qwen2_5OmniActionCausalLMOutputWithPast(Qwen2_5OmniThinkerCausalLMOutputWithPast):
    action_outputs: torch.FloatTensor | None = None


class Qwen2_5OmniThinkerForActionModelling(
    Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin
):
    def __call__(self, *args, **kwargs) -> Qwen2_5OmniActionCausalLMOutputWithPast:
        """
        Override the __call__ method to return the custom output type.
        """
        return super().__call__(*args, **kwargs)

    config_class = Qwen2_5OmniThinkerActionConfig
    base_model_prefix = "thinker"
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]  # noqa: RUF012

    def __init__(self, config: Qwen2_5OmniThinkerActionConfig):
        super().__init__(config)

        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.action_token_embedding = nn.Embedding(1, config.text_config.hidden_size)

        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        # `bias=True` here creates NaN gradients when fsdp is enabled + torch.use_deterministic_algorithms(True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 6, bias=False),
        )

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.l2_loss = nn.MSELoss(reduce=False)

        for param in self.parameters():
            param.requires_grad = not self.config.freeze_network

        for param in self.action_head.parameters():
            param.requires_grad = not self.config.freeze_action_head

        for param in self.action_token_embedding.parameters():
            param.requires_grad = not self.config.freeze_action_embedding

        for param in self.visual.parameters():
            param.requires_grad = not self.config.freeze_vision

        # TODO: When training print number of trainable parameters
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths
                if audio_feature_lengths is not None
                else feature_attention_mask.sum(-1)
            )
        )
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state

        if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError(
                "length of audio_features should match audio_output_lengths"
            )

        return audio_features

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        # labels: Optional[torch.LongTensor] = None,
        cursor_path: torch.Tensor | None = None,
        action_tokens: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_audio_in_video: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        video_second_per_grid: torch.LongTensor | None = None,
    ) -> tuple | Qwen2_5OmniActionCausalLMOutputWithPast:
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
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

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
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
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
            # If we are in decode
            if action_tokens.shape[1] == inputs_embeds.shape[1]:
                # Otherwise this breaks torch.compile graph
                mask = action_tokens.unsqueeze(-1)  # [B, L, 1]
                action_vec = self.action_token_embedding.weight[0]  # [hidden_size]
                action_vec = action_vec.view(1, 1, -1).expand_as(
                    inputs_embeds
                )  # [B, S, D]
                inputs_embeds = torch.where(mask, action_vec, inputs_embeds)

                # inputs_embeds[action_tokens] = self.action_token_embedding.weight[0]

        # 2. Merge text , audios , image and video
        if (
            input_ids is not None
            and input_ids.shape[1] != 1
            and pixel_values_videos is not None
        ):  # Prefill stage
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # check_nans(video_embeds, "video_embeds")
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # check_nans(inputs_embeds, "input_embeds")
        audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + self.rope_deltas
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # If we are in decode, we only multiply out on the first k tokens
        attention_mask[:, : action_tokens.shape[1]] = (
            attention_mask[:, : action_tokens.shape[1]] * ~action_tokens
        )
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        mask_kwargs = {
            "config": self.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        full_attention_mask = create_causal_mask(**mask_kwargs)
        # Allow action tokens to attend to themselves but not be attended to.
        if full_attention_mask.shape[2] == action_tokens.shape[1]:
            unmasked_value = True if full_attention_mask.dtype == torch.bool else 0.0
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
        # The sliding window alternating layers are not always activated depending on the config
        if self.model.has_sliding_layers:
            raise NotImplementedError()
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

        outputs = self.model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)  # [B, S, V]
        action_outputs = self.action_head(hidden_states)  # [B, S, 6]

        assert return_dict, (
            "return_dict should be True for Qwen2_5OmniActionCausalLMOutputWithPast"
        )

        return Qwen2_5OmniActionCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            action_outputs=action_outputs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


class Qwen2_5OmniActionModel(Qwen2_5OmniPreTrainedModel, GenerationMixin):
    config_class = Qwen2_5OmniThinkerActionConfig
    base_model_prefix = "model"
    thinker: Qwen2_5OmniThinkerForActionModelling

    def __init__(self, config: Qwen2_5OmniThinkerActionConfig):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForActionModelling(config)

        self.post_init()

    def forward(self, *args, **kwargs) -> Qwen2_5OmniActionCausalLMOutputWithPast:
        """
        Override the forward method to return the custom output type.
        """
        return self.thinker(*args, **kwargs)


async def main():
    from transformers import AutoTokenizer

    from modeling.data.video_action import ActionDataSample, fetch_data

    # default: Load the model on the available device(s)
    t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    config = Qwen2_5OmniThinkerConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    data = await fetch_data(
        num_actions=3,
        path="gs://induction-labs/jonathan/synth/cursor_follow_v1/sample_3.zarr",
        seq_length=2048,
    )
    data = ActionDataSample.combine_batch([data])
    inputs = data.qwen_inputs.model_dump()

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")
    print(data)
    with torch.no_grad():
        result = model.forward(
            # cursor_path=data.cursor_path,
            # action_tokens=data.action_tokens,
            **inputs,
        )
        result2 = model.generate(**inputs)

    print(result, result2)
    print(t.decode(result2[0], skip_special_tokens=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
