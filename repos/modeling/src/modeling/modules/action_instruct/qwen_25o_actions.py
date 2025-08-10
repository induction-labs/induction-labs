from dataclasses import dataclass
from typing import cast

import torch
from synapse.actions.keyboard_tokenizer import Tokenizer
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
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionEncoder,
)

from modeling.config import (
    AttentionImplementation,
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
        freeze_keyboard_embedding=False,
        freeze_keyboard_head=False,
        use_fun_mask=False,
        **kwargs,
    ):
        self.freeze_network = freeze_network
        self.freeze_vision = freeze_vision
        self.freeze_action_head = freeze_action_head
        self.freeze_action_embedding = freeze_action_embedding
        self.freeze_keyboard_embedding = freeze_keyboard_embedding
        self.freeze_keyboard_head = freeze_keyboard_head
        self.use_fun_mask = use_fun_mask

        super().__init__(**kwargs)


@dataclass
class Qwen2_5OmniActionCausalLMOutputWithPast(Qwen2_5OmniThinkerCausalLMOutputWithPast):
    action_outputs: torch.FloatTensor | None = None
    keyboard_outputs: torch.FloatTensor | None = None
    position_ids: torch.LongTensor | None = None


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

        self.config = cast(Qwen2_5OmniThinkerActionConfig, self.config)

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
        self.keyboard_tokenizer = Tokenizer.load(
            "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
        )
        self.keyboard_vocab = self.keyboard_tokenizer.vocab_size
        self.keyboard_head = nn.Linear(
            config.text_config.hidden_size, self.keyboard_vocab, bias=False
        )
        # TODO: fix this loading code
        # self.keyboard_head.weight = nn.Parameter(torch.load("/home/ubuntu/induction-labs/repos/modeling/output_head_7b.pt"))
        self.keyboard_embedding = nn.Embedding(
            self.keyboard_vocab, config.text_config.hidden_size
        )
        # self.keyboard_embedding.weight = nn.Parameter(torch.load("/home/ubuntu/induction-labs/repos/modeling/embedding_dict_7b.pt"))

        # `bias=True` here creates NaN gradients when fsdp is enabled + torch.use_deterministic_algorithms(True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, 6, bias=False),
        )

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None

        for param in self.parameters():
            param.requires_grad = not self.config.freeze_network

        for param in self.action_head.parameters():
            param.requires_grad = not self.config.freeze_action_head

        for param in self.action_token_embedding.parameters():
            param.requires_grad = not self.config.freeze_action_embedding

        for param in self.visual.parameters():
            param.requires_grad = not self.config.freeze_vision

        for param in self.keyboard_head.parameters():
            param.requires_grad = not self.config.freeze_keyboard_head

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
        action_tokens: torch.Tensor | None = None,
        keyboard_token_mask: torch.Tensor | None = None,
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
            if (
                action_tokens is not None
                and action_tokens.shape[1] == inputs_embeds.shape[1]
            ):
                # Otherwise this breaks torch.compile graph
                mask = action_tokens.unsqueeze(-1)  # [B, L, 1]
                action_vec = self.action_token_embedding.weight[0]  # [hidden_size]
                action_vec = action_vec.view(1, 1, -1).expand_as(
                    inputs_embeds
                )  # [B, S, D]
                inputs_embeds = torch.where(mask, action_vec, inputs_embeds)

            if (
                keyboard_token_mask is not None
                and keyboard_token_mask.shape[1] == inputs_embeds.shape[1]
            ):
                # Otherwise this breaks torch.compile graph
                kbd_mask = keyboard_token_mask.bool()
                kbd_ids = input_ids[kbd_mask]
                kbd_embeds = self.keyboard_embedding(kbd_ids)
                kbd_full = torch.zeros_like(inputs_embeds)
                kbd_full[kbd_mask] = kbd_embeds
                inputs_embeds = torch.where(
                    kbd_mask.unsqueeze(-1), kbd_full, inputs_embeds
                )

        # 2. Merge text , audios , image and video
        if (
            input_ids is not None
            and input_ids.shape[1] != 1
            and pixel_values_videos is not None
        ):  # Prefill stage
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # check_nans(video_embeds, "video_embeds")
            n_video_tokens = (input_ids == self.config.video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            assert n_video_tokens == n_video_features, (
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
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
        audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                between_video_token_mask = torch.zeros_like(input_ids)
                if keyboard_token_mask is not None:
                    between_video_token_mask = keyboard_token_mask

                if action_tokens is not None:
                    between_video_token_mask = between_video_token_mask | action_tokens

                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    between_video_token_mask=between_video_token_mask,
                    use_audio_in_video=use_audio_in_video,
                    audio_seqlens=audio_feature_lengths,
                    second_per_grids=video_second_per_grid,
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

        if self.config.use_fun_mask:
            assert (
                self.config._attn_implementation
                != AttentionImplementation.FLASH_ATTENTION_2
            ), (
                "Fun mask is not supported with Flash Attention 2, please use SDPA or Eager"
            )
            if action_tokens is not None:
                # If we are in decode, we only multiply out on the first k tokens
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
                "config": self.model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            full_attention_mask = create_causal_mask(**mask_kwargs)
            print(full_attention_mask.shape, inputs_embeds.shape)
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
            # The sliding window alternating layers are not always activated depending on the config
            if self.model.has_sliding_layers:
                raise NotImplementedError()
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )
            attention_mask = causal_mask_mapping

        outputs = self.model(
            # attention_mask=causal_mask_mapping,
            attention_mask=attention_mask,
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

        # logits = self.lm_head(hidden_states)  # [B, S, V]

        action_outputs = None
        if action_tokens is not None:
            action_outputs = self.action_head(hidden_states)  # [B, S, 6]

        token_outputs = None
        if keyboard_token_mask is not None:
            token_outputs = self.keyboard_head(hidden_states)  # [B, S, T]

        assert return_dict, (
            "return_dict should be True for Qwen2_5OmniActionCausalLMOutputWithPast"
        )

        return Qwen2_5OmniActionCausalLMOutputWithPast(
            # loss=loss,
            # logits=logits,
            position_ids=position_ids,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            action_outputs=action_outputs,
            keyboard_outputs=token_outputs,
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

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        between_video_token_mask: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: torch.LongTensor | None = None,
        second_per_grids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            # initialize placeholder for position_ids
            # all ones
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                # process a single batch
                # only care about unmasked tokens
                input_ids = input_ids[attention_mask[i] == 1]
                keyboard_mask_for_sample = between_video_token_mask[i]
                shifted_keyboard_mask = torch.concat(
                    [
                        keyboard_mask_for_sample[1:],
                        torch.zeros([1]).to(keyboard_mask_for_sample.device),
                    ],
                    dim=0,
                )
                # create index of [actions, 2] for start end of the keyboard tokens
                start_end_keyboard_idx = (
                    torch.argwhere(keyboard_mask_for_sample != shifted_keyboard_mask)
                    .squeeze(1)
                    .reshape(-1, 2)
                    + 1
                )

                image_nums, video_nums, audio_nums = 0, 0, 0
                keyboard_nums = 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]

                # number of audio sequences
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                # number of image sequences
                image_nums = (vision_tokens == image_token_id).sum()
                # number of video sequences
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                # where to start scanning from
                st = 0
                # counter for remaining images, videos and audios
                remain_images, remain_videos, remain_audios = (
                    image_nums,
                    video_nums,
                    audio_nums,
                )
                multimodal_nums = (
                    image_nums + audio_nums
                    if use_audio_in_video
                    else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        # process audio tokens
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1)
                                + st_idx
                            )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        bos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        audio_len = (
                            (audio_seqlens[audio_idx] - 1) // 2 + 1 - 2
                        ) // 2 + 1
                        llm_pos_ids = (
                            torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        eos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        # process image tokens
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1)
                                + st_idx
                            )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        bos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * 1 * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx,
                            image_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                        )
                        image_len = image_grid_thw[image_idx].prod() // (
                            spatial_merge_size**2
                        )
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        eos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_video and not use_audio_in_video:
                        # process video tokens without audio
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            # normal text tokens between videos
                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1)
                                + st_idx
                            )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        bos_len = 1
                        # bos position embedding (like text)
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).long()
                        _num_frames = (
                            second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        )
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                        )
                        llm_grid_h = grid_hs[video_idx] // spatial_merge_size
                        llm_grid_w = grid_ws[video_idx] // spatial_merge_size
                        # inject the position ids for text in between videos
                        # print(llm_pos_ids.shape, llm_grid_h, llm_grid_w)
                        # print(llm_pos_ids)
                        # print(st_idx, num_frames, torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds)
                        video_len = video_grid_thw[video_idx].prod() // (
                            spatial_merge_size**2
                        )

                        # interleave keyboard token pos ids
                        for frame in llm_pos_ids.view(
                            3, -1, llm_grid_h * llm_grid_w
                        ).permute(1, 0, 2):
                            llm_pos_ids_list.append(frame)
                            keyboard_start, keyboard_end = start_end_keyboard_idx[
                                keyboard_nums
                            ]
                            idxs = (
                                torch.arange(
                                    frame[0, -1],
                                    frame[0, -1] + keyboard_end - keyboard_start,
                                ).repeat(3, 1)
                                + 1
                            )  # add one so no overlap with the first video token
                            llm_pos_ids_list.append(idxs)
                            video_len += keyboard_end - keyboard_start
                            keyboard_nums += 1

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        eos_len = 1
                        # eos position embedding (like text)
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    elif min_ed == ed_video and use_audio_in_video:
                        # process video tokens with audio
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1)
                                + st_idx
                            )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        bos_len = 1
                        # why two bos?
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        audio_len = (
                            (audio_seqlens[audio_idx] - 1) // 2 + 1 - 2
                        ) // 2 + 1
                        audio_llm_pos_ids = (
                            torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                        )

                        t_ntoken_per_chunk = int(
                            position_id_per_seconds * seconds_per_chunk
                        )
                        video_chunk_indexes = self.get_chunked_index(
                            video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx
                        )
                        audio_chunk_indexes = self.get_chunked_index(
                            audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx
                        )
                        sub_len = 0
                        for j in range(
                            max(len(video_chunk_indexes), len(audio_chunk_indexes))
                        ):
                            video_chunk_index = (
                                video_chunk_indexes[j]
                                if j < len(video_chunk_indexes)
                                else None
                            )
                            audio_chunk_index = (
                                audio_chunk_indexes[j]
                                if j < len(audio_chunk_indexes)
                                else None
                            )
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[
                                        :, video_chunk_index[0] : video_chunk_index[1]
                                    ]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[
                                        :, audio_chunk_index[0] : audio_chunk_index[1]
                                    ]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (
                            spatial_merge_size**2
                        )

                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        eos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        st += (
                            text_len + bos_len * 2 + audio_len + video_len + eos_len * 2
                        )

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

                # write results of single batch to position_ids
                # for the masked positions keep the position_ids as 1
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = (
                max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
            )

            return position_ids, mrope_position_deltas


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
    from transformers.generation.configuration_utils import GenerationConfig

    from modeling.data.video_action import (
        ActionDataSample,
        VideoProcessorConfig,
        calc_min_num_tokens_for_n_actions,
        fetch_data,
        make_raw_prompt,
    )

    # default: Load the model on the available device(s)
    t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    config = Qwen2_5OmniThinkerActionConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    model = Qwen2_5OmniThinkerForActionModelling.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = model.to(torch.cuda.current_device())
    processor_config = VideoProcessorConfig.Qwen25O()

    # VideoProcessorConfig
    raw_prompt = make_raw_prompt(
        processor_config,
    )
    num_tokens = calc_min_num_tokens_for_n_actions(
        840 * 476, 1, raw_prompt, processor_config
    )

    data, _ = await fetch_data(
        path="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_25.zarr",
        seq_length=num_tokens,
        config=processor_config,
        raw_prompt=raw_prompt,
        start=6,
    )

    data = ActionDataSample.combine_batch([data])
    data = data.to_device("cuda")
    inputs = data.qwen_inputs.model_dump()
    # inputs["second_per_grid_ts"] = inputs.pop("video_second_per_grid")

    with torch.no_grad():
        # result = model.forward(**inputs, action_tokens=data.action_tokens)
        result2 = model.generate(
            **inputs,
            generation_config=GenerationConfig(max_new_tokens=400),
            action_tokens=data.action_tokens,
        )

    # print(result, result2)
    print(t.decode(result2[0], skip_special_tokens=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
