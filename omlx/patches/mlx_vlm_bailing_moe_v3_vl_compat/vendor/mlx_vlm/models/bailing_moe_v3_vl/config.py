import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

from ..base import BaseModelConfig

AttentionKind = Literal["kda", "mla"]


def _config_kwargs(config_cls, params):
    return {
        key: value
        for key, value in params.items()
        if key in inspect.signature(config_cls).parameters
    }


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_moe_vit"
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    disable_merger_proj: bool = True
    deepstack_visual_indexes: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.model_type != "qwen3_moe_vit":
            raise ValueError(f"Unsupported vision model type: {self.model_type}")
        if self.spatial_merge_size != 2:
            raise ValueError("bailing_moe_v3_vl requires fixed 2x2 vision packing")


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "bailing_moe_v3"
    vocab_size: int = 157184
    hidden_size: int = 2560
    intermediate_size: int = 6144
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    rope_theta: float = 6000000.0
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    first_k_dense_replace: int = 2
    num_experts: int = 512
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 768
    moe_shared_expert_intermediate_size: int = 768
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    layer_group_size: int = 6
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    short_conv_kernel_size: int = 4
    kda_safe_gate: bool = True
    kda_lower_bound: float | None = -5.0
    gated_attention_proj_granularity_type: str = "head_wise"
    mrope_section: list[int] = field(default_factory=lambda: [8, 12, 12])
    expert_swiglu_limit_list: list[float] = field(default_factory=list)
    share_expert_swiglu_limit_list: list[float] = field(default_factory=list)
    video_start_token: int = 157160
    layer_plan: tuple[AttentionKind, ...] = field(init=False)

    def __post_init__(self):
        if self.layer_group_size <= 0:
            raise ValueError("layer_group_size must be positive")
        if self.score_function != "sigmoid":
            raise ValueError("bailing_moe_v3_vl requires sigmoid expert routing")
        if self.num_shared_experts != 1:
            raise ValueError("bailing_moe_v3_vl requires exactly one shared expert")
        if self.gated_attention_proj_granularity_type != "head_wise":
            raise ValueError("Only head-wise MLA output gating is supported")
        if sum(self.mrope_section) * 2 != self.qk_rope_head_dim:
            raise ValueError(
                "mrope_section must cover half of qk_rope_head_dim "
                f"({self.mrope_section} for {self.qk_rope_head_dim})"
            )
        self.rope_scaling = dict(self.rope_scaling or {})
        self.rope_scaling.setdefault("type", "default")
        self.rope_scaling["mrope_section"] = list(self.mrope_section)
        self.expert_swiglu_limit_list = (
            list(self.expert_swiglu_limit_list) + [0.0] * self.num_hidden_layers
        )[: self.num_hidden_layers]
        self.share_expert_swiglu_limit_list = (
            list(self.share_expert_swiglu_limit_list) + [0.0] * self.num_hidden_layers
        )[: self.num_hidden_layers]
        if any(limit < 0 for limit in self.expert_swiglu_limit_list):
            raise ValueError("expert_swiglu_limit_list values must be non-negative")
        if any(limit < 0 for limit in self.share_expert_swiglu_limit_list):
            raise ValueError(
                "share_expert_swiglu_limit_list values must be non-negative"
            )
        self.layer_plan = tuple(
            "mla"
            if (index + 1) % self.layer_group_size == 0
            or index
            >= self.num_hidden_layers // self.layer_group_size * self.layer_group_size
            else "kda"
            for index in range(self.num_hidden_layers)
        )


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "bailing_moe_v3_vl"
    image_token_id: int = 157157
    video_token_id: int = 156909
    vision_start_token_id: int = 157158
    vision_end_token_id: int = 157159
    video_start_token_id: int = 157160
    mrope_section: list[int] = field(default_factory=lambda: [8, 12, 12])
    image_token_index: int | None = None
    video_token_index: int | None = None
    skip_vision: bool = False
    quantization: dict[str, Any] | None = None
    quantization_config: dict[str, Any] | None = None

    def __post_init__(self):
        if self.model_type != "bailing_moe_v3_vl":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        self.video_start_token_id = self.text_config.video_start_token
        self.text_config.mrope_section = list(self.mrope_section)
        self.text_config.__post_init__()
        if self.quantization is None:
            self.quantization = self.quantization_config
        elif (
            self.quantization_config is not None
            and self.quantization_config != self.quantization
        ):
            raise ValueError("quantization and quantization_config must agree")
        self.quantization_config = self.quantization

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        mrope_section = list(params.get("mrope_section", [8, 12, 12]))
        text = params.get("text_config")
        if isinstance(text, dict):
            text = dict(text)
            text["mrope_section"] = mrope_section
            params["text_config"] = TextConfig(**_config_kwargs(TextConfig, text))
        vision = params.get("vision_config")
        if isinstance(vision, dict):
            params["vision_config"] = VisionConfig(
                **_config_kwargs(VisionConfig, vision)
            )
        params["mrope_section"] = mrope_section
        return cls(**_config_kwargs(cls, params))
