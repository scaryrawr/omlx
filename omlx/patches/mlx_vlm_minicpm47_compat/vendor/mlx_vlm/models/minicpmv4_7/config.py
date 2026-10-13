from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig
from ..qwen3_5.config import sanitize_quantization_config
from ..qwen3_5_moe.config import TextConfig


def validate_downsample_mode(mode):
    if mode not in ("4x", "16x"):
        raise ValueError("MiniCPM-V 4.7 downsample_mode must be '4x' or '16x'")
    return mode


@dataclass
class VisionConfig(BaseModelConfig):
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 980
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    insert_layer_id: int = 6
    window_kernel_size: tuple = (2, 2)
    model_type: str = "minicpmv4_7_vision"

    def __post_init__(self):
        self.window_kernel_size = tuple(self.window_kernel_size)
        if self.window_kernel_size != (2, 2):
            raise ValueError("MiniCPM-V 4.7 requires a 2x2 vision window")
        if not 0 <= self.insert_layer_id < self.num_hidden_layers:
            raise ValueError("insert_layer_id must identify a vision encoder layer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Vision hidden_size must be divisible by attention heads")
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("MiniCPM-V 4.7 vision requires gelu_pytorch_tanh")

    @property
    def window_hidden_size(self):
        return self.hidden_size * 4

    @property
    def window_intermediate_size(self):
        return self.intermediate_size * 4


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "minicpmv4_7"
    insert_layer_id: int = 6
    downsample_mode: str = "16x"
    mrope_mode: str = "canvas"
    merge_kernel_size: tuple = (2, 2)
    merger_times: int = 1
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_start_id: int = 248078
    image_end_id: int = 248079
    slice_start_id: int = 248088
    slice_end_id: int = 248089
    newline_id: int = 198
    tie_word_embeddings: bool = False
    eos_token_id: Optional[list] = None
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        self.text_config = deepcopy(self.text_config)
        self.vision_config = deepcopy(self.vision_config)
        validate_downsample_mode(self.downsample_mode)
        if self.model_type != "minicpmv4_7" or self.mrope_mode != "canvas":
            raise ValueError("Expected MiniCPM-V 4.7 with canvas M-RoPE")
        self.merge_kernel_size = tuple(self.merge_kernel_size)
        if self.merge_kernel_size != (2, 2) or self.merger_times != 1:
            raise ValueError("MiniCPM-V 4.7 supports one final 2x2 merger")
        self.vision_config.insert_layer_id = self.insert_layer_id
        self.vision_config.__post_init__()
        if self.text_config.model_type != "qwen3_5_moe_text":
            raise ValueError("MiniCPM-V 4.7 35B-A3B requires Qwen3.5 MoE text")
        interval = self.text_config.full_attention_interval
        if type(interval) is not int or interval <= 0:
            raise ValueError("full_attention_interval must be a positive integer")
        if (
            self.text_config.num_hidden_layers
            < self.text_config.full_attention_interval
        ):
            raise ValueError("The hybrid text model requires a full-attention layer")
        if self.tie_word_embeddings != self.text_config.tie_word_embeddings:
            raise ValueError("Root and text tie_word_embeddings must agree")
        self.quantization = sanitize_quantization_config(self.quantization)
        self.quantization_config = sanitize_quantization_config(
            self.quantization_config
        )
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id or [248044, 248046]

    @classmethod
    def from_dict(cls, params):
        params = deepcopy(params)
        text = params.get("text_config")
        vision = params.get("vision_config")
        if not isinstance(text, (dict, TextConfig)):
            raise ValueError("MiniCPM-V 4.7 requires text_config")
        if isinstance(text, dict):
            interval = text.get("full_attention_interval", 4)
            layers = text.get("num_hidden_layers", 0)
            if type(interval) is not int or interval <= 0:
                raise ValueError("full_attention_interval must be a positive integer")
            expected = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(layers)
            ]
            if text.get("layer_types", expected) != expected:
                raise ValueError("Unsupported MiniCPM-V 4.7 hybrid layer layout")
            if text.get("hidden_act", "silu") != "silu":
                raise ValueError("MiniCPM-V 4.7 text requires silu")
            if not text.get("attn_output_gate", True):
                raise ValueError("MiniCPM-V 4.7 requires gated full attention")
            params["text_config"] = TextConfig.from_dict(text)
        if isinstance(vision, dict):
            vision["insert_layer_id"] = params.get("insert_layer_id", 6)
            params["vision_config"] = VisionConfig.from_dict(vision)
        elif not isinstance(vision, VisionConfig):
            raise ValueError("MiniCPM-V 4.7 requires vision_config")
        return super().from_dict(params)
