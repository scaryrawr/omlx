import mlx.core as mx
from mlx import nn

from ..qwen3_vl.vision import (
    PatchEmbed,
    Qwen3VLMoEVisionBlock,
    VisionRotaryEmbedding,
)
from ..qwen3_vl.vision import (
    VisionModel as Qwen3VLVisionModel,
)
from .config import VisionConfig


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.packed_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x).reshape(-1, self.packed_size)


class VisionModel(Qwen3VLVisionModel):
    def __init__(self, config: VisionConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = [Qwen3VLMoEVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        self.deepstack_visual_indexes = []
        self.deepstack_merger_list = []
