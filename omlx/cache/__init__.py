# SPDX-License-Identifier: Apache-2.0
"""
Cache module - unified cache management for oMLX.

This package contains cache implementations including:
- Prefix caching for KV state reuse
- Paged cache for memory-efficient KV storage
- VLM cache for vision-language model optimization
- SSD cache for disk-based persistence
"""

# Stats
# Factory
from .factory import CacheConfig, CacheFactory

# Hybrid cache config
from .hybrid_cache import (
    LayerCacheConfig,
    ModelCacheConfig,
    create_default_kvcache_config,
)

# Interfaces
from .interface import CacheManager

# Paged cache implementations
from .paged_cache import (
    BlockHash,
    BlockHashToBlockMap,
    BlockTable,
    CacheBlock,
    FreeKVCacheBlockQueue,
    PagedCacheManager,
    compute_block_hash,
)

# Paged SSD cache implementations
from .paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    PagedSSDCacheManager,
    parse_size,
)

# Prefix cache implementations (SSD-only)
from .prefix_cache import (
    BlockAwarePrefixCache,
    BlockCacheEntry,
)

# Managers
from .recovery import CacheRecoveryManager
from .stats import (
    BaseCacheStats,
    PagedCacheStats,
    PagedSSDCacheStats,
    VLMCacheStats,
)

# Type handlers
from .type_handlers import (
    ArraysCacheHandler,
    CacheListHandler,
    CacheStateInfo,
    CacheType,
    CacheTypeHandler,
    DefaultCacheHandler,
    KVCacheHandler,
    RotatingKVCacheHandler,
    SizedArraysCache,
)

# Type registry
from .type_registry import CacheTypeRegistry

# Vision feature cache
from .vision_feature_cache import (
    VisionFeatureSSDCache,
    VisionFeatureSSDEntry,
)

__all__ = [
    # Stats
    "BaseCacheStats",
    "PagedCacheStats",
    "VLMCacheStats",
    "PagedSSDCacheStats",
    # Interfaces
    "CacheManager",
    # Paged cache
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "FreeKVCacheBlockQueue",
    "BlockHashToBlockMap",
    "BlockHash",
    "compute_block_hash",
    # Prefix cache (SSD-only)
    "BlockAwarePrefixCache",
    "BlockCacheEntry",
    # Paged SSD cache
    "PagedSSDCacheManager",
    "PagedSSDBlockMetadata",
    "PagedSSDCacheIndex",
    "parse_size",
    # Vision feature cache
    "VisionFeatureSSDCache",
    "VisionFeatureSSDEntry",
    # Managers
    "CacheRecoveryManager",
    # Factory
    "CacheConfig",
    "CacheFactory",
    # Type handlers
    "CacheType",
    "CacheTypeHandler",
    "CacheStateInfo",
    "KVCacheHandler",
    "RotatingKVCacheHandler",
    "ArraysCacheHandler",
    "CacheListHandler",
    "DefaultCacheHandler",
    "SizedArraysCache",
    # Type registry
    "CacheTypeRegistry",
    # Hybrid cache config
    "LayerCacheConfig",
    "ModelCacheConfig",
    "create_default_kvcache_config",
]
