# SPDX-License-Identifier: Apache-2.0
"""Lightweight checkpoint classification shared by discovery and integrations."""

# Known embedding model types from mlx-embeddings
EMBEDDING_MODEL_TYPES = {
    "embedding_gemma2",
    "bert",
    "xlm-roberta",
    "xlm_roberta",
    "modernbert",
    "siglip",
    "colqwen2_5",
    "colqwen2-5",
}

# Speculative-decoding "helper" checkpoints (dFlash / MTP / assistant drafters)
# are never meant to be served as standalone chat models. Some declare a
# distinctive top-level model_type — an ``*_assistant`` (e.g. gemma4_assistant)
# or ``*_mtp`` (e.g. qwen3_5_mtp) marker — but DFlash draft checkpoints declare
# a plain model_type (e.g. ``qwen3``) and are only distinguishable by their
# architecture name (``DFlashDraftModel``) or a drafter-only config block
# (``dflash_config``). Keep these in sync with the drafter resolution in
# engine_pool.py (~1498) and the dflash gate in engine/dflash.py when new
# drafter families are added.
HELPER_CONFIG_MODEL_TYPE_SUFFIXES = ("_assistant", "_mtp")


def is_helper_config_model_type(config_model_type: str | None) -> bool:
    """True when ``config_model_type`` marks a speculative-decoding drafter.

    These are the raw top-level ``model_type`` values from a checkpoint's
    config.json (e.g. ``gemma4_assistant``, ``qwen3_5_mtp``). Note this misses
    DFlash drafts, whose model_type is a plain ``qwen3`` — use
    :func:`omlx.model_discovery.is_helper_model_config` when the full config dict is available.
    """
    if not isinstance(config_model_type, str) or not config_model_type:
        return False
    mt = config_model_type.lower()
    return mt.endswith(HELPER_CONFIG_MODEL_TYPE_SUFFIXES)
