# SPDX-License-Identifier: Apache-2.0
"""Lightweight checkpoint classification shared by discovery and integrations."""

NEMOTRON_OMNI_MODEL_TYPES = frozenset(
    {"nemotron_h_nano_omni", "nemotronh_nano_omni_reasoning_v3", "nemotron_h_omni"}
)


def is_nemotron_omni(model_type: str | None) -> bool:
    return (
        isinstance(model_type, str) and model_type.lower() in NEMOTRON_OMNI_MODEL_TYPES
    )


def is_nemotron_modelopt_source(config: dict) -> bool:
    quant = config.get("quantization_config") or {}
    if not isinstance(quant, dict) or not is_nemotron_omni(config.get("model_type")):
        return False
    return (
        quant.get("quant_method") in {"modelopt", "modelopt_mixed"}
        or (
            quant.get("quant_method") == "fp8"
            and quant.get("moe_quant_algo") == "NVFP4"
        )
    ) and quant.get("quant_algo") in {"NVFP4", "W4A16_NVFP4", "MIXED_PRECISION"}


def normalize_nemotron_config(config: dict) -> dict:
    """Expose native aliases and layer geometry without importing an engine."""
    if not is_nemotron_omni(config.get("model_type")):
        return config
    result = dict(config)
    result["model_type"] = config["model_type"].lower()
    text = dict(result.get("text_config") or result.pop("llm_config", None) or {})
    result.pop("llm_config", None)
    pattern = (
        text.get("block_configs")
        or text.get("hybrid_override_pattern")
        or text.get("layers_block_type")
    )
    if pattern:
        text["num_hidden_layers"] = len(pattern)
    if result.get("vision_final_layernorm") is None:
        result["vision_final_layernorm"] = (
            int(text.get("num_nextn_predict_layers", 0) or 0) > 0
        )
    result["text_config"] = text
    return result


# Known embedding model types served by mlx-vlm or oMLX native loaders
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
