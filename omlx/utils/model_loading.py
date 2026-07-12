# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
from mlx.utils import tree_flatten

from .safetensors_shards import select_safetensors_weight_files

logger = logging.getLogger(__name__)

_VLM_TEXT_PREFIX = "language_model."
# HF/checkpoint order vs runtime module-tree order for the VLM text stack.
# ``sanitize`` swaps the former to the latter; class_predicate matches the latter.
_CKPT_TEXT_PREFIX = "model.language_model."
_RUNTIME_TEXT_PREFIX = "language_model.model."
_JANG_CONFIG_FILENAMES = (
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
)
_JANG_SUPPORTED_BITS = (8, 6, 5, 4, 3, 2)
_JANG_COMMON_GROUP_SIZES = (16, 32, 64, 96, 128, 256)
_JANG_QUANT_SHAPE_CANDIDATES = tuple(
    (bits, group_size)
    for bits in _JANG_SUPPORTED_BITS
    for group_size in _JANG_COMMON_GROUP_SIZES
)
_JANG_QUANT_METADATA_KEYS = {
    "actual_bits",
    "actual_bits_per_weight",
    "actual_bit_width",
    "actual_bit_widths",
    "actualbitsperweight",
    "actualbitwidth",
    "actualbitwidths",
    "actualbits",
    "backend",
    "attention_bits",
    "attentionbits",
    "bit_width",
    "bit_widths_used",
    "bitwidth",
    "bitwidthsused",
    "bits",
    "block_size",
    "blocksize",
    "codebook_vq",
    "cca_conv_bits",
    "ccaconvbits",
    "embed_bits",
    "embedbits",
    "effective_bits",
    "effective_bits_per_weight",
    "effectivebits",
    "effectivebitsperweight",
    "expert_layout",
    "expertlayout",
    "family",
    "lm_head_bits",
    "lmheadbits",
    "mxtqbits",
    "mxtqbitsbyrole",
    "mxtqdownbits",
    "mxtqgateupbits",
    "group_size",
    "groupsize",
    "linear_class",
    "method",
    "mode",
    "mxtq_bits",
    "mxtq_bits_by_role",
    "mxtq_down_bits",
    "mxtq_gate_up_bits",
    "mxtq_seed",
    "norms_residual_bits",
    "normsresidualbits",
    "norm_convention",
    "profile",
    "quant_method",
    "quantization_scheme",
    "quantization_backend",
    "quantization_mode",
    "router_bits",
    "routerbits",
    "routed_expert_bit_plan",
    "routed_expert_bit_plans",
    "routed_expert_group_size",
    "routed_expert_layer_bits",
    "routed_expert_bits",
    "routed_experts",
    "routedexpertbitplan",
    "routedexpertbitplans",
    "routedexpertbits",
    "routedexpertgroupsize",
    "routedexpertlayerbits",
    "routed_layer_bits",
    "routedlayerbits",
    "scoring_method",
    "selective_passthrough",
    "target_bits",
    "target_bit_width",
    "tier_bits",
    "tied_embedding",
    "top_level_default",
    "topleveldefault",
    "turboquant",
    "weight_format",
    "weightformat",
}
JangCodec = Literal["affine_jang", "mxfp", "jangtq", "unknown_jang"]


@dataclass(frozen=True)
class JangQuantizationMetadata:
    """Normalized metadata for a JANG-family sidecar and adjacent model config."""

    model_path: Path
    sidecar_path: Path
    sidecar: dict[str, Any]
    model_config: dict[str, Any] | None
    codec: JangCodec
    markers: tuple[str, ...]
    profile: str | None
    target_bits: Any | None
    actual_bits: Any | None
    block_size: Any | None
    group_size: Any | None
    mxtq_bits: Any | None
    model_family: str | None
    capabilities: dict[str, Any] | None
    chat: dict[str, Any] | None
    cache_type: str | None
    cache_subtype: str | None
    draft_strategy: str | None
    drafter_path: str | None
    draft_branching_budget: int | None
    draft_block_size: int | None
    reasoning_parser: str | None
    tool_parser: str | None
    think_in_template: bool | None
    supports_thinking: bool | None
    supports_tools: bool | None
    supports_text: bool | None
    supports_vision: bool | None
    supports_video: bool | None
    supports_audio: bool | None
    drop_earlier_reasoning: bool | None
    chat_encoder: str | None
    chat_has_tokenizer_template: bool | None
    chat_bos_token: str | None
    chat_bos_token_id: int | None
    chat_eos_token: str | None
    chat_eos_token_id: int | None
    chat_role_tokens: dict[str, str] | None
    chat_reasoning_modes: tuple[str, ...] | None
    chat_reasoning_default_mode: str | None
    chat_reasoning_thinking_start: str | None
    chat_reasoning_thinking_end: str | None
    chat_reasoning_effort_levels: tuple[str | None, ...] | None
    chat_tool_dsml_token: str | None
    chat_tool_calls_block: str | None
    chat_tool_invoke_block: str | None
    chat_tool_parameter_block: str | None
    chat_tool_output_tag: str | None
    sampling_temperature: float | None
    sampling_top_p: float | None
    sampling_max_new_tokens: int | None
    routed_expert_bit_plans: tuple[Any, ...]
    is_vlm: bool
    is_kimi_vlm: bool
    is_deepseek_v4: bool
    bundle_path: Path | None = None
    embedded_sidecar: bool = False


_MLX_LM_LOAD_CONFIG_PATCHED = False
_DSV4_FLASH_REQUIRED_ROPE_SCALING = {
    "type": "yarn",
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "beta_fast": 32,
    "beta_slow": 1,
}
_DSV4_CRITICAL_CONTROL_RE = re.compile(
    r"^(hc_head_(?:fn|base|scale)|"
    r"layers\.\d+\.hc_(?:attn|ffn)_(?:fn|base|scale)|"
    r"layers\.\d+\.attn\.attn_sink|"
    r"layers\.\d+\.ffn\.gate\.bias)$"
)
_DSV4_SWITCH_MLP_TQ_BITS_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.switch_mlp\." r"(gate_proj|up_proj|down_proj)\.tq_bits$"
)
_MTP_WEIGHT_RE = re.compile(r"(^|\.)(?:mtp|mtp_layers|nextn|next_n)(?:\.|_)")
_DSV4_FAST_LOAD_DISABLE_ENV = "JANGTQ_DISABLE_DSV4_FAST_LOAD"
_DSV4_FAST_LOAD_NOTICE_LOGGED = False
_JANG_ATTENTION_QUANT_KEY_RE = re.compile(
    r"(\.(?:q|k|v|o)_proj$|"
    r"\.(?:q_a|q_b|kv_a|kv_b)_proj(?:_with_mqa)?$|"
    r"\.(?:wq|wk|wv|wo)$|"
    r"\.attention\.[^.]+$)"
)
_JANG_ROUTED_QUANT_KEY_RE = re.compile(
    r"(\.switch_mlp\.|\.experts(?:\.|$)|\.mlp\.experts\.|"
    r"\.routed_experts?\.|\.expert_\d+\.)"
)
_JANG_HIGH_BIT_QUANT_KEY_RE = re.compile(
    r"(^embed_tokens$|\.embed_tokens$|^lm_head$|\.lm_head$|"
    r"\.shared_experts?\.|\.shared_expert\.)"
)


# mlx_lm.load dropped trust_remote_code in some releases. Check once at
# import time so call sites can pass it safely across versions.
def _mlx_lm_load_accepts_trust_remote_code() -> bool:
    try:
        from mlx_lm import load as _lm_load

        return "trust_remote_code" in inspect.signature(_lm_load).parameters
    except Exception:
        return False


_LM_LOAD_ACCEPTS_TRC = _mlx_lm_load_accepts_trust_remote_code()


def lm_load_compat(path_or_repo: str, *, trust_remote_code: bool = False, **kwargs):
    """Wrapper around mlx_lm.load that forwards trust_remote_code only when supported."""
    from mlx_lm import load

    if _LM_LOAD_ACCEPTS_TRC:
        kwargs["trust_remote_code"] = trust_remote_code
    return load(path_or_repo, **kwargs)


def expand_per_layer_quant_keys(cfg: dict) -> dict:
    """Add module-tree-path variants of per-layer quantization keys.

    mlx-lm's ``nn.quantize`` class_predicate matches the runtime module-tree
    path directly (``if p in config["quantization"]``), but oQ / HF
    checkpoints key per-layer overrides by other conventions:

    - bare safetensors tensor base name (``"lm_head"``), which the VLM text
      tree nests under ``language_model.`` (``"language_model.lm_head"``).
    - HF checkpoint order ``model.language_model.layers.N.*``, which
      ``sanitize`` swaps to module-tree order
      ``language_model.model.layers.N.*``.

    Without the matching variant the lookup misses, the global bits are used,
    and the layer is built at the wrong bit-width.

    Mutates *cfg* in place and returns it for convenience.
    """
    for config_key in ("quantization", "quantization_config"):
        quant = cfg.get(config_key)
        if not isinstance(quant, dict):
            continue
        extras: dict[str, dict] = {}
        for key, val in quant.items():
            if not isinstance(val, dict):
                continue
            if key.startswith(_CKPT_TEXT_PREFIX):
                # model.language_model.X -> language_model.model.X
                variant = _RUNTIME_TEXT_PREFIX + key[len(_CKPT_TEXT_PREFIX) :]
            elif key.startswith(_VLM_TEXT_PREFIX):
                # language_model.X -> X
                variant = key[len(_VLM_TEXT_PREFIX) :]
            else:
                # X -> language_model.X
                variant = _VLM_TEXT_PREFIX + key
            if variant not in quant and variant not in extras:
                extras[variant] = val
        if extras:
            quant.update(extras)
    return cfg


def expand_glm_moe_dsa_fused_quant_keys(cfg: dict) -> dict:
    """Add quantization specs for GLM DSA fused MoE gate/up layers.

    The oMLX GLM DSA patch fuses ``switch_mlp.gate_proj`` and
    ``switch_mlp.up_proj`` into ``switch_mlp.gate_up_proj``.  mlx-lm's loader
    chooses a module's quantizer from ``config["quantization"][path]`` before
    falling back to the global quantization settings.  GLM-5.1-MXFP4-Q8 ships
    per-layer MXFP4 specs for the split gate/up modules, but no fused path
    entry, so the fallback incorrectly quantizes ``gate_up_proj`` as affine and
    strict loading asks for missing ``gate_up_proj.biases`` tensors.

    Mutates *cfg* in place and returns it for convenience.
    """
    if cfg.get("model_type") != "glm_moe_dsa":
        return cfg

    for config_key in ("quantization", "quantization_config"):
        quant = cfg.get(config_key)
        if not isinstance(quant, dict):
            continue

        extras: dict[str, dict] = {}
        for gate_path, gate_spec in list(quant.items()):
            if not gate_path.endswith(".mlp.switch_mlp.gate_proj"):
                continue
            if not isinstance(gate_spec, dict):
                continue

            base_path = gate_path[: -len(".gate_proj")]
            up_path = f"{base_path}.up_proj"
            fused_path = f"{base_path}.gate_up_proj"
            if fused_path in quant:
                continue

            up_spec = quant.get(up_path)
            if isinstance(up_spec, dict) and up_spec == gate_spec:
                extras[fused_path] = dict(gate_spec)

        if extras:
            quant.update(extras)

    return cfg


def _patch_mlx_lm_load_config() -> None:
    """Wrap ``mlx_lm.utils.load_config`` to expand per-layer quant keys."""
    global _MLX_LM_LOAD_CONFIG_PATCHED
    if _MLX_LM_LOAD_CONFIG_PATCHED:
        return

    try:
        import mlx_lm.utils as _lu
    except ImportError:
        return

    _original = _lu.load_config

    def _patched(model_path, *args, **kwargs):
        cfg = _original(model_path, *args, **kwargs)
        expand_per_layer_quant_keys(cfg)
        expand_glm_moe_dsa_fused_quant_keys(cfg)
        patch_jangtq_weight_format_from_artifacts(cfg, Path(model_path))
        patch_jang_quantization_from_shapes(cfg, Path(model_path))
        return cfg

    _lu.load_config = _patched
    _MLX_LM_LOAD_CONFIG_PATCHED = True


def maybe_apply_pre_load_patches(
    model_name: str,
    model_settings: Any | None = None,
    for_vlm: bool = False,
) -> None:
    """Apply patches that need to run *before* mlx_lm.load() runs.

    Dispatches:

    - DeepSeek V4 patch (PR 1192) when ``config.json`` declares a
      ``deepseek_v4*`` model_type.
    - Step 3.7 Flash text-only wrapper (PR 1325) when ``config.json``
      declares ``model_type == "step3p7"``.
    - Llama 4 attention offset patch when ``config.json`` declares
      ``model_type == "llama4"`` directly or under ``text_config``.
    - GLM-5.2 ``glm_moe_dsa`` patch (mlx-lm PR 1410) when ``config.json``
      declares ``model_type == "glm_moe_dsa"``. Required because pinned
      mlx-lm exposes it as a bare DeepSeek-V3.2 subclass and cannot load
      checkpoints whose shared DSA layers carry no indexer weights.
    - Native MTP patch (PR 990 + PR 15) when the config declares MTP heads
      on a supported model_type. Always applied for sanitize correctness;
      head attachment is gated by ``model_settings.mtp_enabled``.
    - mlx-vlm side MTP runtime + nested-visual patches when ``for_vlm`` is
      True. Required so persisted ``mtp.*`` weights can bind to the
      LanguageModel tree even when ``mtp_enabled`` is False (otherwise
      strict load fails on a Qwen3.6 *-mtp VLM and the engine falls back
      to LLM, losing vision). VLMBatchedEngine passes ``for_vlm=True``;
      BatchedEngine / DFlashEngine / LLM loaders keep the default.
    - mlx-vlm MoE VLM sanitize patch when ``for_vlm`` is True and the
      checkpoint is a Qwen3.6 MoE VLM without declared MTP heads.
      Pre-converted mlx-lm exports ship ``switch_mlp`` weights; stock
      mlx-vlm ``sanitize`` unconditionally pops ``experts.gate_up_proj``
      and crashes with KeyError unless the mlx_vlm_mtp sanitize replacement
      is installed first. ``for_vlm=True`` is only passed by
      ``VLMBatchedEngine``, so no separate ``vision_config`` gate is needed.
    Both patches inject modules into ``sys.modules`` and replace mlx-lm
    internals; gating keeps non-affected models at zero cost.

    Safe to call repeatedly; the patches are idempotent.
    """
    # Reset the process-wide MTP flag so non-MTP-compatible models (or
    # models with mtp_enabled=False) are not polluted by a prior model
    # load that left the flag True.
    from ..patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(False)

    _patch_mlx_lm_load_config()

    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for pre-load patch dispatch: %s", config_path, e
        )
        return

    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.startswith("deepseek_v4"):
        from ..patches.deepseek_v4 import apply_deepseek_v4_patch

        if apply_deepseek_v4_patch():
            logger.info("DeepSeek V4 pre-load patch applied for %s", model_name)

    if model_type == "step3p7":
        from ..patches.step3p7 import apply_step3p7_patch

        if apply_step3p7_patch():
            logger.info("Step 3.7 pre-load patch applied for %s", model_name)

    if model_type == "hy_v3":
        from ..patches.hy_v3 import apply_hy_v3_patch

        if apply_hy_v3_patch():
            logger.info("Hy3 pre-load patch applied for %s", model_name)

    text_config = config.get("text_config")
    text_model_type = (
        text_config.get("model_type") if isinstance(text_config, dict) else None
    )
    if model_type == "llama4" or text_model_type == "llama4":
        from ..patches.llama4_attention import apply_llama4_attention_patch

        if apply_llama4_attention_patch():
            logger.info("Llama 4 attention patch applied for %s", model_name)

    if model_type == "glm_moe_dsa":
        from ..patches.glm_moe_dsa import apply_glm_moe_dsa_patch

        if apply_glm_moe_dsa_patch():
            logger.info("GLM MoE DSA pre-load patch applied for %s", model_name)

    minimax_m3_types = {"minimax_m3", "minimax_m3_vl"}
    if for_vlm and (
        model_type in minimax_m3_types or text_model_type in minimax_m3_types
    ):
        from ..patches.mlx_vlm_minimax_m3_compat import (
            apply_mlx_vlm_minimax_m3_compat_patch,
        )

        if apply_mlx_vlm_minimax_m3_compat_patch():
            logger.info(
                "MiniMax M3 mlx-vlm compatibility patch applied for %s",
                model_name,
            )

        from ..patches.minimax_m3_sparse_attention import (
            apply_minimax_m3_sparse_attention_patch,
        )

        if apply_minimax_m3_sparse_attention_patch():
            logger.info(
                "MiniMax M3 sparse attention patch applied for %s",
                model_name,
            )

    # Apply the MTP patch whenever the model has MTP heads on a compatible
    # model_type — even when mtp_enabled is False. The patch is required
    # for *sanitize correctness*: stock mlx-lm Model.sanitize triggers a
    # +1 norm shift whenever it sees mtp.* keys (assuming a raw HF
    # checkpoint), which double-shifts an already-converted MLX model and
    # corrupts the output (garbage tokens). PR 990's sanitize gates the
    # shift on "unsanitized conv1d" instead.
    #
    # Whether the model actually attaches an MTP head — and therefore
    # whether BatchGenerator runs the MTP draft+verify cycle — is gated
    # by a process-wide flag set just before mlx_lm.load() runs. With
    # mtp_enabled=False the patch is still active so sanitize behaves
    # correctly, but Model.__init__ skips ``self.mtp = MTPModule(args)``;
    # the resulting model is indistinguishable from a stock model that
    # never had MTP heads.
    jang_dropped_mtp = _jang_sidecar_declares_mtp_dropped(config_path.parent)
    jang_mtp_runtime_blocked = _jang_sidecar_declares_mtp_runtime_blocked(
        config_path.parent
    )
    mtp_config = _config_without_mtp_heads(config) if jang_dropped_mtp else config

    if _is_mtp_compatible(mtp_config, model_type):
        mtp_enabled = bool(
            model_settings is not None and getattr(model_settings, "mtp_enabled", False)
        )
        if jang_mtp_runtime_blocked and mtp_enabled:
            logger.warning(
                "mtp_enabled=True for %s but JANG metadata blocks native MTP "
                "after runtime validation; loading retained MTP weights without "
                "speculative execution",
                model_name,
            )
            mtp_enabled = False
        from ..patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            set_mtp_active,
            set_mtp_depth,
        )

        if apply_mlx_lm_mtp_patch():
            set_mtp_active(mtp_enabled)
            # mtp_num_draft_tokens is the MAX draft depth; an adaptive
            # controller picks 1..max per sequence from rolling accept/latency
            # estimates, so prose/chat settles at 1 and predictable text
            # climbs. Set it to 1 for a fixed depth-1 cycle. Note: depth >= 2
            # verify forwards route through the verify-shape qmm kernels
            # (M >= 3), whose numerics can diverge from the unrouted path at
            # bf16 tail-ULP level.
            depth = getattr(model_settings, "mtp_num_draft_tokens", None)
            set_mtp_depth(int(depth) if depth else 3)
            if mtp_enabled:
                logger.info(
                    "Native MTP patch applied for %s (model_type=%s, active)",
                    model_name,
                    model_type,
                )
            else:
                logger.debug(
                    "Native MTP patch applied for %s for sanitize correctness "
                    "(model has MTP heads but mtp_enabled=False; head not attached)",
                    model_name,
                )

        # mlx-vlm side: only relevant when entering through VLMBatchedEngine
        # (e.g. ``qwen3_5_moe`` with vision_config). The mlx-lm patch alone
        # can't attach an MTP head to the mlx-vlm classes — apply the
        # parallel runtime patch so MTPModule is instantiated on
        # ``LanguageModel.__init__``.
        #
        # Applied regardless of ``mtp_enabled``: with MTP off, persisted
        # ``mtp.*`` weights still need a binding site on the language model
        # tree or mlx-vlm's strict load_weights fails with "parameters not
        # in model" (issue #1404). MTP decode invocation stays gated by
        # ``is_mtp_active()`` downstream, so MTP off + module attached
        # behaves identically to a stock no-MTP model at inference time
        # (with a small constant memory cost for the unused MTPModule).
        #
        # ``for_vlm=False`` skips this branch on BatchedEngine / DFlashEngine
        # paths so mlx-vlm classes are not touched when the load goes
        # through mlx-lm only.
        if for_vlm:
            try:
                from ..patches.mlx_vlm_mtp import (
                    apply_mlx_vlm_mtp_patch,
                    apply_mlx_vlm_mtp_runtime_patch,
                    set_mtp_attach_enabled,
                )
            except Exception:
                pass
            else:
                # Decide attach-vs-skip BEFORE applying the runtime patch
                # because the patch wraps ``LanguageModel.__init__`` which
                # reads the flag at instantiation. Some Qwen3.6 MoE VLM
                # exports (unsloth UD MLX builds, issue #1426) declare
                # ``mtp_num_hidden_layers > 0`` in config.json but ship no
                # ``mtp.*`` weights; attaching MTPModule there causes
                # strict load_weights to fail with "Missing N parameters"
                # and silently downgrade the engine to LLM, dropping
                # vision. Scan the index for actual mtp.* keys and skip
                # attachment when they're absent.
                has_mtp_weights = _checkpoint_has_mtp_weights(model_name)
                set_mtp_attach_enabled(has_mtp_weights)

                # Sanitize-preservation patch runs unconditionally: the
                # stock mlx-vlm Model.sanitize strips every ``mtp.*`` key,
                # so without this an MTP head with persisted weights would
                # load at random init (0% accept). When mtp.* weights are
                # absent the patch is a no-op on the affected paths.
                if apply_mlx_vlm_mtp_patch():
                    if mtp_enabled:
                        logger.info(
                            "mlx-vlm MTP sanitize patch applied for %s",
                            model_name,
                        )
                    else:
                        logger.debug(
                            "mlx-vlm MTP sanitize patch applied for %s "
                            "(mtp_enabled=False; allows persisted mtp.* "
                            "weights to bind)",
                            model_name,
                        )
                if apply_mlx_vlm_mtp_runtime_patch():
                    if not has_mtp_weights:
                        logger.info(
                            "mlx-vlm runtime MTP patch applied for %s "
                            "(config declares mtp heads but checkpoint "
                            "ships no mtp.* weights; MTPModule attachment "
                            "skipped to keep strict load_weights happy)",
                            model_name,
                        )
                    elif mtp_enabled:
                        logger.info(
                            "mlx-vlm runtime MTP patch applied for %s",
                            model_name,
                        )
                    else:
                        logger.debug(
                            "mlx-vlm runtime MTP patch applied for %s "
                            "(mtp_enabled=False; head attached for weight "
                            "load only)",
                            model_name,
                        )
    elif model_settings is not None and getattr(model_settings, "mtp_enabled", False):
        if jang_dropped_mtp:
            logger.warning(
                "mtp_enabled=True for %s but JANG metadata marks MTP weights as "
                "dropped; MTP path will be inactive",
                model_name,
            )
        else:
            logger.warning(
                "mtp_enabled=True for %s but model is incompatible "
                "(model_type=%r, mtp_heads=%s); MTP path will be inactive",
                model_name,
                model_type,
                _has_mtp_heads(config),
            )

    # Pre-converted mlx-lm Qwen3.6 MoE VLMs (e.g. mlx-community mxfp4) ship
    # switch_mlp weights under language_model.model.* and often declare
    # mtp_num_hidden_layers=0. The mlx_vlm_mtp sanitize replacement skips
    # unfuse when switch_mlp is already present; stock mlx-vlm sanitize
    # unconditionally pops experts.gate_up_proj and VLM load fails with
    # KeyError → LLM fallback (vision silently dropped, issue #1261). That
    # sanitize patch was previously only wired through _is_mtp_compatible
    # above; apply it here for non-MTP MoE VLMs. Runtime MTP patch stays in
    # the branch above.
    if (
        for_vlm
        and model_type
        and model_type.startswith("qwen3_5_moe")
        and not _is_mtp_compatible(mtp_config, model_type)
    ):
        try:
            from ..patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch
        except Exception as e:
            logger.debug("qwen3_6 MoE VLM sanitize patch import failed: %s", e)
        else:
            if apply_mlx_vlm_mtp_patch():
                logger.debug(
                    "mlx-vlm qwen3_6 MoE VLM sanitize patch applied for %s "
                    "(no MTP heads; switch_mlp load correctness)",
                    model_name,
                )

    # qwen3_5_moe covers Qwen3.6 too (HF config sets model_type=qwen3_5_moe).
    # The nested-visual sanitize wrap remaps language_model.model.visual.*
    # to vision_tower.* for Qwen3.6's nested ViT layout. Wraps whichever
    # Model.sanitize is current (stock mlx-vlm or mlx_vlm_mtp runtime), so
    # the call has to land after apply_mlx_vlm_mtp_runtime_patch above.
    # VLM-only: dflash / mlx-lm paths never instantiate mlx-vlm classes,
    # so touching them there is just dead weight.
    if for_vlm and model_type and model_type.startswith("qwen3_5_moe"):
        try:
            from ..patches.qwen3_6_nested_visual import (
                apply_qwen3_6_nested_visual_patch,
            )
        except Exception as e:
            logger.debug("qwen3_6 nested-visual patch import failed: %s", e)
        else:
            if apply_qwen3_6_nested_visual_patch():
                logger.info(
                    "qwen3_6 nested-visual sanitize wrap applied for %s",
                    model_name,
                )


def _has_mtp_heads(config: dict) -> bool:
    """True iff the model config declares any MTP head layers."""
    if int(config.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(config.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    text_cfg = config.get("text_config") or {}
    if int(text_cfg.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    return int(text_cfg.get("num_nextn_predict_layers", 0) or 0) > 0


def _config_without_mtp_heads(config: dict[str, Any]) -> dict[str, Any]:
    patched = dict(config)
    for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        patched[key] = 0
    text_cfg = patched.get("text_config")
    if isinstance(text_cfg, dict):
        patched["text_config"] = dict(text_cfg)
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
            patched["text_config"][key] = 0
    return patched


def _jang_sidecar_declares_mtp_dropped(model_path: Path) -> bool:
    try:
        metadata = read_jang_metadata(model_path)
    except ValueError as exc:
        logger.debug("Could not inspect JANG MTP metadata for %s: %s", model_path, exc)
        return False
    if metadata is None:
        return False
    return _jang_metadata_declares_mtp_dropped(metadata.sidecar, metadata.model_config)


def _jang_sidecar_declares_mtp_runtime_blocked(model_path: Path) -> bool:
    """Return True when JANG metadata disables an otherwise intact MTP runtime."""

    try:
        metadata = read_jang_metadata(model_path)
    except ValueError as exc:
        logger.debug("Could not inspect JANG MTP metadata for %s: %s", model_path, exc)
        return False
    if metadata is None:
        return False
    return _jang_metadata_declares_mtp_runtime_blocked(
        metadata.sidecar, metadata.model_config
    )


def _jang_metadata_declares_mtp_dropped(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> bool:
    drop_modes = {
        "drop",
        "dropped",
        "absent",
        "missing",
        "missing_weight",
        "missing_weights",
        "metadata_only",
        "config_only",
        "config_only_missing_weights",
        "disabled",
        "metadata_only_missing_weights",
        "preserved_disabled",
        "runtime_disabled",
        "runtime_unwired",
        "unwired",
    }
    for source in (jang_config, model_config):
        if not isinstance(source, dict):
            continue
        for item in _iter_dicts(source):
            if _coerce_jang_bool(item.get("drop_mtp")) is True:
                return True
            mtp = item.get("mtp")
            if isinstance(mtp, dict) and mtp.get("kept") is False:
                return True
            runtime = item.get("runtime")
            if not isinstance(runtime, dict):
                continue
            mode = runtime.get("mtp_mode")
            normalized_mode = mode.strip().lower() if isinstance(mode, str) else None
            if normalized_mode == "absent":
                return True
            if runtime.get("bundle_has_mtp") is False and normalized_mode in drop_modes:
                return True
    return False


def _jang_metadata_declares_mtp_runtime_blocked(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> bool:
    """Return True for a measured native-MTP runtime block.

    ``runtime.native_mtp_blocked`` preserves MTP tensors for compatible
    loaders while preventing a known regression in speculative throughput.
    It is distinct from dropped-MTP metadata, which means the artifact has no
    usable MTP weights at all.
    """

    for source in (jang_config, model_config):
        if not isinstance(source, dict):
            continue
        for item in _iter_dicts(source):
            runtime = item.get("runtime")
            if not isinstance(runtime, dict):
                continue
            blocked = _mapping_value(
                runtime,
                "native_mtp_blocked",
                "nativeMtpBlocked",
            )
            blocked_bool = _coerce_jang_bool(blocked)
            if blocked_bool is True:
                return True
            if blocked_bool is False:
                continue
            if isinstance(blocked, Mapping):
                if _coerce_jang_bool(
                    _mapping_value(blocked, "blocked", "disabled")
                ) is True:
                    return True
                if _coerce_jang_bool(_mapping_value(blocked, "enabled")) is False:
                    return True
            if isinstance(blocked, str) and blocked.strip():
                return True
    return False


_MTP_WEIGHT_PREFIXES = (
    "mtp.",
    "mtp_layers.",
    "language_model.mtp.",
    "language_model.mtp_layers.",
    "model.mtp.",
    "model.mtp_layers.",
    "model.language_model.mtp.",
    "model.language_model.mtp_layers.",
)
_SAFETENSORS_INDEX_NAMES = (
    "model.safetensors.index.json",
    "consolidated.safetensors.index.json",
    "model.jang.index.json",
    "model.jjqf.index.json",
    "model.mxq.index.json",
)
_JANGTQ_RUNTIME_SIDECAR = "jangtq_runtime.safetensors"


def _iter_safetensors_index_paths(model_path: Path):
    for name in _SAFETENSORS_INDEX_NAMES:
        path = model_path / name
        if path.exists():
            yield path


def _is_mtp_weight_key(
    key: str, prefixes: tuple[str, ...] = _MTP_WEIGHT_PREFIXES
) -> bool:
    return (
        key.startswith(prefixes) or _MTP_WEIGHT_RE.search(key) is not None
    )


def _nextn_weight_prefixes(model_path: str | Path) -> tuple[str, ...]:
    """Weight-key prefixes for MTP layers stored as extra decoder layers.

    DeepSeek-V3-style checkpoints (GLM-5.2 among them) keep their MTP head
    as ``model.layers.<num_hidden_layers + i>.*`` rather than ``mtp.*``;
    the model patch's sanitize remaps them at load/convert time, so for
    detection purposes those layers count as MTP weights.
    """
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return ()
    cfgs = (config, config.get("text_config") or {})
    n_mtp = max(int(c.get("num_nextn_predict_layers", 0) or 0) for c in cfgs)
    if n_mtp <= 0:
        return ()
    n_main = max(int(c.get("num_hidden_layers", 0) or 0) for c in cfgs)
    if n_main <= 0:
        return ()
    return tuple(f"model.layers.{n_main + i}." for i in range(n_mtp))


def _checkpoint_has_mtp_weights(model_path: str | Path) -> bool:
    """True iff the checkpoint at *model_path* ships any MTP weight tensor.

    Matches both the ``mtp.*`` naming and the nextn layout (extra decoder
    layers past ``num_hidden_layers``, see ``_nextn_weight_prefixes``).

    Some Qwen3.6 MoE VLM exports declare ``mtp_num_hidden_layers > 0`` in
    ``config.json`` but strip the MTP weights during conversion (e.g.
    ``unsloth/Qwen3.6-35B-A3B-UD-MLX-*bit``). Attaching ``MTPModule`` for
    such a checkpoint causes mlx-vlm's strict ``load_weights`` to fail with
    "Missing N parameters: language_model.mtp.*", the engine falls back to
    LLM, and vision is silently dropped (issue #1426).

    Reads known safetensors index files when present (no shard I/O). Falls back
    to safetensors shard metadata headers. Returns False when neither resolves —
    callers treat that as "no MTP weights" (the conservative choice: skip
    MTPModule attachment).
    """
    p = Path(model_path)
    if not p.is_dir():
        return False

    prefixes = _MTP_WEIGHT_PREFIXES + _nextn_weight_prefixes(p)

    for index_path in _iter_safetensors_index_paths(p):
        try:
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map") or {}
            if any(_is_mtp_weight_key(str(k), prefixes) for k in weight_map):
                return True
        except Exception as e:
            logger.debug("Failed to read %s for mtp weight scan: %s", index_path, e)

    shards = select_safetensors_weight_files(p)
    if not shards:
        return False
    try:
        import safetensors
    except Exception as e:
        logger.debug("safetensors import failed for mtp weight scan: %s", e)
        return False

    for shard in shards:
        try:
            with safetensors.safe_open(str(shard), framework="numpy") as f:
                for k in f.keys():  # noqa: SIM118 - safetensors handles expose keys()
                    if _is_mtp_weight_key(str(k), prefixes):
                        return True
        except Exception as e:
            logger.debug("Failed to read %s header for mtp weight scan: %s", shard, e)
    return False


def _is_mtp_compatible(config: dict, model_type: str | None) -> bool:
    """Decide whether the native MTP patch can be applied to this model.

    Supports Qwen3.5/3.6 (mlx-lm PR 990), DeepSeek-V4-Flash (Blaizzy/mlx-lm
    fork PR 15) and GLM-5.2 (glm_moe_dsa). The model also has to declare
    MTP heads in the config; otherwise the patch is a no-op.
    """
    if not _has_mtp_heads(config):
        return False
    if not model_type:
        return False
    return (
        model_type.startswith("qwen3_5")
        or model_type.startswith("qwen3_6")
        or model_type.startswith("deepseek_v4")
        or model_type == "glm_moe_dsa"
    )


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_settings: Any | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    trust_remote_code = (
        bool(getattr(model_settings, "trust_remote_code", False))
        if model_settings is not None
        else False
    )
    return lm_load_compat(
        model_name,
        tokenizer_config=tokenizer_config,
        trust_remote_code=trust_remote_code,
    )


def _collect_mx_arrays(
    value: Any, arrays: list[mx.array], seen: dict[int, Any]
) -> None:
    """Collect mx arrays from MLX pytrees plus custom object attributes."""
    obj_id = id(value)
    if obj_id in seen:
        return
    seen[obj_id] = value

    if isinstance(value, mx.array):
        arrays.append(value)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            _collect_mx_arrays(item, arrays, seen)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_mx_arrays(item, arrays, seen)
        return

    _collect_mx_arrays_from_module_apis(value, arrays, seen)

    attrs = getattr(value, "__dict__", None)
    if attrs:
        _collect_mx_arrays(attrs, arrays, seen)


def _collect_mx_arrays_from_module_apis(
    value: Any, arrays: list[mx.array], seen: dict[int, Any]
) -> None:
    """Collect arrays exposed through MLX module APIs."""
    for name in ("parameters", "trainable_parameters", "buffers", "leaf_modules"):
        method = getattr(value, name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except TypeError:
            continue
        _collect_mx_arrays(result, arrays, seen)


def materialize_lazy_state(model: Any) -> None:
    """Force-evaluate every mx.array in the model tree on the loader thread.

    mlx-vlm's load() runs `mx.eval(model.language_model.parameters())`, which
    leaves frozen buffers (RoPE freqs and similar) plus sibling sub-trees
    (vision_tower, audio_tower) as lazy arrays bound to the loader thread's
    default stream. When a different thread (e.g. an EngineCore per-engine
    executor introduced in #1304) later runs forward, mx.eval hits "no
    Stream(gpu, X) in current thread" because those lazy ops target a stream
    that only exists on the loader thread. Materializing the whole tree here
    makes every leaf array safe to read from any thread afterwards.
    """
    arrays: list[mx.array] = []
    seen: dict[int, Any] = {}
    for item in tree_flatten(model):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], mx.array):
            _collect_mx_arrays(item[1], arrays, seen)
    _collect_mx_arrays_from_module_apis(model, arrays, seen)
    _collect_mx_arrays(model, arrays, seen)
    logger.debug("Materializing %d MLX array(s) from model state", len(arrays))
    if arrays:
        mx.eval(*arrays)


def apply_post_load_transforms(model: Any, model_settings: Any = None) -> Any:
    """Apply optional post-load model transforms based on settings.

    Currently supports:
    - IndexCache: skip redundant indexer computation in DSA layers

    Args:
        model: A loaded mlx-lm model instance.
        model_settings: A ModelSettings instance (or None).

    Returns:
        The (possibly patched) model.
    """
    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, index_cache_freq)
        if applied:
            logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model


def _find_jang_config_path(model_path: Path) -> Path | None:
    for parent in (model_path, model_path / "target"):
        for name in _JANG_CONFIG_FILENAMES:
            path = parent / name
            if path.exists():
                return path
    return None


def _read_embedded_jang_config(
    model_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(model_config, dict):
        return None
    jang_config = model_config.get("jang")
    return jang_config if isinstance(jang_config, dict) else None


def _read_jang_config(config_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(config_path.read_text())
    except Exception as e:
        raise ValueError(f"Could not read JANG config {config_path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"JANG config {config_path} must contain a JSON object")
    return data


def _read_json_object_or_none(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _mapping_value(source: Mapping[str, Any] | None, *keys: str) -> Any | None:
    if not isinstance(source, Mapping):
        return None
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _jangspec_bundle_root(model_path: Path, sidecar_path: Path) -> Path | None:
    if sidecar_path.parent.name != "target":
        return None
    root = sidecar_path.parent.parent
    if model_path not in {root, sidecar_path.parent}:
        return None
    return root if (root / "jangspec.json").is_file() else None


def _jang_payload_path(model_path: Path, sidecar_path: Path) -> Path:
    """Return the directory that should be handed to jang_tools loaders."""

    if sidecar_path.parent != model_path:
        return sidecar_path.parent
    return model_path


def _normalized_jang_loader_sidecar(
    metadata: JangQuantizationMetadata,
) -> dict[str, Any]:
    """Return sidecar metadata normalized for older ``jang_tools`` loaders."""

    sidecar = dict(metadata.sidecar)
    if metadata.codec == "jangtq" and metadata.mxtq_bits is not None:
        sidecar.setdefault("weight_format", "mxtq")
        if not any(key in sidecar for key in ("mxtq_bits", "mxtqBits")):
            sidecar["mxtq_bits"] = metadata.mxtq_bits
    return sidecar


def _materialize_embedded_jang_sidecar(metadata: JangQuantizationMetadata) -> None:
    if not (metadata.embedded_sidecar or metadata.sidecar_path.name == "config.json"):
        return
    sidecar_path = metadata.model_path / "jang_config.json"
    if sidecar_path.exists():
        return
    try:
        sidecar_path.write_text(
            json.dumps(
                _normalized_jang_loader_sidecar(metadata), indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            "JANG sidecar metadata is embedded in config.json but could not be materialized "
            f"to {sidecar_path} for jang_tools loading: {exc}"
        ) from exc


def _jang_loader_needs_normalized_sidecar(
    metadata: JangQuantizationMetadata,
) -> bool:
    """Return True when an existing sidecar needs alias backfill for loaders."""

    if metadata.bundle_path is not None:
        return False
    if metadata.sidecar_path.name == "config.json":
        return False
    if metadata.codec != "jangtq" or metadata.mxtq_bits is None:
        return False
    return not any(key in metadata.sidecar for key in ("mxtq_bits", "mxtqBits"))


def _temporary_jang_loader_path(
    metadata: JangQuantizationMetadata,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return a loader path whose sidecar uses the schema expected by jang_tools."""

    if not _jang_loader_needs_normalized_sidecar(metadata):
        return metadata.model_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="omlx-jang-loader-")
    temp_path = Path(temp_dir.name)
    for child in metadata.model_path.iterdir():
        if child.name == "jang_config.json":
            continue
        (temp_path / child.name).symlink_to(
            child,
            target_is_directory=child.is_dir(),
        )
    (temp_path / "jang_config.json").write_text(
        json.dumps(_normalized_jang_loader_sidecar(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return temp_path, temp_dir


def _retain_jang_loader_temp_dir(
    result: tuple[Any, Any],
    temp_dir: tempfile.TemporaryDirectory[str] | None,
) -> tuple[Any, Any]:
    if temp_dir is None:
        return result
    model, processor = result
    for owner in (model, processor):
        try:
            owner._omlx_jang_loader_temp_dir = temp_dir
            return result
        except AttributeError:
            continue
    logger.warning(
        "Could not retain temporary normalized JANG loader directory; "
        "loaded artifacts may rely on already-open file handles."
    )
    return result


def _normalize_jang_marker(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    normalized = "".join(ch if ch.isalnum() else "_" for ch in text)
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or None


def _canonical_jang_marker(marker: str) -> str:
    return marker.replace("_", "")


def _add_jang_marker(markers: list[str], value: Any) -> None:
    if isinstance(value, str):
        marker = _normalize_jang_marker(value)
        if marker and marker not in markers:
            markers.append(marker)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _add_jang_marker(markers, item)


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_dicts(nested)


def _has_jang_v2_metadata_signal(jang_config: dict[str, Any]) -> bool:
    """Return True for modern JANG v2 sidecars that omit legacy format markers."""

    version = _mapping_value(
        jang_config,
        "version",
        "format_version",
        "formatVersion",
        "jang_version",
        "jangVersion",
    )
    if version is not None:
        return True

    quantization = jang_config.get("quantization")
    if not isinstance(quantization, dict):
        quantization = jang_config.get("quantization_config")
    if not isinstance(quantization, dict):
        return False

    return any(
        _mapping_value(quantization, *keys) is not None
        for keys in (
            ("bits", "bit_width", "bitWidth", "target_bits", "targetBits"),
            ("group_size", "groupSize", "block_size", "blockSize"),
            ("bit_widths_used", "bitWidthsUsed"),
        )
    )


def _first_nested_value(sources: tuple[Any, ...], keys: set[str]) -> Any | None:
    normalized_keys = {_normalize_jang_marker(key) or key for key in keys}
    for source in sources:
        for item in _iter_dicts(source):
            for key, value in item.items():
                normalized_key = _normalize_jang_marker(key)
                if normalized_key in normalized_keys and value is not None:
                    return value
    return None


def _all_nested_values(sources: tuple[Any, ...], keys: set[str]) -> tuple[Any, ...]:
    normalized_keys = {_normalize_jang_marker(key) or key for key in keys}
    values: list[Any] = []
    for source in sources:
        for item in _iter_dicts(source):
            for key, value in item.items():
                normalized_key = _normalize_jang_marker(key)
                if normalized_key in normalized_keys and value is not None:
                    values.append(value)
    return tuple(values)


def _first_nested_string(sources: tuple[Any, ...], keys: set[str]) -> str | None:
    value = _first_nested_value(sources, keys)
    return value if isinstance(value, str) else None


def _collect_jang_markers(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
    sidecar_path: Path,
) -> tuple[str, ...]:
    markers: list[str] = []

    stem = sidecar_path.stem
    if stem.endswith("_config") or stem.endswith("_cfg"):
        sidecar_hint = stem.rsplit("_", 1)[0]
        if sidecar_hint in {"jjqf", "mxq"}:
            _add_jang_marker(markers, sidecar_hint)

    marker_keys = {
        "format",
        "weight_format",
        "weightFormat",
        "profile",
        "jang_profile",
        "jangProfile",
    }
    nested_marker_keys = marker_keys | {"family", "mode", "method"}
    for source in (jang_config, model_config):
        if not isinstance(source, dict):
            continue
        for key in marker_keys:
            _add_jang_marker(markers, source.get(key))
        quantization = source.get("quantization")
        if isinstance(quantization, dict):
            for key in nested_marker_keys:
                _add_jang_marker(markers, quantization.get(key))
        quantization_config = source.get("quantization_config")
        if isinstance(quantization_config, dict):
            for key in nested_marker_keys | {"quant_method"}:
                _add_jang_marker(markers, quantization_config.get(key))
        text_config = source.get("text_config")
        if isinstance(text_config, dict):
            for key in marker_keys:
                _add_jang_marker(markers, text_config.get(key))
            for nested in ("quantization", "quantization_config"):
                nested_quantization = text_config.get(nested)
                if isinstance(nested_quantization, dict):
                    for key in nested_marker_keys | {"quant_method"}:
                        _add_jang_marker(markers, nested_quantization.get(key))
        metadata = source.get("metadata")
        if isinstance(metadata, dict):
            for item in _iter_dicts(metadata):
                for key in nested_marker_keys:
                    _add_jang_marker(markers, item.get(key))

    if not markers and _has_jang_v2_metadata_signal(jang_config):
        _add_jang_marker(markers, "jang")

    return tuple(markers)


def _jang_model_family(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> str | None:
    """Return the JANG-stamped model-family hint when present."""

    sources = (jang_config, model_config)
    for source in (jang_config, model_config):
        if not isinstance(source, dict):
            continue
        family = source.get("model_family")
        if isinstance(family, str) and family.strip():
            return family
        family = source.get("family")
        if isinstance(family, str) and family.strip():
            return family
        capabilities = source.get("capabilities")
        if isinstance(capabilities, dict):
            family = capabilities.get("family")
            if isinstance(family, str) and family.strip():
                return family

    for source in sources:
        if not isinstance(source, dict):
            continue
        source_model = _mapping_value(source, "source_model", "sourceModel")
        if isinstance(source_model, dict):
            family = source_model.get("architecture") or source_model.get("family")
            if isinstance(family, str) and family.strip():
                return family

    for source in sources:
        if not isinstance(source, dict):
            continue
        architecture = source.get("architecture")
        if isinstance(architecture, dict):
            family = architecture.get("type") or architecture.get("text_model_type")
            if isinstance(family, str) and family.strip():
                return family

    for source in sources:
        if not isinstance(source, dict):
            continue
        text_config = source.get("text_config")
        if isinstance(text_config, dict):
            family = text_config.get("model_type")
            if isinstance(family, str) and family.strip():
                return family

    for source in sources:
        if not isinstance(source, dict):
            continue
        family = source.get("model_type")
        if isinstance(family, str) and family.strip():
            return family
    return None


def _jang_capabilities(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the first JANG capabilities block stamped by converter metadata."""

    for source in (jang_config, model_config):
        if not isinstance(source, dict):
            continue
        capabilities = source.get("capabilities")
        if isinstance(capabilities, dict):
            return capabilities
    return None


def _jang_chat(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the first JANG chat block stamped by converter metadata."""

    for source in (jang_config, model_config):
        if not isinstance(source, dict):
            continue
        chat = source.get("chat")
        if isinstance(chat, dict):
            return chat
    return None


def _nested_dict(source: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else None


def _first_string_value(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_bool_value(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _first_int_value(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_float_value(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _string_tuple_value(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    items = tuple(item for item in value if isinstance(item, str))
    return items or None


def _optional_string_tuple_value(value: Any) -> tuple[str | None, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    items: list[str | None] = []
    for item in value:
        if item is None or isinstance(item, str):
            items.append(item)
    return tuple(items) if items else None


def _string_dict_value(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    items = {str(key): val for key, val in value.items() if isinstance(val, str)}
    return items or None


_JANG_MEDIA_MODALITIES = {
    "audio",
    "image",
    "images",
    "image_text",
    "multimodal",
    "video",
    "vision",
    "vision_language",
    "visual",
    "vl",
    "vlm",
}
_JANG_TEXT_RUNTIME_STATUSES = {
    "weights_preserved_text_runtime",
    "preserved_weights_text_runtime",
    "preserved_disabled",
    "text_only",
    "text_runtime",
    "unwired",
}


def _coerce_jang_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _jang_modalities_declare_active_media(modalities: Any) -> bool:
    if isinstance(modalities, str):
        marker = _normalize_jang_marker(modalities)
        return marker in _JANG_MEDIA_MODALITIES
    if isinstance(modalities, dict):
        for key, value in modalities.items():
            marker = _normalize_jang_marker(key)
            if marker in _JANG_MEDIA_MODALITIES and _coerce_jang_bool(value) is True:
                return True
        return False
    if isinstance(modalities, (list, tuple, set)):
        return any(
            (
                (_normalize_jang_marker(item) in _JANG_MEDIA_MODALITIES)
                if isinstance(item, str)
                else False
            )
            for item in modalities
        )
    return False


def _jang_modalities_declare_no_active_media(modalities: Any) -> bool:
    if not isinstance(modalities, dict):
        return False
    saw_disabled_media = False
    for key, value in modalities.items():
        marker = _normalize_jang_marker(key)
        if marker not in _JANG_MEDIA_MODALITIES:
            continue
        flag = _coerce_jang_bool(value)
        if flag is True:
            return False
        if flag is False:
            saw_disabled_media = True
    return saw_disabled_media


def _jang_config_declares_active_media(jang_config: dict[str, Any]) -> bool:
    if _coerce_jang_bool(_mapping_value(jang_config, "has_audio", "hasAudio")) is True:
        return True
    if _coerce_jang_bool(_mapping_value(jang_config, "has_image", "hasImage")) is True:
        return True
    if (
        _coerce_jang_bool(_mapping_value(jang_config, "has_vision", "hasVision"))
        is True
    ):
        return True
    if _coerce_jang_bool(_mapping_value(jang_config, "has_video", "hasVideo")) is True:
        return True
    return _jang_modalities_declare_active_media(
        _mapping_value(jang_config, "modalities", "modality")
    )


def _jang_cache_subtype(
    capabilities: dict[str, Any] | None,
    sources: tuple[Any, ...],
) -> str | None:
    runtime = _first_nested_value(sources, {"runtime"})
    runtime = runtime if isinstance(runtime, dict) else None
    cache_topology = _nested_dict(runtime, "cache_topology")
    return _first_string_value(
        capabilities.get("cache_subtype") if isinstance(capabilities, dict) else None,
        capabilities.get("cacheSubType") if isinstance(capabilities, dict) else None,
        capabilities.get("cacheSubtype") if isinstance(capabilities, dict) else None,
        runtime.get("cache_subtype") if runtime else None,
        runtime.get("cacheSubtype") if runtime else None,
        cache_topology.get("family") if cache_topology else None,
    )


def _jang_draft_block(capabilities: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(capabilities, dict):
        return None
    for key in (
        "draft",
        "drafter",
        "speculative",
        "speculative_decoding",
        "speculativeDecoding",
    ):
        value = capabilities.get(key)
        if isinstance(value, dict):
            return value
    return None


def _jang_draft_strategy(capabilities: dict[str, Any] | None) -> str | None:
    draft = _jang_draft_block(capabilities)
    return _first_string_value(
        capabilities.get("draft_strategy") if isinstance(capabilities, dict) else None,
        capabilities.get("draftStrategy") if isinstance(capabilities, dict) else None,
        draft.get("strategy") if draft else None,
        draft.get("draft_strategy") if draft else None,
        draft.get("draftStrategy") if draft else None,
    )


def _jang_drafter_path(capabilities: dict[str, Any] | None) -> str | None:
    draft = _jang_draft_block(capabilities)
    return _first_string_value(
        capabilities.get("drafter_path") if isinstance(capabilities, dict) else None,
        capabilities.get("drafterPath") if isinstance(capabilities, dict) else None,
        draft.get("path") if draft else None,
        draft.get("drafter_path") if draft else None,
        draft.get("drafterPath") if draft else None,
    )


def _jang_draft_branching_budget(capabilities: dict[str, Any] | None) -> int | None:
    draft = _jang_draft_block(capabilities)
    return _first_int_value(
        (
            capabilities.get("branching_budget")
            if isinstance(capabilities, dict)
            else None
        ),
        capabilities.get("branchingBudget") if isinstance(capabilities, dict) else None,
        draft.get("branching_budget") if draft else None,
        draft.get("branchingBudget") if draft else None,
    )


def _jang_draft_block_size(capabilities: dict[str, Any] | None) -> int | None:
    draft = _jang_draft_block(capabilities)
    return _first_int_value(
        capabilities.get("block_size") if isinstance(capabilities, dict) else None,
        capabilities.get("blockSize") if isinstance(capabilities, dict) else None,
        draft.get("block_size") if draft else None,
        draft.get("blockSize") if draft else None,
    )


def _jang_reasoning_parser(
    capabilities: dict[str, Any] | None,
    chat: dict[str, Any] | None,
) -> str | None:
    reasoning = _nested_dict(capabilities, "reasoning")
    chat_reasoning = _nested_dict(chat, "reasoning")
    return _first_string_value(
        (
            capabilities.get("reasoning_parser")
            if isinstance(capabilities, dict)
            else None
        ),
        capabilities.get("reasoningParser") if isinstance(capabilities, dict) else None,
        reasoning.get("parser") if reasoning else None,
        chat_reasoning.get("parser") if chat_reasoning else None,
    )


def _jang_tool_parser(
    capabilities: dict[str, Any] | None,
    chat: dict[str, Any] | None,
) -> str | None:
    tools = _nested_dict(capabilities, "tools")
    tool_calling = _nested_dict(chat, "tool_calling")
    return _first_string_value(
        tool_calling.get("parser") if tool_calling else None,
        capabilities.get("tool_parser") if isinstance(capabilities, dict) else None,
        capabilities.get("toolParser") if isinstance(capabilities, dict) else None,
        tools.get("parser") if tools else None,
    )


def _jang_think_in_template(
    capabilities: dict[str, Any] | None,
    chat: dict[str, Any] | None,
) -> bool | None:
    reasoning = _nested_dict(capabilities, "reasoning")
    chat_reasoning = _nested_dict(chat, "reasoning")
    return _first_bool_value(
        (
            capabilities.get("think_in_template")
            if isinstance(capabilities, dict)
            else None
        ),
        capabilities.get("thinkInTemplate") if isinstance(capabilities, dict) else None,
        reasoning.get("think_in_template") if reasoning else None,
        reasoning.get("thinkInTemplate") if reasoning else None,
        chat_reasoning.get("think_in_template") if chat_reasoning else None,
        chat_reasoning.get("thinkInTemplate") if chat_reasoning else None,
        chat_reasoning.get("supported") if chat_reasoning else None,
    )


def _jang_supports_thinking(
    capabilities: dict[str, Any] | None,
    chat: dict[str, Any] | None,
) -> bool | None:
    reasoning = _nested_dict(capabilities, "reasoning")
    chat_reasoning = _nested_dict(chat, "reasoning")
    return _first_bool_value(
        (
            capabilities.get("supports_thinking")
            if isinstance(capabilities, dict)
            else None
        ),
        (
            capabilities.get("supportsThinking")
            if isinstance(capabilities, dict)
            else None
        ),
        reasoning.get("supported") if reasoning else None,
        chat_reasoning.get("supported") if chat_reasoning else None,
    )


def _jang_supports_tools(
    capabilities: dict[str, Any] | None,
    chat: dict[str, Any] | None,
) -> bool | None:
    tools = _nested_dict(capabilities, "tools")
    tool_calling = _nested_dict(chat, "tool_calling")
    return _first_bool_value(
        capabilities.get("supports_tools") if isinstance(capabilities, dict) else None,
        capabilities.get("supportsTools") if isinstance(capabilities, dict) else None,
        tools.get("supported") if tools else None,
        tool_calling.get("supported") if tool_calling else None,
    )


def _jang_supports_text(capabilities: dict[str, Any] | None) -> bool | None:
    return _first_bool_value(
        capabilities.get("supports_text") if isinstance(capabilities, dict) else None,
        capabilities.get("supportsText") if isinstance(capabilities, dict) else None,
    )


def _jang_supports_vision(capabilities: dict[str, Any] | None) -> bool | None:
    return _first_bool_value(
        (
            capabilities.get("supports_vision")
            if isinstance(capabilities, dict)
            else None
        ),
        capabilities.get("supportsVision") if isinstance(capabilities, dict) else None,
        capabilities.get("supports_image") if isinstance(capabilities, dict) else None,
        capabilities.get("supportsImage") if isinstance(capabilities, dict) else None,
    )


def _jang_supports_video(capabilities: dict[str, Any] | None) -> bool | None:
    return _first_bool_value(
        capabilities.get("supports_video") if isinstance(capabilities, dict) else None,
        capabilities.get("supportsVideo") if isinstance(capabilities, dict) else None,
    )


def _jang_supports_audio(capabilities: dict[str, Any] | None) -> bool | None:
    return _first_bool_value(
        capabilities.get("supports_audio") if isinstance(capabilities, dict) else None,
        capabilities.get("supportsAudio") if isinstance(capabilities, dict) else None,
    )


def _jang_drop_earlier_reasoning(chat: dict[str, Any] | None) -> bool | None:
    reasoning = _nested_dict(chat, "reasoning")
    if reasoning is None:
        return None
    return _first_bool_value(
        reasoning.get("drop_earlier_reasoning"),
        reasoning.get("dropEarlierReasoning"),
    )


def _jang_chat_role_tokens(chat: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(chat, dict):
        return None
    return _string_dict_value(
        _mapping_value(chat, "role_tokens", "roleTokens"),
    )


def _jang_chat_reasoning_modes(chat: dict[str, Any] | None) -> tuple[str, ...] | None:
    reasoning = _nested_dict(chat, "reasoning")
    return _string_tuple_value(reasoning.get("modes") if reasoning else None)


def _jang_chat_reasoning_effort_levels(
    chat: dict[str, Any] | None,
) -> tuple[str | None, ...] | None:
    reasoning = _nested_dict(chat, "reasoning")
    return _optional_string_tuple_value(
        _mapping_value(reasoning, "reasoning_effort_levels", "reasoningEffortLevels")
        if reasoning
        else None
    )


def _jang_chat_sampling_defaults(chat: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(chat, dict):
        return None
    sampling = _nested_dict(chat, "sampling_defaults")
    if sampling is None:
        sampling = _nested_dict(chat, "samplingDefaults")
    return sampling


def _jang_mxtq_bits(sources: tuple[Any, ...]) -> Any | None:
    def normalize(value: Any) -> Any:
        if isinstance(value, int):
            return {"routed_expert": value}
        if not isinstance(value, dict) or not value:
            return value
        routed = value.get("routed_expert")
        if routed is not None:
            return value
        if any(key in value for key in ("gate_proj", "up_proj", "down_proj")):
            return {"routed_expert": dict(value)}
        return value

    bits = _first_nested_value(
        sources,
        {
            "mxtq_bits",
            "mxtqBits",
            "mxtq_bits_by_role",
            "mxtqBitsByRole",
            "routed_expert_bits",
            "routedExpertBits",
        },
    )
    if bits is not None:
        return normalize(bits)

    routed_plan_default = _jang_routed_expert_bit_plan_default(sources)
    if routed_plan_default is not None:
        return normalize(routed_plan_default)

    gate_up_bits = _first_nested_value(sources, {"mxtq_gate_up_bits", "mxtqGateUpBits"})
    down_bits = _first_nested_value(sources, {"mxtq_down_bits", "mxtqDownBits"})
    if gate_up_bits is None and down_bits is None:
        return None

    projected: dict[str, Any] = {}
    if gate_up_bits is not None:
        projected["gate_proj"] = gate_up_bits
        projected["up_proj"] = gate_up_bits
    if down_bits is not None:
        projected["down_proj"] = down_bits
    return {"routed_expert": projected}


def _jang_routed_expert_bit_plan_default(sources: tuple[Any, ...]) -> Any | None:
    """Return the default bits from scoped routed-expert plan metadata."""

    for source in sources:
        for item in _iter_dicts(source):
            plan = item.get("routed_expert_bit_plan")
            if not isinstance(plan, dict):
                plan = item.get("routedExpertBitPlan")
            if not isinstance(plan, dict):
                continue
            value = _mapping_value(plan, "default", "default_bits", "defaultBits")
            if value is not None:
                return value
    return None


def _jang_bit_widths_used(sources: tuple[Any, ...]) -> tuple[int, ...]:
    widths: list[int] = []
    for value in _all_nested_values(sources, {"bit_widths_used", "bitWidthsUsed"}):
        if not isinstance(value, (list, tuple, set)):
            continue
        for item in value:
            try:
                width = int(item)
            except (TypeError, ValueError):
                continue
            if width in _JANG_SUPPORTED_BITS and width not in widths:
                widths.append(width)
    return tuple(widths)


def _jang_mxtq_bits_with_codec(
    sources: tuple[Any, ...], codec: JangCodec, model_path: Path
) -> Any | None:
    bits = _jang_mxtq_bits(sources)
    if bits is not None or codec != "jangtq":
        return bits

    bit_widths = _jang_bit_widths_used(sources)
    if bit_widths:
        return min(bit_widths)

    codebook_bits = _jangtq_runtime_codebook_bits(model_path)
    if codebook_bits:
        return min(codebook_bits)
    return None


def _jang_source_model_candidates(
    metadata: JangQuantizationMetadata,
) -> tuple[Any, ...]:
    """Return source-model tokenizer hints in sidecar-first precedence order."""

    candidates: list[Any] = []
    for source in (metadata.sidecar, metadata.model_config):
        if not isinstance(source, dict):
            continue
        for key in (
            "source_model",
            "sourceModel",
            "base_model",
            "baseModel",
            "tokenizer_source",
            "tokenizerSource",
        ):
            value = source.get(key)
            if value is not None:
                candidates.append(value)
    return tuple(candidates)


def _index_has_tq_packed_entry(model_path: Path) -> bool:
    for index_path in _iter_safetensors_index_paths(model_path):
        try:
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map") or {}
            if isinstance(weight_map, dict) and any(
                _has_jangtq_tensor_marker(str(key))
                or _has_jangtq_tensor_marker(str(value))
                for key, value in weight_map.items()
            ):
                return True
        except Exception as e:
            logger.debug("Failed to read %s for JANGTQ scan: %s", index_path, e)
    return False


_JANGTQ_TENSOR_MARKERS = (
    ".tq_bits",
    ".tq_packed",
    ".tq_codebook",
    ".tq_hadamard_seed",
    ".tq_norms",
    ".tq_signs",
)
_JANGTQ_ARTIFACT_FILENAMES = {
    "jangpress-prestacked.safetensors",
    _JANGTQ_RUNTIME_SIDECAR,
    "jangtq_stacked.json",
    "jangtq_stacked.safetensors",
}
_JANGTQ_RUNTIME_CODEBOOK_RE = re.compile(r"^codebook\.[^.]+\.([1-9]\d*)$")
_JANGTQ_RUNTIME_SIGNS_RE = re.compile(r"^signs\.[^.]+\.[^.]+$")


def _has_jangtq_tensor_marker(value: str) -> bool:
    return any(marker in value for marker in _JANGTQ_TENSOR_MARKERS)


def _has_jangtq_artifact_markers(model_path: Path) -> bool:
    if any((model_path / name).exists() for name in _JANGTQ_ARTIFACT_FILENAMES):
        return True
    if _index_has_tq_packed_entry(model_path):
        return True
    if any(
        _has_jangtq_tensor_marker(path.name)
        for path in model_path.glob("*.safetensors")
    ):
        return True
    return _safetensors_has_tq_packed_key(model_path)


def _jangtq_runtime_key_sets(model_path: Path) -> tuple[set[str], set[str]]:
    """Return codebook/sign keys advertised by a JANGTQ runtime sidecar."""

    sidecar_path = model_path / _JANGTQ_RUNTIME_SIDECAR
    if not sidecar_path.is_file():
        return set(), set()

    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(
            f"safetensors is required to inspect {_JANGTQ_RUNTIME_SIDECAR}: {exc}"
        ) from exc

    try:
        with safe_open(str(sidecar_path), framework="numpy") as handle:
            keys = {str(key) for key in handle.keys()}  # noqa: SIM118
    except Exception as exc:
        raise RuntimeError(
            f"JANGTQ runtime sidecar {sidecar_path} is unreadable: {exc}"
        ) from exc

    codebook_keys = {key for key in keys if _JANGTQ_RUNTIME_CODEBOOK_RE.match(key)}
    signs_keys = {key for key in keys if _JANGTQ_RUNTIME_SIGNS_RE.match(key)}
    return codebook_keys, signs_keys


def _jangtq_runtime_codebook_bits(model_path: Path) -> tuple[int, ...]:
    sidecar_path = model_path / _JANGTQ_RUNTIME_SIDECAR
    if not sidecar_path.is_file():
        return ()
    try:
        codebook_keys, _ = _jangtq_runtime_key_sets(model_path)
    except RuntimeError as exc:
        logger.debug("Could not infer JANGTQ runtime codebook bits: %s", exc)
        return ()

    bits: set[int] = set()
    for key in codebook_keys:
        match = _JANGTQ_RUNTIME_CODEBOOK_RE.match(key)
        if match is not None:
            bits.add(int(match.group(1)))
    return tuple(sorted(bits))


def _preflight_jangtq_runtime_sidecar(metadata: JangQuantizationMetadata) -> None:
    """Fail early when a present JANGTQ runtime sidecar is malformed."""

    sidecar_path = metadata.model_path / _JANGTQ_RUNTIME_SIDECAR
    if not sidecar_path.is_file():
        return

    codebook_keys, signs_keys = _jangtq_runtime_key_sets(metadata.model_path)
    issues: list[str] = []
    if not codebook_keys:
        issues.append("no codebook.<in_features>.<bits> tensors")
    if not signs_keys:
        issues.append("no signs.<in_features>.<seed> tensors")
    if issues:
        raise RuntimeError(
            f"JANGTQ runtime sidecar {sidecar_path} is incomplete: "
            + "; ".join(issues)
            + ". Rebuild or re-download the JANGTQ bundle."
        )


def _safetensors_has_tq_packed_key(model_path: Path) -> bool:
    try:
        from safetensors import safe_open
    except Exception as exc:
        logger.debug("safetensors import failed for JANGTQ artifact scan: %s", exc)
        return False

    for shard in select_safetensors_weight_files(model_path):
        try:
            with safe_open(str(shard), framework="numpy") as handle:
                if any(
                    _has_jangtq_tensor_marker(str(key))
                    for key in handle.keys()  # noqa: SIM118
                ):
                    return True
        except Exception as exc:
            logger.debug("Failed to scan %s for JANGTQ keys: %s", shard, exc)
    return False


def _classify_jang_codec(
    markers: tuple[str, ...],
    sources: tuple[Any, ...],
    model_path: Path,
) -> JangCodec:
    canonical = {_canonical_jang_marker(marker) for marker in markers}
    explicit_affine = (
        any(
            marker in {"affine", "jang_affine"}
            or (marker.startswith("jang_") and "tq" not in marker)
            for marker in markers
        )
        or _first_nested_value(sources, {"quantization_backend", "quantizationBackend"})
        == "mx.quantize"
    )
    explicit_jangtq = any(
        marker in {"affinemxtq", "jangtq", "mxtq", "turboquant"}
        or marker.startswith("jangtq")
        or marker.startswith("mxtq")
        for marker in canonical
    )
    bit_accounting = (
        _first_nested_value(
            sources,
            {
                "mxtq_bits",
                "mxtqBits",
                "mxtq_bits_by_role",
                "mxtqBitsByRole",
                "routed_expert_bits",
                "routedExpertBits",
                "routed_expert_layer_bits",
                "routedExpertLayerBits",
                "mxtq_gate_up_bits",
                "mxtqGateUpBits",
                "mxtq_down_bits",
                "mxtqDownBits",
            },
        )
        is not None
        or _jang_routed_expert_bit_plan_default(sources) is not None
    )
    if (
        explicit_jangtq
        or (bit_accounting and not explicit_affine)
        or (
            _first_nested_value(sources, {"tq_layout", "tqLayout"}) is not None
            and not explicit_affine
        )
        or _has_jangtq_artifact_markers(model_path)
    ):
        return "jangtq"

    mxfp_markers = {"mxfp4", "mxfp8", "nvfp4", "fp4", "fp8"}
    if canonical & mxfp_markers:
        return "mxfp"

    affine_markers = {"jang", "jjqf", "mxq", "affine", "jangaffine"}
    if canonical & affine_markers:
        return "affine_jang"
    if any(marker.startswith("jang_") for marker in markers):
        return "affine_jang"

    return "unknown_jang"


def _jang_method_marker_declares_jang(marker: str) -> bool:
    """Return True for method names that are exclusive to JANG-family loaders."""

    canonical = _canonical_jang_marker(marker)
    return (
        marker.startswith("jang")
        or marker.startswith("mxtq")
        or canonical in {"jjqf", "mxq", "turboquant"}
    )


def _config_has_jang_v2_metadata_signal(config: dict[str, Any]) -> bool:
    """Return True when config.json carries modern JANG v2 metadata only."""

    version = _mapping_value(
        config,
        "version",
        "format_version",
        "formatVersion",
        "jang_version",
        "jangVersion",
    )
    if version is None:
        return False

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return False

    bit_widths = _mapping_value(quantization, "bit_widths_used", "bitWidthsUsed")
    if isinstance(bit_widths, (list, tuple, set)) and bit_widths:
        return True

    profile = _normalize_jang_marker(_mapping_value(quantization, "profile"))
    if profile is not None and (
        profile.startswith("jang_") or profile.startswith("jangtq")
    ):
        return True

    backend = _normalize_jang_marker(
        _mapping_value(
            quantization,
            "quantization_backend",
            "quantizationBackend",
        )
    )
    if backend == "mx_quantize":
        return True

    has_average_bits = (
        _mapping_value(
            quantization,
            "target_bits",
            "targetBits",
            "target_bit_width",
            "targetBitWidth",
            "actual_bits",
            "actualBits",
            "actual_bits_per_weight",
            "actualBitsPerWeight",
            "actual_bit_width",
            "actualBitWidth",
            "effective_bits",
            "effectiveBits",
            "effective_bits_per_weight",
            "effectiveBitsPerWeight",
        )
        is not None
    )
    has_group_size = (
        _mapping_value(
            quantization,
            "block_size",
            "blockSize",
            "group_size",
            "groupSize",
        )
        is not None
    )
    return has_average_bits and has_group_size


def _config_declares_jang_metadata(
    model_path: Path,
    config: dict[str, Any],
) -> bool:
    """Return True when config.json itself carries JANG-format evidence."""

    if _has_jangtq_artifact_markers(model_path):
        return True
    if _config_has_jang_v2_metadata_signal(config):
        return True

    sources: tuple[Any, ...] = (config,)
    marker_values = _all_nested_values(
        sources,
        {
            "format",
            "weight_format",
            "weightFormat",
            "profile",
            "jang_profile",
            "jangProfile",
            "tq_layout",
            "tqLayout",
            "family",
        },
    )
    known_markers = {
        "affine",
        "affinemxtq",
        "fp4",
        "fp8",
        "jang",
        "jangaffine",
        "jangq",
        "jangtq",
        "jjqf",
        "mxfp4",
        "mxfp8",
        "mxtq",
        "mxtq2",
        "mxtq4",
        "turboquant",
        "mxq",
        "nvfp4",
    }
    for value in marker_values:
        marker = _normalize_jang_marker(value)
        if marker is None:
            continue
        canonical = _canonical_jang_marker(marker)
        if (
            marker.startswith("jang_")
            or marker.startswith("jangtq")
            or marker.startswith("mxtq")
            or canonical in known_markers
        ):
            return canonical != "mlx"

    method_values = _all_nested_values(
        sources,
        {"method", "quant_method", "quantMethod"},
    )
    for value in method_values:
        marker = _normalize_jang_marker(value)
        if marker is not None and _jang_method_marker_declares_jang(marker):
            return True

    if _first_nested_value(sources, {"tq_layout", "tqLayout"}) is not None:
        return True
    return (
        _first_nested_value(
            sources,
            {
                "mxtq_bits",
                "mxtqBits",
                "mxtq_bits_by_role",
                "mxtqBitsByRole",
                "routed_expert_bits",
                "routedExpertBits",
                "routed_expert_layer_bits",
                "routedExpertLayerBits",
                "mxtq_gate_up_bits",
                "mxtqGateUpBits",
                "mxtq_down_bits",
                "mxtqDownBits",
            },
        )
        is not None
    )


def _sanitize_grouped_conv1d_layout(weights: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Return weights with leftover HF-layout grouped Conv1d tensors transposed."""

    fixed: dict[Any, Any] | None = None
    for key, value in weights.items():
        shape = getattr(value, "shape", None)
        if (
            "conv1d.weight" in str(key)
            and getattr(value, "ndim", None) == 3
            and len(shape or ()) == 3
            and shape[1] == 1
            and shape[-1] != 1
        ):
            if fixed is None:
                fixed = dict(weights)
            fixed[key] = mx.transpose(value, axes=(0, 2, 1))
    return weights if fixed is None else fixed


def _sanitize_grouped_conv1d_weight_payload(weights: Any) -> Any:
    if isinstance(weights, Mapping):
        return _sanitize_grouped_conv1d_layout(weights)
    if not isinstance(weights, (list, tuple)):
        return weights

    items: list[tuple[Any, Any]] = []
    for item in weights:
        if not isinstance(item, tuple) or len(item) != 2:
            return weights
        items.append(item)

    sanitized = _sanitize_grouped_conv1d_layout(dict(items))
    if sanitized is weights:
        return weights
    return type(weights)(sanitized.items())


@contextmanager
def _scoped_jang_grouped_conv1d_load_weights_sanitize() -> Iterator[None]:
    """Patch MLX module weight loading during JANG loads to fix old SSM layouts."""

    import mlx.nn as nn

    original = nn.Module.load_weights

    def _load_weights_with_grouped_conv1d_sanitize(self, weights, *args, **kwargs):
        return original(
            self,
            _sanitize_grouped_conv1d_weight_payload(weights),
            *args,
            **kwargs,
        )

    nn.Module.load_weights = _load_weights_with_grouped_conv1d_sanitize  # type: ignore[assignment, method-assign]
    try:
        yield
    finally:
        nn.Module.load_weights = original  # type: ignore[assignment, method-assign]


def _model_has_jang_quant_shape_signal(
    model_path: Path,
    config: dict[str, Any],
) -> bool:
    """Return True when shape-derived quant metadata repair is safe to run."""

    if _find_jang_config_path(model_path) is not None:
        return True
    for key in ("jang", "jang_version", "jang_profile"):
        if key in config:
            return True
    for value in _all_nested_values(
        (config,), {"format", "weight_format", "weightFormat"}
    ):
        marker = _normalize_jang_marker(value)
        if marker in {
            "jang",
            "jang_affine",
            "jangq",
            "mxfp4",
            "mxfp8",
            "mxq",
            "jjqf",
        }:
            return True
    for value in _all_nested_values(
        (config,), {"method", "quant_method", "quantMethod"}
    ):
        marker = _normalize_jang_marker(value)
        if marker is not None and _jang_method_marker_declares_jang(marker):
            return True
    return _config_declares_jang_metadata(model_path, config)


def _infer_jang_quantization_from_shapes(
    model_path: Path,
    *,
    model_config: Mapping[str, Any] | None = None,
    allowed_bits: set[int] | None = None,
    group_size_hints: set[int] | None = None,
    config_quantization: Mapping[str, Any] | None = None,
    trust_top_level_claim: bool = False,
) -> dict[str, tuple[int, int]]:
    """Infer affine quantization ``(bits, group_size)`` from safetensor shapes."""

    shards = [
        path
        for path in select_safetensors_weight_files(model_path)
        if path.name
        not in {
            "jangtq_runtime.safetensors",
            "jangtq_stacked.safetensors",
        }
    ]
    if not shards:
        return {}

    try:
        from safetensors import safe_open
    except Exception as exc:
        logger.debug("safetensors import failed for JANG shape walk: %s", exc)
        return {}

    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        try:
            with safe_open(str(shard), framework="numpy") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open exposes keys()
                    if key.endswith(".importance") or ".tq_" in key:
                        continue
                    shapes[str(key)] = tuple(
                        int(dim) for dim in handle.get_slice(key).get_shape()
                    )
        except Exception as exc:
            logger.debug("Failed JANG quant shape scan for %s: %s", shard, exc)
            return {}

    quantized_shapes: dict[str, tuple[int, int]] = {}
    for scales_key, scales_shape in shapes.items():
        if not scales_key.endswith(".scales") or not scales_shape:
            continue
        base = scales_key[: -len(".scales")]
        weight_shape = shapes.get(base + ".weight")
        if not weight_shape:
            continue
        packed = int(weight_shape[-1])
        scale_groups = int(scales_shape[-1])
        if packed <= 0 or scale_groups <= 0:
            continue
        quantized_shapes[base] = (packed, scale_groups)

    uniform_group_size = _infer_uniform_jang_group_size(
        quantized_shapes,
        allowed_bits=allowed_bits,
        group_size_hints=group_size_hints,
    )
    top_level_claim = _quantization_default_pair(config_quantization)

    inferred: dict[str, tuple[int, int]] = {}
    for base, (packed, scale_groups) in quantized_shapes.items():
        pair = _infer_jang_quant_pair_from_shapes(
            packed,
            scale_groups,
            module_name=base,
            allowed_bits=allowed_bits,
            group_size_hints=group_size_hints,
            preferred_pair=_jang_quantization_claim_for_module(
                config_quantization,
                base,
            ),
            top_level_pair=top_level_claim if trust_top_level_claim else None,
            expected_input_dim=_jang_expected_input_dim_for_module(
                model_config,
                base,
            ),
            preferred_group_size=uniform_group_size,
        )
        if pair is not None:
            inferred[base] = pair
    return inferred


def _jang_quant_pair_candidates_from_shapes(
    packed: int,
    scale_groups: int,
    *,
    allowed_bits: set[int] | None,
    group_size_hints: set[int] | None,
) -> list[tuple[int, int]]:
    numerator = packed * 32
    allowed = allowed_bits or set(_JANG_SUPPORTED_BITS)
    matches: list[tuple[int, int]] = []

    for bits, group_size in _JANG_QUANT_SHAPE_CANDIDATES:
        if bits not in allowed:
            continue
        if group_size_hints is not None and group_size not in group_size_hints:
            continue
        if _jang_quant_pair_matches_packed_shape(
            packed,
            scale_groups,
            bits=bits,
            group_size=group_size,
        ):
            matches.append((bits, group_size))

    if matches:
        return matches

    # JANG Studio exposes block-size overrides, so newer bundles may use a
    # non-standard group size. Only derive those when the sidecar/config gives
    # enough hints to avoid confusing e.g. 5-bit/64 with 4-bit/80.
    if allowed_bits is None and group_size_hints is None:
        return []
    for bits in _JANG_SUPPORTED_BITS:
        if bits not in allowed:
            continue
        denominator = bits * scale_groups
        if denominator <= 0 or numerator % denominator != 0:
            continue
        group_size = numerator // denominator
        if group_size <= 0:
            continue
        if group_size_hints is not None and group_size not in group_size_hints:
            continue
        matches.append((bits, group_size))
    return matches


def _infer_uniform_jang_group_size(
    quantized_shapes: Mapping[str, tuple[int, int]],
    *,
    allowed_bits: set[int] | None,
    group_size_hints: set[int] | None,
) -> int | None:
    """Return a bundle-wide group size when shapes pin one unambiguously."""

    if not quantized_shapes:
        return None

    viable_by_module: list[set[int]] = []
    distinct_products: set[int] = set()
    for packed, scale_groups in quantized_shapes.values():
        if packed <= 0 or scale_groups <= 0:
            continue
        candidates = _jang_quant_pair_candidates_from_shapes(
            packed,
            scale_groups,
            allowed_bits=allowed_bits,
            group_size_hints=group_size_hints,
        )
        if not candidates:
            continue
        viable_by_module.append({group_size for _, group_size in candidates})
        product = packed * 32
        if product % scale_groups == 0:
            distinct_products.add(product // scale_groups)

    if not viable_by_module or len(distinct_products) < 2:
        return None

    intersection = set(viable_by_module[0])
    for viable in viable_by_module[1:]:
        intersection &= viable
        if not intersection:
            return None
    if len(intersection) == 1:
        return next(iter(intersection))
    return None


def _infer_jang_quant_pair_from_shapes(
    packed: int,
    scale_groups: int,
    *,
    module_name: str | None = None,
    allowed_bits: set[int] | None,
    group_size_hints: set[int] | None,
    preferred_pair: tuple[int, int] | None = None,
    top_level_pair: tuple[int, int] | None = None,
    expected_input_dim: int | None = None,
    preferred_group_size: int | None = None,
) -> tuple[int, int] | None:
    matches = _jang_quant_pair_candidates_from_shapes(
        packed,
        scale_groups,
        allowed_bits=allowed_bits,
        group_size_hints=group_size_hints,
    )
    if not matches:
        return None
    for pair in (preferred_pair, top_level_pair):
        if pair is not None and pair in matches:
            return pair
    if expected_input_dim is not None:
        expected_matches = [
            pair
            for pair in matches
            if _jang_quant_pair_matches_exact_input_dim(
                packed,
                scale_groups,
                bits=pair[0],
                group_size=pair[1],
                input_dim=expected_input_dim,
            )
        ]
        if len(expected_matches) == 1:
            return expected_matches[0]
        if expected_matches:
            return _choose_jang_quant_pair_for_module(
                module_name or "",
                expected_matches,
            )
    if preferred_group_size is not None:
        for pair in matches:
            if pair[1] == preferred_group_size:
                return pair
    if len(matches) == 1:
        return matches[0]
    return _choose_jang_quant_pair_for_module(module_name or "", matches)


def _jang_quantization_claim_for_module(
    quantization: Mapping[str, Any] | None,
    module_name: str,
) -> tuple[int, int] | None:
    if not isinstance(quantization, Mapping):
        return None
    for key in _jang_quantization_key_variants(module_name):
        pair = _quantization_default_pair(quantization.get(key))
        if pair is not None:
            return pair
    return None


def _choose_jang_quant_pair_for_module(
    module_name: str,
    matches: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Resolve ambiguous shape-derived JANG quantization matches."""

    if _JANG_ROUTED_QUANT_KEY_RE.search(module_name):
        return min(matches, key=lambda pair: (pair[0], pair[1]))
    if _JANG_HIGH_BIT_QUANT_KEY_RE.search(module_name):
        return max(matches, key=lambda pair: (pair[0], -pair[1]))
    if _JANG_ATTENTION_QUANT_KEY_RE.search(module_name):
        return max(matches, key=lambda pair: (pair[0], -pair[1]))
    return max(matches, key=lambda pair: (pair[0], -pair[1]))


def _jang_expected_input_dim_for_module(
    config: Mapping[str, Any] | None,
    module_name: str,
) -> int | None:
    """Return an architecture input dim that can disambiguate JANG shapes."""

    if not isinstance(config, Mapping):
        return None

    model_type = str(config.get("model_type") or "").lower()
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        text_config = config
    text_model_type = str(text_config.get("model_type") or "").lower()
    qwen_types = {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "qwen3_5_vl",
        "qwen3_6",
        "qwen3_6_moe",
    }
    if model_type not in qwen_types and text_model_type not in qwen_types:
        return None

    hidden = _int_mapping_value(text_config, "hidden_size")
    intermediate = _int_mapping_value(text_config, "intermediate_size")
    num_heads = _int_mapping_value(text_config, "num_attention_heads")
    head_dim = _int_mapping_value(text_config, "head_dim")
    linear_v_heads = _int_mapping_value(text_config, "linear_num_value_heads")
    linear_v_dim = _int_mapping_value(text_config, "linear_value_head_dim")

    name = module_name
    for prefix in (
        "language_model.model.",
        "model.language_model.",
        "language_model.",
        "model.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    if name.endswith("embed_tokens") or name.endswith("lm_head"):
        return hidden

    if ".linear_attn." in name:
        if name.endswith((".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a")):
            return hidden
        if (
            name.endswith(".out_proj")
            and linear_v_heads
            and linear_v_dim
        ):
            return linear_v_heads * linear_v_dim

    if ".self_attn." in name:
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return hidden
        if name.endswith(".o_proj") and num_heads and head_dim:
            return num_heads * head_dim

    if ".mlp." in name:
        if name.endswith((".gate_proj", ".up_proj")):
            return hidden
        if name.endswith(".down_proj"):
            return intermediate

    return None


def _int_mapping_value(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _jang_quant_pair_matches_exact_input_dim(
    packed: int,
    scale_groups: int,
    *,
    bits: int,
    group_size: int,
    input_dim: int,
) -> bool:
    if input_dim <= 0:
        return False
    return (
        scale_groups == (input_dim + group_size - 1) // group_size
        and packed == (input_dim * bits + 31) // 32
    )


def _jang_quant_pair_matches_packed_shape(
    packed: int,
    scale_groups: int,
    *,
    bits: int,
    group_size: int,
) -> bool:
    if packed <= 0 or scale_groups <= 0 or bits <= 0 or group_size <= 0:
        return False
    min_in_dim = (scale_groups - 1) * group_size + 1
    max_in_dim = scale_groups * group_size
    min_packed = (min_in_dim * bits + 31) // 32
    max_packed = (max_in_dim * bits + 31) // 32
    return min_packed <= packed <= max_packed


def _dominant_quantization_pair(
    inferred: Mapping[str, tuple[int, int]],
) -> tuple[int, int] | None:
    if not inferred:
        return None
    counts = Counter(inferred.values())
    best_count = max(counts.values())
    candidates = {pair for pair, count in counts.items() if count == best_count}
    for pair in _JANG_QUANT_SHAPE_CANDIDATES:
        if pair in candidates:
            return pair
    return next(iter(candidates))


def _quantization_default_pair(
    quantization: Mapping[str, Any] | None,
) -> tuple[int, int] | None:
    if not isinstance(quantization, Mapping):
        return None

    bits = _strict_positive_int_mapping_value(
        quantization,
        "bits",
        "bit_width",
        "bitWidth",
    )
    group_size = _strict_positive_int_mapping_value(
        quantization,
        "block_size",
        "blockSize",
        "blocksize",
        "group_size",
        "groupSize",
        "groupsize",
    )
    if bits is None or group_size is None:
        return None
    return bits, group_size


def _strict_positive_int_mapping_value(
    source: Mapping[str, Any],
    *keys: str,
) -> int | None:
    for key in keys:
        value = source.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            if value.is_integer() and value > 0:
                return int(value)
            continue
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            return parsed if parsed > 0 else None
    return None


def _jang_quantization_key_variants(key: str) -> tuple[str, ...]:
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            variants.append(value)

    add(key)
    if ".attn." in key:
        add(key.replace(".attn.", ".self_attn."))
    elif key.endswith(".attn"):
        add(key[: -len(".attn")] + ".self_attn")

    if key.startswith("language_model.model."):
        add(key[len("language_model.") :])
        add(key[len("language_model.model.") :])
    elif key.startswith("language_model."):
        add(key[len("language_model.") :])
    elif key.startswith("model."):
        add("language_model." + key)
    else:
        add("model." + key)
        add("language_model." + key)
        add("language_model.model." + key)
    return tuple(variants)


def _is_quantization_layer_override_key(key: str, value: Any) -> bool:
    marker = _normalize_jang_marker(key) or key
    return marker not in _JANG_QUANT_METADATA_KEYS and isinstance(value, dict)


def patch_jangtq_weight_format_from_artifacts(
    config: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    """Force JANGTQ config markers when runtime artifacts identify the bundle."""

    if not _has_jangtq_artifact_markers(model_path):
        return config

    config["weight_format"] = "mxtq"

    def patch_nested_quantization(target: dict[str, Any]) -> None:
        for key in ("quantization", "quantization_config"):
            quantization = target.get(key)
            if isinstance(quantization, dict):
                quantization["weight_format"] = "mxtq"

    patch_nested_quantization(config)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config["weight_format"] = "mxtq"
        patch_nested_quantization(text_config)
    return config


def patch_jang_quantization_from_shapes(
    config: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    """Repair JANG mixed-bit affine quantization metadata from safetensor headers.

    JANG-family converters may carry stale, stripped, or operator-restamped
    ``config.json["quantization"]`` blocks. The safetensor ``.weight`` and
    ``.scales`` shapes are authoritative for affine MX/JANG weights, so this
    patches the in-memory config before MLX constructs ``QuantizedLinear``
    modules. It never writes to the model directory.
    """

    if not _model_has_jang_quant_shape_signal(model_path, config):
        return config

    allowed_bits, group_size_hints = _jang_shape_inference_hints(model_path, config)
    config_quantization = config.get("quantization")
    config_quantization = (
        config_quantization if isinstance(config_quantization, Mapping) else None
    )
    inferred = _infer_jang_quantization_from_shapes(
        model_path,
        model_config=config,
        allowed_bits=allowed_bits,
        group_size_hints=group_size_hints,
        config_quantization=config_quantization,
        trust_top_level_claim=allowed_bits is None or len(allowed_bits) <= 1,
    )
    if not inferred:
        return config

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        quantization = {}
        config["quantization"] = quantization

    default_pair = _dominant_quantization_pair(inferred)
    if default_pair is None:
        return config

    default_bits, default_group_size = default_pair
    quantization["bits"] = default_bits
    quantization["group_size"] = default_group_size
    mode = quantization.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        quantization["mode"] = "affine"
        mode = "affine"

    patched = 0
    existing_layer_keys = {
        key
        for key, value in quantization.items()
        if _is_quantization_layer_override_key(key, value)
    }
    for key in existing_layer_keys:
        if key not in inferred:
            quantization.pop(key, None)

    for key, (bits, group_size) in sorted(inferred.items()):
        needs_per_layer_override = (bits, group_size) != (
            default_bits,
            default_group_size,
        )
        existing = quantization.get(key)
        existing_pair = None
        if isinstance(existing, dict):
            existing_pair = _quantization_default_pair(existing)
        if not needs_per_layer_override and existing_pair in {
            None,
            (bits, group_size),
        }:
            continue

        entry: dict[str, Any] = {"bits": bits, "group_size": group_size}
        if isinstance(mode, str):
            entry["mode"] = mode
        changed = False
        for variant in _jang_quantization_key_variants(key):
            if variant == key and existing_pair == (bits, group_size):
                continue
            if quantization.get(variant) != entry:
                quantization[variant] = dict(entry)
                changed = True
        if changed:
            patched += 1

    if patched:
        logger.info(
            "JANG quantization shape walk produced %d per-layer override(s) "
            "over default bits=%s group_size=%s for %s",
            patched,
            default_bits,
            default_group_size,
            model_path,
        )
    return config


def _jang_shape_inference_hints(
    model_path: Path,
    config: dict[str, Any],
) -> tuple[set[int] | None, set[int] | None]:
    sources: list[Any] = [config]
    group_hint_sources: list[Any] = []
    sidecar_path = _find_jang_config_path(model_path)
    if sidecar_path is not None:
        sidecar = _read_json_object_or_none(sidecar_path)
        if sidecar is not None:
            sources.insert(0, sidecar)
            group_hint_sources.append(sidecar)
    elif isinstance(config.get("jang"), dict):
        group_hint_sources.append(config["jang"])
    elif _config_declares_jang_metadata(model_path, config):
        group_hint_sources.append(config)

    bit_widths = set(_jang_bit_widths_used(tuple(sources)))
    group_sizes: set[int] = set()
    for value in _all_nested_values(
        tuple(group_hint_sources),
        {
            "block_size",
            "blockSize",
            "blocksize",
            "group_size",
            "groupSize",
            "groupsize",
            "routed_expert_group_size",
            "routedExpertGroupSize",
        },
    ):
        try:
            group_size = int(value)
        except (TypeError, ValueError):
            continue
        if group_size > 0:
            group_sizes.add(group_size)

    return (bit_widths or None), (group_sizes or None)


def _jang_config_declares_no_vision(jang_config: dict[str, Any]) -> bool:
    if (
        _coerce_jang_bool(_mapping_value(jang_config, "has_vision", "hasVision"))
        is False
    ):
        return True
    if _jang_modalities_declare_no_active_media(jang_config.get("modalities")):
        return True
    capabilities = jang_config.get("capabilities")
    if isinstance(capabilities, dict) and _jang_capabilities_declare_no_active_media(
        capabilities
    ):
        return True
    architecture = jang_config.get("architecture")
    return (
        isinstance(architecture, dict)
        and _coerce_jang_bool(_mapping_value(architecture, "has_vision", "hasVision"))
        is False
    )


def _jang_capabilities_declares_vision(
    capabilities: dict[str, Any] | None,
) -> bool:
    if not isinstance(capabilities, dict):
        return False
    if not _jang_capabilities_have_active_multimodal_runtime(capabilities):
        return False
    if (
        _coerce_jang_bool(_mapping_value(capabilities, "has_vision", "hasVision"))
        is True
    ):
        return True
    if _coerce_jang_bool(_mapping_value(capabilities, "has_image", "hasImage")) is True:
        return True
    if _coerce_jang_bool(_mapping_value(capabilities, "has_video", "hasVideo")) is True:
        return True
    if (
        _coerce_jang_bool(
            _mapping_value(capabilities, "supports_audio", "supportsAudio")
        )
        is True
    ):
        return True
    if (
        _coerce_jang_bool(
            _mapping_value(capabilities, "supports_image", "supportsImage")
        )
        is True
    ):
        return True
    if (
        _coerce_jang_bool(
            _mapping_value(capabilities, "supports_vision", "supportsVision")
        )
        is True
    ):
        return True
    if (
        _coerce_jang_bool(
            _mapping_value(capabilities, "supports_video", "supportsVideo")
        )
        is True
    ):
        return True
    if _jang_modalities_declare_active_media(capabilities.get("modalities")):
        return True
    return _jang_modalities_declare_active_media(capabilities.get("modality"))


def _jang_capabilities_declare_no_active_media(
    capabilities: dict[str, Any],
) -> bool:
    if _jang_modalities_declare_no_active_media(capabilities.get("modalities")):
        return True

    saw_disabled_media = False
    for key in (
        ("has_audio", "hasAudio"),
        ("has_image", "hasImage"),
        ("has_vision", "hasVision"),
        ("has_video", "hasVideo"),
        ("supports_audio", "supportsAudio"),
        ("supports_image", "supportsImage"),
        ("supports_vision", "supportsVision"),
        ("supports_video", "supportsVideo"),
    ):
        flag = _coerce_jang_bool(_mapping_value(capabilities, *key))
        if flag is True:
            return False
        if flag is False:
            saw_disabled_media = True
    return saw_disabled_media


def _jang_capabilities_have_active_multimodal_runtime(
    capabilities: dict[str, Any],
) -> bool:
    status = _mapping_value(capabilities, "multimodal_status", "multimodalStatus")
    if (
        isinstance(status, str)
        and status.strip().lower() in _JANG_TEXT_RUNTIME_STATUSES
    ):
        return False
    unwired = _mapping_value(capabilities, "unwired_modalities", "unwiredModalities")
    return not (isinstance(unwired, (list, tuple, set)) and len(unwired) > 0)


def _detect_jang_vlm_model(
    model_path: Path,
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> bool:
    if _jang_config_declares_no_vision(jang_config):
        return False

    capabilities = _jang_capabilities(jang_config, model_config)
    if isinstance(
        capabilities, dict
    ) and not _jang_capabilities_have_active_multimodal_runtime(capabilities):
        return False
    if isinstance(capabilities, dict) and _jang_capabilities_declare_no_active_media(
        capabilities
    ):
        return False

    if _jang_config_declares_active_media(jang_config):
        return True

    if model_config is not None and (
        "vision_config" in model_config
        or "vit_config" in model_config
        or bool(model_config.get("mm_vision_tower"))
        or "audio_config" in model_config
        or "video_config" in model_config
    ):
        return True

    architecture = jang_config.get("architecture")
    if (
        isinstance(architecture, dict)
        and _coerce_jang_bool(_mapping_value(architecture, "has_vision", "hasVision"))
        is True
    ):
        return True
    if _jang_capabilities_declares_vision(
        _jang_capabilities(jang_config, model_config)
    ):
        return True

    return (model_path / "preprocessor_config.json").exists() or (
        model_path / "video_preprocessor_config.json"
    ).exists()


def _iter_model_identity_strings(
    model_path: Path,
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
):
    yield from model_path.parts[-3:]
    for source in (model_config, jang_config):
        if not isinstance(source, dict):
            continue
        for key in ("model_type", "_name_or_path", "name_or_path"):
            value = source.get(key)
            if isinstance(value, str):
                yield value
        model_family = source.get("model_family")
        if isinstance(model_family, str):
            yield model_family
        family = source.get("family")
        if isinstance(family, str):
            yield family
        capabilities = source.get("capabilities")
        if isinstance(capabilities, dict) and isinstance(
            capabilities.get("family"),
            str,
        ):
            yield capabilities["family"]
        source_model = _mapping_value(source, "source_model", "sourceModel")
        if isinstance(source_model, str):
            yield source_model
        elif isinstance(source_model, dict):
            for key in (
                "architecture",
                "family",
                "hub_id",
                "hubID",
                "id",
                "model",
                "name",
                "repo_id",
                "repoID",
            ):
                value = source_model.get(key)
                if isinstance(value, str):
                    yield value
        architecture = source.get("architecture")
        if isinstance(architecture, dict):
            for key in ("type", "text_model_type", "model_type"):
                value = architecture.get(key)
                if isinstance(value, str):
                    yield value
        architectures = source.get("architectures")
        if isinstance(architectures, list):
            for architecture in architectures:
                if isinstance(architecture, str):
                    yield architecture
        text_config = source.get("text_config")
        if isinstance(text_config, dict) and isinstance(
            text_config.get("model_type"),
            str,
        ):
            yield text_config["model_type"]


def _is_kimi_vlm_model(
    model_path: Path,
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> bool:
    for value in _iter_model_identity_strings(model_path, jang_config, model_config):
        normalized = _normalize_jang_marker(value)
        if normalized is None or "kimi" not in normalized:
            continue
        parts = normalized.split("_")
        has_vl = "vl" in parts or "vision" in parts
        has_k2_6 = "k2_6" in normalized or "k26" in normalized
        if has_vl or has_k2_6:
            return True
    return False


def _is_deepseek_v4_model(
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> bool:
    for value in _iter_model_identity_strings(Path(), jang_config, model_config):
        if _normalize_jang_marker(value) == "deepseek_v4":
            return True
    return False


def read_jang_metadata(model_path: str | Path) -> JangQuantizationMetadata | None:
    """Read and normalize JANG-family sidecar metadata for *model_path*."""

    path = Path(model_path)
    sidecar_path = _find_jang_config_path(path)
    bundle_path = None
    embedded_sidecar = False
    model_config: dict[str, Any] | None
    if sidecar_path is None:
        embedded = _read_embedded_jang_metadata(path)
        if embedded is None:
            return None
        (
            payload_path,
            bundle_path,
            model_config,
            jang_config,
            embedded_sidecar,
        ) = embedded
        sidecar_path = payload_path / "config.json"
    else:
        bundle_path = _jangspec_bundle_root(path, sidecar_path)
        payload_path = _jang_payload_path(path, sidecar_path)
        jang_config = _read_jang_config(sidecar_path)
        model_config = _read_json_object_or_none(payload_path / "config.json")
        if model_config is None and payload_path != path:
            model_config = _read_json_object_or_none(path / "config.json")
    sources: tuple[Any, ...] = (
        jang_config,
        model_config if model_config is not None else {},
    )
    markers = _collect_jang_markers(jang_config, model_config, sidecar_path)
    is_kimi_vlm = _is_kimi_vlm_model(payload_path, jang_config, model_config)
    is_vlm = (
        _detect_jang_vlm_model(payload_path, jang_config, model_config) or is_kimi_vlm
    )

    capabilities = _jang_capabilities(jang_config, model_config)
    chat = _jang_chat(jang_config, model_config)
    chat_reasoning = _nested_dict(chat, "reasoning")
    chat_tool_calling = _nested_dict(chat, "tool_calling")
    chat_sampling_defaults = _jang_chat_sampling_defaults(chat)
    model_family = _jang_model_family(jang_config, model_config)
    reasoning_parser = _jang_reasoning_parser(capabilities, chat)
    think_in_template = _jang_think_in_template(capabilities, chat)
    codec = _classify_jang_codec(markers, sources, payload_path)

    return JangQuantizationMetadata(
        model_path=payload_path,
        sidecar_path=sidecar_path,
        sidecar=jang_config,
        model_config=model_config,
        codec=codec,
        markers=markers,
        profile=_first_nested_string(sources, {"profile"}),
        target_bits=_first_nested_value(
            sources,
            {
                "target_bits",
                "targetBits",
                "target_bit_width",
                "targetBitWidth",
                "bits",
                "bit_width",
                "bitWidth",
            },
        ),
        actual_bits=_first_nested_value(
            sources,
            {
                "actual_bits",
                "actualBits",
                "actual_bits_per_weight",
                "actualBitsPerWeight",
                "effective_bits",
                "effectiveBits",
                "effective_bits_per_weight",
                "effectiveBitsPerWeight",
                "actual_bit_width",
                "actualBitWidth",
            },
        ),
        block_size=_first_nested_value(
            sources,
            {"block_size", "blockSize", "blocksize"},
        ),
        group_size=_first_nested_value(
            sources,
            {"group_size", "groupSize", "groupsize"},
        ),
        mxtq_bits=_jang_mxtq_bits_with_codec(sources, codec, payload_path),
        model_family=model_family,
        capabilities=capabilities,
        chat=chat,
        cache_type=_first_nested_string(sources, {"cache_type", "cacheType"}),
        cache_subtype=_jang_cache_subtype(capabilities, sources),
        draft_strategy=_jang_draft_strategy(capabilities),
        drafter_path=_jang_drafter_path(capabilities),
        draft_branching_budget=_jang_draft_branching_budget(capabilities),
        draft_block_size=_jang_draft_block_size(capabilities),
        reasoning_parser=(
            None
            if (
                think_in_template is False
                and model_family is not None
                and _normalize_jang_marker(model_family)
                in {"lfm2", "lfm2_moe", "lfm2_5", "lfm25"}
            )
            else reasoning_parser
        ),
        tool_parser=_jang_tool_parser(capabilities, chat),
        think_in_template=think_in_template,
        supports_thinking=_jang_supports_thinking(capabilities, chat),
        supports_tools=_jang_supports_tools(capabilities, chat),
        supports_text=_jang_supports_text(capabilities),
        supports_vision=_jang_supports_vision(capabilities),
        supports_video=_jang_supports_video(capabilities),
        supports_audio=_jang_supports_audio(capabilities),
        drop_earlier_reasoning=_jang_drop_earlier_reasoning(chat),
        chat_encoder=_first_string_value(chat.get("encoder") if chat else None),
        chat_has_tokenizer_template=_first_bool_value(
            _mapping_value(
                chat,
                "has_tokenizer_chat_template",
                "hasTokenizerChatTemplate",
            )
            if chat
            else None
        ),
        chat_bos_token=_first_string_value(
            _mapping_value(chat, "bos_token", "bosToken") if chat else None,
        ),
        chat_bos_token_id=_first_int_value(
            _mapping_value(chat, "bos_token_id", "bosTokenId") if chat else None,
        ),
        chat_eos_token=_first_string_value(
            _mapping_value(chat, "eos_token", "eosToken") if chat else None,
        ),
        chat_eos_token_id=_first_int_value(
            _mapping_value(chat, "eos_token_id", "eosTokenId") if chat else None,
        ),
        chat_role_tokens=_jang_chat_role_tokens(chat),
        chat_reasoning_modes=_jang_chat_reasoning_modes(chat),
        chat_reasoning_default_mode=_first_string_value(
            _mapping_value(chat_reasoning, "default_mode", "defaultMode")
            if chat_reasoning
            else None
        ),
        chat_reasoning_thinking_start=_first_string_value(
            _mapping_value(chat_reasoning, "thinking_start", "thinkingStart")
            if chat_reasoning
            else None
        ),
        chat_reasoning_thinking_end=_first_string_value(
            _mapping_value(chat_reasoning, "thinking_end", "thinkingEnd")
            if chat_reasoning
            else None
        ),
        chat_reasoning_effort_levels=_jang_chat_reasoning_effort_levels(chat),
        chat_tool_dsml_token=_first_string_value(
            _mapping_value(chat_tool_calling, "dsml_token", "dsmlToken")
            if chat_tool_calling
            else None
        ),
        chat_tool_calls_block=_first_string_value(
            _mapping_value(chat_tool_calling, "tool_calls_block", "toolCallsBlock")
            if chat_tool_calling
            else None
        ),
        chat_tool_invoke_block=_first_string_value(
            _mapping_value(chat_tool_calling, "invoke_block", "invokeBlock")
            if chat_tool_calling
            else None
        ),
        chat_tool_parameter_block=_first_string_value(
            _mapping_value(chat_tool_calling, "parameter_block", "parameterBlock")
            if chat_tool_calling
            else None
        ),
        chat_tool_output_tag=_first_string_value(
            _mapping_value(chat_tool_calling, "tool_output_tag", "toolOutputTag")
            if chat_tool_calling
            else None
        ),
        sampling_temperature=_first_float_value(
            (
                chat_sampling_defaults.get("temperature")
                if chat_sampling_defaults
                else None
            ),
        ),
        sampling_top_p=_first_float_value(
            _mapping_value(chat_sampling_defaults, "top_p", "topP")
            if chat_sampling_defaults
            else None
        ),
        sampling_max_new_tokens=_first_int_value(
            _mapping_value(
                chat_sampling_defaults,
                "max_new_tokens",
                "maxNewTokens",
            )
            if chat_sampling_defaults
            else None
        ),
        routed_expert_bit_plans=_all_nested_values(
            sources,
            {
                "routed_expert_bit_plan",
                "routed_expert_bit_plans",
                "routed_expert_layer_bits",
                "routedExpertBitPlan",
                "routedExpertBitPlans",
                "routedExpertLayerBits",
                "routed_expert_bits",
                "routedExpertBits",
                "mxtq_bits_by_role",
                "mxtqBitsByRole",
                "mxtq_gate_up_bits",
                "mxtqGateUpBits",
                "mxtq_down_bits",
                "mxtqDownBits",
                "expert_bit_plan",
                "expert_bit_plans",
                "expert_bits",
            },
        ),
        is_vlm=is_vlm,
        is_kimi_vlm=is_kimi_vlm,
        is_deepseek_v4=_is_deepseek_v4_model(jang_config, model_config),
        bundle_path=bundle_path,
        embedded_sidecar=embedded_sidecar,
    )


def _read_embedded_jang_metadata(
    path: Path,
) -> tuple[Path, Path | None, dict[str, Any], dict[str, Any], bool] | None:
    candidates: list[tuple[Path, Path | None]] = [(path, None)]
    target_path = path / "target"
    if (path / "jangspec.json").is_file() and (target_path / "config.json").exists():
        candidates.insert(0, (target_path, path))

    for payload_path, bundle_path in candidates:
        model_config = _read_json_object_or_none(payload_path / "config.json")
        jang_config = _read_embedded_jang_config(model_config)
        if jang_config is not None and model_config is not None:
            return payload_path, bundle_path, model_config, jang_config, True
        if model_config is not None and _config_declares_jang_metadata(
            payload_path,
            model_config,
        ):
            return payload_path, bundle_path, model_config, model_config, False
    return None


def _safe_resolved_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve()
    except Exception:
        return expanded


def _quantization_dicts(source: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
    if not isinstance(source, dict):
        return
    for key in ("quantization", "quantization_config"):
        value = source.get(key)
        if isinstance(value, dict):
            yield value


def _has_explicit_empty_jang_bit_widths_used(
    metadata: JangQuantizationMetadata,
) -> bool:
    for source in (metadata.sidecar, metadata.model_config):
        if isinstance(source, dict):
            bit_widths = source.get("bit_widths_used")
            if bit_widths is None:
                bit_widths = source.get("bitWidthsUsed")
            if isinstance(bit_widths, list) and len(bit_widths) == 0:
                return True
        for quantization in _quantization_dicts(source):
            bit_widths = quantization.get("bit_widths_used")
            if bit_widths is None:
                bit_widths = quantization.get("bitWidthsUsed")
            if isinstance(bit_widths, list) and len(bit_widths) == 0:
                return True
    return False


def _bit_width_list_has_supported_width(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            width = int(item)
        except (TypeError, ValueError):
            continue
        if width in _JANG_SUPPORTED_BITS:
            return True
    return False


def _has_nonempty_jang_bit_widths_used(metadata: JangQuantizationMetadata) -> bool:
    for source in (metadata.sidecar, metadata.model_config):
        if isinstance(source, dict):
            bit_widths = source.get("bit_widths_used")
            if bit_widths is None:
                bit_widths = source.get("bitWidthsUsed")
            if _bit_width_list_has_supported_width(bit_widths):
                return True
        for quantization in _quantization_dicts(source):
            bit_widths = quantization.get("bit_widths_used")
            if bit_widths is None:
                bit_widths = quantization.get("bitWidthsUsed")
            if _bit_width_list_has_supported_width(bit_widths):
                return True
    return False


def _mxfp_sidecar_has_loader_metadata(metadata: JangQuantizationMetadata) -> bool:
    if _has_nonempty_jang_bit_widths_used(metadata):
        return True
    if metadata.is_vlm or metadata.is_deepseek_v4:
        return True

    loader_keys = {
        "architecture",
        "capabilities",
        "chat",
        "family",
        "has_audio",
        "has_image",
        "has_video",
        "has_vision",
        "modality",
        "modalities",
        "model_family",
        "runtime",
        "source_model",
    }
    for key, value in metadata.sidecar.items():
        normalized = _normalize_jang_marker(key) or key
        if normalized in loader_keys and value is not None:
            return True
    return False


def _should_use_standard_loader_for_jang_sidecar(
    metadata: JangQuantizationMetadata,
) -> bool:
    if metadata.codec == "jangtq":
        return False
    if metadata.codec == "mxfp" and not _mxfp_sidecar_has_loader_metadata(metadata):
        logger.info(
            "JANG sidecar %s declares MXFP without loader metadata; "
            "using the standard MLX loader for %s",
            metadata.sidecar_path,
            metadata.model_path,
        )
        return True
    if _has_explicit_empty_jang_bit_widths_used(metadata):
        logger.info(
            "JANG sidecar %s has empty quantization.bit_widths_used; "
            "using the standard MLX loader for %s",
            metadata.sidecar_path,
            metadata.model_path,
        )
        return True
    if _has_plain_mlx_jang_weight_format(metadata):
        logger.info(
            "JANG sidecar %s declares format/weight_format='mlx'; using the standard "
            "MLX loader for %s",
            metadata.sidecar_path,
            metadata.model_path,
        )
        return True
    return False


def _has_plain_mlx_jang_weight_format(metadata: JangQuantizationMetadata) -> bool:
    for value in _all_nested_values(
        (metadata.sidecar, metadata.model_config if metadata.model_config else {}),
        {"format", "weight_format", "weightFormat"},
    ):
        marker = _normalize_jang_marker(value)
        if marker == "mlx":
            return True
    return False


_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
)


def _has_tokenizer_artifacts(path: Path) -> bool:
    return any((path / name).is_file() for name in _TOKENIZER_FILENAMES)


def _has_usable_local_jang_tokenizer(path: Path) -> bool:
    if not _has_tokenizer_artifacts(path):
        return False
    tokenizer_config = _read_json_object_or_none(path / "tokenizer_config.json") or {}
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if (
        isinstance(tokenizer_class, str)
        and _normalize_jang_marker(tokenizer_class)
        in {"tiktokentokenizer", "tiktokentokenizerfast", "tokenizersbackend"}
    ):
        return (path / "tokenizer.json").is_file()
    return True


def _repo_id_from_jang_source_model(source_model: Any) -> str | None:
    def _valid_repo_id(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip().removeprefix("hf://")
        if not text or text.startswith(("/", ".")):
            return None
        if text.count("/") != 1:
            return None
        org, name = text.split("/", 1)
        if not org or not name:
            return None
        return f"{org}/{name}"

    direct = _valid_repo_id(source_model)
    if direct is not None:
        return direct
    if not isinstance(source_model, dict):
        return None

    for key in (
        "repo_id",
        "repoID",
        "hub_id",
        "hubID",
        "id",
        "name",
        "path",
        "_name_or_path",
    ):
        direct = _valid_repo_id(source_model.get(key))
        if direct is not None:
            return direct

    org = source_model.get("org") or source_model.get("organization")
    name = source_model.get("name") or source_model.get("model")
    if isinstance(org, str) and isinstance(name, str):
        name = name.strip().removeprefix("/")
        if org.strip() and name and "/" not in name:
            return f"{org.strip()}/{name}"
    return None


def _local_path_from_jang_source_model(
    source_model: Any,
    *,
    base_path: Path,
) -> Path | None:
    def _candidate(value: Any) -> Path | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text or text.startswith("hf://"):
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = base_path / path
        return path if path.is_dir() else None

    direct = _candidate(source_model)
    if direct is not None:
        return direct
    if not isinstance(source_model, dict):
        return None
    for key in ("local_path", "localPath", "path", "_name_or_path", "name_or_path"):
        direct = _candidate(source_model.get(key))
        if direct is not None:
            return direct
    return None


def _hf_hub_cache_root() -> Path:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _iter_hf_snapshot_dirs(repo_id: str) -> Iterator[Path]:
    cache_dir = _hf_hub_cache_root() / ("models--" + repo_id.replace("/", "--"))
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return
    yield from sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _resolve_jang_tokenizer_source(metadata: JangQuantizationMetadata) -> Path | None:
    if _has_usable_local_jang_tokenizer(metadata.model_path):
        return metadata.model_path

    for source_model in _jang_source_model_candidates(metadata):
        local_path = _local_path_from_jang_source_model(
            source_model,
            base_path=metadata.model_path,
        )
        if local_path is not None and _has_tokenizer_artifacts(local_path):
            return local_path

        repo_id = _repo_id_from_jang_source_model(source_model)
        if repo_id is None:
            continue

        for snapshot in _iter_hf_snapshot_dirs(repo_id):
            if _has_tokenizer_artifacts(snapshot):
                return snapshot
    return None


@contextmanager
def _scoped_jang_source_tokenizer_fallback(
    metadata: JangQuantizationMetadata,
    loader_path: Path | None = None,
) -> Iterator[None]:
    """Temporarily route tokenizer loading to a local JANG tokenizer source."""

    tokenizer_source = _resolve_jang_tokenizer_source(metadata)
    if tokenizer_source is None:
        yield
        return

    targets = {_safe_resolved_path(metadata.model_path)}
    if loader_path is not None:
        targets.add(_safe_resolved_path(loader_path))
    resolved_source = _safe_resolved_path(tokenizer_source)
    if all(resolved_source == target for target in targets):
        yield
        return

    try:
        import transformers
    except ImportError:
        yield
        return

    original = transformers.AutoTokenizer.from_pretrained

    def _from_pretrained_with_source_fallback(
        path: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if _safe_resolved_path(path) not in targets:
            return original(path, *args, **kwargs)
        logger.info(
            "Loading tokenizer for JANG bundle %s from cached source model %s",
            metadata.model_path,
            tokenizer_source,
        )
        return original(tokenizer_source, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = _from_pretrained_with_source_fallback  # type: ignore[method-assign, assignment]
    try:
        yield
    finally:
        transformers.AutoTokenizer.from_pretrained = original  # type: ignore[method-assign]


def _dsv4_bundle_config(model_path: str | Path) -> dict[str, Any] | None:
    config = _read_json_object_or_none(Path(model_path) / "config.json")
    if config is None or config.get("model_type") != "deepseek_v4":
        return None
    return config


def _dsv4_weight_map(model_path: str | Path) -> dict[str, str]:
    """Return tensor-key -> safetensors filename for a DSV4 bundle."""

    bundle = Path(model_path)
    for index_path in _iter_safetensors_index_paths(bundle):
        try:
            data = json.loads(index_path.read_text())
        except Exception as e:
            raise RuntimeError(
                f"Could not read safetensors index {index_path}: {e}"
            ) from e
        weight_map_data = data.get("weight_map")
        if not isinstance(weight_map_data, dict):
            raise RuntimeError(
                f"Safetensors index {index_path} has no weight_map object"
            )
        return {str(key): str(value) for key, value in weight_map_data.items()}

    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(f"safetensors is required for DSV4 header audit: {e}") from e

    weight_map: dict[str, str] = {}
    for shard in select_safetensors_weight_files(bundle):
        if shard.name == "jangtq_runtime.safetensors":
            continue
        try:
            with safe_open(str(shard), framework="numpy") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open exposes keys()
                    weight_map[str(key)] = shard.name
        except Exception as e:
            raise RuntimeError(f"Could not read safetensors header {shard}: {e}") from e
    return weight_map


def _audit_dsv4_control_tensor_dtypes(model_path: str | Path) -> dict[str, Any]:
    """Header-only audit for DSV4 control tensors that must remain F32."""

    report: dict[str, Any] = {
        "checked": False,
        "critical_count": 0,
        "non_f32_count": 0,
        "non_f32_examples": [],
        "error": None,
    }
    if _dsv4_bundle_config(model_path) is None:
        return report

    report["checked"] = True
    try:
        weight_map = _dsv4_weight_map(model_path)
        critical_keys = sorted(
            key for key in weight_map if _DSV4_CRITICAL_CONTROL_RE.match(key)
        )
        report["critical_count"] = len(critical_keys)
        if not critical_keys:
            report["error"] = "No DSV4 critical control tensors found in safetensors."
            return report

        from safetensors import safe_open

        handles: dict[str, Any] = {}
        try:
            for key in critical_keys:
                filename = weight_map[key]
                handle = handles.get(filename)
                if handle is None:
                    handle = safe_open(
                        str(Path(model_path) / filename), framework="numpy"
                    )
                    handle.__enter__()
                    handles[filename] = handle
                dtype = handle.get_slice(key).get_dtype()
                if dtype != "F32":
                    report["non_f32_count"] += 1
                    if len(report["non_f32_examples"]) < 12:
                        report["non_f32_examples"].append({"key": key, "dtype": dtype})
        finally:
            for handle in handles.values():
                handle.__exit__(None, None, None)
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
    return report


def _legacy_dsv4_slug_hint(model_path: str | Path) -> str:
    """Return a corrected-bundle hint for the retracted legacy DSV4 slug."""

    bundle = Path(model_path)
    config = _read_json_object_or_none(bundle / "config.json") or {}
    jang_config = _read_json_object_or_none(bundle / "jang_config.json") or {}
    haystack = " ".join(
        str(value)
        for value in (
            bundle,
            config.get("_name_or_path", ""),
            config.get("name_or_path", ""),
            jang_config.get("source_model", ""),
            jang_config.get("_name_or_path", ""),
        )
    )
    if (
        "DeepSeek-V4-Flash-JANGTQ" in haystack
        and "V3-F32" not in haystack
        and "V3_F32" not in haystack
    ):
        return (
            " The legacy DeepSeek-V4-Flash-JANGTQ bundle was retracted after "
            "F16 critical-control tensors were found; download a corrected "
            "V3-F32 / V3_F32 DSV4 bundle instead."
        )
    return ""


def _validate_dsv4_control_tensors(model_path: str | Path) -> None:
    """Raise a clear load-time error for known-bad DSV4 control tensors."""

    report = _audit_dsv4_control_tensor_dtypes(model_path)
    if not report.get("checked"):
        return
    if report.get("error"):
        raise RuntimeError(
            "DSV4 bundle integrity audit failed before load: "
            f"{report['error']}. Rebuild or re-download the DSV4 bundle."
            f"{_legacy_dsv4_slug_hint(model_path)}"
        )
    bad_count = int(report.get("non_f32_count") or 0)
    if bad_count:
        examples = ", ".join(
            f"{item['key']}={item['dtype']}"
            for item in report.get("non_f32_examples", [])[:6]
        )
        raise RuntimeError(
            "DSV4 bundle is known-bad: "
            f"{bad_count}/{report.get('critical_count')} critical control tensors "
            "are not F32. DSV4 mHC/Sinkhorn/router/sink tensors must retain "
            "source precision; F16/BF16 copies can cause long-context drift, "
            "language salad, and repetition loops. Rebuild with a DSV4-safe "
            f"converter or use a corrected bundle.{_legacy_dsv4_slug_hint(model_path)} "
            f"Examples: {examples}"
        )


def _validate_dsv4_artifact_bit_plan(model_path: str | Path) -> None:
    """Raise for DSV4 JANGTQ bit-plan or runtime sidecar inconsistencies."""

    report = _audit_dsv4_artifact_bit_plan(model_path)
    if not report.get("checked"):
        return
    issues = list(report.get("issues") or [])
    if issues:
        sample = "; ".join(str(issue) for issue in issues[:5])
        raise RuntimeError(
            "DSV4 JANGTQ artifact audit failed before load: "
            f"{sample}. Rebuild or re-download the DSV4 bundle."
            f"{_legacy_dsv4_slug_hint(model_path)}"
        )


def _dsv4_coerce_layer_bit_plan(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        return {}

    raw = value.get("routed_layer_bits")
    if raw is None:
        raw = value.get("routedLayerBits")
    if raw is None:
        raw = value.get("routed_expert_layer_bits")
    if raw is None:
        raw = value.get("routedExpertLayerBits")
    if isinstance(raw, dict):
        value = raw
    elif any(not str(key).isdigit() for key in value):
        return {}

    out: dict[str, int] = {}
    try:
        for key, bit_value in value.items():
            out[str(int(key))] = int(bit_value)
    except (TypeError, ValueError):
        return {}
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _dsv4_metadata_routed_layer_bits(
    config: dict[str, Any],
    jang_config: dict[str, Any],
) -> dict[str, int]:
    jang_quant = jang_config.get("quantization")
    if not isinstance(jang_quant, dict):
        jang_quant = {}
    jang_routed_experts = jang_quant.get("routed_experts")
    if not isinstance(jang_routed_experts, dict):
        jang_routed_experts = {}
    jang_bit_plan = jang_routed_experts.get("bit_plan")
    if not isinstance(jang_bit_plan, dict):
        jang_bit_plan = {}

    config_quant = config.get("quantization")
    if not isinstance(config_quant, dict):
        config_quant = {}
    config_routed_plan = config_quant.get("routed_expert_bit_plan")
    if not isinstance(config_routed_plan, dict):
        config_routed_plan = {}

    candidates: list[Any] = [
        jang_bit_plan.get("routed_layer_bits"),
        jang_quant.get("routed_layer_bits"),
        jang_config.get("routed_layer_bits"),
        jang_config.get("routed_expert_layer_bits"),
        jang_config.get("routedExpertLayerBits"),
        config_routed_plan.get("routed_layer_bits"),
        config_routed_plan.get("routed_expert_layer_bits"),
        config_routed_plan.get("routedExpertLayerBits"),
        config_quant.get("routed_layer_bits"),
        config_quant.get("routed_expert_layer_bits"),
        config_quant.get("routedExpertLayerBits"),
        config.get("routed_layer_bits"),
        config.get("routed_expert_layer_bits"),
        config.get("routedExpertLayerBits"),
    ]
    candidates.extend(
        _all_nested_values(
            (jang_config, config),
            {
                "routed_layer_bits",
                "routedLayerBits",
                "routed_expert_layer_bits",
                "routedExpertLayerBits",
                "routed_expert_bit_plan",
                "routedExpertBitPlan",
                "routed_expert_bit_plans",
                "routedExpertBitPlans",
            },
        )
    )
    for candidate in candidates:
        coerced = _dsv4_coerce_layer_bit_plan(candidate)
        if coerced:
            return coerced
    return {}


_DSV4_ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _normalize_dsv4_projection_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    aliases = {
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
    }
    name = aliases.get(name, name)
    return name if name in _DSV4_ROUTED_PROJECTIONS else None


def _dsv4_coerce_projection_bit_plan(value: Any) -> dict[str, int]:
    if isinstance(value, int):
        return {projection: int(value) for projection in _DSV4_ROUTED_PROJECTIONS}
    if not isinstance(value, dict) or not value:
        return {}

    for wrapper in ("routed_expert", "routed_experts", "default", "default_bits"):
        nested = value.get(wrapper)
        nested_plan = _dsv4_coerce_projection_bit_plan(nested)
        if nested_plan:
            return nested_plan

    out: dict[str, int] = {}
    try:
        for raw_key, raw_bits in value.items():
            projection = _normalize_dsv4_projection_name(raw_key)
            if projection is None:
                continue
            if raw_bits is None:
                continue
            out[projection] = int(raw_bits)
    except (TypeError, ValueError):
        return {}
    return dict(sorted(out.items()))


def _dsv4_metadata_routed_projection_bits(
    config: dict[str, Any],
    jang_config: dict[str, Any],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Return default and per-layer routed projection bit plans from metadata."""

    sources = (jang_config, config)
    default_candidates: list[Any] = [
        jang_config.get("mxtq_bits"),
        config.get("mxtq_bits"),
        jang_config.get("mxtq_bits_by_role"),
        config.get("mxtq_bits_by_role"),
        jang_config.get("mxtqBits"),
        config.get("mxtqBits"),
        jang_config.get("mxtqBitsByRole"),
        config.get("mxtqBitsByRole"),
        jang_config.get("routed_expert_bits"),
        config.get("routed_expert_bits"),
        jang_config.get("routedExpertBits"),
        config.get("routedExpertBits"),
    ]

    gate_up_bits = _first_nested_value(sources, {"mxtq_gate_up_bits", "mxtqGateUpBits"})
    down_bits = _first_nested_value(sources, {"mxtq_down_bits", "mxtqDownBits"})
    if gate_up_bits is not None or down_bits is not None:
        projected: dict[str, Any] = {}
        if gate_up_bits is not None:
            projected["gate_proj"] = gate_up_bits
            projected["up_proj"] = gate_up_bits
        if down_bits is not None:
            projected["down_proj"] = down_bits
        default_candidates.append(projected)

    for source in sources:
        runtime = source.get("runtime") if isinstance(source, dict) else None
        if isinstance(runtime, dict):
            default_candidates.extend(
                [
                    runtime.get("mxtq_bits"),
                    runtime.get("mxtqBits"),
                    runtime.get("mxtq_bits_by_role"),
                    runtime.get("mxtqBitsByRole"),
                    runtime.get("routed_expert_bits"),
                    runtime.get("routedExpertBits"),
                ]
            )
            routed_plan = runtime.get("routed_expert_bit_plan") or runtime.get(
                "routedExpertBitPlan"
            )
            if isinstance(routed_plan, dict):
                default_candidates.append(routed_plan.get("default"))
                default_candidates.append(routed_plan.get("default_bits"))
                default_candidates.append(routed_plan.get("defaultBits"))

    for source in sources:
        for item in _iter_dicts(source):
            plan = item.get("routed_expert_bit_plan")
            if not isinstance(plan, dict):
                plan = item.get("routedExpertBitPlan")
            if not isinstance(plan, dict):
                continue
            default_candidates.extend(
                [
                    plan.get("default"),
                    plan.get("default_bits"),
                    plan.get("defaultBits"),
                ]
            )

    default_bits: dict[str, int] = {}
    for candidate in default_candidates:
        default_bits = _dsv4_coerce_projection_bit_plan(candidate)
        if default_bits:
            break

    layer_overrides: dict[str, dict[str, int]] = {}
    for source in sources:
        for item in _iter_dicts(source):
            plan = item.get("routed_expert_bit_plan")
            if not isinstance(plan, dict):
                continue
            overrides = plan.get("layer_overrides") or plan.get("layerOverrides")
            if not isinstance(overrides, dict):
                continue
            for raw_layer, raw_plan in overrides.items():
                try:
                    layer = str(int(raw_layer))
                except (TypeError, ValueError):
                    continue
                projection_plan = _dsv4_coerce_projection_bit_plan(raw_plan)
                if projection_plan:
                    layer_overrides[layer] = projection_plan
    return default_bits, dict(
        sorted(layer_overrides.items(), key=lambda item: int(item[0]))
    )


def _dsv4_expected_projection_bits_for_layer(
    layer: str,
    default_projection_bits: dict[str, int],
    projection_overrides: dict[str, dict[str, int]],
) -> dict[str, int]:
    expected = dict(default_projection_bits)
    expected.update(projection_overrides.get(layer, {}))
    return {key: expected[key] for key in _DSV4_ROUTED_PROJECTIONS if key in expected}


def _dsv4_routed_default_bits(
    config: dict[str, Any],
    jang_config: dict[str, Any],
) -> int:
    for source in (jang_config, config):
        nested_bits = _first_nested_value(
            (source,),
            {
                "mxtq_bits",
                "mxtqBits",
                "mxtq_bits_by_role",
                "mxtqBitsByRole",
                "routed_expert_bits",
                "routedExpertBits",
            },
        )
        if (
            isinstance(nested_bits, dict)
            and nested_bits.get("routed_expert") is not None
        ):
            routed_bits = nested_bits["routed_expert"]
            if isinstance(routed_bits, dict):
                values = [int(value) for value in routed_bits.values()]
                return min(values) if values else 2
            return int(routed_bits)
        if isinstance(nested_bits, int):
            return int(nested_bits)
        routed_plan_default = _jang_routed_expert_bit_plan_default((source,))
        if routed_plan_default is not None:
            projection_plan = _dsv4_coerce_projection_bit_plan(routed_plan_default)
            if projection_plan:
                return min(projection_plan.values())
            try:
                return int(routed_plan_default)
            except (TypeError, ValueError):
                pass
        mxtq_bits = source.get("mxtq_bits")
        if isinstance(mxtq_bits, dict) and mxtq_bits.get("routed_expert") is not None:
            return int(mxtq_bits["routed_expert"])
        quantization = source.get("quantization")
        if not isinstance(quantization, dict):
            continue
        quant_mxtq_bits = quantization.get("mxtq_bits")
        if (
            isinstance(quant_mxtq_bits, dict)
            and quant_mxtq_bits.get("routed_expert") is not None
        ):
            return int(quant_mxtq_bits["routed_expert"])
        routed_experts = quantization.get("routed_experts")
        if isinstance(routed_experts, dict) and routed_experts.get("bits") is not None:
            return int(routed_experts["bits"])
        if quantization.get("routed_expert_bits") is not None:
            return int(quantization["routed_expert_bits"])
        bit_widths = _jang_bit_widths_used((source,))
        if bit_widths:
            return min(bit_widths)
    return 2


def _audit_dsv4_artifact_bit_plan(model_path: str | Path) -> dict[str, Any]:
    """Audit DSV4 routed TQ bit plans and required runtime sidecar keys."""

    bundle = Path(model_path)
    config = _dsv4_bundle_config(bundle)
    jang_config = _read_json_object_or_none(bundle / "jang_config.json") or {}
    report: dict[str, Any] = {
        "checked": False,
        "routed_layer_bits": {},
        "routed_projection_bits": {},
        "routed_bit_counts": {},
        "metadata_routed_layer_bits": {},
        "metadata_routed_projection_bits": {},
        "metadata_routed_projection_layer_overrides": {},
        "metadata_matches_actual": None,
        "metadata_layer_value_mismatches": {},
        "metadata_projection_value_mismatches": {},
        "actual_overrides_missing_from_metadata": {},
        "projection_mismatches": [],
        "missing_projection_layers": [],
        "sidecar": {
            "present": False,
            "required_keys": [],
            "missing_keys": [],
        },
        "issues": [],
    }
    if config is None:
        return report
    report["checked"] = True

    try:
        from safetensors import safe_open

        weight_map = _dsv4_weight_map(bundle)
    except Exception as e:
        report["issues"].append(f"{type(e).__name__}: {e}")
        return report

    seed = int(jang_config.get("mxtq_seed") or config.get("mxtq_seed") or 42)
    by_layer: dict[str, dict[str, int]] = {}
    required_sidecar_keys: set[str] = set()

    def _read_scalar(key: str) -> int:
        with safe_open(str(bundle / weight_map[key]), framework="numpy") as handle:
            tensor = handle.get_tensor(key)
        return int(tensor.reshape(-1)[0])

    def _read_shape(key: str) -> tuple[int, ...] | None:
        filename = weight_map.get(key)
        if filename is None:
            return None
        with safe_open(str(bundle / filename), framework="numpy") as handle:
            return tuple(int(dim) for dim in handle.get_slice(key).get_shape())

    for key in sorted(weight_map):
        match = _DSV4_SWITCH_MLP_TQ_BITS_RE.match(key)
        if match is None:
            continue
        layer, projection = match.groups()
        try:
            bits = _read_scalar(key)
        except Exception as e:
            report["issues"].append(f"failed to read {key}: {type(e).__name__}: {e}")
            continue
        by_layer.setdefault(layer, {})[projection] = bits

        packed_shape = _read_shape(key[: -len(".tq_bits")] + ".tq_packed")
        if packed_shape is not None and len(packed_shape) >= 3 and bits > 0:
            values_per_u32 = max(1, 32 // bits)
            in_features = int(packed_shape[-1]) * values_per_u32
            required_sidecar_keys.add(f"codebook.{in_features}.{bits}")
            required_sidecar_keys.add(f"signs.{in_features}.{seed}")

    default_projection_bits, projection_overrides = (
        _dsv4_metadata_routed_projection_bits(config, jang_config)
    )
    report["metadata_routed_projection_bits"] = default_projection_bits
    report["metadata_routed_projection_layer_overrides"] = projection_overrides

    expected_projections = set(_DSV4_ROUTED_PROJECTIONS)
    routed_layer_bits: dict[str, int] = {}
    routed_projection_bits: dict[str, dict[str, int]] = {}
    for layer, projections in sorted(by_layer.items(), key=lambda item: int(item[0])):
        missing = sorted(expected_projections - set(projections))
        if missing:
            report["missing_projection_layers"].append(
                {"layer": layer, "missing": missing}
            )
            report["issues"].append(
                f"layer {layer} missing routed tq_bits projections: "
                + ", ".join(missing)
            )
            continue
        projection_report = dict(sorted(projections.items()))
        routed_projection_bits[layer] = projection_report
        values = set(projections.values())
        if len(values) != 1:
            expected = _dsv4_expected_projection_bits_for_layer(
                layer,
                default_projection_bits,
                projection_overrides,
            )
            if expected != projection_report:
                report["projection_mismatches"].append(
                    {"layer": layer, "projections": projection_report}
                )
                report["issues"].append(
                    f"layer {layer} routed projections have mixed tq_bits without "
                    f"matching JANG metadata: {projection_report}"
                )
            continue
        routed_layer_bits[layer] = next(iter(values))

    report["routed_layer_bits"] = routed_layer_bits
    report["routed_projection_bits"] = routed_projection_bits
    bit_counts: dict[str, int] = {}
    for layer, projections in routed_projection_bits.items():
        if layer in routed_layer_bits:
            bits = routed_layer_bits[layer]
            bit_counts[str(bits)] = bit_counts.get(str(bits), 0) + 1
            continue
        for bits in projections.values():
            bit_counts[str(bits)] = bit_counts.get(str(bits), 0) + 1
    report["routed_bit_counts"] = dict(
        sorted(bit_counts.items(), key=lambda item: int(item[0]))
    )

    default_bits = _dsv4_routed_default_bits(config, jang_config)
    actual_overrides = {
        layer: bits
        for layer, bits in routed_layer_bits.items()
        if int(bits) != default_bits
    }
    metadata = _dsv4_metadata_routed_layer_bits(config, jang_config)
    metadata_mismatches = {
        layer: {"metadata": bits, "actual": routed_layer_bits.get(layer)}
        for layer, bits in metadata.items()
        if routed_layer_bits.get(layer) != bits
    }
    missing_actual_overrides = {
        layer: bits
        for layer, bits in actual_overrides.items()
        if metadata.get(layer) != bits
    }
    metadata_projection_mismatches = {
        layer: {"metadata": expected, "actual": routed_projection_bits.get(layer)}
        for layer, expected in (
            (
                layer,
                _dsv4_expected_projection_bits_for_layer(
                    layer,
                    default_projection_bits,
                    projection_overrides,
                ),
            )
            for layer in routed_projection_bits
        )
        if expected and routed_projection_bits.get(layer) != expected
    }
    report["metadata_routed_layer_bits"] = metadata
    report["metadata_layer_value_mismatches"] = metadata_mismatches
    report["metadata_projection_value_mismatches"] = metadata_projection_mismatches
    report["actual_overrides_missing_from_metadata"] = missing_actual_overrides
    report["metadata_matches_actual"] = (
        not metadata_mismatches
        and not missing_actual_overrides
        and not metadata_projection_mismatches
    )
    if metadata_projection_mismatches:
        report["issues"].append(
            "metadata routed projection bit plan does not match actual headers "
            f"{metadata_projection_mismatches}"
        )
    if metadata and not report["metadata_matches_actual"]:
        report["issues"].append(
            f"metadata routed_layer_bits {metadata} does not match actual headers "
            f"(value_mismatches={metadata_mismatches}, "
            f"missing_non_default_overrides={missing_actual_overrides})"
        )

    sidecar_path = bundle / "jangtq_runtime.safetensors"
    required_sorted = sorted(required_sidecar_keys)
    report["sidecar"]["present"] = sidecar_path.is_file()
    report["sidecar"]["required_keys"] = required_sorted
    if required_sorted:
        if not sidecar_path.is_file():
            report["sidecar"]["missing_keys"] = required_sorted
            report["issues"].append(
                "sidecar missing: jangtq_runtime.safetensors is required for "
                f"{len(required_sorted)} observed routed TQ codebook/sign keys"
            )
        else:
            try:
                with safe_open(str(sidecar_path), framework="numpy") as handle:
                    present = set(handle.keys())
                missing_keys = [key for key in required_sorted if key not in present]
                report["sidecar"]["missing_keys"] = missing_keys
                if missing_keys:
                    report["issues"].append(
                        "sidecar missing required JANGTQ runtime keys: "
                        + ", ".join(missing_keys)
                    )
            except Exception as e:
                report["sidecar"]["missing_keys"] = required_sorted
                report["issues"].append(f"sidecar unreadable: {type(e).__name__}: {e}")

    return report


def _is_dsv4_flash_like_config(
    config: dict[str, Any],
    *,
    model_path: str | Path | None = None,
) -> bool:
    """Return True for DSV4-Flash-like configs that use compressed RoPE."""

    if config.get("model_type") != "deepseek_v4":
        return False

    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list):
        return False
    try:
        if not any(int(ratio or 0) > 0 for ratio in ratios):
            return False
    except (TypeError, ValueError):
        return False

    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if _int_or_none(config.get("hidden_size")) not in (None, 4096):
        return False
    if _int_or_none(config.get("head_dim")) not in (None, 512):
        return False
    rope_head_dim = _int_or_none(
        config.get("qk_rope_head_dim", config.get("rope_head_dim"))
    )
    if rope_head_dim not in (None, 64):
        return False

    if model_path is not None and "DeepSeek-V4-Flash" in str(model_path):
        return True
    return _int_or_none(config.get("compress_rope_theta")) == 160000


def _normalize_dsv4_jangtq_config(
    config: dict[str, Any],
    *,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Repair stale DSV4-Flash compressed-RoPE metadata without mutating input."""

    if not _is_dsv4_flash_like_config(config, model_path=model_path):
        return config, False

    repaired = dict(config)
    rope_scaling = config.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or not rope_scaling:
        repaired["rope_scaling"] = dict(_DSV4_FLASH_REQUIRED_ROPE_SCALING)
        return repaired, True

    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    if rope_type not in ("yarn", "deepseek_yarn"):
        return config, False

    missing = [
        key
        for key in (
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
        )
        if key not in rope_scaling
    ]
    if not missing:
        return config, False

    normalized = dict(_DSV4_FLASH_REQUIRED_ROPE_SCALING)
    normalized.update(rope_scaling)
    repaired["rope_scaling"] = normalized
    return repaired, True


@contextmanager
def _scoped_dsv4_jangtq_load_config(model_path: Path) -> Iterator[None]:
    """Temporarily normalize ``mlx_lm.utils.load_config`` for one DSV4 bundle."""

    import mlx_lm.utils as mlx_lm_utils

    target = _safe_resolved_path(model_path)
    original = mlx_lm_utils.load_config

    def _load_config_with_dsv4_repairs(path: Any, *args: Any, **kwargs: Any) -> Any:
        config = original(path, *args, **kwargs)
        if _safe_resolved_path(path) != target:
            return config
        repaired, changed = _normalize_dsv4_jangtq_config(
            config,
            model_path=target,
        )
        if changed:
            logger.warning(
                "DeepSeek V4 JANGTQ config repair: injected conservative "
                "DeepSeek-V4-Flash YaRN rope_scaling for %s",
                model_path,
            )
        return repaired

    mlx_lm_utils.load_config = _load_config_with_dsv4_repairs  # type: ignore[assignment]
    try:
        yield
    finally:
        mlx_lm_utils.load_config = original


@contextmanager
def _scoped_dsv4_jangtq_tokenizer_fallback(model_path: Path) -> Iterator[None]:
    """Temporarily fall back to local tokenizer.json for one DSV4 bundle."""

    import transformers

    target = _safe_resolved_path(model_path)
    original = transformers.AutoTokenizer.from_pretrained

    def _from_pretrained_with_local_dsv4_fallback(
        path: Any, *args: Any, **kwargs: Any
    ) -> Any:
        try:
            return original(path, *args, **kwargs)
        except (AttributeError, ValueError) as exc:
            current = _safe_resolved_path(path)
            if current != target:
                raise

            config = _read_json_object_or_none(current / "config.json") or {}
            tokenizer_json = current / "tokenizer.json"
            if (
                config.get("model_type") != "deepseek_v4"
                or not tokenizer_json.is_file()
            ):
                raise

            message = str(exc)
            if not any(
                marker in message
                for marker in (
                    "deepseek_v4",
                    "DeepSeek",
                    "rope_scaling",
                    "max_position_embeddings",
                    "Unrecognized model",
                )
            ):
                raise

            from transformers import PreTrainedTokenizerFast

            logger.warning(
                "DeepSeek V4 JANGTQ tokenizer repair: AutoTokenizer rejected "
                "config metadata for %s; loading local tokenizer.json instead: %s",
                model_path,
                exc,
            )
            return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))

    patched = _from_pretrained_with_local_dsv4_fallback
    transformers.AutoTokenizer.from_pretrained = patched  # type: ignore[method-assign, assignment]
    try:
        yield
    finally:
        transformers.AutoTokenizer.from_pretrained = original  # type: ignore[method-assign]


def _dsv4_total_memory_gb_decimal() -> float | None:
    """Return host RAM in decimal GB when it can be determined."""

    try:
        import psutil

        return float(psutil.virtual_memory().total) / 1e9
    except Exception:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return float(page_size) * float(page_count) / 1e9
    except (AttributeError, OSError, ValueError):
        return None


def _configure_dsv4_pool_quant_default() -> str:
    """Default DSV4 to the native materialized pool codec unless overridden."""

    os.environ.setdefault("DSV4_LONG_CTX", "1")
    if "DSV4_POOL_QUANT" not in os.environ:
        os.environ["DSV4_POOL_QUANT"] = "1"
    return os.environ["DSV4_POOL_QUANT"]


def _install_dsv4_memory_defaults() -> None:
    """Install conservative DSV4 JANGTQ/MLX memory defaults."""

    if sys.platform == "darwin" and "JANGTQ_WIRED_LIMIT_GB" not in os.environ:
        total_gb = _dsv4_total_memory_gb_decimal()
        if total_gb is not None and total_gb > 0:
            target_gb = int(total_gb * 0.52)
            target_gb = max(48, min(target_gb, 160))
            os.environ["JANGTQ_WIRED_LIMIT_GB"] = str(target_gb)

    cache_limit_raw = os.environ.setdefault("DSV4_MLX_CACHE_LIMIT_GB", "8")
    try:
        cache_gb = float(cache_limit_raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid DSV4_MLX_CACHE_LIMIT_GB=%r; expected GB value",
            cache_limit_raw,
        )
        return

    set_cache_limit = getattr(mx, "set_cache_limit", None)
    if cache_gb > 0 and callable(set_cache_limit):
        try:
            set_cache_limit(int(cache_gb * 1024**3))
        except Exception as exc:
            logger.warning("Failed to set DSV4 MLX cache limit: %s", exc)


def _configure_dsv4_runtime_defaults() -> None:
    """Configure DSV4 runtime defaults while preserving explicit overrides."""

    _configure_dsv4_pool_quant_default()
    _install_dsv4_memory_defaults()


def _import_dsv4_mlx_register() -> None:
    """Eagerly import DSV4 MLX registration so loaders can resolve the model."""

    try:
        importlib.import_module("jang_tools.dsv4.mlx_register")
    except ImportError as exc:
        raise ImportError(
            "DeepSeek V4 JANG loading requires jang_tools.dsv4.mlx_register. "
            "Install or upgrade the 'jang' package with DSV4 support."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "DeepSeek V4 JANG runtime registration failed while importing "
            f"jang_tools.dsv4.mlx_register: {exc}"
        ) from exc


def _patch_dsv4_pool_quant_cache_methods() -> None:
    """Install singleton batching methods on JANG's pool-quant DSV4 cache."""
    try:
        from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    except ImportError:
        return

    if not hasattr(PoolQuantizedV4Cache, "extract"):

        def _extract_singleton(self, idx: int):
            if int(idx) != 0:
                raise IndexError(
                    f"{type(self).__name__} singleton cache only has row 0"
                )
            return self

        PoolQuantizedV4Cache.extract = _extract_singleton

    if not hasattr(PoolQuantizedV4Cache, "filter"):

        def _filter_singleton(self, batch_indices):
            try:
                n = len(batch_indices)
            except TypeError:
                n = int(getattr(batch_indices, "shape", (0,))[0] or 0)
            if n == 0:
                self.keys = None
                self.values = None
                self.offset = 0
                self.compressor_state = {
                    "buffer_kv": None,
                    "buffer_gate": None,
                    "pooled": None,
                }
                self.indexer_state = {
                    "buffer_kv": None,
                    "buffer_gate": None,
                    "pooled": None,
                }
                return
            if n == 1:
                return
            raise NotImplementedError(
                f"{type(self).__name__}.filter with batch_size={n} > 1 is not "
                "implemented"
            )

        PoolQuantizedV4Cache.filter = _filter_singleton


def _verify_dsv4_attention_contract() -> None:
    """Verify the installed JANG DSV4 attention path has required fixes."""

    try:
        from jang_tools.dsv4.mlx_model import DeepseekV4Attention
    except Exception as exc:
        raise RuntimeError(
            "DSV4 installed jang_tools attention contract could not be verified "
            f"because imports failed: {exc}"
        ) from exc

    try:
        source = inspect.getsource(DeepseekV4Attention.__call__)
    except Exception as exc:
        raise RuntimeError(
            "DSV4 installed jang_tools attention contract could not be verified "
            f"because source inspection failed: {exc}"
        ) from exc

    required = {
        "symmetric_mask_trim": "if attn_mask.shape[-1] > full_kv.shape[2]",
        "per_query_compressed_pool_mask": "comp_mask = comp_mask & selected",
        "indexer_topk_threshold": "pooled.shape[1] > self.indexer.index_topk",
    }
    missing = [name for name, marker in required.items() if marker not in source]
    if missing:
        raise RuntimeError(
            "DSV4 installed jang_tools is too old for production DSV4 runtime: "
            "DeepseekV4Attention is missing "
            + ", ".join(missing)
            + ". Update jang_tools so DSV4 includes the native mask-width and "
            "per-query compressed-pool attention fixes."
        )


def _iter_model_modules(model: Any) -> Iterator[tuple[str, Any]]:
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        yield from named_modules()
        return

    leaf_modules = getattr(model, "leaf_modules", None)
    if callable(leaf_modules):
        try:
            modules = leaf_modules()
            if isinstance(modules, dict):
                for name, module in modules.items():
                    yield str(name), module
            else:
                for idx, module in enumerate(modules):
                    yield str(idx), module
            return
        except Exception:
            pass

    seen: set[int] = set()

    def _walk(name: str, value: Any) -> Iterator[tuple[str, Any]]:
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        yield name, value
        if isinstance(value, (str, bytes, bytearray)):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_name = f"{name}.{key}" if name else str(key)
                yield from _walk(child_name, child)
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                child_name = f"{name}.{idx}" if name else str(idx)
                yield from _walk(child_name, child)
            return
        children = getattr(value, "children", None)
        if callable(children):
            try:
                child_values = children()
            except Exception:
                child_values = None
            if isinstance(child_values, dict):
                for key, child in child_values.items():
                    child_name = f"{name}.{key}" if name else str(key)
                    yield from _walk(child_name, child)

    yield from _walk("", model)


def _audit_dsv4_switchglu_contract(model: Any) -> None:
    """Fail loudly if DSV4 routed experts lost limited-SwiGLU semantics."""

    try:
        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
        from mlx_lm.models.switch_layers import SwitchGLU
    except Exception as exc:
        raise RuntimeError(
            "DSV4 routed-expert contract audit could not import SwitchGLU/"
            f"TurboQuant classes: {exc}"
        ) from exc

    tq_switchglus: list[str] = []
    bad_limit: list[tuple[str, float]] = []
    for name, module in _iter_model_modules(model):
        if not isinstance(module, SwitchGLU):
            continue
        if not (
            isinstance(getattr(module, "gate_proj", None), TurboQuantSwitchLinear)
            and isinstance(getattr(module, "up_proj", None), TurboQuantSwitchLinear)
            and isinstance(getattr(module, "down_proj", None), TurboQuantSwitchLinear)
        ):
            continue

        tq_switchglus.append(name)
        activation = getattr(module, "activation", None)
        try:
            swiglu_limit = float(getattr(activation, "swiglu_limit", 0.0) or 0.0)
        except (TypeError, ValueError):
            swiglu_limit = 0.0
        if abs(swiglu_limit - 10.0) > 1e-6:
            bad_limit.append((name, swiglu_limit))

    if not tq_switchglus:
        raise RuntimeError(
            "DSV4 routed-expert contract audit found zero TurboQuant SwitchGLU "
            "modules. A DSV4 JANGTQ bundle must hydrate routed experts before "
            "inference."
        )
    if bad_limit:
        sample = ", ".join(
            f"{name or '<root>'}={limit:g}" for name, limit in bad_limit[:5]
        )
        raise RuntimeError(
            "DSV4 routed-expert contract audit failed: limited-SwiGLU "
            f"swiglu_limit=10 missing on {len(bad_limit)}/{len(tq_switchglus)} "
            f"TurboQuant SwitchGLU modules ({sample})."
        )


def _install_dsv4_canonical_chat_template(tokenizer: Any, model_path: Path) -> None:
    """Install the canonical DSV4 chat-template shim when available."""

    try:
        from omlx.patches.deepseek_v4 import dsv4_chat_encoder
    except Exception as exc:
        logger.debug("DSV4 canonical chat-template shim unavailable: %s", exc)
        return

    install = getattr(dsv4_chat_encoder, "install_canonical_chat_template", None)
    if not callable(install):
        return
    try:
        install(tokenizer, model_path)
    except Exception as exc:
        logger.warning(
            "Could not install canonical DSV4 chat-template shim for %s: %s",
            model_path,
            exc,
        )


def _env_flag_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_install_dsv4_fast_load_patch(model_path: str | Path | None = None) -> bool:
    """Evaluate DSV4 fast-load support, but do not install sidecar patches.

    Upstream vMLX has performance-only DSV4 stacked-sidecar fast-load machinery.
    oMLX intentionally leaves that disabled until tests can prove parity with
    the installed streaming hydrate path. This explicit no-op gate lets loader
    integration call it safely without model-dir or cache-dir sidecar writes.
    """

    if _env_flag_enabled(os.environ.get(_DSV4_FAST_LOAD_DISABLE_ENV)):
        logger.debug(
            "DSV4 sidecar fast-load patch disabled by %s=1; using installed "
            "jang_tools hydrate path unchanged",
            _DSV4_FAST_LOAD_DISABLE_ENV,
        )
        return False

    try:
        load_jangtq = importlib.import_module("jang_tools.load_jangtq")
    except ImportError as exc:
        logger.debug("Could not inspect jang_tools DSV4 fast-load hooks: %s", exc)
        return False

    hydrate = getattr(load_jangtq, "_hydrate_dsv4_jangtq_streaming", None)
    if not callable(hydrate):
        logger.debug(
            "DSV4 sidecar fast-load patch not installed: installed "
            "jang_tools.load_jangtq has no _hydrate_dsv4_jangtq_streaming hook"
        )
        return False

    global _DSV4_FAST_LOAD_NOTICE_LOGGED
    if not _DSV4_FAST_LOAD_NOTICE_LOGGED:
        logger.info(
            "DSV4 sidecar fast-load patch not installed for %s: parity with "
            "the installed jang_tools streaming hydrate hook is not proven; "
            "using the streaming hydrate path unchanged",
            model_path if model_path is not None else "DSV4 bundle",
        )
        _DSV4_FAST_LOAD_NOTICE_LOGGED = True
    return False


def _load_deepseek_v4_jang(
    model_path: Path,
    metadata: JangQuantizationMetadata,
) -> tuple[Any, Any]:
    """Load DeepSeek V4 JANG codecs through DSV4 runtime contracts."""

    is_jangtq = metadata.codec == "jangtq"
    if metadata.codec not in {"jangtq", "affine_jang", "mxfp"}:
        raise ValueError(
            f"Unsupported DeepSeek V4 JANG quantization codec {metadata.codec!r} "
            f"for {model_path}"
        )

    if is_jangtq:
        _validate_dsv4_control_tensors(model_path)
        _validate_dsv4_artifact_bit_plan(model_path)

    _configure_dsv4_runtime_defaults()
    _import_dsv4_mlx_register()
    _patch_dsv4_pool_quant_cache_methods()
    _verify_dsv4_attention_contract()
    if is_jangtq:
        _maybe_install_dsv4_fast_load_patch(model_path)

    if is_jangtq:
        from jang_tools.load_jangtq import load_jangtq_model as load_model
    else:
        from jang_tools.loader import load_jang_model as load_model

    with (
        _scoped_dsv4_jangtq_load_config(model_path),
        _scoped_dsv4_jangtq_tokenizer_fallback(model_path),
    ):
        model, tokenizer = cast(tuple[Any, Any], load_model(model_path))
    if is_jangtq:
        _audit_dsv4_switchglu_contract(model)
    _install_dsv4_canonical_chat_template(tokenizer, model_path)
    return model, tokenizer


def _is_laguna_jang_model(metadata: JangQuantizationMetadata) -> bool:
    """Return whether metadata identifies the text-only Laguna runtime."""

    model_config = metadata.model_config or {}
    model_types = [model_config.get("model_type")]
    text_config = model_config.get("text_config")
    if isinstance(text_config, Mapping):
        model_types.append(text_config.get("model_type"))

    return any(
        _normalize_jang_marker(value) == "laguna"
        for value in (*model_types, metadata.model_family)
    )


def _load_laguna_jang(metadata: JangQuantizationMetadata) -> tuple[Any, Any]:
    """Load a Laguna JANG bundle through its architecture-specific runtime."""

    try:
        from jang_tools.laguna.runtime import load as load_laguna
    except ImportError as exc:
        raise ImportError(
            "Laguna JANG loading requires the jang_tools.laguna runtime. "
            'Install or upgrade with: pip install "jang[vlm]>=2.5.31".'
        ) from exc

    model, _, bundle_format = load_laguna(str(metadata.model_path))
    model.config = dict(metadata.model_config or {"model_type": "laguna"})

    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(
        str(metadata.model_path),
        trust_remote_code=False,
    )

    from .generation_config import load_generation_config_token_ids

    eos_token_ids = load_generation_config_token_ids(
        metadata.model_path,
        "eos_token_id",
    )
    if eos_token_ids:
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        tokenizer = TokenizerWrapper(
            tokenizer,
            eos_token_ids=sorted(eos_token_ids),
        )

    logger.info(
        "Loaded Laguna JANG bundle %s with %s format",
        metadata.model_path,
        bundle_format,
    )
    return model, tokenizer


def _load_jang_quantization(
    model_path: Path,
    *,
    is_vlm: bool,
    jang_metadata: JangQuantizationMetadata | None = None,
) -> tuple[Any, Any]:
    if jang_metadata is None:
        jang_metadata = read_jang_metadata(model_path)
    if jang_metadata is None:
        raise ValueError(f"No JANG config found in {model_path}")

    load_path = jang_metadata.model_path
    bundle_path = jang_metadata.bundle_path
    model_is_vlm = jang_metadata.is_vlm
    if model_is_vlm and not is_vlm:
        raise ValueError(
            "JANG loader was asked to load a VLM artifact as text-only. "
            f"Use the VLM engine for: {load_path}"
        )
    if is_vlm and not model_is_vlm:
        raise ValueError(
            "JANG loader was asked to load a text-only model through the VLM path: "
            f"{load_path}"
        )

    if jang_metadata.codec == "unknown_jang":
        markers = ", ".join(jang_metadata.markers) or "none"
        profile = jang_metadata.profile or "none"
        raise ValueError(
            "Unsupported JANG quantization codec for "
            f"{load_path}. Sidecar: {jang_metadata.sidecar_path.name}; "
            f"markers: {markers}; profile: {profile}. "
            "Install or upgrade to a newer 'jang' package that supports this "
            "JANG sidecar format."
        )

    if jang_metadata.codec == "jangtq":
        _preflight_jangtq_runtime_sidecar(jang_metadata)

    try:
        if _is_laguna_jang_model(jang_metadata):
            return _load_laguna_jang(jang_metadata)

        _materialize_embedded_jang_sidecar(jang_metadata)
        with _scoped_jang_grouped_conv1d_load_weights_sanitize():
            if bundle_path is not None:
                try:
                    from jang_tools.jangspec.bundle_loader import (
                        load_jang_model_from_bundle,
                    )
                except ImportError:
                    logger.debug(
                        "JANGSpec bundle loader unavailable; falling back to %s",
                        load_path,
                    )
                else:
                    with _scoped_jang_source_tokenizer_fallback(
                        jang_metadata, bundle_path
                    ):
                        return cast(
                            tuple[Any, Any],
                            load_jang_model_from_bundle(bundle_path),
                        )

            load_path, temp_dir = _temporary_jang_loader_path(jang_metadata)

            if jang_metadata.is_deepseek_v4 and not is_vlm:
                return _retain_jang_loader_temp_dir(
                    _load_deepseek_v4_jang(load_path, jang_metadata),
                    temp_dir,
                )

            with _scoped_jang_source_tokenizer_fallback(jang_metadata, load_path):
                if jang_metadata.codec == "jangtq" and is_vlm:
                    if jang_metadata.is_kimi_vlm:
                        from jang_tools.load_jangtq_kimi_vlm import (
                            load_jangtq_kimi_vlm_model,
                        )

                        return _retain_jang_loader_temp_dir(
                            cast(
                                tuple[Any, Any],
                                load_jangtq_kimi_vlm_model(load_path),
                            ),
                            temp_dir,
                        )

                    from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model

                    return _retain_jang_loader_temp_dir(
                        cast(tuple[Any, Any], load_jangtq_vlm_model(load_path)),
                        temp_dir,
                    )
                if jang_metadata.codec == "jangtq":
                    from jang_tools.load_jangtq import load_jangtq_model

                    return _retain_jang_loader_temp_dir(
                        cast(tuple[Any, Any], load_jangtq_model(load_path)),
                        temp_dir,
                    )
                if jang_metadata.codec in {"affine_jang", "mxfp"} and is_vlm:
                    from jang_tools.loader import load_jang_vlm_model

                    return _retain_jang_loader_temp_dir(
                        cast(tuple[Any, Any], load_jang_vlm_model(load_path)),
                        temp_dir,
                    )
                if jang_metadata.codec in {"affine_jang", "mxfp"}:
                    from jang_tools.loader import load_jang_model

                    return _retain_jang_loader_temp_dir(
                        cast(tuple[Any, Any], load_jang_model(load_path)),
                        temp_dir,
                    )
        raise ValueError(
            f"Unsupported JANG quantization codec {jang_metadata.codec!r} "
            f"for {load_path}"
        )
    except ImportError as e:
        raise ImportError(
            "This model uses JANG quantization, but the 'jang' package "
            "is not installed. Run 'uv sync' from a source checkout or "
            'install with: pip install "jang[vlm]>=2.5.31".'
        ) from e


def maybe_load_custom_quantization(
    model_name: str,
    *,
    is_vlm: bool,
) -> tuple[Any, Any] | None:
    """Load models that require a custom upstream quantization loader.

    Returns ``None`` when the model does not declare a known custom
    quantization method. The custom loaders (e.g. paroquant) handle
    their own tokenizer/processor wiring, so omlx's tokenizer_config
    and trust_remote_code are not forwarded.
    """
    model_path = Path(model_name)
    jang_metadata = read_jang_metadata(model_path)
    if jang_metadata is not None:
        if _should_use_standard_loader_for_jang_sidecar(jang_metadata):
            return None
        return _load_jang_quantization(
            model_path,
            is_vlm=is_vlm,
            jang_metadata=jang_metadata,
        )

    config_path = model_path / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for custom quantization dispatch: %s",
            config_path,
            e,
        )
        return None

    quant_config = config.get("quantization_config")
    quant_method = quant_config.get("quant_method") if quant_config else None

    if not quant_method:
        return None

    if quant_method.lower() == "paroquant":
        try:
            from paroquant.inference.backends.mlx.load import load as paro_load
        except ImportError as e:
            raise ImportError(
                "This model uses ParoQuant. Install it separately with: "
                'pip install "paroquant[mlx]"'
            ) from e

        model, processor, loaded_is_vlm = paro_load(model_name, force_text=not is_vlm)
        if is_vlm and not loaded_is_vlm:
            raise ValueError(
                "ParoQuant loader returned a text-only model for VLM load: "
                f"{model_name}"
            )
    else:
        # The quant method may be already supported by mlx-lm; simply return None.
        return None

    return model, processor
