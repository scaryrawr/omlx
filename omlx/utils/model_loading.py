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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

_VLM_TEXT_PREFIX = "language_model."
_JANG_CONFIG_FILENAMES = (
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
)
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
    routed_expert_bit_plans: tuple[Any, ...]
    is_vlm: bool
    is_kimi_vlm: bool
    is_deepseek_v4: bool

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
    r"^layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.tq_bits$"
)
_DSV4_FAST_LOAD_DISABLE_ENV = "JANGTQ_DISABLE_DSV4_FAST_LOAD"
_DSV4_FAST_LOAD_NOTICE_LOGGED = False


def expand_per_layer_quant_keys(cfg: dict) -> dict:
    """Add ``language_model.``-prefixed variants of per-layer quantization keys.

    oQ writes per-layer overrides keyed by safetensors tensor base name
    (e.g. ``"lm_head"``), but ``nn.quantize``'s class_predicate receives
    model-tree paths (``"language_model.lm_head"``).  Without the prefixed
    variant the lookup misses and the global bits are used, causing a
    shape mismatch at ``load_weights``.

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
            prefixed = _VLM_TEXT_PREFIX + key
            if not key.startswith(_VLM_TEXT_PREFIX) and prefixed not in quant:
                extras[prefixed] = val
            elif key.startswith(_VLM_TEXT_PREFIX):
                short = key[len(_VLM_TEXT_PREFIX) :]
                if short not in quant:
                    extras[short] = val
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
    These patches inject modules into ``sys.modules`` and replace mlx-lm
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

    text_config = config.get("text_config")
    text_model_type = (
        text_config.get("model_type") if isinstance(text_config, dict) else None
    )
    if model_type == "llama4" or text_model_type == "llama4":
        from ..patches.llama4_attention import apply_llama4_attention_patch

        if apply_llama4_attention_patch():
            logger.info("Llama 4 attention patch applied for %s", model_name)

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
    if _is_mtp_compatible(config, model_type):
        mtp_enabled = bool(
            model_settings is not None and getattr(model_settings, "mtp_enabled", False)
        )
        from ..patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            set_mtp_active,
        )

        if apply_mlx_lm_mtp_patch():
            set_mtp_active(mtp_enabled)
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
        and not _is_mtp_compatible(config, model_type)
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


_MTP_WEIGHT_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "model.language_model.mtp.",
)


def _checkpoint_has_mtp_weights(model_path: str | Path) -> bool:
    """True iff the checkpoint at *model_path* ships any ``mtp.*`` weight tensor.

    Some Qwen3.6 MoE VLM exports declare ``mtp_num_hidden_layers > 0`` in
    ``config.json`` but strip the MTP weights during conversion (e.g.
    ``unsloth/Qwen3.6-35B-A3B-UD-MLX-*bit``). Attaching ``MTPModule`` for
    such a checkpoint causes mlx-vlm's strict ``load_weights`` to fail with
    "Missing N parameters: language_model.mtp.*", the engine falls back to
    LLM, and vision is silently dropped (issue #1426).

    Reads ``model.safetensors.index.json`` when present (no shard I/O).
    Falls back to the first safetensors shard's metadata header. Returns
    False when neither resolves — callers treat that as "no MTP weights"
    (the conservative choice: skip MTPModule attachment).
    """
    p = Path(model_path)
    if not p.is_dir():
        return False

    index_path = p / "model.safetensors.index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map") or {}
            return any(k.startswith(_MTP_WEIGHT_PREFIXES) for k in weight_map)
        except Exception as e:
            logger.debug("Failed to read %s for mtp weight scan: %s", index_path, e)

    shards = sorted(p.glob("*.safetensors"))
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
                for k in f.keys():  # noqa: SIM118 - safe_open exposes keys(), not dict iteration
                    if k.startswith(_MTP_WEIGHT_PREFIXES):
                        return True
        except Exception as e:
            logger.debug("Failed to read %s header for mtp weight scan: %s", shard, e)
    return False


def _is_mtp_compatible(config: dict, model_type: str | None) -> bool:
    """Decide whether the native MTP patch can be applied to this model.

    Phase 1 supports Qwen3.5/3.6 (mlx-lm PR 990) and DeepSeek-V4-Flash
    (Blaizzy/mlx-lm fork PR 15). The model also has to declare MTP heads
    in the config; otherwise the patch is a no-op.
    """
    if not _has_mtp_heads(config):
        return False
    if not model_type:
        return False
    return (
        model_type.startswith("qwen3_5")
        or model_type.startswith("qwen3_6")
        or model_type.startswith("deepseek_v4")
    )


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_settings: Any | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    from mlx_lm import load

    trust_remote_code = (
        bool(getattr(model_settings, "trust_remote_code", False))
        if model_settings is not None
        else False
    )
    return load(
        model_name,
        tokenizer_config=tokenizer_config,
        trust_remote_code=trust_remote_code,
    )


def _collect_mx_arrays(value: Any, arrays: list[mx.array], seen: dict[int, Any]) -> None:
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
    for name in _JANG_CONFIG_FILENAMES:
        path = model_path / name
        if path.exists():
            return path
    return None


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

    marker_keys = {"format", "weight_format", "profile"}
    nested_marker_keys = marker_keys | {"mode", "method"}
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
        metadata = source.get("metadata")
        if isinstance(metadata, dict):
            for item in _iter_dicts(metadata):
                for key in nested_marker_keys:
                    _add_jang_marker(markers, item.get(key))

    return tuple(markers)


def _index_has_tq_packed_entry(model_path: Path) -> bool:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map") or {}
            if isinstance(weight_map, dict):
                return any(
                    ".tq_packed" in str(key) or ".tq_packed" in str(value)
                    for key, value in weight_map.items()
                )
        except Exception as e:
            logger.debug("Failed to read %s for JANGTQ scan: %s", index_path, e)
    return False


def _has_jangtq_artifact_markers(model_path: Path) -> bool:
    if (model_path / "jangtq_runtime.safetensors").exists():
        return True
    if _index_has_tq_packed_entry(model_path):
        return True
    return any(".tq_packed" in path.name for path in model_path.glob("*.tq_packed*"))


def _classify_jang_codec(
    markers: tuple[str, ...],
    sources: tuple[Any, ...],
    model_path: Path,
) -> JangCodec:
    canonical = {_canonical_jang_marker(marker) for marker in markers}

    if (
        any("jangtq" in marker or "mxtq" in marker for marker in canonical)
        or _first_nested_value(sources, {"mxtq_bits"}) is not None
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


def _detect_jang_vlm_model(
    model_path: Path,
    jang_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> bool:
    if model_config is not None and (
        "vision_config" in model_config
        or "vit_config" in model_config
        or bool(model_config.get("mm_vision_tower"))
    ):
        return True

    architecture = jang_config.get("architecture")
    if isinstance(architecture, dict) and architecture.get("has_vision") is True:
        return True

    return (model_path / "preprocessor_config.json").exists()


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
    for source in (model_config, jang_config):
        if not isinstance(source, dict):
            continue
        model_type = source.get("model_type")
        if isinstance(model_type, str) and model_type.lower() == "deepseek_v4":
            return True
    return False


def read_jang_metadata(model_path: str | Path) -> JangQuantizationMetadata | None:
    """Read and normalize JANG-family sidecar metadata for *model_path*."""

    path = Path(model_path)
    sidecar_path = _find_jang_config_path(path)
    if sidecar_path is None:
        return None

    jang_config = _read_jang_config(sidecar_path)
    model_config = _read_json_object_or_none(path / "config.json")
    sources: tuple[Any, ...] = (
        jang_config,
        model_config if model_config is not None else {},
    )
    markers = _collect_jang_markers(jang_config, model_config, sidecar_path)
    is_kimi_vlm = _is_kimi_vlm_model(path, jang_config, model_config)
    is_vlm = _detect_jang_vlm_model(path, jang_config, model_config) or is_kimi_vlm

    return JangQuantizationMetadata(
        model_path=path,
        sidecar_path=sidecar_path,
        sidecar=jang_config,
        model_config=model_config,
        codec=_classify_jang_codec(markers, sources, path),
        markers=markers,
        profile=_first_nested_string(sources, {"profile"}),
        target_bits=_first_nested_value(
            sources,
            {"target_bits", "target_bit_width", "bits", "bit_width"},
        ),
        actual_bits=_first_nested_value(
            sources,
            {"actual_bits", "effective_bits", "actual_bit_width"},
        ),
        block_size=_first_nested_value(sources, {"block_size", "blocksize"}),
        group_size=_first_nested_value(sources, {"group_size", "groupsize"}),
        mxtq_bits=_first_nested_value(sources, {"mxtq_bits"}),
        routed_expert_bit_plans=_all_nested_values(
            sources,
            {
                "routed_expert_bit_plan",
                "routed_expert_bit_plans",
                "routed_expert_bits",
                "expert_bit_plan",
                "expert_bit_plans",
                "expert_bits",
            },
        ),
        is_vlm=is_vlm,
        is_kimi_vlm=is_kimi_vlm,
        is_deepseek_v4=_is_deepseek_v4_model(jang_config, model_config),
    )


def _safe_resolved_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve()
    except Exception:
        return expanded


def _dsv4_bundle_config(model_path: str | Path) -> dict[str, Any] | None:
    config = _read_json_object_or_none(Path(model_path) / "config.json")
    if config is None or config.get("model_type") != "deepseek_v4":
        return None
    return config


def _dsv4_weight_map(model_path: str | Path) -> dict[str, str]:
    """Return tensor-key -> safetensors filename for a DSV4 bundle."""

    bundle = Path(model_path)
    index_path = bundle / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            data = json.loads(index_path.read_text())
        except Exception as e:
            raise RuntimeError(f"Could not read safetensors index {index_path}: {e}") from e
        weight_map_data = data.get("weight_map")
        if not isinstance(weight_map_data, dict):
            raise RuntimeError(f"Safetensors index {index_path} has no weight_map object")
        return {str(key): str(value) for key, value in weight_map_data.items()}

    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(f"safetensors is required for DSV4 header audit: {e}") from e

    weight_map: dict[str, str] = {}
    for shard in sorted(bundle.glob("*.safetensors")):
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
                    handle = safe_open(str(Path(model_path) / filename), framework="numpy")
                    handle.__enter__()
                    handles[filename] = handle
                dtype = handle.get_slice(key).get_dtype()
                if dtype != "F32":
                    report["non_f32_count"] += 1
                    if len(report["non_f32_examples"]) < 12:
                        report["non_f32_examples"].append(
                            {"key": key, "dtype": dtype}
                        )
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
        config_routed_plan.get("routed_layer_bits"),
        config_quant.get("routed_layer_bits"),
        config.get("routed_layer_bits"),
    ]
    candidates.extend(
        _all_nested_values(
            (jang_config, config),
            {"routed_layer_bits", "routed_expert_bit_plan", "routed_expert_bit_plans"},
        )
    )
    for candidate in candidates:
        coerced = _dsv4_coerce_layer_bit_plan(candidate)
        if coerced:
            return coerced
    return {}


def _dsv4_routed_default_bits(
    config: dict[str, Any],
    jang_config: dict[str, Any],
) -> int:
    for source in (jang_config, config):
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
    return 2


def _audit_dsv4_artifact_bit_plan(model_path: str | Path) -> dict[str, Any]:
    """Audit DSV4 routed TQ bit plans and required runtime sidecar keys."""

    bundle = Path(model_path)
    config = _dsv4_bundle_config(bundle)
    jang_config = _read_json_object_or_none(bundle / "jang_config.json") or {}
    report: dict[str, Any] = {
        "checked": False,
        "routed_layer_bits": {},
        "routed_bit_counts": {},
        "metadata_routed_layer_bits": {},
        "metadata_matches_actual": None,
        "metadata_layer_value_mismatches": {},
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
            report["issues"].append(
                f"failed to read {key}: {type(e).__name__}: {e}"
            )
            continue
        by_layer.setdefault(layer, {})[projection] = bits

        packed_shape = _read_shape(key[: -len(".tq_bits")] + ".tq_packed")
        if packed_shape is not None and len(packed_shape) >= 3 and bits > 0:
            values_per_u32 = max(1, 32 // bits)
            in_features = int(packed_shape[-1]) * values_per_u32
            required_sidecar_keys.add(f"codebook.{in_features}.{bits}")
            required_sidecar_keys.add(f"signs.{in_features}.{seed}")

    expected_projections = {"gate_proj", "up_proj", "down_proj"}
    routed_layer_bits: dict[str, int] = {}
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
        values = set(projections.values())
        if len(values) != 1:
            projection_report = dict(sorted(projections.items()))
            report["projection_mismatches"].append(
                {"layer": layer, "projections": projection_report}
            )
            report["issues"].append(
                f"layer {layer} routed projections have mixed tq_bits: "
                f"{projection_report}"
            )
            continue
        routed_layer_bits[layer] = next(iter(values))

    report["routed_layer_bits"] = routed_layer_bits
    bit_counts: dict[str, int] = {}
    for bits in routed_layer_bits.values():
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
    report["metadata_routed_layer_bits"] = metadata
    report["metadata_layer_value_mismatches"] = metadata_mismatches
    report["actual_overrides_missing_from_metadata"] = missing_actual_overrides
    report["metadata_matches_actual"] = (
        not metadata_mismatches and not missing_actual_overrides
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
                report["issues"].append(
                    f"sidecar unreadable: {type(e).__name__}: {e}"
                )

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
        for key in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow")
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
        sample = ", ".join(f"{name or '<root>'}={limit:g}" for name, limit in bad_limit[:5])
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


def _load_jang_quantization(model_path: Path, *, is_vlm: bool) -> tuple[Any, Any]:
    jang_metadata = read_jang_metadata(model_path)
    if jang_metadata is None:
        raise ValueError(f"No JANG config found in {model_path}")

    model_is_vlm = jang_metadata.is_vlm
    if model_is_vlm and not is_vlm:
        raise ValueError(
            "JANG loader was asked to load a VLM artifact as text-only. "
            f"Use the VLM engine for: {model_path}"
        )
    if is_vlm and not model_is_vlm:
        raise ValueError(
            "JANG loader was asked to load a text-only model through the VLM path: "
            f"{model_path}"
        )

    if jang_metadata.codec == "unknown_jang":
        markers = ", ".join(jang_metadata.markers) or "none"
        profile = jang_metadata.profile or "none"
        raise ValueError(
            "Unsupported JANG quantization codec for "
            f"{model_path}. Sidecar: {jang_metadata.sidecar_path.name}; "
            f"markers: {markers}; profile: {profile}. "
            "Install or upgrade to a newer 'jang' package that supports this "
            "JANG sidecar format."
        )

    try:
        if jang_metadata.is_deepseek_v4 and not is_vlm:
            return _load_deepseek_v4_jang(model_path, jang_metadata)

        if jang_metadata.codec == "jangtq" and is_vlm:
            if jang_metadata.is_kimi_vlm:
                from jang_tools.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model

                return cast(tuple[Any, Any], load_jangtq_kimi_vlm_model(model_path))

            from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model

            return cast(tuple[Any, Any], load_jangtq_vlm_model(model_path))
        if jang_metadata.codec == "jangtq":
            from jang_tools.load_jangtq import load_jangtq_model

            return cast(tuple[Any, Any], load_jangtq_model(model_path))
        if jang_metadata.codec in {"affine_jang", "mxfp"} and is_vlm:
            from jang_tools.loader import load_jang_vlm_model

            return cast(tuple[Any, Any], load_jang_vlm_model(model_path))
        if jang_metadata.codec in {"affine_jang", "mxfp"}:
            from jang_tools.loader import load_jang_model

            return cast(tuple[Any, Any], load_jang_model(model_path))
        raise ValueError(
            f"Unsupported JANG quantization codec {jang_metadata.codec!r} "
            f"for {model_path}"
        )
    except ImportError as e:
        raise ImportError(
            "This model uses JANG quantization, but the 'jang' package "
            "is not installed. Run 'uv sync' from a source checkout or "
            'install with: pip install "jang[vlm]>=2.5.29".'
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
    if _find_jang_config_path(model_path) is not None:
        return _load_jang_quantization(model_path, is_vlm=is_vlm)

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
