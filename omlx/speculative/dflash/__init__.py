# SPDX-License-Identifier: Apache-2.0
"""DFlash block drafter for the batched Lightning MTP verify path.

The drafter replaces the embedded MTP head as the draft source inside
``omlx.patches.mlx_lm_mtp``: the shared verify forward, acceptance, cache
rollback and emission stay the same, and this module only turns captured
target hidden states into the next draft block for every row.

The draft model is loaded by mlx-vlm: DFlash2, DFlash v1, Laguna or the Muse
Glimmer assistant. Its ``draft_block`` takes the target hidden states of the
tokens committed since the previous call and appends them to a per-row
sliding-window context cache, so each row keeps its own cache here and the
verify path feeds it the accepted positions after every cycle. The forward
here mirrors those classes, including their projection, gating and mask
differences.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import get_model_path

from ...model_settings import MAX_LIGHTNING_MTP_DRAFT_TOKENS
from ...patches import qwen35_packed_linear
from .drafter import DFlashDrafter
from .variants import apply_partial_rope, load_mask_embedding

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..dspark_drafter import DSparkDrafter


def resolve_block_size(
    model: nn.Module, requested: int | None, max_rows: int | None = None
) -> int:
    """Clamp the block (anchor + proposals) to what Lightning MTP verifies.

    ``max_rows`` is the target's own verify limit, when it has one.
    """
    trained = int(getattr(model.config, "block_size", 0) or 0)
    block = int(requested or trained or 0)
    if block <= 1:
        raise ValueError("DFlash block size must cover the anchor and one proposal")
    limit = MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1
    if max_rows:
        limit = min(limit, int(max_rows))
    if trained and block > trained:
        block = trained
    return min(block, limit)


def resolve_verify_mode(mode: str | None) -> bool:
    """Whether ``dflash_verify_mode`` lets the verified part of a block adapt."""
    if mode in (None, "adaptive"):
        return True
    if mode in ("dflash", "off"):
        return False
    raise ValueError(f"unknown dflash_verify_mode {mode!r}")


def context_bytes_per_row(config: dict, window: int | None = None) -> int:
    """Bytes one request's drafter context can hold, from the draft config.

    A row keeps its K/V ring and the projected history for the window, and
    while it decodes without drafting it queues up to a window of raw layer
    captures. bf16 throughout.
    """
    hidden = int(config["hidden_size"])
    heads = int(config.get("num_attention_heads") or 1)
    head_dim = int(config.get("head_dim") or hidden // heads)
    kv_heads = int(config.get("num_key_value_heads") or heads)
    dflash_config = config.get("dflash_config") or {}
    captures = len(
        dflash_config.get("target_layer_ids") or config.get("target_layer_ids") or ()
    )
    window = int(
        window
        or dflash_config.get("draft_window_size")
        or config.get("draft_window_size")
        or config.get("sliding_window")
        or 2048
    )
    layers = int(config["num_hidden_layers"])
    per_token = 2 * (layers * 2 * kv_heads * head_dim + hidden + captures * hidden)
    return window * per_token


def quantize_drafter(model: nn.Module, *, bits: int, group_size: int) -> int:
    """Affine-quantize the drafter's linear layers; returns the layer count."""
    quantized = 0

    def predicate(path: str, module: nn.Module) -> bool:
        nonlocal quantized
        if not isinstance(module, nn.Linear):
            return False
        if module.weight.shape[-1] % group_size:
            return False
        quantized += 1
        return True

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return quantized


def load_dflash_drafter(
    path: str,
    target_model: Any,
    *,
    tokenizer: Any = None,
    target_path: str | None = None,
    block_size: int | None = None,
    quant_enabled: bool = False,
    quant_bits: int = 4,
    quant_group_size: int = 64,
    verify_mode: str | None = None,
    window_size: int | None = None,
) -> DFlashDrafter | DSparkDrafter:
    """Load a DFlash checkpoint through mlx-vlm and bind it to ``target_model``.

    ``target_model`` is the mlx-vlm model whose ``language_model`` owns the
    embeddings and lm_head the drafter borrows. ``window_size`` overrides the
    checkpoint's context window.
    """
    adaptive = resolve_verify_mode(verify_mode)
    is_minicpm = (
        getattr(getattr(target_model, "config", None), "model_type", None)
        == "minicpmv4_7"
    )
    if is_minicpm:
        from ..minicpm_dspark import (
            MiniCPMDSparkDraftModel,
            load_dspark_model,
        )
        from ..minicpm_pairing import validate_model_pair

        if not Path(path).is_dir():
            raise ValueError("MiniCPM DSpark requires an existing local helper checkpoint.")
        dspark_class = MiniCPMDSparkDraftModel
        if not callable(getattr(dspark_class, "propose_block", None)):
            raise ValueError("Native DSpark proposal support is incomplete.")
        validate_model_pair(target_path, path)
        model, kind = load_dspark_model(path), "dflash"
        if not isinstance(model, dspark_class):
            raise ValueError("MiniCPM-V 4.7 requires its paired DSpark helper, not DFlash.")
    else:
        model, kind = load_drafter(path, kind="dflash")
    if kind != "dflash":
        raise ValueError(f"{path} resolved to drafter kind {kind!r}, expected 'dflash'")
    if quant_enabled and not getattr(model.config, "quantization", None):
        count = quantize_drafter(model, bits=quant_bits, group_size=quant_group_size)
        logger.info(
            "DFlash drafter quantized: %d linear layers at %d bits (group %d)",
            count,
            quant_bits,
            quant_group_size,
        )
    local = get_model_path(path)
    config = json.loads((local / "config.json").read_text())
    if not is_minicpm:
        apply_partial_rope(model, config)
    mask_path = local / "mask_embedding.pt"
    mask_embedding = (
        load_mask_embedding(mask_path, int(config["hidden_size"]))
        if mask_path.is_file()
        else None
    )
    # Laguna checks the target vocabulary through the tokenizer when the
    # target (an mlx-lm model) carries no config.
    if (
        tokenizer is not None
        and "tokenizer" in inspect.signature(model.bind).parameters
    ):
        model.bind(target_model, tokenizer=tokenizer)
    else:
        model.bind(target_model)
    language_model = getattr(target_model, "language_model", target_model)
    block = resolve_block_size(
        model,
        (
            getattr(model, "dflash_default_block_size", None)
            if is_minicpm and block_size is None
            else block_size
        ),
        getattr(language_model, "_omlx_mtp_max_verify_rows", None),
    )
    mx.eval(model.parameters())
    if not is_minicpm and qwen35_packed_linear.enabled(target_model):
        packed = qwen35_packed_linear.pack_drafter(model)
        logger.info("DFlash drafter packed 4-bit projections: %d layers", packed)
    drafter: DFlashDrafter | DSparkDrafter
    if is_minicpm:
        from ..dspark_drafter import DSparkDrafter

        drafter = DSparkDrafter(model, block_size=block, source_path=path)
    else:
        drafter = DFlashDrafter(
            model,
            block_size=block,
            source_path=path,
            adaptive=adaptive,
            window_size=window_size,
            value_scale=(config.get("dflash_config") or {}).get("attention_value_scale"),
            mask_embedding=mask_embedding,
        )
    logger.info(
        "DFlash drafter loaded: path=%s kind=%s block=%d adaptive=%s target_layers=%s",
        path,
        drafter.kind,
        block,
        adaptive if not is_minicpm else False,
        drafter.target_layer_ids,
    )
    return drafter


def attach_drafter(language_model: Any, drafter: DFlashDrafter | DSparkDrafter) -> None:
    """Expose the drafter to Lightning MTP through the language model markers."""
    language_model._omlx_drafter = drafter
    language_model._omlx_mtp_decode_enabled = True
    language_model._omlx_mtp_multi_request = not getattr(
        language_model, "_omlx_drafter_single_row", False
    )
    language_model._omlx_mtp_batch_rollback = True
    language_model._omlx_mtp_chain = True
    language_model._omlx_mtp_depth = drafter.depth
    language_model._omlx_mtp_head_clone = False
    if drafter.kind == "dspark":
        language_model._omlx_mtp_depth_fixed = True


def detach_drafter(language_model: Any) -> DFlashDrafter | DSparkDrafter | None:
    drafter = getattr(language_model, "_omlx_drafter", None)
    if drafter is None:
        return None
    drafter.clear()
    language_model._omlx_drafter = None
    language_model._omlx_mtp_decode_enabled = False
    if drafter.kind == "dspark":
        language_model._omlx_mtp_depth_fixed = False
    return drafter


__all__ = [
    "DFlashDrafter",
    "attach_drafter",
    "context_bytes_per_row",
    "detach_drafter",
    "load_dflash_drafter",
    "quantize_drafter",
    "resolve_block_size",
    "resolve_verify_mode",
]
