# SPDX-License-Identifier: Apache-2.0
"""MTP-head prompt priming: fold the prompt into the head cache during prefill.

Without priming the MTP head starts generation with an empty KV cache — its
first drafts see none of the prompt and acceptance starts context-starved,
recovering only as committed generation tokens accumulate (MTPLX measured
committed-history priming at 0.90 acceptance vs 0.26 unprimed on depth-1
real-code prompts). This module rides the existing prefill forwards: every
backbone chunk forward already computes the trunk-normed hidden for all chunk
positions, so the (hidden[t], token[t+1]) pairs the head history needs are
available for free. Each chunk is folded into a head cache immediately and
the chunk hidden is discarded — only a single (1, 1, H) pending row carries
across chunks.

Transport: contexts live in a cache-list-identity keyed map on the patched
language-model instance (the ``host``). Cache-entry attributes cannot carry
them — mlx-lm's
insert merge rebuilds every layer cache that lacks filter/extract support
(all of DeepSeek-V4's and GLM-5.2's CacheList entries, and TurboQuant
replaces KVCache entries at end of prefill) — while the model instance is
the one object every forward and the activation both see. The outer cache
list stays stable while a request's prefill chunks stream, so separate
singleton prefills can retain their own timelines while the engine
interleaves them. mlx-lm's ``PromptProcessingBatch.split`` and
``GenerationBatch.extend`` rebuild the outer list as requests move between
queues; the BatchGenerator patch calls ``transfer_ctx`` at those ownership
boundaries.

Fail-safe invariant: every capture verifies the anchor offset advanced
contiguously since the previous capture (``expected_offset``). Any rewind,
trim, request switch, or unknown cache path breaks the equality and
invalidates the context, degrading to the current unprimed behaviour —
never to a wrong history.

Capture sites (each calls :func:`maybe_capture` after the backbone forward):

- mlx-lm qwen3_5 text path: the patched ``TextModel.__call__``
  (``qwen35_model``), which computes the trunk-normed hidden inline.
- mlx-vlm qwen3_5 path: a wrap on the inner ``Qwen3_5Model.__call__``
  (``qwen35_vlm_runtime``), whose return value *is* the trunk-normed
  hidden; the MoE inner model inherits it. The outer ``LanguageModel`` is
  reached via a weakref stamped at init.
- DeepSeek-V4 (``deepseek_v4_model``): the patched ``Model.__call__``
  requests ``return_raw_hidden`` and passes the raw 4D Hyper-stream hidden
  (the head input variant; no trunk norm).
- GLM-5.2 (``glm_moe_dsa_model``): the patched ``Model.__call__`` passes
  the post-final-norm hidden it already computes.

All sites skip ``return_hidden=True`` forwards (MTP verify cycles and the
activation forward in ``_post_init_mtp``); the final (hidden[prompt[-1]],
main_tok) pair is folded by :func:`take_primed` at activation instead.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# The MTP head is fed the trunk's *post-norm* hidden and chains on its own
# post-norm output. Measured on Qwen3.6-27B this accepts a few points higher
# than PR 990's pre-norm at every depth. Draft-side only, so output identity
# is unaffected regardless. Priming folds must use the same variant as the
# decode-time history folds in batch_generator, hence the single definition
# here.
HEAD_HIDDEN_POST_NORM = True

_CTX_ATTR = "_omlx_mtp_prime_ctx"

_MAX_CONTEXTS = 64

_SUPPRESS = threading.local()


def priming_enabled() -> bool:
    """Prompt priming is on by default for MTP-enabled models."""
    return os.environ.get("OMLX_MTP_PROMPT_PRIMING", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def prime_window() -> int:
    """Max prompt length (tokens) to prime; 0 = unlimited.

    Escape hatch for the head-cache memory cost on very long prompts (one
    full-attention layer of KV over the prompt). Prompts longer than the
    window run unprimed — a coarse cap, not a suffix window, because the
    prompt length is unknown while chunks stream through the model.
    """
    try:
        return max(0, int(os.environ.get("OMLX_MTP_PRIME_WINDOW", "0")))
    except ValueError:
        return 0


@contextmanager
def suppress_capture():
    """Disable capture on this thread for the duration of the block."""
    _SUPPRESS.value = True
    try:
        yield
    finally:
        _SUPPRESS.value = False


def _suppressed() -> bool:
    return bool(getattr(_SUPPRESS, "value", False))


@dataclass
class _PrimeCtx:
    """Streaming priming state owned by one outer prompt-cache list."""

    cache_id: int = 0
    mtp_cache: List[Any] = field(default_factory=list)
    # Head-input hidden of the newest seen token, (1, 1, ..., H) — pairs
    # with the first token of the next chunk (or main_tok at activation).
    pending_hidden: Optional[Any] = None
    # Folded (hidden, next_token) pairs == head-cache offset.
    folded: int = 0
    # Anchor cache offset observed after the last captured forward. The next
    # capture requires offset_now - S == expected_offset (contiguity).
    expected_offset: int = 0
    valid: bool = True


def _anchor(cache: Optional[List[Any]]) -> Optional[Any]:
    """First cache entry with a scalar offset.

    Container layers (``CacheList``-style, exposing ``.caches`` — DeepSeek-V4
    and GLM-5.2 backbones) are searched one level deep: the container itself
    has no offset but its first sub-cache (RotatingKVCache / KVCache) does.
    """
    if not cache:
        return None
    for c in cache:
        if _read_offset(c) is not None:
            return c
        for sub in getattr(c, "caches", ()) or ():
            if _read_offset(sub) is not None:
                return sub
    return None


def _read_offset(entry: Any) -> Optional[int]:
    offset = getattr(entry, "offset", None)
    if type(offset) is int:
        return offset
    if offset is not None and getattr(offset, "size", 0) == 1:
        try:
            return int(offset.reshape(()).item())
        except Exception:
            return None
    return None


def _activation_offset(cache: Optional[List[Any]]) -> Optional[int]:
    """Attention-layer offset at MTP activation, tolerant of batch caches.

    Between the last capture and activation, ``insert()`` runs mlx-lm's
    cache merge: scalar ``KVCache`` entries without singleton passthrough
    become batch caches whose ``offset`` is a 1-element ``mx.array``. The
    one-off ``int()`` sync here is activation-time only, never per-chunk.
    """
    if not cache:
        return None

    for c in cache:
        got = _read_offset(c)
        if got is not None:
            return got
        for sub in getattr(c, "caches", ()) or ():
            got = _read_offset(sub)
            if got is not None:
                return got
    return None


def _host_candidates(model: Any):
    """The model itself plus the wrapped language model, if any.

    Mirrors ``batch_generator._resolve_mtp_chain_depth``: the host that
    carries the slot is the patched language-model instance — the outer
    adapter / VLM wrapper for qwen paths, the Model itself for DeepSeek/GLM.
    """
    yield model
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            yield inner


def _find_ctx(
    model: Any, cache: Optional[List[Any]] = None
) -> Optional[_PrimeCtx]:
    for host in _host_candidates(model):
        contexts = getattr(host, _CTX_ATTR, None)
        if not isinstance(contexts, dict):
            continue
        if cache is not None:
            ctx = contexts.get(id(cache))
            if ctx is not None and ctx.cache_id == id(cache):
                return ctx
            continue
        if contexts:
            return next(reversed(contexts.values()))
    return None


def drop_ctx(model: Any, cache: Optional[List[Any]] = None) -> None:
    """Remove one cache-owned context, or every context when cache is None."""
    if model is None:
        return
    for host in _host_candidates(model):
        contexts = getattr(host, _CTX_ATTR, None)
        if not isinstance(contexts, dict):
            if contexts is None:
                continue
            try:
                delattr(host, _CTX_ATTR)
            except AttributeError:
                pass
            continue
        if cache is None:
            contexts.clear()
        else:
            ctx = contexts.get(id(cache))
            if ctx is not None and ctx.cache_id == id(cache):
                contexts.pop(id(cache), None)
        if not contexts:
            try:
                delattr(host, _CTX_ATTR)
            except AttributeError:
                pass


def transfer_ctx(
    model: Any,
    source_cache: Optional[List[Any]],
    target_cache: Optional[List[Any]],
) -> None:
    """Move a live context when mlx-lm rebuilds the outer cache list."""
    if model is None or source_cache is None or target_cache is None:
        return
    if source_cache is target_cache:
        return
    for host in _host_candidates(model):
        contexts = getattr(host, _CTX_ATTR, None)
        if not isinstance(contexts, dict):
            continue
        ctx = contexts.get(id(source_cache))
        if ctx is None or ctx.cache_id != id(source_cache):
            continue
        contexts.pop(id(source_cache), None)
        ctx.cache_id = id(target_cache)
        contexts[id(target_cache)] = ctx
        return


def _host_eligible(host: Any) -> bool:
    return bool(
        getattr(host, "_omlx_mtp_decode_enabled", False)
        and getattr(host, "_omlx_mtp_chain", False)
        and getattr(host, "mtp", None) is not None
    )


def capture_eligible(host: Any, cache: Optional[List[Any]]) -> bool:
    """Cheap pre-check for capture sites that must decide the forward shape.

    The DeepSeek/GLM backbones only expose the head-input hidden when asked
    (``return_raw_hidden``), so their patched ``__call__`` consults this
    before choosing the call form. Everything here is re-checked inside
    :func:`maybe_capture`; this exists purely to keep the ineligible path
    identical to stock.
    """
    return (
        not _suppressed()
        and priming_enabled()
        and cache is not None
        and _host_eligible(host)
    )


def maybe_capture(
    host: Any, inputs: Any, normed: Any, cache: Optional[List[Any]]
) -> None:
    """Fold this forward's (hidden, next_token) pairs into the priming cache.

    ``host`` is the patched language model (mlx-lm ``TextModel`` or mlx-vlm
    ``LanguageModel``) exposing ``mtp`` / ``model.embed_tokens`` /
    ``make_mtp_cache``. ``normed`` is the trunk-normed hidden for all
    positions of ``inputs`` (1, S, H). Host-side bookkeeping only — the head
    forward is dispatched lazily and no GPU sync happens here.

    Call sites guard the cheap negatives (return_hidden / n_confirmed /
    inputs_embeds / batch>1) before calling; everything here re-checks what
    is load-bearing and bails silently, so a miss degrades to unprimed.
    """
    if _suppressed() or not priming_enabled():
        return
    if cache is None or not _host_eligible(host):
        return
    if inputs is None or getattr(inputs, "ndim", 0) != 2 or inputs.shape[0] != 1:
        return
    anchor = _anchor(cache)
    if anchor is None:
        return

    import mlx.core as mx

    seq_len = int(inputs.shape[1])
    offset_after = _read_offset(anchor)  # forward already ran; includes S
    if offset_after is None:
        return

    ctx = _find_ctx(host, cache)
    if ctx is not None and (
        not ctx.valid or ctx.expected_offset != offset_after - seq_len
    ):
        # Rewind / trim / unknown path on this cache: never guess.
        drop_ctx(host, cache)
        ctx = None
    window = prime_window()
    if window and offset_after > window:
        if ctx is not None:
            drop_ctx(host, cache)
        return
    if ctx is None:
        if seq_len <= 1:
            # A lone decode step cannot start a prompt timeline.
            return
        # Keep only the identity token. Holding the outer list here pins every
        # main-model KV tensor on the long-lived model if a handoff is missed.
        # Offset contiguity below still rejects stale or reused identities.
        ctx = _PrimeCtx(cache_id=id(cache), mtp_cache=host.make_mtp_cache())
        if not ctx.mtp_cache:
            return
        contexts = getattr(host, _CTX_ATTR, None)
        if not isinstance(contexts, dict):
            contexts = {}
            setattr(host, _CTX_ATTR, contexts)
        elif len(contexts) >= _MAX_CONTEXTS:
            contexts.pop(next(iter(contexts)))
        contexts[id(cache)] = ctx

    if ctx.pending_hidden is not None:
        if seq_len > 1:
            pairs_hidden = mx.concatenate(
                [ctx.pending_hidden, normed[:, :-1]], axis=1
            )
        else:
            pairs_hidden = ctx.pending_hidden
        pairs_tokens = inputs
    else:
        if seq_len <= 1:
            ctx.pending_hidden = normed[:, -1:]
            ctx.expected_offset = offset_after
            return
        pairs_hidden = normed[:, :-1]
        pairs_tokens = inputs[:, 1:]

    # Fold through the public mtp_forward so every family's head layout
    # (module, block list, CacheList head caches) is handled by its own
    # model patch. The returned logits are never evaluated — nothing pulls
    # on them, so the lm_head tail costs nothing.
    host.mtp_forward(pairs_hidden, pairs_tokens, ctx.mtp_cache, logits_keep=1)
    ctx.folded += int(pairs_tokens.shape[1])
    ctx.pending_hidden = normed[:, -1:]
    ctx.expected_offset = offset_after
    # Materialize the head-cache buffers per chunk so the fold graph never
    # accumulates across a long prefill; the (1,1,H) pending row is evaluated
    # alongside so the chunk's full hidden can be freed.
    evals = [ctx.pending_hidden]
    flat = []
    for c in ctx.mtp_cache:
        subs = getattr(c, "caches", None)
        flat.extend(subs if subs else (c,))
    for c in flat:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        if keys is not None:
            evals.append(keys)
        if values is not None:
            evals.append(values)
    mx.async_eval(evals)


def take_primed(
    model: Any,
    cache: Optional[List[Any]],
    main_tok: Any,
) -> Optional[tuple]:
    """Pop the priming context at MTP activation and finish the seam.

    Called from ``_post_init_mtp`` after its 1-token backbone forward at
    ``main_tok`` (which capture skipped — it runs with return_hidden=True).
    Validates that the context is contiguous up to exactly that forward,
    folds the final (hidden[prompt[-1]], main_tok) pair through the public
    ``mtp_forward`` (adapter/outer-model level), and returns
    ``(mtp_cache, hist_offset)`` — or None, in which case the caller keeps
    the current unprimed behaviour.
    """
    # Hosts with their own priming shape (inkling's sliding-window
    # multi-block fold) own the whole activation seam.
    for host in _host_candidates(model):
        hook = getattr(host, "mtp_take_primed", None)
        if callable(hook):
            return hook(cache, main_tok)
    ctx = _find_ctx(model, cache)
    if ctx is None:
        return None
    drop_ctx(model, cache)
    if not (ctx.valid and ctx.folded > 0 and ctx.pending_hidden is not None):
        return None
    offset = _activation_offset(cache)
    if offset is None or ctx.expected_offset != offset - 1:
        logger.debug(
            "MTP priming discarded: seam offset mismatch (ctx=%s cache=%s)",
            ctx.expected_offset,
            offset,
        )
        return None
    try:
        model.mtp_forward(
            ctx.pending_hidden,
            main_tok.reshape(1, 1),
            ctx.mtp_cache,
            logits_keep=1,
        )
    except Exception as exc:
        logger.debug("MTP priming discarded: seam fold failed: %s", exc)
        return None
    return ctx.mtp_cache, ctx.folded + 1


def prime_ctx_stats(
    model: Any, cache: Optional[List[Any]] = None
) -> Optional[int]:
    """Folded pair count of a live context (introspection / tests)."""
    ctx = _find_ctx(model, cache)
    return ctx.folded if ctx is not None else None


__all__ = [
    "HEAD_HIDDEN_POST_NORM",
    "priming_enabled",
    "prime_window",
    "suppress_capture",
    "maybe_capture",
    "take_primed",
    "drop_ctx",
    "transfer_ctx",
    "prime_ctx_stats",
]
