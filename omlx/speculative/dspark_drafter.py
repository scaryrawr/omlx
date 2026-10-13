# SPDX-License-Identifier: Apache-2.0
"""Native DSpark proposals with request-owned, untruncated context caches."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from ..patches.mlx_lm_mtp import batch_generator as bg
from .dflash.drafter import _concat_captured


@dataclass
class _DSparkRow:
    cache: list[Any]
    pending: list[mx.array] = field(default_factory=list)
    fed: int = 0
    confidences: mx.array | None = None


class DSparkDrafter:
    """Adapt native proposals to Lightning MTP without replacing their sampler."""

    kind = "dspark"
    window = sys.maxsize

    def __init__(self, model: Any, *, block_size: int, source_path: str):
        if not callable(getattr(model, "propose_block", None)):
            raise ValueError(
                "DSpark requires the native proposal/probability/confidence contract."
            )
        self.model = model
        self.source_path = source_path
        self.target_layer_ids = list(model.config.target_layer_ids)
        self.block_size = block_size
        self.depth = block_size - 1
        self._rows: dict[Any, _DSparkRow] = {}
        self._request_seeds: dict[str, list[mx.array]] = {}
        self.scope_uids: tuple | None = None

    def _row(self, uid: Any) -> _DSparkRow:
        if uid not in self._rows:
            self._rows[uid] = _DSparkRow(cache=self.model.make_cache())
        return self._rows[uid]

    def seed(self, uid: Any, captured: Sequence[mx.array]) -> None:
        if captured:
            self._row(uid).pending.append(_concat_captured(captured))

    def observe(self, uids: Iterable[Any], captured: Sequence[mx.array] | None) -> None:
        if captured:
            for index, uid in enumerate(uids):
                self.seed(uid, [layer[index : index + 1] for layer in captured])

    def seed_request(self, request_id: str, captured: Sequence[mx.array]) -> None:
        if captured:
            self._request_seeds.setdefault(request_id, []).append(
                _concat_captured(captured)
            )

    def bind_uid(self, request_id: str, uid: Any) -> None:
        seeds = self._request_seeds.pop(request_id, None)
        if seeds:
            self._row(uid).pending.extend(seeds)

    def release_request(self, request_id: str) -> None:
        self._request_seeds.pop(request_id, None)

    def release(self, uids: Iterable[Any]) -> None:
        for uid in uids:
            self._rows.pop(uid, None)

    def clear(self) -> None:
        self._rows.clear()
        self._request_seeds.clear()

    @contextmanager
    def decode_scope(self, uids: Iterable[Any]):
        previous = self.scope_uids
        self.scope_uids = tuple(uids)
        try:
            yield
        finally:
            self.scope_uids = previous

    def context_length(self, uid: Any) -> int:
        row = self._rows.get(uid)
        return 0 if row is None else row.fed

    def draft(self, jobs: Sequence[tuple]) -> None:
        stochastic = [
            gen_batch is not None and not bg._is_greedy(gen_batch)
            for gen_batch, *_ in jobs
        ]
        if any(
            sampled
            and (
                len(jobs) > 1
                or len(getattr(job[0], "uids", ())) > 1
                or (self.scope_uids is not None and len(self.scope_uids) > 1)
            )
            for job, sampled in zip(jobs, stochastic)
        ):
            raise ValueError("Batched sampled DSpark is not supported.")
        started = time.perf_counter()
        for job, sampled in zip(jobs, stochastic):
            gen_batch, state, captured, committed, _prev_buf = job
            row = self._row(state.uid)
            if captured:
                row.pending.append(_concat_captured(captured))
            if not row.pending:
                raise RuntimeError(f"DSpark has no target context for uid={state.uid!r}")
            hidden = (
                row.pending[0]
                if len(row.pending) == 1
                else mx.concatenate(row.pending, axis=1)
            )
            sampler = (
                bg._resolve_draft_sampler(gen_batch, state)
                if sampled
                else lambda logits: mx.argmax(logits, axis=-1)
            )
            probabilities = (
                (lambda logprobs, sampler=sampler: mx.exp(
                    bg._accept_lp_for(sampler, logprobs)
                ))
                if sampled
                else None
            )
            proposal = self.model.propose_block(
                committed.reshape(-1)[-1:].astype(mx.int32),
                hidden,
                row.cache,
                self.block_size,
                sampler,
                proposal_probs=probabilities,
            )
            if proposal.block_size != self.block_size or proposal.tokens.shape != (
                1,
                self.depth,
            ):
                raise RuntimeError("Native DSpark returned an inconsistent block width.")
            if sampled and proposal.proposal_probs is None:
                raise RuntimeError("Sampled DSpark did not return its proposal law.")
            state.drafts = proposal.tokens.reshape(-1).astype(mx.uint32)
            state.draft_lps = []
            state.draft_accept_lps = (
                [mx.log(proposal.proposal_probs[0, index]) for index in range(self.depth)]
                if sampled
                else []
            )
            row.confidences = proposal.confidences
            row.fed += int(hidden.shape[1])
            row.pending.clear()
            mx.async_eval(state.drafts)
        share = (time.perf_counter() - started) * 1000 / max(1, len(jobs))
        for _, state, *_ in jobs:
            if state.stats is not None:
                state.stats.mtp_head_ms += share
