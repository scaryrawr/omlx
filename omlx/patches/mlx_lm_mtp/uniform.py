# SPDX-License-Identifier: Apache-2.0
"""Common-frontier target sampling for uniform recurrent-cache rollback."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class UniformAcceptance:
    accepted_drafts: int
    successors: tuple[int, ...]


def sample_uniform_prefix(
    drafts: Sequence[Sequence[int]],
    sample_position: Callable[[int], Sequence[int]],
    stop_position: Callable[[int, Sequence[int]], bool],
) -> UniformAcceptance:
    depth = len(drafts[0])
    for position in range(depth + 1):
        targets = tuple(sample_position(position))
        if (
            position == depth
            or stop_position(position, targets)
            or any(target != row[position] for target, row in zip(targets, drafts))
        ):
            return UniformAcceptance(position, targets)
    raise AssertionError("Uniform verification did not reach its frontier")


def advance_uniform(batch, depth, rows, replacements, *, cache=None):
    """Commit one exact native verify at the cohort's common sampled frontier."""
    import mlx.core as mx

    from . import batch_generator as bg

    whole_batch = cache is not None
    if cache is None:
        cache = bg._merge_row_caches([row.prompt_cache for _, row, _ in rows])
    inputs = mx.stack(
        [mx.concatenate([state.next_main, state.drafts]) for _, _, state in rows]
    )
    logits, hidden, undo, _ = bg._call_backbone_captured(
        batch.model, inputs, cache, n_confirmed=1
    )
    if undo is None:
        raise RuntimeError("Nemotron verification did not return exact rollback state")
    drafts = inputs[:, 1:].tolist()
    distributions = [[] for _ in rows]
    matchers = [copy(row._matchers[0]) for _, row, _ in rows]

    def sample_position(position):
        sampled = []
        for i, (_, row, _) in enumerate(rows):
            processors = bg._proc_list(row)
            previous = (
                row._token_context[0].update_and_fetch(
                    inputs[i, position : position + 1]
                )
                if processors is not None
                else None
            )
            value = bg._apply_processors(
                processors, previous, logits[i : i + 1, position, :]
            )
            lp = bg._logprobs(value)
            distributions[i].append(lp[0])
            token = bg._resolve_sampler(row)(lp).reshape(-1).tolist()
            sampled.append(int(token[0]))
        return sampled

    def stop_position(position, targets):
        terminal = False
        for (_, row, _), matcher, token in zip(rows, matchers, targets):
            stopped = bool(matcher.advance(token))
            remaining = row.max_tokens[0] - row._num_tokens[0]
            terminal |= stopped or position + 1 >= remaining
        return terminal

    decision = sample_uniform_prefix(drafts, sample_position, stop_position)
    accepted = decision.accepted_drafts
    # A failed target/head commit fails the cohort. Never publish a queue or
    # reinterpret partially advanced native state as an ordinary-decode fallback.
    batch.model.rollback_speculative_cache(cache, undo, accepted, depth + 1)
    staged = []
    for i, (_, row, state) in enumerate(rows):
        next_state = copy(state)
        next_state.queue = deque()
        next_state.stats = deepcopy(state.stats)
        next_state.mtp_cache = bg._clone_mtp_head_cache(state.mtp_cache)
        bg._mtp_head_trim_to(next_state.mtp_cache, state.hist_offset)
        committed_ids = drafts[i][:accepted] + [decision.successors[i]]
        committed = mx.array(committed_ids, dtype=mx.uint32)
        previous = (
            row._token_context[0].tokens if bg._proc_list(row) is not None else None
        )
        bg._chain_next_drafts(
            row, next_state, hidden[i : i + 1, : accepted + 1], committed, previous
        )
        next_state.next_main = committed[-1:]
        next_state.stats.cycles += 1
        next_state.stats.accepts += accepted
        next_state.stats.rejects += int(accepted < depth)
        while len(next_state.stats.depth_drafted) < depth:
            next_state.stats.depth_drafted.append(0)
            next_state.stats.depth_accepted.append(0)
        for j in range(depth):
            next_state.stats.depth_drafted[j] += int(j <= accepted)
            next_state.stats.depth_accepted[j] += int(j < accepted)
        for j, token in enumerate(committed_ids):
            source = (
                "draft" if j < accepted else "bonus" if accepted == depth else "verify"
            )
            next_state.queue.append((token, distributions[i][j], source))
        staged.append(next_state)
    mx.eval(
        [entry.state for entry in cache],
        [[entry.state for entry in state.mtp_cache] for state in staged],
        [state.drafts for state in staged],
    )
    for row_index, ((index, row, state), next_state) in enumerate(zip(rows, staged)):
        state.__dict__.update(next_state.__dict__)
        batch._token_context[index] = row._token_context[0]
        if not whole_batch:
            replacements[index] = [entry.extract(row_index) for entry in cache]
    if whole_batch:
        batch.prompt_cache = cache
    bg._clear_rollback(cache)
    return whole_batch


def terminal_cache(batch, row_index, token_id, *, pending):
    """Return exactly the emitted native prefix without sampling a successor."""
    import mlx.core as mx

    cache = batch.extract_cache(row_index)
    if pending:
        result = batch.model(
            mx.array([[token_id]], dtype=mx.uint32), cache=cache, return_hidden=True
        )
        mx.eval(result.logits, [entry.state for entry in cache])
    return cache
