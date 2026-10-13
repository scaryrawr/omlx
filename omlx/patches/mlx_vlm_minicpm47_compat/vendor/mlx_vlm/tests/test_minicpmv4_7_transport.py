"""Request-local position transport and tiny DSpark exactness, without assets."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace as NS
from typing import get_args, get_type_hints
from unittest.mock import Mock

import mlx.core as mx

mx.set_default_device(mx.cpu)

# isort: split
import numpy as np
import pytest

from mlx_vlm.generate import ar
from mlx_vlm.generate import dispatch
from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.generate.types import GenerateKwargs
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import BatchKVCache, KVCache, RotatingKVCache
from mlx_vlm.models.qwen3_5_moe.config import TextConfig
from mlx_vlm.models.qwen3_5_moe.language import LanguageModel
from mlx_vlm.speculative import dflash, utils
from mlx_vlm.speculative.cache_state import (
    SpeculativeCacheTransaction,
    start_speculative_cache,
)
from mlx_vlm.speculative.drafters.dspark.config import DSparkConfig
from mlx_vlm.speculative.drafters.dspark.dspark import DSparkDraftModel, DSparkProposal


def greedy(logits):
    return mx.argmax(logits, axis=-1)


class DeltaTarget:
    def __init__(self, delta):
        self.language_model = self
        self.delta = delta
        self.calls = []
        self.config = NS(eos_token_id=None)

    def chunked_prefill_policy(self, **kwargs):
        return True

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        batch, length = input_ids.shape
        positions = mx.broadcast_to(mx.arange(length)[None, None], (3, batch, length))
        return InputEmbeddingsFeatures(
            inputs_embeds=input_ids[..., None].astype(mx.float32),
            position_ids=positions,
            rope_deltas=mx.full((batch, 1), self.delta, dtype=mx.int32),
        )

    def __call__(self, inputs, cache, inputs_embeds=None, **kwargs):
        offset = cache[0].offset
        self.calls.append((offset, int(inputs.shape[1]), dict(kwargs)))
        batch, length = inputs.shape
        kv = mx.zeros((batch, 1, length, 1))
        cache[0].update_and_fetch(kv, kv)
        return LanguageModelOutput(
            logits=mx.broadcast_to(mx.array([0.0, 1.0, 0.0]), (batch, length, 3)),
            hidden_states=[mx.zeros((batch, length, 4))],
        )


@pytest.mark.parametrize("chunk", [None, 2])
def test_single_decode_retains_only_request_delta(chunk):
    target = DeltaTarget(7)
    result = list(
        ar.generate_step(
            mx.array([[1, 2, 1, 2, 1]]),
            target,
            None,
            None,
            prompt_cache=[KVCache()],
            max_tokens=3,
            temperature=0,
            prefill_step_size=chunk,
            image_bound="prefill-only",
        )
    )
    assert len(result) == 3
    prefill = [call for call in target.calls if call[0] < 5]
    decode = [call for call in target.calls if call[0] >= 5]
    assert sum(call[1] for call in prefill) == 5
    assert decode
    for _, _, kwargs in target.calls:
        assert kwargs["rope_deltas"].tolist() == [[7]]
    for _, _, kwargs in decode:
        assert "position_ids" not in kwargs
        assert "image_bound" not in kwargs
    assert not hasattr(target, "_rope_deltas")
    assert not hasattr(target, "_position_ids")


def test_single_apc_explicit_metadata_and_request_isolation():
    for delta in [9, -3, 9]:
        target = DeltaTarget(100)
        cache = KVCache()
        kv = mx.zeros((1, 1, 3, 1))
        cache.update_and_fetch(kv, kv)
        positions = mx.broadcast_to(mx.arange(5)[None, None], (3, 1, 5))
        list(
            ar.generate_step(
                mx.array([[1, 2]]),
                target,
                None,
                None,
                prompt_cache=[cache],
                max_tokens=2,
                temperature=0,
                prefill_step_size=None,
                position_ids=positions,
                rope_deltas=mx.array([[delta]]),
            )
        )
        assert target.calls[0][2]["position_ids"].tolist() == positions.tolist()
        assert all(call[2]["rope_deltas"].item() == delta for call in target.calls)
        assert "position_ids" not in target.calls[-1][2]


def test_single_and_server_dispatch_transport(monkeypatch):
    seen = []

    def rounds(*args, **kwargs):
        seen.append(kwargs["rope_deltas"].tolist())
        yield 2, None

    def rounds_batch(*args, **kwargs):
        seen.append(kwargs["rope_deltas"].tolist())
        yield [2, 2], None

    monkeypatch.setattr(utils, "_dflash_rounds", rounds)
    monkeypatch.setattr(utils, "_dflash_rounds_batch", rounds_batch)
    drafter = NS(config=NS(target_layer_ids=[0]))
    output = NS(hidden_states=[mx.zeros((1, 1, 4))])
    list(
        utils.run_speculative_rounds(
            NS(),
            drafter,
            [],
            mx.array([[1]]),
            mx.array([1]),
            mx.zeros(3),
            output,
            draft_kind="dflash",
            max_tokens=2,
            sampler=greedy,
            sampler_is_greedy=True,
            rope_deltas=mx.array([[7]]),
        )
    )
    for batch in [1, 2]:
        list(
            utils.run_speculative_server_rounds(
                NS(),
                drafter,
                [],
                mx.zeros((batch, 1, 4)),
                draft_kind="dflash",
                first_bonus=mx.ones((batch,), dtype=mx.int32),
                max_tokens=2,
                sampler=greedy,
                greedy_sampling=True,
                rope_deltas=mx.arange(batch)[:, None] + 5,
            )
        )
    assert seen == [[[7]], [[5]], [[5], [6]]]


class RowCache:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, keep):
        self.rows = [self.rows[index] for index in keep.tolist()]


def test_speculative_pending_reorder_tracks_cache_and_delta(monkeypatch):
    seen = []

    def rounds(*args, **kwargs):
        seen.append((args[2][0].rows, kwargs["rope_deltas"].tolist()))
        yield [1, 1], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", rounds)
    batch = ar.SpeculativeGenerationBatch(
        model=NS(),
        draft_model=NS(),
        draft_kind="dflash",
        uids=[10, 20, 30],
        first_tokens=mx.array([1, 2, 3]),
        prompt_cache=[RowCache([10, 20, 30])],
        sampler=greedy,
        stop_criteria=lambda token: False,
        max_tokens=[4, 4, 4],
        hidden=mx.zeros((3, 1, 4)),
        shared_kv_states=None,
        prompt_tokens=mx.array([[1], [2], [3]]),
        greedy_sampling=True,
        rope_deltas=mx.array([[5], [7], [9]]),
    )
    batch.filter([2, 0])
    assert batch.uids == [30, 10]
    assert batch._rope_deltas.tolist() == [[9], [5]]
    batch._start_rounds()
    next(batch._rounds_iter)
    assert seen == [([30, 10], [[9], [5]])]
    with pytest.raises(RuntimeError, match="reorder"):
        batch.filter([1, 0])


def test_prompt_delta_rows_never_broadcast_or_merge_missing_rows():
    with pytest.raises(ValueError, match="shape"):
        ar._split_prompt_kwargs_per_row({"rope_deltas": mx.array([[7]])}, 2)
    rows = [
        {"inputs_embeds": mx.zeros((1, 2, 4)), "rope_deltas": mx.array([[5]])},
        {"inputs_embeds": mx.zeros((1, 1, 4)), "rope_deltas": mx.array([[9]])},
    ]
    _, kwargs = ar._merge_prefill_prompt_kwargs(rows, [[1, 2], [3]])
    assert kwargs["rope_deltas"].tolist() == [[5], [9]]
    rows[1].pop("rope_deltas")
    with pytest.raises(ValueError, match="Every merged"):
        ar._merge_prefill_prompt_kwargs(rows, [[1, 2], [3]])
    with pytest.raises(ValueError, match="shape"):
        ar.PromptProcessingBatch._capture_rope_deltas_from_prompt_kwargs(
            {"rope_deltas": mx.array([[7]])}, NS(), 2
        )


@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
def test_other_drafters_preserve_legacy_metadata_and_reject_stateless_targets(kind):
    deltas = mx.array([[7]])
    legacy = NS(_rope_deltas=deltas)
    utils._validate_request_deltas(legacy, kind, deltas, 1)
    assert legacy._rope_deltas is deltas
    utils._validate_request_deltas(NS(), kind, None, 1)
    with pytest.raises(ValueError, match="only by DFlash"):
        utils._validate_request_deltas(NS(), kind, deltas, 1)


def test_ordinary_delta_merge_rejection_does_not_mutate_rows():
    left = ar.GenerationBatch.empty(NS(), greedy, lambda token: False)
    right = ar.GenerationBatch.empty(NS(), greedy, lambda token: False)
    left.uids, right.uids = [10], [20]
    left.max_tokens, right.max_tokens = [3], [3]
    left._num_tokens, right._num_tokens = [0], [0]
    left._rope_deltas = mx.array([[5]])
    with pytest.raises(RuntimeError, match="mixes"):
        left.extend(right)
    assert left.uids == [10]
    assert right.uids == [20]
    right._rope_deltas = mx.array([[9]])
    left.extend(right)
    assert left._rope_deltas.tolist() == [[5], [9]]
    left.filter([1, 0])
    assert left.uids == [20, 10]
    assert left._rope_deltas.tolist() == [[9], [5]]


def test_batch_round_filters_deltas_with_cache_rows():
    class Target(DeltaTarget):
        def __call__(self, inputs, cache, **kwargs):
            transaction = start_speculative_cache(cache, inputs.shape[1])
            output = super().__call__(inputs, cache, **kwargs)
            output.gdn_states = transaction
            return output

        def rollback_speculative_cache(self, *args):
            raise AssertionError("transaction must own rollback")

        def speculative_verify_dflash_hidden(self, *args):
            raise AssertionError(
                "request deltas cannot enter metadata-free convenience"
            )

        def speculative_argmax_from_hidden(self, hidden):
            raise AssertionError("metadata-free convenience is unsupported")

    class Drafter:
        prefer_requested_block_size = True
        config = NS(block_size=2, target_layer_ids=[0])

        def __init__(self):
            self.accept_lens, self.draft_lens = [], []

        def reset(self, target):
            return []

        def make_cache(self):
            return []

        def draft_block(self, bonus, hidden, cache, block_size, sampler, dtype):
            return mx.ones((1, block_size - 1), dtype=dtype)

    target = Target(0)
    cache = BatchKVCache(left_padding=[0, 0])
    list(
        dflash._dflash_rounds_batch(
            target,
            Drafter(),
            [cache],
            mx.zeros((2, 1, 4)),
            first_bonus=mx.array([1, 1]),
            max_tokens=5,
            sampler=greedy,
            stop_check=lambda row, token: row == 0,
            greedy_sampling=True,
            rope_deltas=mx.array([[5], [9]]),
        )
    )
    observed = [call[2]["rope_deltas"].tolist() for call in target.calls]
    assert observed[0] == [[5], [9]]
    assert observed[1:] and all(deltas == [[9]] for deltas in observed[1:])


def tiny_hybrid(native=False):
    if native:
        from mlx_vlm.models.minicpmv4_7.config import ModelConfig
        from mlx_vlm.models.minicpmv4_7.language import LanguageModel as NativeLanguage
        from mlx_vlm.tests.test_minicpmv4_7 import tiny_config

        config = ModelConfig.from_dict(tiny_config())
        return NativeLanguage(config.text_config, config)
    return LanguageModel(
        TextConfig(
            model_type="qwen3_5_moe",
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=3,
            full_attention_interval=2,
            num_experts=2,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=32,
            moe_intermediate_size=16,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=128,
            rope_parameters={
                "type": "default",
                "mrope_section": [1, 0, 0],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
            },
        )
    )


def assert_cache_equal(actual, expected):
    for actual_cache, expected_cache in zip(actual, expected):
        for a, b in zip(actual_cache.state, expected_cache.state):
            if a is None or b is None:
                assert a is b
            else:
                np.testing.assert_allclose(
                    np.array(a), np.array(b), rtol=2e-5, atol=2e-5
                )


@pytest.mark.parametrize("accepted", [0, 1, 2])
@pytest.mark.parametrize("rotating", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_real_hybrid_capture_verifier_keeps_transaction_and_delta(
    accepted, rotating, native
):
    mx.random.seed(41)
    model = tiny_hybrid(native)
    actual, expected = model.make_cache(), model.make_cache()
    if rotating:
        actual[-1], expected[-1] = (
            RotatingKVCache(max_size=4, keep=0),
            RotatingKVCache(max_size=4, keep=0),
        )
    prompt = mx.array([[1, 2, 3, 4, 5]])
    delta = mx.array([[7]])
    positions = mx.broadcast_to(mx.arange(5)[None, None], (3, 1, 5)) + 7
    for caches in [actual, expected]:
        model(prompt, cache=caches, position_ids=positions, rope_deltas=delta)
    block = mx.array([[6, 7, 8]])
    output, transaction = dflash._dflash_verify(
        model, block, actual, [0, 1], rope_deltas=delta
    )
    assert isinstance(transaction, SpeculativeCacheTransaction)
    assert output.gdn_states is transaction
    assert [hidden.shape for hidden in output.hidden_states] == [(1, 3, 16)] * 2
    transaction.commit([accepted + 1])
    model(block[:, : accepted + 1], cache=expected, rope_deltas=delta)
    next_token = mx.array([[9]])
    actual_logits = model(next_token, cache=actual, rope_deltas=delta).logits
    expected_logits = model(next_token, cache=expected, rope_deltas=delta).logits
    np.testing.assert_allclose(
        np.array(actual_logits), np.array(expected_logits), rtol=2e-5, atol=2e-5
    )
    assert_cache_equal(actual, expected)


@pytest.mark.parametrize("native", [False, True])
def test_post_verification_sampler_exception_aborts_real_hybrid(native):
    model = tiny_hybrid(native)
    cache = model.make_cache()
    model(
        mx.array([[1, 2, 3]]),
        cache=cache,
        rope_deltas=mx.array([[4]]),
        position_ids=mx.broadcast_to(mx.arange(3)[None, None], (3, 1, 3)) + 4,
    )
    before = deepcopy(cache)

    def fail(logits):
        raise RuntimeError("sample failed")

    with pytest.raises(RuntimeError, match="sample failed"):
        dflash._dflash_verify_greedy(
            model,
            mx.array([[4, 5, 6]]),
            cache,
            [0, 1],
            fail,
            rope_deltas=mx.array([[4]]),
        )
    assert_cache_equal(cache, before)


def tiny_dspark(**kwargs):
    config = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        block_size=4,
        proposal_length=3,
        mask_token_id=31,
        target_layer_ids=[0, 1],
        num_target_layers=2,
        markov_rank=4,
        enable_confidence_head=True,
    )
    config.update(kwargs)
    return DSparkDraftModel(DSparkConfig(**config))


@pytest.mark.parametrize("from_anchor", [True, False])
def test_real_dspark_proposals_keep_conditioned_law_and_confidence(from_anchor):
    mx.random.seed(9)
    target = tiny_hybrid()
    drafter = tiny_dspark(sample_from_anchor=from_anchor)
    drafter.bind(target)
    hidden = mx.zeros((1, 3, 32))
    inputs = mx.array([[2, 31, 31]]) if from_anchor else mx.array([[2, 31, 31, 31]])
    draft_hidden = drafter._hidden(inputs, hidden, drafter.make_cache())
    if not from_anchor:
        draft_hidden = draft_hidden[:, 1:]
    base_logits = drafter._logits(draft_hidden)
    probabilities = lambda logprobs: mx.softmax(logprobs / 0.7, axis=-1)
    result = drafter.propose_block(
        2, hidden, drafter.make_cache(), 4, greedy, proposal_probs=probabilities
    )
    previous = mx.array([2])
    for step in range(3):
        conditioned = drafter.markov_head.apply_step_logits(
            base_logits[:, step], previous
        )
        expected = mx.softmax(conditioned.astype(mx.float32) / 0.7, axis=-1)
        np.testing.assert_allclose(
            np.array(result.proposal_probs[:, step]), np.array(expected), atol=1e-6
        )
        previous = result.tokens[:, step]
    expected_confidence = drafter.confidence(draft_hidden, mx.array([2]), result.tokens)
    np.testing.assert_allclose(
        np.array(result.confidences), np.array(expected_confidence), atol=1e-6
    )
    assert result.block_size == 4
    assert result.sample_from_anchor is from_anchor
    assert result.tokens.shape == result.confidences.shape == (1, 3)
    legacy = drafter.propose_block(2, hidden, drafter.make_cache(), 4, greedy)
    assert legacy.proposal_probs is None
    assert drafter.draft_block(2, hidden, drafter.make_cache(), 4, greedy).tolist() == (
        legacy.tokens.tolist()
    )
    with pytest.raises(ValueError, match="cap"):
        drafter.propose_block(2, hidden, drafter.make_cache(), 5, greedy)


@pytest.mark.parametrize(
    "tokens,accepted,expected",
    [
        ([0, 0], 0, [1]),
        ([1, 0], 1, [1, 1]),
        ([1, 1], 2, [1, 1, 1]),
    ],
)
def test_sampled_correction_accept_none_partial_all(tokens, accepted, expected):
    proposal = DSparkProposal(
        tokens=mx.array([tokens]),
        proposal_probs=mx.array([[[1.0, 0.0], [1.0, 0.0]]]),
        confidences=None,
        block_size=3,
        sample_from_anchor=True,
    )
    # For token 1 use q=1 so acceptance is deterministic too.
    proposal.proposal_probs[0, :, :] = mx.array(
        [[1 - token, token] for token in tokens]
    )
    logits = mx.array([[[-float("inf"), 0.0]] * 3])
    actual = dflash._sample_dspark_target_walk(logits, proposal, mx.exp, 3)
    assert actual == (accepted, expected)


def test_sampled_correction_uses_residual_not_original_target(monkeypatch):
    proposal = DSparkProposal(
        tokens=mx.array([[0]]),
        proposal_probs=mx.array([[[0.8, 0.2]]]),
        confidences=None,
        block_size=2,
        sample_from_anchor=True,
    )
    monkeypatch.setattr(mx.random, "uniform", lambda **kwargs: mx.array(0.9))
    seen = []

    def categorical(logprobs, **kwargs):
        seen.append(mx.exp(logprobs).tolist())
        return mx.array(1)

    monkeypatch.setattr(mx.random, "categorical", categorical)
    result = dflash._sample_dspark_target_walk(
        mx.log(mx.array([[[0.2, 0.8], [0.5, 0.5]]])),
        proposal,
        mx.exp,
        2,
    )
    assert result == (0, [1])
    np.testing.assert_allclose(seen, [[0.0, 1.0]], atol=1e-6)


def test_sampled_correction_preserves_target_marginal():
    mx.random.seed(85)
    logits = mx.log(mx.array([[[0.2, 0.8], [0.5, 0.5]]]))
    q = mx.array([[[0.8, 0.2]]])
    count = 0
    for _ in range(2000):
        tokens = mx.random.categorical(mx.log(q)).astype(mx.int32)
        proposal = DSparkProposal(tokens, q, None, 2, True)
        _, emitted = dflash._sample_dspark_target_walk(logits, proposal, mx.exp, 1)
        count += emitted[0] == 1
    assert abs(count / 2000 - 0.8) < 0.04


class ProbabilitySampler:
    def probabilities(self, logprobs):
        return mx.softmax(logprobs / 0.7, axis=-1)

    def __call__(self, logprobs):
        return mx.random.categorical(mx.log(self.probabilities(logprobs)))


@pytest.mark.parametrize("native", [False, True])
def test_real_hybrid_dspark_sampled_rounds_use_proposal_result(monkeypatch, native):
    mx.random.seed(28)
    target = tiny_hybrid(native)
    drafter = tiny_dspark(num_target_layers=len(target.layers))
    delta = mx.array([[5]])
    inputs = mx.array([[1, 2, 3]])
    caches = target.make_cache()
    output = target(
        inputs,
        cache=caches,
        rope_deltas=delta,
        position_ids=mx.broadcast_to(mx.arange(3)[None, None], (3, 1, 3)) + 5,
        capture_layer_ids=[0, 1],
    )
    sampler = ProbabilitySampler()
    first = sampler(output.logits[:, -1])
    seen = []
    propose = drafter.propose_block

    def observe(*args, **kwargs):
        result = propose(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(drafter, "propose_block", observe)
    result = list(
        utils.run_speculative_rounds(
            target,
            drafter,
            caches,
            inputs,
            first,
            mx.zeros(32),
            output,
            draft_kind="dflash",
            max_tokens=7,
            sampler=sampler,
            rope_deltas=delta,
        )
    )
    assert len(result) == 7
    assert seen
    assert all(proposal.proposal_probs is not None for proposal in seen)
    assert all(proposal.confidences.shape == proposal.tokens.shape for proposal in seen)
    assert not any(getattr(cache, "is_speculating", False) for cache in caches)


@pytest.mark.parametrize("chunk", [None, 2])
def test_native_greedy_ar_and_dspark_full_generation_transport(chunk):
    mx.random.seed(74)
    language = tiny_hybrid(native=True)
    delta = mx.array([[7]])

    def embeddings(input_ids, pixel_values=None, **kwargs):
        length = input_ids.shape[1]
        positions = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length)) + 7
        return InputEmbeddingsFeatures(
            inputs_embeds=language.model.embed_tokens(input_ids),
            position_ids=positions,
            rope_deltas=delta,
        )

    target = NS(
        language_model=language, config=language.config, get_input_embeddings=embeddings
    )
    drafter = tiny_dspark(num_target_layers=4)
    inputs = mx.array([[1, 3, 4, 5, 6]])
    options = dict(
        max_tokens=9,
        temperature=0,
        prefill_step_size=chunk,
    )
    expected = list(ar.generate_step(inputs, target, None, None, **options))
    actual = list(
        ar.generate_step(
            inputs,
            target,
            None,
            None,
            draft_model=drafter,
            draft_kind="dflash",
            draft_block_size=3,
            **options,
        )
    )
    assert [token for token, _ in actual] == [token for token, _ in expected]
    assert not hasattr(language, "_rope_deltas")
    assert not hasattr(language, "_position_ids")


def test_native_greedy_batched_dispatch_matches_isolated_delta_rows():
    mx.random.seed(66)
    language = tiny_hybrid(native=True)
    drafter = tiny_dspark(num_target_layers=4)
    inputs = mx.array([[1, 3, 4], [5, 6, 7]])
    deltas = mx.array([[7], [-2]])
    positions = mx.broadcast_to(mx.arange(3)[None, None], (3, 2, 3)) + deltas[None]
    caches = ar._make_cache(language, [0, 0])
    output = language(
        inputs,
        cache=caches,
        position_ids=positions,
        rope_deltas=deltas,
        capture_layer_ids=[0, 1],
    )
    first = greedy(output.logits[:, -1])
    rows = [[], []]
    for tokens, _ in utils.run_speculative_rounds(
        language,
        drafter,
        caches,
        inputs,
        first,
        mx.zeros((2, 32)),
        output,
        draft_kind="dflash",
        max_tokens=8,
        sampler=greedy,
        sampler_is_greedy=True,
        draft_block_size=3,
        rope_deltas=deltas,
    ):
        for row, token in enumerate(tokens):
            if token is not None:
                rows[row].append(token)
    for row in range(2):
        cache = language.make_cache()
        out = language(
            inputs[row : row + 1],
            cache=cache,
            position_ids=positions[:, row : row + 1],
            rope_deltas=deltas[row : row + 1],
        )
        expected = []
        for _ in range(8):
            token = greedy(out.logits[:, -1])
            expected.append(token.item())
            out = language(
                token[:, None], cache=cache, rope_deltas=deltas[row : row + 1]
            )
        assert rows[row] == expected
    assert not hasattr(language, "_rope_deltas")


def test_positioned_probability_callback_matches_filtered_sampling_law():
    from mlx_vlm.sample_utils import apply_top_k, apply_top_p

    logprobs = mx.log(mx.array([[0.05, 0.15, 0.3, 0.5]]))
    sampler = ar._PositionedTargetSampler(temperature=0.7, top_p=0.8, top_k=3, seed=5)
    scaled = apply_top_k(logprobs, 3) / 0.7
    normalized = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    expected = mx.softmax(apply_top_p(normalized, 0.8), axis=-1)
    np.testing.assert_allclose(
        np.array(sampler.probabilities(logprobs)), np.array(expected), atol=1e-6
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"top_p": 0.8, "top_k": 3, "min_p": 0.1},
        {"typical_p": 0.7, "top_n_sigma": 1.0},
        {"p_less": True},
        {"min_p": 0.9},
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_standard_sampler_probability_callback_is_its_actual_law(
    monkeypatch, options, dtype
):
    from mlx_vlm import sample_utils

    seen = []

    def categorical(logprobs, temp):
        seen.append(logprobs * (1 / temp))
        return mx.argmax(logprobs, axis=-1)

    monkeypatch.setattr(sample_utils, "categorical_sampling", categorical)
    sampler = sample_utils.make_sampler(temp=0.7, **options)
    logprobs = mx.log(mx.array([[0.03, 0.17, 0.3, 0.5]], dtype=dtype))
    sampler(logprobs)
    expected = mx.softmax(seen[0].astype(mx.float32), axis=-1)
    np.testing.assert_allclose(
        np.array(sampler.probabilities(logprobs)), np.array(expected), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(sampler.probabilities(logprobs).sum(axis=-1)), [1.0], atol=1e-6
    )


def test_standard_sampler_rejects_stochastic_xtc_probability_callback():
    from mlx_vlm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.7, xtc_probability=0.5)
    with pytest.raises(ValueError, match="stochastic XTC"):
        sampler.probabilities(mx.zeros((1, 4)))


@pytest.mark.parametrize("chunk", [None, 2])
def test_native_default_sampled_generation_uses_exact_dspark(chunk):
    mx.random.seed(13)
    language = tiny_hybrid(native=True)
    drafter = tiny_dspark(num_target_layers=4)

    def embeddings(input_ids, pixel_values=None, **kwargs):
        return InputEmbeddingsFeatures(
            inputs_embeds=language.model.embed_tokens(input_ids),
            position_ids=mx.broadcast_to(
                mx.arange(input_ids.shape[1])[None, None], (3, 1, input_ids.shape[1])
            )
            + 7,
            rope_deltas=mx.array([[7]]),
        )

    target = NS(
        language_model=language, config=language.config, get_input_embeddings=embeddings
    )
    result = list(
        ar.generate_step(
            mx.array([[1, 3, 4, 5, 6]]),
            target,
            None,
            None,
            max_tokens=8,
            temperature=0.7,
            top_p=0.9,
            top_k=8,
            min_p=0.01,
            draft_model=drafter,
            draft_kind="dflash",
            draft_block_size=3,
            prefill_step_size=chunk,
        )
    )
    assert len(result) == 8
    assert drafter.accept_lens
    assert not hasattr(language, "_rope_deltas")


def test_unsupported_dspark_sampling_fails_before_first_token():
    drafter = tiny_dspark()
    rounds = utils.run_speculative_rounds(
        NS(),
        drafter,
        [],
        mx.array([[1]]),
        mx.array([1]),
        mx.zeros(32),
        NS(hidden_states=[mx.zeros((1, 1, 16))] * 2),
        draft_kind="dflash",
        max_tokens=2,
        sampler=greedy,
    )
    with pytest.raises(ValueError, match="probabilities"):
        next(rounds)
    sampler = NS(probabilities=mx.exp)
    with pytest.raises(ValueError, match="batched"):
        dflash._dspark_probability_sampler(drafter, sampler, False, 2)
    drafter.config.block_size_policy = "adaptive"
    assert dflash._dspark_probability_sampler(drafter, sampler, True) is None
    with pytest.raises(ValueError, match="adaptive"):
        dflash._dspark_probability_sampler(drafter, sampler, False)


def test_explicit_cached_prefix_provider_receives_full_canvas_metadata(monkeypatch):
    language = tiny_hybrid(native=True)
    full_ids = mx.array([[1, 3, 4]])
    mask = mx.array([[1, 1, 1]])
    kwargs = {
        "target_sizes": mx.array([[4, 4]]),
        "target_sizes_videos": mx.array([[8, 4]]),
        "mm_token_type_ids": mx.zeros_like(full_ids),
        "downsample_mode": "16x",
    }
    positions = mx.broadcast_to(mx.arange(3)[None, None], (3, 1, 3))
    deltas = mx.array([[7]])
    provider = Mock(return_value=(positions, deltas))
    monkeypatch.setattr(language, "get_rope_index", provider)
    assert dispatch._prime_cached_prefix_rope_state(
        NS(language_model=language), full_ids, mask, kwargs
    )
    args, supplied = provider.call_args
    assert args[0] is full_ids
    assert supplied["target_sizes"] is kwargs["target_sizes"]
    assert supplied["target_sizes_videos"] is kwargs["target_sizes_videos"]
    assert supplied["mm_token_type_ids"] is kwargs["mm_token_type_ids"]
    assert supplied["attention_mask"] is mask
    assert supplied["downsample_mode"] == "16x"
    assert kwargs["position_ids"] is positions
    assert kwargs["rope_deltas"] is deltas
    assert not hasattr(language, "_position_ids")
    assert not hasattr(language, "_rope_deltas")
    kwargs["attention_mask"] = mask
    assert dispatch._prime_cached_prefix_rope_state(
        NS(language_model=language), full_ids, None, kwargs
    )
    assert provider.call_args.kwargs["attention_mask"] is mask
    provider.side_effect = ValueError("invalid native canvas")
    with pytest.raises(ValueError, match="invalid native canvas"):
        dispatch._prime_cached_prefix_rope_state(
            NS(language_model=language), full_ids, mask, {}
        )
    with pytest.raises(ValueError, match="require get_rope_index"):
        dispatch._prime_cached_prefix_rope_state(
            NS(language_model=NS(requires_explicit_rope_deltas=True)),
            full_ids,
            mask,
            {},
        )


@pytest.mark.parametrize("reuse_source", ["turn", "apc"])
@pytest.mark.parametrize("chunk", [None, 2])
def test_native_dispatch_cold_and_warm_visual_prefix_match(
    monkeypatch, reuse_source, chunk
):
    from mlx_vlm.models.minicpmv4_7.config import ModelConfig
    from mlx_vlm.models.minicpmv4_7.minicpmv4_7 import Model
    from mlx_vlm.tests.test_minicpmv4_7 import tiny_config

    mx.random.seed(93)
    model = Model(ModelConfig.from_dict(tiny_config()))
    full_ids = mx.array([[10, 8, 11, 10, 9, 11, 20, 21, 22]])
    mask = mx.ones_like(full_ids)
    grids = mx.array([[4, 4]])
    pixels = mx.zeros((3, 2, 32))
    image_features = model.get_image_features(pixels, grids, "16x")
    video_features = model.get_image_features(pixels, grids, "16x")
    metadata = {
        "target_sizes": grids,
        "target_sizes_videos": grids,
        "mm_token_type_ids": mx.array([[0, 1, 0, 0, 2, 0, 0, 0, 0]]),
        "cached_image_features": image_features,
        "cached_video_features": video_features,
        "downsample_mode": "16x",
        "attention_mask": mask,
    }
    features = model.get_input_embeddings(full_ids, **metadata)
    prefix = 6
    warm_cache = model.language_model.make_cache()
    model.language_model(
        full_ids[:, :prefix],
        inputs_embeds=features.inputs_embeds[:, :prefix],
        position_ids=features.position_ids[..., :prefix],
        rope_deltas=features.rope_deltas,
        mask=mask[:, :prefix],
        cache=warm_cache,
    )
    mx.eval([entry.state for entry in warm_cache])

    detokenizer = NS(
        last_segment="",
        add_token=lambda *args, **kwargs: None,
        finalize=lambda: None,
    )
    processor = NS(tokenizer=NS(stopping_criteria=lambda token: False))
    monkeypatch.setattr(dispatch, "wired_limit", lambda *args: nullcontext())
    monkeypatch.setattr(
        dispatch, "make_streaming_detokenizer", lambda processor: detokenizer
    )
    observations = []
    original_generate_step = dispatch.generate_step

    def observe(*args, **kwargs):
        observations.append((args, dict(kwargs), []))
        for token, logprobs in original_generate_step(*args, **kwargs):
            copied = mx.array(logprobs)
            mx.eval(copied)
            observations[-1][2].append(copied)
            yield token, logprobs

    monkeypatch.setattr(dispatch, "generate_step", observe)
    options = dict(
        input_ids=full_ids,
        mask=mask,
        max_tokens=4,
        temperature=0,
        prefill_step_size=chunk,
        **metadata,
    )
    cold = list(dispatch.stream_generate(model, processor, "", **options))
    if reuse_source == "turn":
        state = PromptCacheState()
        state.update(full_ids[0, :prefix].tolist(), warm_cache)
        options["prompt_cache_state"] = state
    else:
        coordinator = Mock()
        coordinator.enabled = True
        coordinator.is_checkpoint = False
        coordinator.lookup.return_value = {
            "prefix_len": prefix,
            "warm_cache": warm_cache,
        }
        coordinator.materialize_single.return_value = warm_cache
        monkeypatch.setattr(dispatch._apc, "APCCoordinator", lambda *args: coordinator)
        monkeypatch.setattr(dispatch._apc, "semantic_extra_hash", lambda **kwargs: 0)
        options["apc_manager"] = NS()
    warm = list(dispatch.stream_generate(model, processor, "", **options))
    assert cold[-1].token_ids == warm[-1].token_ids
    assert cold[-1].cached_tokens == 0
    assert warm[-1].cached_tokens == prefix
    cold_call, warm_call = observations
    warm_args, warm_kwargs, warm_logprobs = warm_call
    assert warm_args[0].tolist() == full_ids[:, prefix:].tolist()
    assert warm_args[2] is None
    assert warm_args[3].tolist() == mask[:, prefix:].tolist()
    assert warm_kwargs["attention_mask"].tolist() == mask[:, prefix:].tolist()
    assert "cached_image_features" not in warm_kwargs
    assert "cached_video_features" not in warm_kwargs
    assert "pixel_values_videos" not in warm_kwargs
    assert warm_kwargs["position_ids"].shape == features.position_ids.shape
    np.testing.assert_array_equal(
        np.array(warm_kwargs["position_ids"]), np.array(features.position_ids)
    )
    np.testing.assert_array_equal(
        np.array(warm_kwargs["rope_deltas"]), np.array(features.rope_deltas)
    )
    for actual, expected in zip(warm_logprobs, cold_call[2]):
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=2e-5, atol=2e-5
        )
    assert not hasattr(model.language_model, "_rope_deltas")


def test_native_generation_controls_are_typed():
    hints = get_type_hints(GenerateKwargs)
    assert get_args(hints["downsample_mode"]) == ("4x", "16x")
    assert hints["max_slice_nums"] is int


@pytest.mark.parametrize("padding", [[0, 1], [0, 0]])
def test_native_dspark_rejects_incomplete_or_padded_capture_history(padding):
    language = tiny_hybrid(native=True)
    drafter = tiny_dspark(num_target_layers=4)
    caches = ar._make_cache(language, padding)
    language(
        mx.array([[1, 2, 3], [1, 2, 3]]),
        cache=caches,
        rope_deltas=mx.zeros((2, 1), mx.int32),
    )
    hidden = mx.zeros((2, 2 if padding == [0, 0] else 3, 32))
    with pytest.raises(ValueError, match="capture histor"):
        list(
            dflash._dflash_rounds_batch(
                language,
                drafter,
                caches,
                hidden,
                first_bonus=mx.array([1, 1]),
                max_tokens=4,
                sampler=greedy,
                draft_block_size=3,
                rope_deltas=mx.zeros((2, 1), mx.int32),
            )
        )


@pytest.mark.parametrize("padding", ["none", "left", "right"])
@pytest.mark.parametrize("chunk", [None, 1, 2])
def test_native_masked_chunk_prefill_matches_unpadded_generation(padding, chunk):
    from mlx_vlm.models.minicpmv4_7.config import ModelConfig
    from mlx_vlm.models.minicpmv4_7.minicpmv4_7 import Model
    from mlx_vlm.tests.test_minicpmv4_7 import tiny_config

    mx.random.seed(107)
    target = Model(ModelConfig.from_dict(tiny_config()))
    tokens = [1, 3, 4, 5]
    baseline = list(
        ar.generate_step(
            mx.array([tokens]),
            target,
            None,
            None,
            max_tokens=4,
            temperature=0,
            prefill_step_size=None,
        )
    )
    if padding == "left":
        ids, mask = [0, 0, 0, 0] + tokens, [0] * 4 + [1] * 4
    elif padding == "right":
        ids, mask = tokens + [0, 0, 0, 0], [1] * 4 + [0] * 4
    else:
        ids, mask = tokens, [1] * 4
    actual = list(
        ar.generate_step(
            mx.array([ids]),
            target,
            None,
            mx.array([mask]),
            attention_mask=mx.array([mask]),
            max_tokens=4,
            temperature=0,
            prefill_step_size=chunk,
        )
    )
    assert [token for token, _ in actual] == [token for token, _ in baseline]
    for (_, result), (_, expected) in zip(actual, baseline):
        np.testing.assert_allclose(
            np.array(result),
            np.array(expected),
            rtol=2e-5,
            atol=2e-5,
        )
