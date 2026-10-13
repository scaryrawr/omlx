# SPDX-License-Identifier: Apache-2.0
"""Request lifecycle and authoritative native DSpark proposal handoff."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.model_settings import validate_minicpm_speculation
from omlx.speculative.dspark_drafter import DSparkDrafter
from omlx.utils.sampling import make_sampler


@pytest.fixture(autouse=True)
def cpu_arrays():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _NativeProposalFixture:
    config = SimpleNamespace(target_layer_ids=[3, 11])

    def __init__(self):
        self.calls = []

    def make_cache(self):
        return [SimpleNamespace(offset=0)]

    def propose_block(
        self, last_bonus, hidden, cache, block_size, sampler, *, proposal_probs
    ):
        self.calls.append((last_bonus, hidden, cache))
        cache[0].offset += hidden.shape[1]
        logits = mx.array([[1.0, 3.0, 2.0, -1.0]])
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        depth = block_size - 1
        probabilities = (
            mx.stack([proposal_probs(lp) for _ in range(depth)], axis=1)
            if proposal_probs is not None
            else None
        )
        return SimpleNamespace(
            tokens=mx.full((1, depth), 1, dtype=mx.int32),
            proposal_probs=probabilities,
            confidences=mx.full((1, depth), 0.75),
            block_size=block_size,
            sample_from_anchor=False,
        )


def _state(uid):
    return SimpleNamespace(uid=uid, stats=SimpleNamespace(mtp_head_ms=0.0))


def _captures(length, batch=1):
    return [
        mx.full((batch, length, 2), 3.0),
        mx.full((batch, length, 2), 11.0),
    ]


def test_native_contract_is_required():
    with pytest.raises(ValueError, match="proposal/probability/confidence"):
        DSparkDrafter(SimpleNamespace(), block_size=3, source_path="fixture")


def test_request_seeds_are_not_truncated_and_only_committed_rows_are_fed():
    model = _NativeProposalFixture()
    drafter = DSparkDrafter(model, block_size=3, source_path="fixture")
    state = _state(7)
    drafter.seed_request("request", _captures(3000))
    drafter.bind_uid("request", state.uid)
    drafter.draft([(None, state, _captures(1), mx.array([42]), None)])
    assert model.calls[0][0].tolist() == [42]
    assert model.calls[0][1].shape == (1, 3001, 4)
    assert model.calls[0][1][0, 0].tolist() == [3.0, 3.0, 11.0, 11.0]
    assert drafter.context_length(state.uid) == 3001
    assert state.drafts.tolist() == [1, 1]
    assert state.draft_accept_lps == []
    assert drafter._rows[state.uid].confidences.tolist() == [[0.75, 0.75]]
    drafter.draft([(None, state, _captures(2), mx.array([1, 43]), None)])
    assert model.calls[1][1].shape == (1, 2, 4)
    assert model.calls[0][2] is model.calls[1][2]
    assert drafter.context_length(state.uid) == 3003


def test_batched_greedy_rows_keep_independent_context_and_cache():
    model = _NativeProposalFixture()
    drafter = DSparkDrafter(model, block_size=3, source_path="fixture")
    first, second = _state(7), _state(8)
    drafter.observe([7, 8], _captures(1, batch=2))
    drafter.draft(
        [
            (None, first, _captures(2), mx.array([42]), None),
            (None, second, _captures(3), mx.array([43]), None),
        ]
    )
    assert model.calls[0][2] is not model.calls[1][2]
    assert drafter.context_length(7) == 3
    assert drafter.context_length(8) == 4
    drafter.release([7])
    drafter.release([7])
    assert drafter.context_length(7) == 0
    assert drafter.context_length(8) == 4


def test_sampled_proposal_q_is_native_probability_callback_not_independent_logits():
    model = _NativeProposalFixture()
    drafter = DSparkDrafter(model, block_size=3, source_path="fixture")
    state = _state(7)
    sampler = make_sampler(temp=0.7, top_k=2, top_p=1.0)
    with (
        patch("omlx.speculative.dspark_drafter.bg._is_greedy", return_value=False),
        patch(
            "omlx.speculative.dspark_drafter.bg._resolve_draft_sampler",
            return_value=sampler,
        ),
    ):
        drafter.draft([(object(), state, _captures(2), mx.array([42]), None)])
    assert len(state.draft_accept_lps) == 2
    expected = mx.softmax(mx.array([3.0, 2.0]) / 0.7).tolist()
    for lp in state.draft_accept_lps:
        actual = mx.exp(lp).tolist()
        assert actual == pytest.approx([0.0, *expected, 0.0], abs=1e-6)


def test_sampled_batches_fail_before_native_cache_mutation():
    model = _NativeProposalFixture()
    drafter = DSparkDrafter(model, block_size=3, source_path="fixture")
    with (
        patch("omlx.speculative.dspark_drafter.bg._is_greedy", return_value=False),
        pytest.raises(ValueError, match="Batched sampled"),
    ):
        drafter.draft(
            [
                (object(), _state(7), _captures(1), mx.array([42]), None),
                (object(), _state(8), _captures(1), mx.array([43]), None),
            ]
        )
    assert model.calls == []
    assert drafter._rows == {}


def test_sampled_row_initialization_cannot_hide_its_parent_decode_batch():
    model = _NativeProposalFixture()
    drafter = DSparkDrafter(model, block_size=3, source_path="fixture")
    row = SimpleNamespace(uids=[7])
    with (
        drafter.decode_scope([7, 8]),
        patch("omlx.speculative.dspark_drafter.bg._is_greedy", return_value=False),
        pytest.raises(ValueError, match="Batched sampled"),
    ):
        drafter.draft([(row, _state(7), _captures(1), mx.array([42]), None)])
    assert model.calls == []
    assert drafter._rows == {}


@pytest.mark.parametrize("sampled_row", [0, 1, None])
def test_decode_batch_admission_rejects_sampling_before_target_realign(sampled_row):
    from mlx_lm.generate import GenerationBatch

    from omlx.speculative.dspark_drafter import bg

    assert bg.apply()
    greedy = make_sampler(temp=0.0)
    sampled = make_sampler(temp=0.7)
    samplers = [greedy, greedy]
    if sampled_row is not None:
        samplers[sampled_row] = sampled
    else:
        samplers[1] = None
    realign = MagicMock()
    batch = SimpleNamespace(
        model=SimpleNamespace(
            _omlx_drafter=DSparkDrafter(
                _NativeProposalFixture(), block_size=3, source_path="fixture"
            )
        ),
        uids=[7, 8],
        samplers=samplers,
        fallback_sampler=sampled,
        _omlx_realign_rows=realign,
    )
    with pytest.raises(ValueError, match="Batched sampled"):
        GenerationBatch.next(batch)
    realign.assert_not_called()
    assert batch.model._omlx_drafter._rows == {}


def test_release_before_uid_binding_and_nested_scope_cleanup():
    drafter = DSparkDrafter(
        _NativeProposalFixture(), block_size=3, source_path="fixture"
    )
    drafter.seed_request("cancelled", _captures(2))
    drafter.release_request("cancelled")
    drafter.release_request("cancelled")
    drafter.bind_uid("cancelled", 7)
    assert drafter._rows == {}
    with drafter.decode_scope([7]):
        assert drafter.scope_uids == (7,)
        with pytest.raises(RuntimeError, match="cancelled"), drafter.decode_scope([8]):
            raise RuntimeError("cancelled")
        assert drafter.scope_uids == (7,)
    assert drafter.scope_uids is None
    drafter.seed(7, _captures(2))
    drafter.seed_request("unbound", _captures(3))
    drafter.clear()
    drafter.clear()
    assert drafter._rows == {}
    assert drafter._request_seeds == {}


@pytest.mark.parametrize(
    "override",
    [
        {"dflash_draft_model": None},
        {"dflash_verify_mode": "adaptive"},
        {"dflash_max_ctx": 1024},
        {"dflash_draft_window_size": 2048},
        {"dflash_draft_sink_size": 4},
        {"dflash_ssd_cache": True},
        {"mtp_adaptive_max_depth": 4},
        {"mtp_fixed_depth": 4},
        {"specprefill_enabled": True},
        {"guided_grammar_enabled": True},
        {"mtp_enabled": True},
        {"vlm_mtp_enabled": True},
    ],
)
def test_enabled_native_path_rejects_unsupported_settings(override):
    settings = {"dflash_enabled": True, "dflash_draft_model": "/paired/helper"}
    settings.update(override)
    with pytest.raises(ValueError):
        validate_minicpm_speculation(settings, "minicpmv4_7")
    validate_minicpm_speculation(settings, "qwen3_5")


def test_fixed_defaults_and_ordinary_base_serving_are_unchanged():
    validate_minicpm_speculation({}, "minicpmv4_7")
    validate_minicpm_speculation(
        {"dflash_enabled": True, "dflash_draft_model": "/paired/helper"},
        "minicpmv4_7",
    )


def test_incomplete_native_contract_fails_before_checkpoint_load(tmp_path, monkeypatch):
    from omlx.patches import mlx_vlm_minicpm47_compat as compatibility
    from omlx.speculative import dflash

    monkeypatch.setattr(compatibility, "dspark_model_class", lambda: object)
    target = SimpleNamespace(config=SimpleNamespace(model_type="minicpmv4_7"))
    with (
        patch.object(compatibility, "load_dspark_model") as load,
        pytest.raises(ValueError, match="proposal support is incomplete"),
    ):
        dflash.load_dflash_drafter(str(tmp_path), target)
    load.assert_not_called()


def test_native_helper_attachment_uses_fixed_depth_and_detach_releases_state():
    from omlx.speculative.dflash import attach_drafter, detach_drafter

    drafter = DSparkDrafter(
        _NativeProposalFixture(), block_size=3, source_path="fixture"
    )
    language = SimpleNamespace()
    attach_drafter(language, drafter)
    assert language._omlx_mtp_depth == 2
    assert language._omlx_mtp_depth_fixed
    drafter.seed_request("request", _captures(2))
    assert detach_drafter(language) is drafter
    assert not language._omlx_mtp_decode_enabled
    assert not language._omlx_mtp_depth_fixed
    assert drafter._request_seeds == {}


def test_dspark_prefill_bypasses_kv_prefix_without_hidden_history():
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler

    drafter = DSparkDrafter(
        _NativeProposalFixture(), block_size=3, source_path="fixture"
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = SimpleNamespace(_omlx_drafter=drafter)
    scheduler.block_aware_cache = MagicMock()
    scheduler._prefix_cache_prepared = set()
    scheduler._model_has_unreconstructible_cache = lambda: False
    scheduler._log_prefix_divergence = MagicMock()
    scheduler._try_specprefill_scoring = MagicMock()
    request = Request(
        request_id="request", prompt=[11, 12, 13], sampling_params=SamplingParams()
    )
    request.prompt_token_ids = [11, 12, 13]
    scheduler._prepare_prefix_cache_for_request(request)
    assert request.remaining_tokens == [11, 12, 13]
    assert request.cached_tokens == 0
    scheduler.block_aware_cache.fetch_cache.assert_not_called()


def test_sampled_batch_seed_cannot_bypass_gate_with_single_job():
    model = _NativeProposalFixture()
    drafter = DSparkDrafter(model, block_size=3, source_path="fixture")
    with (
        patch("omlx.speculative.dspark_drafter.bg._is_greedy", return_value=False),
        pytest.raises(ValueError, match="Batched sampled"),
    ):
        drafter.draft(
            [
                (
                    SimpleNamespace(uids=[7, 8]),
                    _state(7),
                    _captures(1),
                    mx.array([42]),
                    None,
                )
            ]
        )
    assert model.calls == []
