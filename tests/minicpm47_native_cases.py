# SPDX-License-Identifier: Apache-2.0
"""Real native/oMLX contracts, collected only by the isolated CPU runner."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.minicpmv4_7 import Model, ModelConfig
from mlx_vlm.tests.test_minicpmv4_7 import hf_weight_fixture, tiny_config
from mlx_vlm.tests.test_minicpmv4_7_processor import Tokenizer
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template

from omlx.engine.vlm import VLMBatchedEngine
from omlx.models.vlm import VLMModelAdapter
from omlx.patches import mlx_vlm_minicpm47_compat as compat
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.scheduler import _bind_text_prefill_rope_delta
from omlx.speculative.dflash import attach_drafter, load_dflash_drafter
from omlx.speculative.dspark_drafter import DSparkDrafter
from omlx.speculative.minicpm_pairing import HELPER, TARGET
from omlx.utils.sampling import make_sampler


def model():
    target = Model(ModelConfig.from_dict(tiny_config()))
    target.load_weights(list(target.sanitize(hf_weight_fixture()).items()), strict=True)
    return target


def test_real_native_parameter_order_keeps_fused_expert_mates_adjacent():
    names = [name for name, _ in tree_flatten(model().parameters())]
    pairs = 0
    for index, name in enumerate(names):
        if name.endswith(".switch_mlp.gate_proj.weight"):
            assert names[index + 1] == name.replace(
                ".gate_proj.weight", ".up_proj.weight"
            )
            pairs += 1
    assert pairs == 4


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pytest.mark.parametrize("accepted", [0, 1, 2])
@pytest.mark.parametrize("rotating", [False, True])
def test_real_omlx_capture_verifier_and_commit(mode, accepted, rotating):
    target = model()
    adapter = VLMModelAdapter(target)
    count = 4 if mode == "4x" else 1
    ids = mx.array([[3, 10] + [8] * count + [11, 4]])
    features = target.get_input_embeddings(
        ids,
        mx.arange(192, dtype=mx.float32).reshape(1, 3, 2, 32) / 192,
        target_sizes=mx.array([[4, 4]]),
        downsample_mode=mode,
        mm_token_type_ids=(ids == 8).astype(mx.int32),
    )
    adapter.set_batch_rope_deltas(features.rope_deltas.reshape(-1))

    def prime():
        cache = target.make_cache()
        if rotating:
            cache = [
                (
                    RotatingKVCache(max_size=4, keep=0)
                    if isinstance(entry, KVCache)
                    else entry
                )
                for entry in cache
            ]
        adapter(
            ids,
            cache=cache,
            inputs_embeds=features.inputs_embeds,
            vlm_extra_kwargs={
                "position_ids": features.position_ids,
                "rope_deltas": features.rope_deltas,
            },
        )
        return cache

    actual_cache, reference_cache = prime(), prime()
    block = mx.array([[5, 6, 7]])
    logits, hidden, transaction, captures = bg._call_backbone_captured(
        adapter, block, actual_cache, n_confirmed=1, capture_layer_ids=[2, 0]
    )
    assert transaction is not None
    expected = [
        target.language_model(
            block[:, index : index + 1],
            cache=reference_cache,
            rope_deltas=features.rope_deltas,
            capture_layer_ids=[0, 2, 3],
            return_hidden=False,
        )
        for index in range(3)
    ]
    np.testing.assert_allclose(
        np.array(logits),
        np.array(mx.concatenate([row.logits for row in expected], 1)),
        atol=3e-5,
    )
    for actual, native_index in zip([*captures, hidden], [1, 0, 2], strict=True):
        np.testing.assert_allclose(
            np.array(actual),
            np.array(
                mx.concatenate([row.hidden_states[native_index] for row in expected], 1)
            ),
            atol=3e-5,
        )
    adapter.rollback_speculative_cache(actual_cache, transaction, accepted, 3)
    replay_cache = prime()
    for index in range(accepted + 1):
        adapter(block[:, index : index + 1], cache=replay_cache)
    token = mx.array([[15]])
    actual = adapter(token, cache=actual_cache)
    replay = adapter(token, cache=replay_cache)
    np.testing.assert_allclose(np.array(actual), np.array(replay), atol=3e-5)


def test_real_raw_text_chunks_rebind_zero_delta_after_interleaved_media():
    target = model()
    adapter = VLMModelAdapter(target)
    ids = mx.array([[3, 4, 5, 6, 7]])
    cache = adapter.make_cache()
    first = []
    for start, end in ((0, 2), (2, 5)):
        adapter.set_batch_rope_deltas(mx.array([-17]))
        _bind_text_prefill_rope_delta(adapter, 0.0)
        first.append(adapter(ids[:, start:end], cache=cache))
    full = target.language_model(ids).logits
    np.testing.assert_allclose(
        np.array(mx.concatenate(first, 1)), np.array(full), atol=3e-5
    )
    adapter.set_batch_rope_deltas(None)
    with pytest.raises(ValueError, match="request-local rope_deltas"):
        adapter(mx.array([[15]]), cache=cache)


@pytest.mark.parametrize("rotating", [False, True])
def test_real_verify_abort_restores_gdn_and_kv_for_next_request(rotating):
    target = model()
    adapter = VLMModelAdapter(target)
    adapter.set_batch_rope_deltas(mx.array([0]))

    def prime():
        cache = adapter.make_cache()
        if rotating:
            cache = [
                (
                    RotatingKVCache(max_size=4, keep=0)
                    if isinstance(entry, KVCache)
                    else entry
                )
                for entry in cache
            ]
        adapter(mx.array([[2, 3, 4]]), cache=cache)
        return cache

    actual_cache, reference_cache = prime(), prime()
    _, _, transaction, _ = bg._call_backbone_captured(
        adapter,
        mx.array([[5, 6, 7]]),
        actual_cache,
        n_confirmed=1,
        capture_layer_ids=[0, 2],
    )
    adapter.abort_speculative_forward()
    adapter.abort_speculative_forward()
    assert not transaction._active
    np.testing.assert_allclose(
        np.array(adapter(mx.array([[15]]), cache=actual_cache)),
        np.array(adapter(mx.array([[15]]), cache=reference_cache)),
        atol=3e-5,
    )


@pytest.mark.parametrize("mode", ["4x", "16x"])
def test_actual_omlx_jinja_processor_encoder_and_tool_parser(mode):
    from mlx_vlm.models.minicpmv4_7 import Processor
    from mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7 import (
        MiniCPMV4_7ImageProcessor,
    )

    template = (
        compat._VENDOR / "mlx_vlm/tests/fixtures/minicpmv4_7_chat_template.jinja"
    ).read_text()

    class OfficialTokenizer(Tokenizer):
        tokens = {
            **Tokenizer.tokens,
            "<|image_pad|>": 8,
            "<|video_pad|>": 9,
            "<image>": 10,
            "</image>": 11,
            "<slice>": 12,
            "</slice>": 13,
            "\n": 2,
            "<tool_call>": 22,
            "</tool_call>": 23,
        }
        chat_template = template

        def get_vocab(self):
            return self.tokens

        def apply_chat_template(self, messages, **kwargs):
            kwargs.pop("tokenize", None)
            kwargs.pop("chat_template", None)
            return render_jinja_template(
                conversations=[messages], chat_template=template, **kwargs
            )[0][0]

    config = tiny_config()
    config["text_config"]["vocab_size"] = 512
    target = Model(ModelConfig.from_dict(config))
    engine = VLMBatchedEngine(model_name="fixture")
    engine._vlm_model = target
    engine._model_type = "minicpmv4_7"
    engine._processor = Processor(
        image_processor=MiniCPMV4_7ImageProcessor(scale_resolution=8, patch_size=2),
        tokenizer=OfficialTokenizer(),
    )
    engine._vision_cache = None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "compare"},
                {"type": "image"},
            ],
        }
    ]
    prepared = engine._prepare_vision_inputs(
        messages,
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        mm_processor_kwargs={"downsample_mode": mode, "max_slice_nums": 1},
    )
    ids, embeddings, extra = prepared[:3]
    assert embeddings.shape == (1, len(ids), 16)
    assert extra["position_ids"].shape == (3, 1, len(ids))
    assert extra["_captured_rope_deltas"].shape == (1, 1)
    assert ids.count(8) == (8 if mode == "4x" else 2)
    tokenizer = engine._processor.tokenizer
    engine._inject_tool_calling(tokenizer)
    assert tokenizer.has_tool_calling
    call = tokenizer.tool_parser(
        "<function=lookup>\n<parameter=query>\nvalue\n</parameter>\n</function>",
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
    )
    assert call["name"] == "lookup"
    assert call["arguments"] == {"query": "value"}


def helper_config():
    return {
        "model_type": "qwen3",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 32,
        "target_model_type": "minicpmv4_7",
        "dflash_config": {
            "projector_type": "dspark",
            "block_size": 3,
            "block_size_policy": "fixed",
            "mask_token_id": 31,
            "target_layer_ids": [0, 1],
            "num_target_layers": 4,
            "markov_rank": 4,
            "enable_confidence_head": True,
        },
    }


def generation_batch(temperature):
    from mlx_lm.generate import BatchGenerator

    target = model()
    adapter = VLMModelAdapter(target)
    adapter.set_batch_rope_deltas(mx.array([0]))
    native = compat._helper_module()
    helper = native.Model(native.ModelConfig.from_dict(helper_config()))
    helper.bind(target)
    drafter = DSparkDrafter(helper, block_size=3, source_path="tiny-fixture")
    attach_drafter(target.language_model, drafter)
    bg.apply()
    sampler = make_sampler(temp=temperature, top_k=2 if temperature else 0)
    cache = adapter.make_cache()
    prefill = adapter(
        mx.array([[2, 3, 4]]), cache=cache, capture_layer_ids=[0, 1]
    )
    drafter.seed_request("fixture", prefill.hidden_states)
    generator = BatchGenerator(
        adapter,
        max_tokens=7,
        sampler=sampler,
        stream=mx.new_stream(mx.cpu),
    )
    uid = generator.insert([[5]], caches=[cache], all_tokens=[[2, 3, 4, 5]])[0]
    drafter.bind_uid("fixture", uid)
    adapter.register_rope_delta(uid, 0.0)
    adapter.set_native_mtp_request_eligible(uid, False)
    return target, adapter, helper, drafter, generator, uid


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_real_omlx_generation_batch_with_native_dspark(temperature):
    target, adapter, helper, drafter, generator, uid = generation_batch(temperature)
    responses = []
    with (
        patch.object(helper, "propose_block", wraps=helper.propose_block) as propose,
        patch.object(
            adapter,
            "rollback_speculative_cache",
            wraps=adapter.rollback_speculative_cache,
        ) as commit,
    ):
        for _ in range(12):
            _, outputs = generator.next()
            responses.extend(outputs)
            if any(output.finish_reason is not None for output in outputs):
                break
        assert propose.call_count > 0
        assert commit.call_count > 0
    assert len(responses) == 7
    assert all(output.uid == uid for output in responses)
    assert responses[-1].finish_reason == "length"
    assert all(output.finish_reason is None for output in responses[:-1])
    assert not adapter._minicpm_speculative_transactions
    if temperature == 0:
        reference_cache = target.make_cache()
        tokens = mx.array([[2, 3, 4, 5]])
        reference = []
        for _ in range(7):
            logits = target.language_model(
                tokens, cache=reference_cache, rope_deltas=mx.array([[0]])
            ).logits
            token = int(mx.argmax(logits[:, -1]).item())
            reference.append(token)
            tokens = mx.array([[token]])
        assert [output.token for output in responses] == reference
    generator.close()
    assert not drafter._rows


def test_real_generation_sampler_failure_aborts_verify_and_cancellation_releases():
    _, adapter, _, drafter, generator, uid = generation_batch(0.7)
    snapshots = []
    original = bg._call_backbone_captured

    def snapshot_forward(model, inputs, cache, **kwargs):
        if kwargs.get("n_confirmed") == 1 and inputs.shape[1] > 1:
            snapshots.append(
                (
                    cache,
                    [
                        [
                            None if value is None else np.array(value)
                            for value in entry.state
                        ]
                        for entry in cache
                    ],
                )
            )
        return original(model, inputs, cache, **kwargs)

    with (
        patch.object(bg, "_call_backbone_captured", side_effect=snapshot_forward),
        patch.object(
            bg, "_stochastic_verify_tokens", side_effect=RuntimeError("fixture sampler")
        ),
        pytest.raises(RuntimeError, match="fixture sampler"),
    ):
        for _ in range(12):
            generator.next()
    assert not adapter._minicpm_speculative_transactions
    assert len(snapshots) == 1
    cache, before = snapshots[0]
    for entry, expected in zip(cache, before, strict=True):
        for value, reference in zip(entry.state, expected, strict=True):
            if reference is None:
                assert value is None
            else:
                np.testing.assert_array_equal(np.array(value), reference)
    generator.remove([uid])
    assert uid not in drafter._rows
    assert not adapter._minicpm_speculative_transactions
    generator.close()


def test_real_omlx_dspark_greedy_batch_with_different_full_histories():
    target, adapter, helper, drafter, generator, first_uid = generation_batch(0.0)
    second_cache = adapter.make_cache()
    prefill = adapter(
        mx.array([[2, 3, 4, 5, 6]]),
        cache=second_cache,
        capture_layer_ids=[0, 1],
    )
    drafter.seed_request("second", prefill.hidden_states)
    second_uid = generator.insert(
        [[7]], caches=[second_cache], all_tokens=[[2, 3, 4, 5, 6, 7]]
    )[0]
    drafter.bind_uid("second", second_uid)
    adapter.register_rope_delta(second_uid, 0.0)
    adapter.set_native_mtp_request_eligible(second_uid, False)
    adapter.set_batch_rope_deltas(mx.array([0, 0]))
    responses = {first_uid: [], second_uid: []}
    with (
        patch.object(helper, "propose_block", wraps=helper.propose_block) as propose,
        patch.object(drafter, "draft", wraps=drafter.draft) as draft,
    ):
        for _ in range(20):
            _, outputs = generator.next()
            for output in outputs:
                responses[output.uid].append(output)
            if all(row and row[-1].finish_reason is not None for row in responses.values()):
                break
        assert propose.call_count > 1
        assert any(len(call.args[0]) == 2 for call in draft.call_args_list)
    for uid, prompt in (
        (first_uid, [2, 3, 4, 5]),
        (second_uid, [2, 3, 4, 5, 6, 7]),
    ):
        cache = target.make_cache()
        tokens = mx.array([prompt])
        expected = []
        for _ in range(7):
            logits = target.language_model(
                tokens, cache=cache, rope_deltas=mx.array([[0]])
            ).logits
            token = int(mx.argmax(logits[:, -1]).item())
            expected.append(token)
            tokens = mx.array([[token]])
        assert [output.token for output in responses[uid]] == expected
        assert responses[uid][-1].finish_reason == "length"
    assert not adapter._minicpm_speculative_transactions
    generator.close()
    assert not drafter._rows


def test_real_helper_strict_load_without_tokenizer_and_native_q_handoff(tmp_path):
    config = helper_config()
    native_module = compat._helper_module()
    helper = native_module.Model(native_module.ModelConfig.from_dict(config))
    config["omlx_source"] = {"repo_id": HELPER[0], "revision": HELPER[1]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        dict(tree_flatten(helper.parameters())),
    )
    target = model()
    engine = VLMBatchedEngine(model_name="tiny-minicpm")
    engine._vlm_model = target
    with patch("omlx.engine.vlm.drafter_target.apply") as install:
        assert engine.prepare_dflash_target() is target
    install.assert_not_called()
    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "config.json").write_text(
        json.dumps({"omlx_source": {"repo_id": TARGET[0], "revision": TARGET[1]}})
    )
    drafter = load_dflash_drafter(
        str(tmp_path), target, target_path=str(target_path), block_size=4
    )
    assert isinstance(drafter, DSparkDrafter)
    captures = target.language_model(
        mx.array([[2, 3, 4]]), capture_layer_ids=[0, 1]
    ).hidden_states
    state = SimpleNamespace(uid=7, stats=None)
    sampler = make_sampler(temp=0.7, top_k=2, top_p=1.0)
    with (
        patch.object(bg, "_is_greedy", return_value=False),
        patch.object(bg, "_resolve_draft_sampler", return_value=sampler),
    ):
        mx.random.seed(9)
        drafter.draft(
            [(SimpleNamespace(uids=[7]), state, captures, mx.array([4]), None)]
        )
    assert state.drafts.shape == (3,)
    assert len(state.draft_accept_lps) == 3
    assert drafter._rows[7].confidences.shape == (1, 3)
    for law in state.draft_accept_lps:
        assert float(mx.sum(mx.exp(law)).item()) == pytest.approx(1.0, abs=1e-5)
        assert int(mx.sum(mx.isfinite(law)).item()) == 2
    drafter.clear()
    assert drafter.context_length(7) == 0
