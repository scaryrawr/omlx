# SPDX-License-Identifier: Apache-2.0
"""MiniCPM request-local media controls and admission plumbing."""

from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from mlx_vlm.models.base import LanguageModelOutput
from PIL import Image

from omlx.api.media_options import MultimodalProcessorOptions
from omlx.engine.base import GenerationOutput
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.model_settings import ModelSettings
from omlx.models.vlm import VLMModelAdapter


@pytest.mark.asyncio
async def test_minicpm_start_propagates_unknown_runtime_guard_before_tensor_load():
    engine = VLMBatchedEngine(model_name="minicpm-fixture")
    with (
        patch(
            "omlx.engine.vlm._read_config_model_type", return_value="minicpmv4_7"
        ),
        patch(
            "omlx.utils.model_loading.maybe_apply_pre_load_patches",
            side_effect=RuntimeError("Unrecognized or partial mlx-vlm"),
        ),
        patch("mlx_vlm.utils.load") as loader,
    ):
        with pytest.raises(RuntimeError, match="Unrecognized or partial"):
            await engine.start()
        loader.assert_not_called()


@pytest.fixture(autouse=True)
def cpu_arrays():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def engine():
    model = VLMBatchedEngine(
        model_name="tiny-minicpm",
        model_settings=ModelSettings(mm_processor_kwargs={"max_slice_nums": 9}),
    )
    model._loaded = True
    model._processor = SimpleNamespace(
        image_token="<|image_pad|>",
        image_processor=SimpleNamespace(max_slice_nums=9),
        apply_chat_template=lambda *args, **kwargs: "<|image_pad|>Describe.",
    )
    features = {
        "inputs_embeds": mx.zeros((1, 4, 2)),
        "position_ids": mx.ones((3, 1, 4), dtype=mx.int32),
        "rope_deltas": mx.array([[3]], dtype=mx.int32),
    }
    model._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type="minicpmv4_7", downsample_mode="16x"),
        language_model=SimpleNamespace(),
        get_input_embeddings=MagicMock(
            return_value=SimpleNamespace(
                inputs_embeds=features["inputs_embeds"],
                to_dict=lambda: dict(features),
            )
        ),
    )
    model._vision_cache = None
    model._format_messages_for_vlm_template = MagicMock(
        return_value=([{"role": "user", "content": "<|image_pad|>Describe."}], [])
    )
    return model


@pytest.fixture
def inputs():
    return {
        "input_ids": mx.array([[1, 99, 99, 2]]),
        "pixel_values": mx.zeros((1, 1)),
        "target_sizes": mx.array([[2, 2]]),
        "mm_token_type_ids": mx.array([[0, 1, 1, 0]]),
    }


def test_compression_reaches_processor_and_encoder_with_request_local_positions(
    engine, inputs
):
    images = [Image.new("RGB", (4, 4))]
    with patch("mlx_vlm.utils.prepare_inputs", return_value=inputs) as prepare:
        result = engine._prepare_vision_inputs(
            [{"role": "user", "content": "<|image_pad|>Describe."}],
            images,
            mm_processor_kwargs={"downsample_mode": "4x", "max_slice_nums": 36},
        )
    assert prepare.call_args.kwargs["downsample_mode"] == "4x"
    assert prepare.call_args.kwargs["max_slice_nums"] == 36
    kwargs = engine._vlm_model.get_input_embeddings.call_args.kwargs
    assert kwargs["downsample_mode"] == "4x"
    assert kwargs["target_sizes"] is inputs["target_sizes"]
    assert kwargs["mm_token_type_ids"] is inputs["mm_token_type_ids"]
    assert result[2]["position_ids"].shape == (3, 1, 4)
    assert result[2]["_captured_rope_deltas"].tolist() == [[3]]
    assert result[4] == 0 and result[5] == [(0, result[3])]
    assert engine._vlm_model.config.downsample_mode == "16x"
    assert engine._processor.image_processor.max_slice_nums == 9
    assert engine._model_settings.mm_processor_kwargs == {"max_slice_nums": 9}


def test_effective_processor_controls_change_prefix_identity(engine, inputs):
    images = [Image.new("RGB", (4, 4))]
    messages = [{"role": "user", "content": "<|image_pad|>Describe."}]
    with patch("mlx_vlm.utils.prepare_inputs", return_value=inputs):
        baseline = engine._prepare_vision_inputs(messages, images)
        compressed = engine._prepare_vision_inputs(
            messages, images, mm_processor_kwargs={"downsample_mode": "4x"}
        )
        sliced = engine._prepare_vision_inputs(
            messages, images, mm_processor_kwargs={"max_slice_nums": 36}
        )
        repeated = engine._prepare_vision_inputs(messages, images)
    assert len({baseline[3], compressed[3], sliced[3]}) == 3
    assert repeated[3] == baseline[3]


def test_unset_slicing_uses_native_image_and_video_defaults(engine):
    engine._model_settings = ModelSettings()
    assert engine._mm_processor_kwargs(None) == {"downsample_mode": "16x"}
    assert engine._mm_processor_kwargs({"max_slice_nums": 1}) == {
        "downsample_mode": "16x",
        "max_slice_nums": 1,
    }


def test_pure_text_initializes_native_request_delta_for_cached_decode(engine):
    engine._format_messages_for_vlm_template.return_value = (
        [{"role": "user", "content": "Describe."}],
        [],
    )
    engine._processor.apply_chat_template = lambda *args, **kwargs: "Describe."
    positions = mx.broadcast_to(mx.arange(3)[None, None, :], (3, 1, 3))
    features = {
        "inputs_embeds": mx.zeros((1, 3, 2)),
        "position_ids": positions,
        "rope_deltas": mx.zeros((1, 1), dtype=mx.int32),
    }
    engine._vlm_model.get_input_embeddings.return_value = SimpleNamespace(
        inputs_embeds=features["inputs_embeds"], to_dict=lambda: dict(features)
    )
    inputs = {
        "input_ids": mx.array([[1, 2, 3]]),
        "mm_token_type_ids": mx.zeros((1, 3), dtype=mx.int32),
    }
    with patch("mlx_vlm.utils.prepare_inputs", return_value=inputs):
        result = engine._prepare_vision_inputs(
            [{"role": "user", "content": "Describe."}], []
        )
    assert result[0] == [1, 2, 3]
    assert result[1] is features["inputs_embeds"]
    assert result[2]["_captured_rope_deltas"].tolist() == [[0]]
    assert result[2]["position_ids"].tolist() == positions.tolist()
    assert result[3:] == (None, 0, [])
    assert engine._vlm_model.get_input_embeddings.call_args.args[1] is None


@pytest.mark.parametrize("model_type", ["minicpmv4_7", "qwen3_5_moe"])
@pytest.mark.parametrize("seq_len", [1, 3])
def test_decode_and_verify_forward_request_deltas_without_changing_qwen_abi(
    model_type, seq_len
):
    calls = []

    def language(input_ids, **kwargs):
        calls.append(kwargs)
        if model_type == "minicpmv4_7":
            assert kwargs["rope_deltas"].shape == (2, 1)
        return SimpleNamespace(logits=mx.zeros((2, seq_len, 8)))

    model = SimpleNamespace(
        language_model=language,
        config=SimpleNamespace(
            model_type=model_type,
            text_config=SimpleNamespace(
                rope_parameters={"mrope_section": [11, 11, 10]}
            ),
        ),
    )
    adapter = VLMModelAdapter(model)
    adapter.set_batch_rope_deltas(mx.array([0, -5]))
    cache = [SimpleNamespace(offset=mx.array([10, 20]))]
    adapter(mx.ones((2, seq_len), dtype=mx.int32), cache=cache)
    if model_type == "minicpmv4_7":
        assert calls[0]["rope_deltas"].tolist() == [[0], [-5]]
    else:
        assert "rope_deltas" not in calls[0]
    assert calls[0]["position_ids"][0].tolist() == [
        list(range(10, 10 + seq_len)),
        list(range(15, 15 + seq_len)),
    ]


@pytest.mark.parametrize("return_hidden", [False, True])
def test_native_captures_preserve_caller_order_and_pre_norm_final(return_hidden):
    calls = []
    transaction = object()

    def language(input_ids, **kwargs):
        calls.append(kwargs)
        assert kwargs["return_hidden"] is False
        return LanguageModelOutput(
            logits=mx.zeros((1, 2, 8)),
            hidden_states=[
                mx.full((1, 2, 4), layer) for layer in kwargs["capture_layer_ids"]
            ],
            gdn_states=transaction,
        )

    language.model = SimpleNamespace(layers=[object() for _ in range(4)])
    adapter = VLMModelAdapter(
        SimpleNamespace(
            language_model=language,
            config=SimpleNamespace(model_type="minicpmv4_7"),
        )
    )
    result = adapter(
        mx.array([[1, 2]]),
        capture_layer_ids=[2, 0],
        return_hidden=return_hidden,
    )
    assert calls[0]["capture_layer_ids"] == ([0, 2, 3] if return_hidden else [0, 2])
    assert [value[0, 0, 0].item() for value in result.hidden_states] == (
        [2, 0, 3] if return_hidden else [2, 0]
    )
    assert result.gdn_states is transaction


def test_minicpm_backbone_uses_capture_bearing_native_exact_verifier():
    from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone_captured

    calls = []
    transaction = object()

    def language(input_ids, **kwargs):
        calls.append(kwargs)
        assert kwargs["speculative_verify"] is True
        assert kwargs["rope_deltas"].tolist() == [[-3]]
        return LanguageModelOutput(
            logits=mx.zeros((1, 3, 8)),
            hidden_states=[
                mx.full((1, 3, 4), layer) for layer in kwargs["capture_layer_ids"]
            ],
            gdn_states=transaction,
        )

    language.model = SimpleNamespace(layers=[object() for _ in range(4)])
    language.speculative_verify_logits = MagicMock(
        side_effect=AssertionError("Convenience verifier loses the capture ABI")
    )
    adapter = VLMModelAdapter(
        SimpleNamespace(
            language_model=language,
            config=SimpleNamespace(model_type="minicpmv4_7"),
        )
    )
    adapter.set_batch_rope_deltas(mx.array([-3]))
    logits, hidden, state, captures = _call_backbone_captured(
        adapter,
        mx.array([[1, 2, 3]]),
        [SimpleNamespace(offset=4)],
        n_confirmed=1,
        capture_layer_ids=[2, 0],
    )
    assert hidden[0, 0, 0].item() == 3
    assert [value[0, 0, 0].item() for value in captures] == [2, 0]
    assert state is transaction
    assert logits.shape == (1, 3, 8)
    language.speculative_verify_logits.assert_not_called()


def test_token_count_runs_exact_processor_without_vision_forward(engine, inputs):
    engine._extract_request_media = MagicMock(
        return_value=(
            [{"role": "user", "content": "<|image_pad|>Describe."}],
            [{"role": "user", "content": "Describe."}],
            [Image.new("RGB", (4, 4))],
            [],
            [],
        )
    )
    with patch("mlx_vlm.utils.prepare_inputs", return_value=inputs) as prepare:
        assert (
            engine.count_chat_tokens(
                [{"role": "user", "content": "Describe."}],
                mm_processor_kwargs={"downsample_mode": "4x"},
            )
            == 4
        )
    assert prepare.call_args.kwargs["downsample_mode"] == "4x"
    engine._vlm_model.get_input_embeddings.assert_not_called()


@pytest.mark.asyncio
async def test_preflight_uses_exact_processor_count_on_executor(engine):
    scheduler = object()
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine._engine = SimpleNamespace(
            _mlx_executor=executor, engine=SimpleNamespace(scheduler=scheduler)
        )
        engine.count_chat_tokens = MagicMock(return_value=128)
        engine._preflight_or_raise_with_eviction = AsyncMock()
        await engine.preflight_chat(
            [{"role": "user", "content": "Describe."}],
            request_id="tiny",
            mm_processor_kwargs={"downsample_mode": "4x"},
        )
    assert engine.count_chat_tokens.call_args.kwargs["mm_processor_kwargs"] == {
        "downsample_mode": "4x"
    }
    engine._preflight_or_raise_with_eviction.assert_awaited_once_with(
        scheduler, num_prompt_tokens=128, request_id="tiny"
    )


@pytest.mark.asyncio
async def test_dspark_preflight_rejects_request_local_sparse_prefill(engine):
    engine._dflash_drafter = SimpleNamespace(kind="dspark")
    with pytest.raises(InvalidRequestError, match="full target context"):
        await engine.preflight_chat([], specprefill=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,match",
    [
        ({"xtc_probability": 0.5, "temperature": 0.7}, "stochastic XTC"),
        ({"compiled_grammar": object()}, "constrained grammar"),
    ],
)
async def test_dspark_unsupported_sampling_fails_before_admission(
    engine, options, match
):
    engine._dflash_drafter = SimpleNamespace(kind="dspark")
    with pytest.raises(InvalidRequestError, match=match):
        await engine.preflight_chat([], **options)
    with pytest.raises(InvalidRequestError, match=match):
        await engine.generate([1], **options)
    with pytest.raises(InvalidRequestError, match=match):
        await anext(engine.stream_generate([1], **options))


def test_dspark_xtc_is_inactive_for_greedy_and_does_not_gate_base_serving(engine):
    engine._dflash_drafter = SimpleNamespace(kind="dspark")
    engine._validate_dspark_request(xtc_probability=0.5, temperature=0.0)
    engine._dflash_drafter = None
    engine._validate_dspark_request(
        xtc_probability=0.5, temperature=0.7, compiled_grammar=object()
    )


def test_audio_is_rejected_before_processor_or_encoder(engine):
    with (
        patch("mlx_vlm.utils.prepare_inputs") as prepare,
        pytest.raises(InvalidRequestError, match="not audio"),
    ):
        engine._prepare_vision_inputs([], [], audio=[object()])
    prepare.assert_not_called()
    engine._vlm_model.get_input_embeddings.assert_not_called()


@pytest.mark.asyncio
async def test_protocol_merge_rejects_non_minicpm_and_preserves_request_precedence():
    from omlx import server

    settings = ModelSettings(
        mm_processor_kwargs={"max_slice_nums": 9, "downsample_mode": "16x"}
    )
    request = SimpleNamespace(
        mm_processor_kwargs=MultimodalProcessorOptions(downsample_mode="4x")
    )
    with pytest.raises(InvalidRequestError, match="only supported"):
        server._request_mm_processor_kwargs(SimpleNamespace(), request, settings)
    model = SimpleNamespace(model_type="minicpmv4_7")
    options = server._request_mm_processor_kwargs(model, request, settings)
    assert options == {"max_slice_nums": 9, "downsample_mode": "4x"}
    model.count_chat_tokens = MagicMock(return_value=4)
    count_threads = []
    model.count_chat_tokens.side_effect = lambda *a, **k: (
        count_threads.append(get_ident()) or 4
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        model._engine = SimpleNamespace(_mlx_executor=executor)
        assert await server._prepare_chat_prompt_for_request(
            model, [], mm_processor_kwargs=options
        ) == (4, None)
    assert count_threads[0] != get_ident()
    assert model.count_chat_tokens.call_args.kwargs["mm_processor_kwargs"] == options


@pytest.mark.asyncio
async def test_tokenization_avoids_encoder_and_gpu_cleanup(engine, inputs):
    engine._extract_request_media = MagicMock(
        return_value=(
            [{"role": "user", "content": [{"type": "image"}]}],
            [{"role": "user", "content": ""}],
            [Image.new("RGB", (4, 4))],
            [],
            [],
        )
    )
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        patch("mlx_vlm.utils.prepare_inputs", return_value=inputs),
        patch("omlx.engine.vlm.mx.synchronize") as synchronize,
        patch("omlx.engine.vlm.mx.clear_cache") as clear_cache,
    ):
        engine._engine = SimpleNamespace(_mlx_executor=executor)
        assert await engine.tokenize_chat([]) == [1, 99, 99, 2]
    engine._vlm_model.get_input_embeddings.assert_not_called()
    synchronize.assert_not_called()
    clear_cache.assert_not_called()


@pytest.fixture
def protocol_engine(monkeypatch):
    from omlx import server

    with ThreadPoolExecutor(max_workers=1) as executor:
        model = MagicMock()
        model.model_type = "minicpmv4_7"
        model.tokenizer = None
        model.is_diffusion_model = False
        model.supports_request_scoped_abort = False
        model._engine = SimpleNamespace(_mlx_executor=executor)
        model.start = AsyncMock()
        model.preflight_chat = AsyncMock()
        model.count_chat_tokens.return_value = 4
        output = GenerationOutput(
            text="Done.", new_text="Done.", prompt_tokens=4, completion_tokens=1
        )
        model.chat = AsyncMock(return_value=output)
        model.streamed_kwargs = []

        async def stream_chat(**kwargs):
            model.streamed_kwargs.append(kwargs)
            yield output

        model.stream_chat = stream_chat
        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            config_model_type="minicpmv4_7", preserve_thinking_default=True
        )
        state = server.ServerState()
        state.engine_pool = pool
        monkeypatch.setattr(server, "_server_state", state)
        monkeypatch.setattr(
            server, "get_engine_for_model", AsyncMock(return_value=model)
        )
        monkeypatch.setattr(server, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(server, "validate_context_window", lambda *a, **k: None)
        monkeypatch.setattr(
            server,
            "get_model_settings_for_request",
            lambda name: ModelSettings(
                mm_processor_kwargs={"downsample_mode": "16x", "max_slice_nums": 9}
            ),
        )
        monkeypatch.setitem(
            server.app.dependency_overrides,
            server.verify_inference_api_key,
            lambda: True,
        )
        yield model


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/messages", "/v1/responses"]
)
def test_protocols_forward_controls_to_count_admission_and_generation(
    protocol_engine, path, stream
):
    from omlx import server

    body = {
        "model": "tiny-minicpm",
        "stream": stream,
        "mm_processor_kwargs": {"downsample_mode": "4x"},
    }
    if path == "/v1/responses":
        body.update(input="Describe.", store=False)
    else:
        body.update(messages=[{"role": "user", "content": "Describe."}], max_tokens=16)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    expected = {"downsample_mode": "4x", "max_slice_nums": 9}
    assert (
        protocol_engine.count_chat_tokens.call_args.kwargs["mm_processor_kwargs"]
        == expected
    )
    assert (
        protocol_engine.preflight_chat.call_args.kwargs["mm_processor_kwargs"]
        == expected
    )
    generated = (
        protocol_engine.streamed_kwargs[0]
        if stream
        else protocol_engine.chat.call_args.kwargs
    )
    assert generated["mm_processor_kwargs"] == expected


def test_anthropic_count_endpoint_uses_same_controls(protocol_engine):
    from omlx import server

    with TestClient(server.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "tiny-minicpm",
                "messages": [{"role": "user", "content": "Describe."}],
                "mm_processor_kwargs": {"downsample_mode": "4x"},
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] == 4
    assert protocol_engine.count_chat_tokens.call_args.kwargs[
        "mm_processor_kwargs"
    ] == {
        "downsample_mode": "4x",
        "max_slice_nums": 9,
    }


def test_formatting_preserves_mixed_visual_order_and_tool_reasoning(engine):
    del engine._format_messages_for_vlm_template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Before"},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": "Between"},
                {"type": "video_url", "video_url": {"url": "ignored"}},
                {"type": "text", "text": "After"},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Think",
            "tool_calls": [],
        },
        {"role": "tool", "content": "Result", "tool_call_id": "call_1"},
    ]
    formatted, ranges = engine._format_messages_for_vlm_template(
        messages, num_images=1, num_videos=1
    )
    assert [part["type"] for part in formatted[0]["content"]] == [
        "text",
        "image",
        "text",
        "video",
        "text",
    ]
    assert formatted[1:] == messages[1:]
    assert "image_url" in messages[0]["content"][1]
    assert ranges == []


def test_system_visuals_are_rejected_without_formatter_fallback(engine):
    del engine._format_messages_for_vlm_template
    with pytest.raises(InvalidRequestError, match="system messages"):
        engine._prepare_vision_inputs(
            [{"role": "system", "content": [{"type": "image"}]}],
            [Image.new("RGB", (4, 4))],
        )


@pytest.mark.parametrize(
    "content, text_parts",
    [
        ("Question", [{"type": "text", "text": "Question"}]),
        (
            [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
            [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
        ),
    ],
)
def test_direct_engine_visuals_are_inserted_on_user_turn(engine, content, text_parts):
    del engine._format_messages_for_vlm_template
    formatted, _ = engine._format_messages_for_vlm_template(
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": content},
        ],
        num_images=2,
    )
    assert formatted[0]["content"] == "System"
    assert formatted[1]["content"] == [
        {"type": "image"},
        {"type": "image"},
        *text_parts,
    ]
