# SPDX-License-Identifier: Apache-2.0
"""Weight-free numerical and end-to-end checks of the Qwen image backport."""

from __future__ import annotations

import ast
import importlib
import json
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from omlx.api import image_routes
from omlx.engine.image import ImageEngine, _image_api
from omlx.patches import mlx_vlm_qwen_image_compat as compat

SIGMAS = [1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "class Model:\n    pass\n",
            "982ac2b3eb77fb75af373b2638159f4a698e3701e533a1c4d69bfe2f582f03f3",
        ),
        (
            "def load():\n    return 1\n",
            "577391320d95bead4088cf34bd51c4f9c18b81d1572ad1ae17b997a4a861e95b",
        ),
        (
            "async def load():\n    return 1\n",
            "cd3a2dc5966095d9d58947f47478e725957eb3c754cbe75254cb654307677f92",
        ),
    ],
)
def test_source_fingerprints_are_python_version_independent(
    source, expected, monkeypatch
):
    from omlx.patches.mlx_vlm_qwen_image_compat.generate_backport import source_hash

    assert source_hash(source) == expected
    tree = ast.parse(source)
    definition = tree.body[0]
    if "type_params" not in definition._fields:
        definition._fields = (*definition._fields, "type_params")
    definition.type_params = []
    monkeypatch.setattr(ast, "parse", lambda _: tree)
    assert source_hash(source) == expected
    definition.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert source_hash(source) != expected


@pytest.fixture(scope="module")
def qwen():
    compat.apply_qwen_image_compat_patch()
    return SimpleNamespace(
        **{
            name: importlib.import_module(f"mlx_vlm.models.qwen_image.{name}")
            for name in ("config", "scheduler", "pipeline", "model", "text_encoder")
        }
    )


@pytest.fixture
def checkpoint(tmp_path):
    configs = {
        "model_index.json": {
            "_class_name": "QwenImage21Pipeline",
            "sample_sigmas": SIGMAS,
        },
        "transformer/config.json": {"_class_name": "QwenImage21Transformer2DModel"},
        "vae/config.json": {
            "z_dim": 4,
            "latents_mean": [0] * 4,
            "latents_std": [1] * 4,
        },
        "scheduler/scheduler_config.json": {
            "use_dynamic_shifting": False,
            "shift": 1,
            "shift_terminal": None,
            "base_image_seq_len": 256,
            "max_image_seq_len": 8192,
            "base_shift": 0.5,
            "max_shift": 0.9,
            "num_train_timesteps": 1000,
        },
    }
    for relative, data in configs.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return tmp_path


def _tiny_pipeline(qwen, checkpoint, monkeypatch):
    calls, references = [], []
    encoder = SimpleNamespace(
        encode=lambda _: mx.zeros((1, 2, 8)),
        encode_edit=lambda _, refs: (
            references.append(len(refs)) or mx.zeros((1, 2, 8)),
            mx.ones((1, 2), dtype=mx.bool_),
        ),
        tokenizer=lambda _: {"input_ids": [1, 2]},
    )
    monkeypatch.setattr(qwen.pipeline, "QwenImageTextEncoder", lambda **_: encoder)

    def transformer(*, timestep, img_shape, **kwargs):
        calls.append((timestep, kwargs))
        return mx.ones((1, img_shape[1] * img_shape[2], 4), dtype=mx.bfloat16)

    transformer.causal_condition = False
    vae = SimpleNamespace(
        encode=lambda pixels: (
            mx.zeros((1, 4, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)),
            None,
        ),
        decode=lambda z: mx.zeros((1, 4, 1, z.shape[-2] * 16, z.shape[-1] * 16)),
    )
    pipeline = qwen.pipeline.QwenImagePipeline(
        variant=qwen.config.get_variant("qwen-image-2.1-turbo"),
        model_path=checkpoint,
        text_encoder=None,
        transformer=transformer,
        vae=vae,
    )
    return pipeline, calls, references


@pytest.mark.parametrize("seq_len", [256, 4096, 16384])
def test_backported_turbo_exact_grid_dtype_timesteps_and_euler(qwen, seq_len):
    scheduler = qwen.scheduler.FlowMatchEulerDiscreteScheduler(
        image_seq_len=seq_len,
        num_inference_steps=30,
        sigmas=SIGMAS,
        use_dynamic_shifting=False,
        shift=1,
        shift_terminal=None,
    )
    expected = np.array([*SIGMAS, 0], dtype=np.float32)
    np.testing.assert_array_equal(np.array(scheduler.sigmas), expected)
    np.testing.assert_array_equal(
        np.array(scheduler.timesteps), expected[:-1] * np.float32(1000)
    )
    assert scheduler.sigmas.dtype == mx.float32
    latent = mx.array([2.0])
    for step in range(8):
        latent = scheduler.step(noise=mx.array([1.0]), latents=latent, step_index=step)
    np.testing.assert_allclose(np.array(latent), [1], atol=1e-7)


def test_explicit_sigmas_are_transformed_and_legacy_dynamic_defaults_survive(qwen):
    cls = qwen.scheduler.FlowMatchEulerDiscreteScheduler
    grid = cls(
        image_seq_len=4096,
        num_inference_steps=30,
        sigmas=[1, 0.5, 0.2],
        use_dynamic_shifting=False,
        shift=2,
        shift_terminal=None,
    )
    np.testing.assert_allclose(np.array(grid.sigmas), [1, 2 / 3, 1 / 3, 0], atol=1e-7)
    grid = cls(image_seq_len=4096, num_inference_steps=4)
    source = np.linspace(1, 0.25, 4, dtype=np.float32)
    mu = qwen.scheduler.calculate_shift(4096)
    shifted = np.exp(mu) / (np.exp(mu) + (1 / source - 1))
    expected = 1 - (1 - shifted) / ((1 - shifted[-1]) / 0.98)
    np.testing.assert_allclose(np.array(grid.sigmas), [*expected, 0], atol=1e-7)


@pytest.mark.parametrize("task", ["generation", "edit"])
@pytest.mark.parametrize("override", [None, [1, 0.6, 0.2]])
def test_real_wrappers_and_sampler_report_effective_steps(
    qwen, checkpoint, monkeypatch, task, override
):
    pipeline, calls, refs = _tiny_pipeline(qwen, checkpoint, monkeypatch)
    api = _image_api()
    cls = (
        qwen.model.QwenImageGenerationModel
        if task == "generation"
        else qwen.model.QwenImageEditModel
    )
    instance = cls(pipeline, "tiny-qwen")
    assert instance.default_sampling.steps == 8
    request_fields = dict(
        prompt="a fox",
        steps=30,
        width=32,
        height=32,
        guidance=1,
        extra={"sigmas": override, "output_resolution": 256},
    )
    if task == "generation":
        result = instance.generate(api.ImageGenerationRequest(**request_fields))
    else:
        paths = []
        for index in range(2):
            path = checkpoint / f"reference-{index}.png"
            Image.new("RGBA", (32, 32), (index * 255, 0, 0, 128)).save(path)
            paths.append(str(path))
        result = instance.edit(
            api.ImageEditRequest(image_paths=paths, **request_fields)
        )
        assert refs == [2]
        assert all(len(kwargs["reference_image_shapes"]) == 2 for _, kwargs in calls)
    expected = SIGMAS if override is None else override
    assert result.steps == len(expected) == len(calls)
    assert result.array.shape == (32, 32, 3 if task == "generation" else 4)
    assert result.array.dtype == mx.uint8
    expected_t = (mx.array(expected) * 1000).astype(mx.bfloat16) / 1000
    np.testing.assert_array_equal(
        np.array(mx.concatenate([t for t, _ in calls]).astype(mx.float32)),
        np.array(expected_t.astype(mx.float32)),
    )
    calls.clear()
    pipeline.generate_array("again", width=32, height=32)
    assert len(calls) == 8
    assert pipeline.sample_sigmas == tuple(SIGMAS)


def test_qwen_engine_routes_execute_saved_recipe_without_weights(
    qwen, checkpoint, monkeypatch, tmp_path
):
    pipeline, calls, references = _tiny_pipeline(qwen, checkpoint, monkeypatch)
    api = _image_api()

    def load(reference, *, task):
        assert reference == str(checkpoint)
        cls = (
            qwen.model.QwenImageGenerationModel
            if task == "generate"
            else qwen.model.QwenImageEditModel
        )
        return cls(pipeline, reference)

    api = replace(api, load_image_model=load)
    monkeypatch.setattr("omlx.engine.image._image_api", lambda: api)
    monkeypatch.setenv("OMLX_IMAGE_TMPDIR", str(tmp_path / "inputs"))
    engine = ImageEngine(
        model_name="tiny-qwen",
        model_path=str(checkpoint),
        image_metadata={"backend": "mlx-vlm", "base_model": "qwen-image-2.1-turbo"},
        tasks=["generation", "edit"],
    )
    entry = SimpleNamespace(
        model_type="image", engine_type="image", tasks=["generation", "edit"]
    )

    class Pool:
        leases = 0

        def get_entry(self, model_id):
            return entry

        async def get_engine(self, model_id, *, _lease):
            assert _lease
            await engine.start()
            self.leases += 1
            return engine

        async def release_engine(self, model_id):
            self.leases -= 1

    pool = Pool()
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(image_routes, "_resolve_model", lambda model: model)
    app = FastAPI()
    app.include_router(image_routes.router)
    with TestClient(app) as client:
        generation = client.post(
            "/v1/images/generations",
            json={
                "model": "tiny-qwen",
                "prompt": "a fox",
                "steps": 30,
                "size": "32x32",
            },
        )
        assert generation.status_code == 200
        assert generation.json()["data"][0]["steps"] == 8
        assert len(calls) == 8
        calls.clear()
        buffer = BytesIO()
        Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
        edit = client.post(
            "/v1/images/edits",
            data={
                "model": "tiny-qwen",
                "prompt": "blue",
                "steps": "30",
                "size": "32x32",
            },
            files=[
                ("image", ("one.png", buffer.getvalue(), "image/png")),
                ("image", ("two.png", buffer.getvalue(), "image/png")),
            ],
        )
        assert edit.status_code == 200
        assert edit.json()["data"][0]["steps"] == 8
        assert references == [2]
        assert len(calls) == 8
        assert pool.leases == 0
        assert not list((tmp_path / "inputs").glob("*"))


def test_source_gate_fixed_noop_unknown_and_repeated_application(qwen, monkeypatch):
    from omlx.patches.mlx_vlm_qwen_image_compat._backport import BACKPORTS

    sources = {
        name: compat._read_source(importlib.import_module(name)) for name in BACKPORTS
    }
    fixed = compat._prepare_sources(sources, BACKPORTS)
    assert compat._prepare_sources(fixed, BACKPORTS) == {}
    altered = {
        **sources,
        next(iter(sources)): "raise RuntimeError('divergent source')\n",
    }
    with pytest.raises(RuntimeError, match="Unsupported"):
        compat._prepare_sources(altered, BACKPORTS)
    partially_fixed = {**sources, next(iter(fixed)): next(iter(fixed.values()))}
    with pytest.raises(RuntimeError, match="partially fixed"):
        compat._prepare_sources(partially_fixed, BACKPORTS)
    cls = qwen.model.QwenImageGenerationModel
    assert compat.apply_qwen_image_compat_patch() is False
    assert qwen.model.QwenImageGenerationModel is cls
    monkeypatch.setattr(compat, "_APPLIED", False)
    monkeypatch.setattr(compat, "_read_source", lambda module: fixed[module.__name__])
    assert compat.apply_qwen_image_compat_patch() is False
    assert qwen.model.QwenImageGenerationModel is cls


@pytest.mark.parametrize("layout", ["legacy", "nested", "both"])
def test_nested_processor_backport_uses_real_image_preprocessing(
    qwen, checkpoint, layout
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    directory = checkpoint / "processor"
    directory.mkdir()
    config = {
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    }
    if layout in ("nested", "both"):
        (directory / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": "Qwen3VLProcessor",
                    "image_processor": config,
                    "video_processor": {"patch_size": 99},
                }
            )
        )
    if layout in ("legacy", "both"):
        (directory / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    **config,
                    "min_pixels": 1024,
                    "max_pixels": 4096,
                }
            )
        )
    encoder = object.__new__(qwen.text_encoder.QwenImageTextEncoder)
    encoder.processor_dir = str(directory)
    encoder._processor = None
    encoder.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    processor = encoder.processor.image_processor
    assert processor.patch_size == 16
    assert processor.min_pixels == (65536 if layout == "nested" else 1024)
    pixels = processor(Image.new("RGB", (32, 32), "white"), return_tensors="np")
    assert np.isfinite(pixels["pixel_values"]).all()


@pytest.mark.parametrize("task", ["generate", "edit"])
def test_renamed_qwen_checkpoint_uses_real_generic_loader_without_downloads(
    qwen, checkpoint, monkeypatch, task
):
    api = _image_api()
    download = importlib.import_module("mlx_vlm.models.qwen_image.download")
    weights = importlib.import_module("mlx_vlm.models.qwen_image.weights")
    monkeypatch.setattr(
        download,
        "snapshot_download",
        lambda **_: pytest.fail("Local Qwen dispatch must not download a checkpoint"),
    )
    for component in ("transformer", "vae", "text_encoder"):
        directory = checkpoint / component
        directory.mkdir(exist_ok=True)
        (directory / "model.safetensors").write_bytes(b"unused")
    (checkpoint / "text_encoder" / "config.json").write_text("{}")
    processor = checkpoint / "processor"
    processor.mkdir()
    (processor / "tokenizer.json").write_text("{}")
    pipeline = SimpleNamespace(
        sample_sigmas=tuple(SIGMAS),
        model_path=checkpoint,
        variant=qwen.config.get_variant("qwen-image-2.1-turbo"),
    )
    received = []

    def from_pretrained(cls, variant, **kwargs):
        received.append(kwargs)
        return pipeline

    monkeypatch.setattr(
        qwen.pipeline.QwenImagePipeline, "from_pretrained", classmethod(from_pretrained)
    )
    # Ensure a class cached before activation cannot redirect to another family.
    model = api.load_image_model(str(checkpoint), task=task)
    expected = (
        qwen.model.QwenImageGenerationModel
        if task == "generate"
        else qwen.model.QwenImageEditModel
    )
    assert isinstance(model, expected)
    assert model.pipeline is pipeline
    assert model.default_sampling.steps == 8
    assert received[0]["model_path"] == checkpoint
    assert (
        download.get_variant("Qwen/Qwen-Image-2.1-Turbo").repo_id
        == "Qwen/Qwen-Image-2.1-Turbo"
    )
    assert (
        weights.get_variant("Qwen/Qwen-Image-2.1-Turbo").repo_id
        == "Qwen/Qwen-Image-2.1-Turbo"
    )


@pytest.mark.parametrize("steps", [0, -1, True, "8", 1.5])
def test_invalid_steps_cannot_hide_behind_checkpoint_recipe(qwen, steps):
    with pytest.raises(ValueError, match="steps"):
        qwen.config.resolve_sampling(steps, None, tuple(SIGMAS))


@pytest.mark.parametrize("sigmas", [[1e-50], [1, 1 - 1e-10]])
def test_float32_collapsed_grid_is_rejected(qwen, sigmas):
    with pytest.raises(ValueError, match="float32"):
        qwen.scheduler.FlowMatchEulerDiscreteScheduler(
            image_seq_len=256,
            num_inference_steps=8,
            sigmas=sigmas,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )
