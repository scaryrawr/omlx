# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the admin Imagine image UI."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
IMAGINE_TEMPLATE = ROOT / "omlx" / "admin" / "templates" / "imagine.html"
NAVBAR_TEMPLATE = ROOT / "omlx" / "admin" / "templates" / "dashboard" / "_navbar.html"
ROUTES = ROOT / "omlx" / "admin" / "routes.py"
I18N_DIR = ROOT / "omlx" / "admin" / "i18n"

REQUIRED_IMAGINE_KEYS = [
    "navbar.tab.imagine",
    "imagine.title",
    "imagine.brand",
    "imagine.nav.chat",
    "imagine.nav.imagine",
    "imagine.api_key_prompt",
    "imagine.select_model",
    "imagine.mode.generate",
    "imagine.mode.edit",
    "imagine.prompt_label",
    "imagine.upload_heading",
    "imagine.advanced.title",
    "imagine.advanced.seed",
    "imagine.advanced.steps",
    "imagine.advanced.guidance",
    "imagine.advanced.size",
    "imagine.advanced.n",
    "imagine.advanced.output_format",
    "imagine.advanced.negative_prompt",
    "imagine.submit.generate",
    "imagine.submit.edit",
    "imagine.results_heading",
    "imagine.error.incorrect_api_key",
    "imagine.error.invalid_image_type",
    "imagine.error.image_too_large",
]


def _imagine_model_selectable(model_type, engine_type=""):
    """Python equivalent of the imagine.html image model filter."""
    t = (model_type or "").lower()
    e = (engine_type or "").lower()
    return t == "image" or e == "image"


def _normalize_tasks(tasks):
    if not isinstance(tasks, list):
        return []
    return [str(task).lower() for task in tasks if task]


def _extract_tasks(model):
    direct_tasks = _normalize_tasks(model.get("tasks"))
    if direct_tasks:
        return direct_tasks
    return _normalize_tasks((model.get("image_metadata") or {}).get("tasks"))


def _status_metadata_maps(models):
    """Python equivalent of the imagine.html /v1/models/status metadata mapping."""
    model_type_map = {}
    model_engine_type_map = {}
    model_task_map = {}
    for model in models:
        model_id = model.get("id")
        if model_id:
            model_type_map[model_id] = model.get("model_type") or "llm"
            model_engine_type_map[model_id] = model.get("engine_type") or ""
            model_task_map[model_id] = _extract_tasks(model)
    for model in models:
        model_id = model.get("model_alias")
        if model_id and model_id not in model_type_map:
            model_type_map[model_id] = model.get("model_type") or "llm"
            model_engine_type_map[model_id] = model.get("engine_type") or ""
            model_task_map[model_id] = _extract_tasks(model)
    return model_type_map, model_engine_type_map, model_task_map


def _supports_task(tasks, task):
    """Strict task metadata policy approved for the Imagine UI."""
    return task in (tasks or [])


def test_imagine_dropdown_filter_keeps_image_model_types():
    assert _imagine_model_selectable("image")
    assert _imagine_model_selectable("llm", engine_type="image")


def test_imagine_dropdown_filter_excludes_non_image_model_types():
    assert not _imagine_model_selectable("llm", engine_type="batched")
    assert not _imagine_model_selectable("vlm", engine_type="vlm")
    assert not _imagine_model_selectable("embedding")
    assert not _imagine_model_selectable("reranker")
    assert not _imagine_model_selectable("audio_transcription")


def test_imagine_task_filter_uses_strict_generation_and_edit_metadata():
    _, _, task_map = _status_metadata_maps(
        [
            {
                "id": "generate-image",
                "model_type": "image",
                "engine_type": "image",
                "tasks": ["generation"],
            },
            {
                "id": "edit-image",
                "model_type": "image",
                "engine_type": "image",
                "tasks": [],
                "image_metadata": {"tasks": ["edit"]},
            },
            {
                "id": "unknown-image",
                "model_type": "image",
                "engine_type": "image",
            },
        ]
    )

    assert _supports_task(task_map["generate-image"], "generation")
    assert not _supports_task(task_map["generate-image"], "edit")
    assert _supports_task(task_map["edit-image"], "edit")
    assert not _supports_task(task_map["edit-image"], "generation")
    assert not _supports_task(task_map["unknown-image"], "generation")
    assert not _supports_task(task_map["unknown-image"], "edit")


def test_imagine_status_metadata_maps_active_aliases_without_collisions():
    model_type_map, engine_type_map, task_map = _status_metadata_maps(
        [
            {
                "id": "chat-model",
                "model_type": "llm",
                "engine_type": "batched",
            },
            {
                "id": "raw-image-model",
                "model_alias": "friendly-image-name",
                "model_type": "image",
                "engine_type": "image",
                "tasks": ["generation"],
            },
            {
                "id": "colliding-image",
                "model_alias": "chat-model",
                "model_type": "image",
                "engine_type": "image",
                "tasks": ["edit"],
            },
        ]
    )

    assert _imagine_model_selectable(
        model_type_map["friendly-image-name"],
        engine_type_map["friendly-image-name"],
    )
    assert task_map["friendly-image-name"] == ["generation"]
    assert model_type_map["chat-model"] == "llm"
    assert engine_type_map["chat-model"] == "batched"


def test_imagine_route_and_nav_are_registered():
    routes_source = ROUTES.read_text()
    nav_source = NAVBAR_TEMPLATE.read_text()

    assert '@router.get("/imagine", response_class=HTMLResponse)' in routes_source
    assert '"imagine.html"' in routes_source
    assert "api_key" in routes_source
    assert 'href="/admin/imagine"' in nav_source
    assert "navbar.tab.imagine" in nav_source


def test_imagine_template_wires_image_endpoints_and_safe_fields():
    source = IMAGINE_TEMPLATE.read_text()

    assert "fetch('/v1/models'" in source
    assert "fetch('/v1/models/status'" in source
    assert "fetch('/v1/images/generations'" in source
    assert "fetch('/v1/images/edits'" in source
    assert "response_format: 'b64_json'" in source
    assert "form.append('image'" in source
    assert "form.append('mask'" not in source
    assert "lora_paths" not in source
    assert "lora_scales" not in source
    for field in (
        "seed",
        "steps",
        "guidance",
        "size",
        "n",
        "output_format",
        "negative_prompt",
    ):
        assert field in source


def test_imagine_template_keeps_results_session_only():
    source = IMAGINE_TEMPLATE.read_text()

    assert "this.results =" in source
    assert "localStorage.setItem('omlx_imagine" not in source
    assert "localStorage.getItem('omlx_imagine" not in source


def test_imagine_template_refilters_model_on_mode_change():
    source = IMAGINE_TEMPLATE.read_text()

    assert "setMode(mode)" in source
    assert "this.ensureValidCurrentModel();" in source
    assert "modeModels()" in source
    assert "supportsTask(model.id, task)" in source


def test_i18n_imagine_keys_present_in_every_language_file():
    for lang_file in I18N_DIR.glob("*.json"):
        translations = json.loads(lang_file.read_text())
        for key in REQUIRED_IMAGINE_KEYS:
            assert key in translations, f"Missing key '{key}' in {lang_file.name}"
            assert translations[key], f"Empty value for '{key}' in {lang_file.name}"
