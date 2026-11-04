# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the admin Imagine image UI."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
WEB_ROOT = ROOT / "apps" / "omlx-web" / "omlx_web"
IMAGINE_TEMPLATE = WEB_ROOT / "templates" / "imagine.html"
CHAT_TEMPLATE = WEB_ROOT / "templates" / "chat.html"
NAVBAR_TEMPLATE = WEB_ROOT / "templates" / "dashboard" / "_navbar.html"
ROUTES = WEB_ROOT / "routes.py"
I18N_DIR = WEB_ROOT / "i18n"

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
    "imagine.checkpoint_schedule_hint",
    "imagine.effective_steps",
    "imagine.advanced.guidance",
    "imagine.advanced.size",
    "imagine.advanced.n",
    "imagine.advanced.output_format",
    "imagine.advanced.negative_prompt",
    "imagine.submit.generate",
    "imagine.submit.edit",
    "imagine.results_heading",
    "imagine.original_image_alt",
    "imagine.timeline_model",
    "imagine.timeline_prompt",
    "imagine.timeline_original",
    "imagine.timeline_edited",
    "imagine.timeline_generated",
    "imagine.error.incorrect_api_key",
    "imagine.error.invalid_image_type",
    "imagine.error.image_too_large",
]


def test_imagine_executes_qwen_recipe_and_request_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute Imagine JavaScript")
    source = IMAGINE_TEMPLATE.read_text().split("{% block scripts %}", 1)[1]
    source = source.split("<script>", 1)[1].split("</script>", 1)[0]
    source = source.replace("{{ api_key | tojson }}", '""')
    harness = (
        r"""
const assert = require('node:assert/strict');
global.localStorage = {getItem: () => null};
global.window = {t: key => key};
"""
        + source
        + r"""
(async () => {
    const app = imagineApp();
    app.apiKeySet = true;
    app.apiKeyInput = 'test';
    app.mergeStatusMetadata([
        {id: 'raw', model_alias: 'turbo', model_type: 'image', engine_type: 'image',
         tasks: ['generation', 'edit'],
         image_metadata: {uses_checkpoint_sigmas: true, checkpoint_steps: 8}},
        {id: 'chat', model_type: 'llm', tasks: []},
        {id: 'collision', model_alias: 'chat', model_type: 'image', tasks: ['edit']}
    ]);
    app.currentModel = 'turbo';
    app.availableModels = [{id: 'turbo'}, {id: 'chat'}];
    assert.equal(app.currentRecipe().checkpoint_steps, 8);
    assert.equal(app.modelTypeMap.chat, 'llm');
    assert.deepEqual(app.modeModels(), [{id: 'turbo'}]);
    app.prompt = 'a fox';
    assert.equal(app.commonPayload().steps, undefined);
    assert.equal(app.commonPayload().guidance, undefined);
    app.settings.steps = '30';
    let generation;
    global.fetch = async (url, options) => {
        assert.equal(url, '/v1/images/generations');
        generation = JSON.parse(options.body);
        return {ok: true, json: async () => ({data: [
            {b64_json: 'AA==', output_format: 'png', steps: 8, guidance: 1, size: '4x2'}
        ]})};
    };
    await app.submit();
    assert.equal(generation.steps, 30);
    assert.equal(generation.sigmas, undefined);
    assert.equal(app.results[0].images[0].steps, 8);
    assert.equal(app.results[0].images[0].guidance, 1);
    app.setMode('edit');
    app.uploadImages = [
        {id: 'one', file: new File(['first'], 'first.png'), preview: 'data:first'},
        {id: 'two', file: new File(['second'], 'second.png'), preview: 'data:second'}
    ];
    global.fetch = async (url, options) => {
        assert.equal(url, '/v1/images/edits');
        assert.deepEqual(options.body.getAll('image').map(file => file.name),
                         ['first.png', 'second.png']);
        assert.equal(options.body.get('sigmas'), null);
        assert.equal(options.body.get('image_strength'), null);
        assert.equal(options.body.get('mask'), null);
        return {ok: true, json: async () => ({data: [
            {b64_json: 'AA==', steps: 8, output_format: 'png'}
        ]})};
    };
    await app.submit();
    assert.equal(app.results[0].mode, 'edit');
    assert.equal(app.results[0].originals.length, 2);
    assert.equal(app.results[0].images[0].steps, 8);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    )
    result = subprocess.run(
        [node], input=harness, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    chat_source = CHAT_TEMPLATE.read_text()

    assert '@router.get("/imagine", response_class=HTMLResponse)' in routes_source
    assert '"imagine.html"' in routes_source
    assert "api_key" in routes_source
    assert 'href="/admin/imagine"' in nav_source
    assert "navbar.tab.imagine" in nav_source
    assert 'href="/admin/imagine"' in chat_source
    assert "navbar.tab.imagine" in chat_source


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


def test_imagine_template_keeps_edit_attempt_timeline_with_originals():
    source = IMAGINE_TEMPLATE.read_text()

    assert "this.results = [attempt, ...this.results]" in source
    assert "attempt.originals" in source
    assert "requestMode === 'edit'" in source
    assert "mode: requestMode" in source
    assert "originalSnapshot" in source
    assert "pendingOriginalPreviews" in source
    assert "image.preview" in source
    assert "previewShouldStayAlive" in source
    assert "clearResults()" in source


def test_imagine_template_does_not_depend_on_unbuilt_arbitrary_tailwind_classes():
    source = IMAGINE_TEMPLATE.read_text()

    assert "imagine-sidebar" in source
    assert "imagine-edit-comparison" in source
    assert "imagine-result-grid" in source
    assert "w-[" not in source
    assert "min-w-[" not in source
    assert "max-w-[" not in source
    assert "h-[" not in source
    assert "z-[" not in source
    assert "grid-cols-[" not in source
    assert "md:grid-cols" not in source
    assert "xl:grid-cols" not in source


def test_imagine_template_refilters_model_on_mode_change():
    source = IMAGINE_TEMPLATE.read_text()

    assert "setMode(mode)" in source
    assert "this.ensureValidCurrentModel();" in source
    assert "modeModels()" in source
    assert "supportsTask(model.id, task)" in source


def test_i18n_imagine_keys_present_in_every_language_file():
    language_files = list(I18N_DIR.glob("*.json"))
    assert language_files, "Imagine translations must be packaged"
    for lang_file in language_files:
        translations = json.loads(lang_file.read_text())
        for key in REQUIRED_IMAGINE_KEYS:
            assert key in translations, f"Missing key '{key}' in {lang_file.name}"
            assert translations[key], f"Empty value for '{key}' in {lang_file.name}"
