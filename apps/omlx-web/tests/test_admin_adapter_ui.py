# SPDX-License-Identifier: Apache-2.0
"""LoRA adapter guidance in the downloader and model manager."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "omlx_web"
CARD_KEYS = ("models.detail.adapter_info", "models.detail.adapter_no_base")
NOTICE_KEY = "models.manager.adapter_issues_heading"
BASE_KEYS = (
    "adapter_base", "adapter_base_hint", "adapter_base_auto", "adapter_base_apply",
    "adapter_base_current", "adapter_base_unload", "adapter_base_saved",
)


def test_adapter_copy_is_available_in_every_locale():
    template = (WEB / "templates/dashboard/_models.html").read_text()
    dashboard = (WEB / "static/js/dashboard.js").read_text()
    retired_keys = ("is_" + "adapter", "lora_" + "warning")

    for key in (*CARD_KEYS, NOTICE_KEY):
        assert key in template
    for key in retired_keys:
        assert key not in template
        assert key not in dashboard

    for locale in sorted((WEB / "i18n").glob("*.json")):
        copy = json.loads(locale.read_text(encoding="utf-8"))
        for key in CARD_KEYS:
            assert copy[key].count("{base}") == 1, (locale.name, key)
        assert copy[NOTICE_KEY], locale.name
        for key in BASE_KEYS:
            assert copy[f"modal.model_settings.{key}"], (locale.name, key)
        assert not any(key in entry for key in retired_keys for entry in copy)


def test_adapter_base_picker_saves_clears_and_reports_errors():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const requests = [];
let fail = false;
let override = null;
let resolved = 'Qwen-oQ4e';
const picker = vm.runInNewContext(source + '\n adapterBasePicker;', {
    window: {t: key => key}, console: {error: () => {}},
    fetch: async (url, options) => {
        assert.equal(url, '/admin/api/adapters/tuned%20adapter/base');
        if (!options) return {ok: true, json: async () => ({
            base_model_id: override, resolved_base_model_id: resolved,
            candidates: ['Qwen-oQ4e', 'Qwen-oQ8'], can_change: true,
        })};
        requests.push(JSON.parse(options.body));
        assert.equal(options.method, 'PUT');
        if (fail) return {ok: false, json: async () => ({detail: 'Unload adapter first.'})};
        override = JSON.parse(options.body).base_model_id;
        resolved = override || 'Qwen-oQ4e';
        return {ok: true, json: async () => ({
            base_model_id: override, adapter: {base_model_id: resolved}, issue: null,
        })};
    },
})('tuned adapter');
picker.formatApiError = value => value;
picker.selectedModel = {id: 'tuned adapter', adapter: {base_model_id: resolved}};
picker.modelSettings = {temperature: 0.5};
picker.loadModels = async () => {
    picker.models = [{id: 'tuned adapter', adapter: {base_model_id: resolved}}];
};
(async () => {
    await picker.init();
    assert.equal(picker.baseId, '');
    assert.equal(picker.resolvedBaseId, 'Qwen-oQ4e');
    assert.deepEqual(Array.from(picker.candidates), ['Qwen-oQ4e', 'Qwen-oQ8']);
    picker.baseId = 'Qwen-oQ8';
    assert.equal(requests.length, 0, 'selection does not submit');
    await picker.saveBase();
    assert.deepEqual(requests, [{base_model_id: 'Qwen-oQ8'}]);
    assert.equal(picker.selectedModel.adapter.base_model_id, 'Qwen-oQ8');
    assert.equal(picker.modelSettings.temperature, 0.5, 'other edits survive');
    await picker.init();
    assert.equal(picker.baseId, 'Qwen-oQ8', 'reopening restores the saved choice');
    picker.baseId = '';
    await picker.init(true);
    assert.equal(picker.baseId, '', 'refresh preserves an unapplied Automatic choice');
    assert.equal(picker.savedBaseId, 'Qwen-oQ8');
    fail = true;
    picker.baseId = '';
    await picker.saveBase();
    assert.equal(picker.error, 'Unload adapter first.');
    assert.equal(picker.savedBaseId, 'Qwen-oQ8');
    fail = false;
    await picker.saveBase();
    assert.deepEqual(requests.at(-1), {base_model_id: null});
    assert.equal(picker.resolvedBaseId, 'Qwen-oQ4e');
    assert.equal(picker.savedBaseId, '');
    assert.equal(picker.error, '');
    picker.canChange = false;
    picker.baseId = 'Qwen-oQ8';
    const count = requests.length;
    await picker.saveBase();
    assert.equal(requests.length, count, 'active adapters cannot be changed');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reload_refreshes_adapter_base_choices_without_losing_edits():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const template = fs.readFileSync('omlx_web/templates/dashboard/_adapter_base_picker.html', 'utf8');
const handler = template.match(/@models-reloaded\.window="([^"]+)"/)?.[1];
assert.ok(handler, 'the mounted picker must handle model rediscovery');
const issues = [{
    id: 'tuned', kind: 'base_missing', reason: 'Download the base.',
    candidates: [], path: '/models/tuned',
}];
let candidates = ['old-local-checkpoint'];
let nextCandidates = ['old-local-checkpoint', 'new-local-checkpoint'];
let refreshed;
let picker;
const context = vm.createContext({
    localStorage: {getItem: () => null},
    window: {
        t: key => key,
        dispatchEvent: event => {
            assert.equal(event.type, 'models-reloaded');
            refreshed = vm.runInNewContext(handler, {
                init: picker.init.bind(picker),
            });
        },
    },
    CustomEvent: class { constructor(type) { this.type = type; } },
    console,
    fetch: async (url, options) => {
        if (url === '/admin/api/reload') {
            assert.equal(options.method, 'POST');
            candidates = nextCandidates;
            return {ok: true};
        }
        if (url === '/admin/api/models') {
            return {ok: true, json: async () => ({models: [], adapter_issues: issues})};
        }
        assert.equal(url, '/admin/api/adapters/tuned/base');
        return {ok: true, json: async () => ({
            candidates: [...candidates], base_model_id: null,
            resolved_base_model_id: null, can_change: true,
        })};
    },
});
vm.runInContext(source, context);
const state = context.dashboard();
state.loadHFModels = async () => {};
picker = context.adapterBasePicker('tuned');
picker.formatApiError = state.formatApiError;
(async () => {
    await state.loadModels();
    await picker.init();
    picker.baseId = 'old-local-checkpoint';
    const mountedIssues = state.adapterIssues;
    await state.reloadModels();
    await refreshed;
    assert.equal(state.adapterIssues, mountedIssues, 'the unchanged notice stays mounted');
    assert.deepEqual(Array.from(picker.candidates), [
        'old-local-checkpoint', 'new-local-checkpoint',
    ]);
    assert.equal(picker.baseId, 'old-local-checkpoint', 'the unapplied choice survives');
    assert.equal(picker.savedBaseId, '', 'refresh does not save the choice');
    nextCandidates = ['new-local-checkpoint'];
    await state.reloadModels();
    await refreshed;
    assert.deepEqual(Array.from(picker.candidates), ['new-local-checkpoint']);
    assert.equal(picker.baseId, '', 'a removed draft choice restores the saved preference');
    assert.equal(picker.savedBaseId, '');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_adapter_base_save_supersedes_an_older_reload_response():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const pendingRead = Promise.withResolvers();
const pendingWrite = Promise.withResolvers();
const issue = {kind: 'base_missing', reason: 'Download the base.'};
let override = 'missing-checkpoint';
let candidates = ['old-local-checkpoint'];
let pauseRead = false;
const picker = vm.runInNewContext(source + '\n adapterBasePicker;', {
    window: {t: key => key}, console,
    fetch: async (url, options) => {
        assert.equal(url, '/admin/api/adapters/tuned/base');
        if (options) {
            assert.equal(options.method, 'PUT');
            assert.deepEqual(JSON.parse(options.body), {base_model_id: null});
            await pendingWrite.promise;
            override = null;
            candidates = ['new-local-checkpoint'];
            return {ok: true, json: async () => ({
                base_model_id: null, adapter: null, issue,
            })};
        }
        const snapshot = {
            base_model_id: override, resolved_base_model_id: null,
            candidates: [...candidates], can_change: true,
        };
        if (pauseRead) {
            pauseRead = false;
            await pendingRead.promise;
        }
        return {ok: true, json: async () => snapshot};
    },
})('tuned');
picker.formatApiError = value => value;
picker.loadModels = async () => { picker.models = []; };
(async () => {
    await picker.init();
    picker.baseId = '';
    const saving = picker.saveBase();
    pauseRead = true;
    const reloading = picker.init(true);
    pendingWrite.resolve();
    await saving;
    pendingRead.resolve();
    await reloading;
    assert.equal(picker.baseId, '', 'the successful Automatic choice stays selected');
    assert.equal(picker.savedBaseId, '', 'the stale read cannot restore the cleared override');
    assert.equal(picker.resolvedBaseId, null);
    assert.deepEqual(Array.from(picker.candidates), ['new-local-checkpoint']);
    assert.equal(picker.status, issue.reason, 'the unresolved notice remains usable');
    assert.equal(picker.error, '');
    assert.equal(picker.loading, false);
    assert.equal(picker.saving, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_adapters_can_show_updates_and_report_unavailable_entries():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const issues = [
    {id: 'same-name', kind: 'base_missing', reason: 'Download the base.',
        candidates: [], path: '/models/a/same-name'},
    {id: 'same-name', kind: 'id_collision', reason: 'Rename the folder.',
        candidates: [], path: '/models/b/same-name'},
];
let payload = {models: [], adapter_issues: issues};
const state = vm.runInNewContext(source + '\n dashboard;', {
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
    window: {t: key => key}, navigator: {language: 'en'}, document: {}, console,
    fetch: async url => {
        assert.equal(url, '/admin/api/models');
        return {ok: true, json: async () => structuredClone(payload)};
    },
})();

const repo = 'harness-hub/LFM3-1.7B-pi';
state.hfModels = [{name: 'LFM3-1.7B-pi', repo_id: repo}];
state.hfModelDetail = {
    repo_id: repo, layout: 'safetensors', update_files: ['adapter_model.safetensors'],
    adapter: {base_repo_id: 'LiquidAI/LFM3-1.7B', rank: 32, local_base_ids: []},
};
assert.equal(state.hfShowsUpdate(), true);
state.hfModelDetail.unsupported_reason = 'Unsupported adapter config.';
assert.equal(state.hfShowsUpdate(), false);

(async () => {
    await state.loadModels();
    assert.deepEqual(state.adapterIssues, issues);
    const firstIssues = state.adapterIssues;
    await state.loadModels();
    assert.equal(state.adapterIssues, firstIssues, 'identical polling must not update the notice');

    payload = {models: []};
    await state.loadModels();
    assert.equal(state.adapterIssues.length, 0);
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
