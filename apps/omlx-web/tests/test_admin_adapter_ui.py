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
        assert not any(key in entry for key in retired_keys for entry in copy)


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
