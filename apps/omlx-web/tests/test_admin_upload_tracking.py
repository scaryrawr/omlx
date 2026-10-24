# SPDX-License-Identifier: Apache-2.0
"""Upload queue discovery and polling regressions without real uploads."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUEUE_KEYS = (
    "queue_loading",
    "queue_error",
    "queue_empty",
    "queue_refresh",
    "queue_activity",
)


def test_upload_tracking_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const timers = new Map();
let nextTimer = 1;
let requests = [];
let respond = async () => ({ok: true, json: async () => ({tasks: []})});
const windowEvents = {};
const browserWindow = {
    t: key => key,
    location: {search: '?tab=models&modelsTab=uploader', hash: ''},
    addEventListener: (name, callback) => {windowEvents[name] = callback;},
};
const state = vm.runInNewContext(source + '\n dashboard;', {
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
    window: browserWindow, navigator: {language: 'en'},
    document: {addEventListener: () => {}}, URLSearchParams,
    console: {error: () => {}},
    fetch: async (url, options) => {
        assert.equal(options, undefined, 'tracking must never mutate uploads');
        requests.push(url);
        return respond();
    },
    setTimeout: (fn, delay) => {
        const id = nextTimer++;
        timers.set(id, {fn, delay});
        return id;
    },
    clearTimeout: id => timers.delete(id),
})();
for (const name of [
    'loadModels', 'loadHFModels', 'loadHFTasks', 'loadOQTasks',
    'startManagerStatusRefresh', 'stopManagerStatusRefresh',
    'stopStatsRefresh', 'stopLogRefresh', 'stopHFRefresh',
    'stopMSRefresh', 'stopOQRefresh', 'syncTabStateToUrl',
]) state[name] = () => {};
let modelLoads = 0;
state.loadUploadOqModels = () => {modelLoads++;};
const active = {task_id: 'upload-1', model_name: 'Example-oQ4', status: 'uploading'};
const queued = {task_id: 'upload-2', model_name: 'Example-oQ6', status: 'pending'};
const flush = () => new Promise(resolve => setImmediate(resolve));
const timer = () => {
    assert.equal(timers.size, 1, 'exactly one refresh timer');
    return [...timers.values()][0];
};
async function tick() {
    const [id, item] = [...timers.entries()][0];
    timers.delete(id);
    await item.fn();
}
(async () => {
    state.mainTab = 'models';
    state.modelsTab = 'uploader';
    respond = async () => {
        assert.equal(state.uploadTasksLoading, true, 'first load is explicit');
        assert.equal(state.uploadTasksLoaded, false, 'do not show empty before success');
        return {ok: true, json: async () => ({tasks: [active, queued]})};
    };
    await state.handleMainTabChange('models');
    respond = async () => ({ok: true, json: async () => ({tasks: [active, queued]})});
    assert.ok(requests.includes('/admin/api/upload/tasks'), 'deep link discovers jobs');
    assert.equal(modelLoads, 1, 'deep link loads available models too');
    assert.equal(state.uploadTasks.length, 2);
    assert.equal(state.uploadTasksLoaded, true);
    assert.equal(state.uploadTasksLoading, false);
    assert.equal(state.uploadTokenValidated, false, 'tracking needs no HF validation');
    assert.equal(timer().delay, 2000);

    state.setModelsTab('uploader');
    await flush();
    assert.equal(timer().delay, 2000, 'repeat entry must not duplicate timers');
    state.startStatsRefresh = () => {};
    state.startLogRefresh = () => {};
    state.$nextTick = () => {};
    for (const tab of ['status', 'logs']) {
        let finishNavigation;
        state[tab === 'status' ? 'loadStats' : 'loadLogs'] =
            () => new Promise(resolve => {finishNavigation = resolve;});
        state.mainTab = tab;
        const leaving = state.handleMainTabChange(tab);
        await flush();
        state.mainTab = 'models';
        await state.handleMainTabChange('models');
        assert.equal(timer().delay, 2000);
        finishNavigation();
        await leaving;
        assert.equal(state._uploadRefreshEnabled, true, 'stale navigation cannot disable tracking');
        assert.equal(timer().delay, 2000, 'returning before navigation completes keeps polling');
    }
    respond = async () => ({ok: true, json: async () => ({
        tasks: [{...active, status: 'completed'}, {...queued, status: 'failed', error: 'failed'}],
    })});
    await tick();
    assert.equal(state.uploadTasks[0].status, 'completed');
    assert.equal(state.uploadTasks[1].status, 'failed');
    assert.equal(timer().delay, 6000, 'terminal jobs switch to idle discovery');

    respond = async () => ({ok: true, json: async () => ({tasks: [queued]})});
    await tick();
    assert.equal(timer().delay, 2000, 'idle discovery finds jobs from another client');
    for (const badResponse of [
        async () => ({ok: false, status: 503}),
        async () => {throw new Error('offline');},
        async () => ({ok: true, json: async () => ({})}),
        async () => ({ok: true, json: async () => ({tasks: [null]})}),
        async () => ({ok: true, json: async () => {throw new Error('invalid JSON');}}),
    ]) {
        respond = badResponse;
        await tick();
        assert.ok(state.uploadTasksError, 'errors must be visible');
        assert.equal(state.uploadTasks.length, 1, 'keep last-known queue');
        assert.equal(state.uploadTasksLoading, false);
        assert.equal(timer().delay, 6000, 'errors retry at the idle cadence');
    }
    respond = async () => ({ok: true, json: async () => ({tasks: []})});
    await state.loadUploadTasks();
    assert.equal(state.uploadTasksError, '');
    assert.equal(state.uploadTasksLoaded, true);
    assert.equal(state.uploadTasks.length, 0);
    assert.equal(timer().delay, 6000);

    state.setModelsTab('manager');
    assert.equal(timers.size, 0, 'leaving Uploader stops polling');
    state.setModelsTab('uploader');
    await flush();
    timer();
    state.mainTab = 'settings';
    await state.handleMainTabChange('settings');
    assert.equal(timers.size, 0, 'leaving Models stops polling');

    state.mainTab = 'models';
    state.modelsTab = 'uploader';
    let release;
    respond = () => new Promise(resolve => {release = resolve;});
    const entering = state.handleMainTabChange('models');
    await flush();
    assert.equal(state.uploadTasksLoading, true);
    const before = requests.length;
    await state.loadUploadTasks();
    assert.equal(requests.length, before, 'do not overlap requests');
    state.setModelsTab('manager');
    release({ok: true, json: async () => ({tasks: [active]})});
    await entering;
    assert.equal(timers.size, 0, 'late response cannot rearm after navigation');
    assert.equal(state.uploadTasksLoading, false);

    state.modelsTab = 'uploader';
    state.uploadTasksLoaded = false;
    state.uploadTasks = [];
    respond = async () => ({ok: false, status: 401});
    await state.handleMainTabChange('models');
    assert.ok(state.uploadTasksError, 'auth failure remains visible');
    assert.equal(state.uploadTasksLoaded, false, 'initial failure is not an empty success');
    assert.equal(timer().delay, 6000, 'failures retry at a bounded cadence');
    state.stopUploadRefresh();
    assert.equal(timers.size, 0);

    for (const name of [
        'applyTheme', 'loadGlobalSettings', 'loadServerInfo',
        'loadProfileFields', 'loadPresets', 'checkForUpdate',
        'fetchClaudeDesktopStatus', 'startUpdateCheckTimer', 'settingsScrollToHash',
    ]) state[name] = () => {};
    state.$watch = () => {};
    respond = async () => ({ok: true, json: async () => ({tasks: [active]})});
    await state.init();
    assert.equal(state.modelsTab, 'uploader');
    timer();
    browserWindow.location.search = '?tab=models&modelsTab=manager';
    windowEvents.popstate();
    assert.equal(timers.size, 0, 'browser history leaving Uploader stops polling');
    const beforeBack = requests.length;
    browserWindow.location.search = '?tab=models&modelsTab=uploader';
    windowEvents.popstate();
    await flush();
    assert.equal(requests.length, beforeBack + 1, 'browser history discovers uploads');
    timer();
    state.stopUploadRefresh();
    assert.ok(requests.every(url => url === '/admin/api/upload/tasks'));
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_upload_queue_has_explicit_states_and_no_fake_percentage():
    html = (ROOT / "omlx_web/templates/dashboard/_models.html").read_text()
    queue = html.split("<!-- Upload Queue -->", 1)[1].split(
        "<!-- Upload Confirmation Modal -->", 1
    )[0]
    assert 'x-show="uploadTasks.length > 0" x-cloak>' not in queue.split(
        "<template", 1
    )[0]
    for key in QUEUE_KEYS:
        if key == "queue_error":
            continue
        assert f"models.uploader.{key}" in queue
    assert "uploadTasksLoading" in queue
    assert "uploadTasksLoaded" in queue
    assert "uploadTasksError" in queue
    assert 'x-show="uploadTasksLoading && !uploadTasksLoaded"' in queue
    assert 'x-show="uploadTasksError"' in queue
    assert (
        'x-show="uploadTasksLoaded && !uploadTasksLoading && !uploadTasksError '
        '&& uploadTasks.length === 0"' in queue
    )
    assert '@click="loadUploadTasks()"' in queue
    assert "task.progress" not in queue
    assert "uploadTokenValidated" not in queue
    assert "animate-pulse" in queue


def test_upload_queue_locales():
    for locale in sorted((ROOT / "omlx_web/i18n").glob("*.json")):
        data = json.loads(locale.read_text())
        for key in QUEUE_KEYS:
            assert data.get(f"models.uploader.{key}"), (locale.name, key)
