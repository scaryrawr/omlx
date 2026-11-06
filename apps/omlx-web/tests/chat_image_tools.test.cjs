// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const web = path.join(__dirname, '../omlx_web');
const template = fs.readFileSync(path.join(web, 'templates/chat.html'), 'utf8');
const store = new Map([['omlx_chat_api_key', 'test-key']]);
global.localStorage = {
    getItem: key => store.get(key) || null,
    setItem: (key, value) => store.set(key, String(value)),
    removeItem: key => store.delete(key),
};
global.window = { innerWidth: 1200, t: key => key, matchMedia: () => ({ matches: false }) };
global.document = { getElementById: () => null };
const markdownOptions = [];
global.marked = {
    use: options => markdownOptions.push(options),
    parse: text => `<p>${text}</p>`,
};
global.DOMPurify = { sanitize: html => html };
const images = vm.runInThisContext(
    fs.readFileSync(path.join(web, 'static/js/chat-image-tools.js'), 'utf8') + '\nChatImageTools;'
);
vm.runInThisContext(fs.readFileSync(path.join(web, 'static/js/chat-history.js'), 'utf8'));
const script = template.split('{% block scripts %}')[1].match(/<script>([\s\S]*?)<\/script>/)[1]
    .replace('{{ api_key | tojson }}', '""');
const chatApp = vm.runInThisContext(script + '\nchatApp;');
const alpineContext = {
    window: { Element: class {} },
    document: { createElement: () => ({}) },
    MutationObserver: class {},
    queueMicrotask: () => {},
};
vm.runInNewContext(fs.readFileSync(path.join(web, 'static/js/alpine.min.js'), 'utf8'), alpineContext);
const reactive = alpineContext.window.Alpine.reactive;
const png = 'data:image/png;base64,c291cmNl';
const upload = { id: 'upload-1', role: 'user', content: [
    { type: 'text', text: 'Edit this' }, { type: 'image_url', image_url: { url: png } },
] };
const controllerScope = (messages = [upload]) => ({
    chatId: 'owner', messages, resultMessageId: 'reserved',
    signal: new AbortController().signal, headers: { Authorization: 'Bearer test-key' },
});
const call = (name, args, id = name) => ({
    id, type: 'function', function: { name, arguments: JSON.stringify(args) },
});
const json = data => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
function sse(calls) {
    const delta = calls
        ? { tool_calls: calls.map((tc, index) => ({ ...tc, index })) }
        : { content: 'Done' };
    const data = {
        choices: [{ delta, finish_reason: calls ? 'tool_calls' : 'stop' }],
    };
    return new Response(`data: ${JSON.stringify(data)}\n\ndata: [DONE]\n\n`);
}

function appFixture(messages = [structuredClone(upload)]) {
    const app = reactive(chatApp());
    app.currentChatId = 'foreground';
    app.currentModel = 'text';
    app.chatHistoryPending = false;
    app.savedChats = new Map();
    app.historyStore = {
        deleted: new Set(),
        save: chat => {
            app.savedChats.set(chat.id, app.cloneData(chat));
            return Promise.resolve([]);
        },
        delete: id => {
            app.savedChats.delete(id);
            app.historyStore.deleted.add(id);

            return Promise.resolve();
        },
        clear: () => {
            app.savedChats.clear();
            return Promise.resolve();
        },
    };
    app.modelTypeMap = { text: 'llm', vision: 'vlm' };
    app.availableModels = [{ id: 'text', name: 'text' }, { id: 'vision', name: 'vision' }];
    app.chatSessions.owner = {
        messages, model: 'text', systemPrompt: '', modelSettingsByModel: {},
        modelSettings: app.defaultModelSettings(), contextTokens: 0,
    };
    app.$refs = {};
    app.$nextTick = fn => fn();
    for (const name of [
        'startStatsPolling', 'stopStatsPolling', 'startPrefillPolling', 'stopPrefillPolling',
        'ensureStatsPollingForCurrentChat', 'scrollToBottom', 'scheduleEnhanceMessages',
    ]) app[name] = () => {};
    return app;
}

function archiveFixture(t, app) {
    const Reader = global.FileReader;
    const alert = global.alert;
    const alerts = [];
    let reader;
    global.FileReader = class {
        constructor() { reader = this; }
        readAsText() {}
    };
    global.alert = message => alerts.push(message);
    t.after(() => {
        global.FileReader = Reader;
        global.alert = alert;
    });
    app.loadChatHistory = async () => {};
    app.historyStore.generation = 0;

    return {
        alerts,
        import(entries) {
            app.importChats({ target: { files: [{}], value: 'archive.json' } });
            return reader.onload({ target: { result: JSON.stringify(entries) } });
        },
    };
}

test('failed deletion keeps the chat available for export and retry', async () => {
    const app = appFixture();
    app.saveCurrentChat('owner');
    app.historyDeleted.add('already-deleted');
    const failure = new Error('Storage write failed');
    let reported;
    app.historyStore.delete = async () => { throw failure; };
    app.reportHistoryError = error => { reported = error; };

    await app.deleteChat('owner', { skipConfirm: true });

    assert.equal(reported, failure);
    assert.ok(app.chatHistory.some(chat => chat.id === 'owner'));
    assert.ok(app.chatSessions.owner);

    assert.ok(!app.historyDeleted.has('owner'));
    assert.ok(app.historyDeleted.has('already-deleted'));
    app.chatSessions.owner.messages.push({ id: 'later', role: 'user', content: 'Keep this after failure' });
    app.saveCurrentChat('owner');
    assert.equal(JSON.parse(JSON.stringify(app.chatHistory))[0].messages.at(-1).content, 'Keep this after failure');

    app.historyStore.delete = async () => {};
    await app.deleteChat('owner', { skipConfirm: true });
    assert.ok(!app.chatHistory.some(chat => chat.id === 'owner'));
    assert.ok(!app.historyDirty.has('owner'));
});

test('failed deletion of the first message preserves context and images for later saves', async () => {
    const app = appFixture([
        structuredClone(upload),
        { id: 'result', role: 'tool', content: 'Generated image', _images: [png] },
    ]);
    app.currentChatId = 'owner';
    app.messages = app.chatSessions.owner.messages;
    const original = app.cloneData(app.messages);

    app.saveCurrentChat('owner');
    await app.saveChatHistory();

    app.historyStore.delete = async () => { throw new Error('Storage unavailable'); };

    app.reportHistoryError = () => {};

    await app.deleteMessage(0);

    assert.deepEqual(app.chatSessions.owner.messages, original);
    assert.equal(app.messages, app.chatSessions.owner.messages);
    app.messages.push({ id: 'later', role: 'user', content: 'Continue with these images' });
    app.saveCurrentChat('owner');
    await app.saveChatHistory();

    const saved = app.savedChats.get('owner');

    assert.deepEqual(saved.messages.slice(0, 2), original);
    assert.equal(saved.messages.at(-1).content, 'Continue with these images');
    assert.deepEqual(JSON.parse(JSON.stringify(app.chatHistory))[0].messages, saved.messages);
});

for (const temporary of [false, true]) {
    test(`an active microphone keeps its chat identity while temporary is ${temporary}`, async () => {
        const app = appFixture([]);
        app.currentChatId = 'owner';
        app.messages = app.chatSessions.owner.messages;
        app.chatSessions.owner.temporary = temporary;
        app.micSession = { chatId: 'owner' };

        await app.toggleTemporaryChat();

        assert.equal(app.currentChatId, 'owner');
        assert.equal(app.isTemporaryChat('owner'), temporary);
        assert.ok(app.chatSessions.owner);
        assert.equal(app.messages, app.chatSessions.owner.messages);
    });
}

test('history writes protect the current chat, streams, and recordings but allow inactive eviction', async () => {
    const app = appFixture();
    app.currentChatId = 'owner';
    app.chatSessions.other = { messages: [] };
    app.chatSessions.recording = { messages: [] };
    app.chatSessions.inactive = { messages: [] };
    app.getStreamSession('other', true).isStreaming = true;
    app.micSession = { chatId: 'recording' };
    let options;

    app.historyStore.save = async (chat, saveOptions) => {
        options = saveOptions;

        return [];
    };

    app.saveCurrentChat('owner');
    await app.saveChatHistory();

    assert.deepEqual(options.protectedIds(), ['owner', 'other', 'recording']);
});

test('a failed image save remains protected after navigating to a smaller chat', async () => {
    const app = appFixture();
    app.currentChatId = 'owner';
    app.messages = app.chatSessions.owner.messages;
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    app.reportHistoryError = () => {};
    app.historyStore.save = async (chat, options) => {
        if (chat.id === 'owner') throw new Error('Image quota exceeded');
        if (options.protectedIds().includes('owner')) throw new Error('Chat history is full');

        app.historyStore.deleted.add('owner');

        return ['owner'];
    };
    app.messages.push({ id: 'later', role: 'user', content: 'Keep my unsaved context' });
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    app.currentChatId = 'smaller';
    app.chatSessions.smaller = { messages: [], model: 'text', modelSettingsByModel: {} };
    app.saveCurrentChat('smaller');
    await app.saveChatHistory();

    assert.ok(app.historyDirty.has('owner'));
    assert.ok(!app.historyDeleted.has('owner'));
    assert.equal(app.savedChats.get('owner').messages.length, 1);

    const exported = JSON.parse(JSON.stringify(app.chatHistory)).find(chat => chat.id === 'owner');

    assert.equal(exported.messages.at(-1).content, 'Keep my unsaved context');
    assert.equal(exported.messages[0].content[1].image_url.url, png);
});

test('queued writes resolve protection from the latest navigation and dirty ownership', async () => {
    const app = appFixture();
    let options;
    app.historyStore.save = async (chat, value) => {
        options = value;
        return [];
    };
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    app.currentChatId = 'newly-open';
    app.historyDirty.set('failed-image', { version: 99, mode: 'save', flight: null });
    const protectedIds = options.protectedIds();

    assert.ok(protectedIds.includes('newly-open'));
    assert.ok(protectedIds.includes('failed-image'));
});

test('failed clear retains live history instead of reporting an empty history', async t => {
    const app = appFixture();
    app.saveCurrentChat('owner');
    app.historyDeleted.add('already-deleted');
    app.chatSessions['already-deleted'] = { messages: [] };
    const failure = new Error('Storage clear failed');
    let reported;
    const confirm = global.confirm;
    global.confirm = () => true;
    t.after(() => { global.confirm = confirm; });
    app.historyStore.clear = async () => { throw failure; };
    app.reportHistoryError = error => { reported = error; };

    await app.clearAllHistory();

    assert.equal(reported, failure);
    assert.ok(app.chatHistory.some(chat => chat.id === 'owner'));
    assert.ok(app.chatSessions.owner);
    assert.deepEqual(app.savedChats.get('owner'), app.cloneData(app.chatHistory.find(chat => chat.id === 'owner')));

    assert.ok(!app.historyDeleted.has('owner'));
    assert.ok(app.historyDeleted.has('already-deleted'));
    app.chatSessions.owner.messages.push({ id: 'later', role: 'user', content: 'Keep this after failure' });
    app.saveCurrentChat('owner');
    assert.equal(JSON.parse(JSON.stringify(app.chatHistory))[0].messages.at(-1).content, 'Keep this after failure');

    app.historyStore.clear = async () => {};
    app.startNewChat = async () => {};
    await app.clearAllHistory();

    assert.equal(app.historyDirty.size, 0);
});

for (const synchronous of [false, true]) {
    test(`failed ${synchronous ? 'synchronous' : 'asynchronous'} writes retry only dirty IDs on an explicit flush`, async () => {
        const app = appFixture();
        const failure = new Error('Storage unavailable');
        const writes = [];
        let failing = true;
        app.reportHistoryError = () => {};
        app.historyStore.save = chat => {
            writes.push(chat.id);
            if (failing) {
                if (synchronous) throw failure;
                return Promise.reject(failure);
            }
            app.savedChats.set(chat.id, app.cloneData(chat));
            return Promise.resolve([]);
        };

        app.saveCurrentChat('owner');
        await app.historyDirty.get('owner')?.flight?.done;
        await new Promise(resolve => setImmediate(resolve));

        assert.deepEqual(writes, ['owner']);
        assert.ok(app.historyDirty.has('owner'));
        assert.equal(app.historyDirty.get('owner').flight, null);
        failing = false;
        await app.saveChatHistory();
        await app.saveChatHistory();

        assert.deepEqual(writes, ['owner', 'owner']);
        assert.equal(app.historyDirty.size, 0);
        assert.equal(app.savedChats.get('owner').messages[0].content[1].image_url.url, png);
    });
}

for (const failFirst of [false, true]) {
    test(`a newer mutation survives an older ${failFirst ? 'failed' : 'successful'} in-flight write`, async () => {
        const app = appFixture();
        const writes = [];
        const releases = [];
        app.reportHistoryError = () => {};
        app.historyStore.save = chat => {
            const snapshot = app.cloneData(chat);
            writes.push(snapshot);
            return new Promise((resolve, reject) => {
                releases.push(() => {
                    if (failFirst && writes.length === 1) reject(new Error('Storage unavailable'));
                    else {
                        app.savedChats.set(chat.id, snapshot);
                        resolve([]);
                    }
                });
            });
        };

        app.saveCurrentChat('owner');
        const olderVersion = app.historyDirty.get('owner').version;
        app.renameDraft = 'Newest title';
        app.saveRenamedChat('owner');
        app.chatSessions.owner.messages.push({ id: 'later', role: 'user', content: 'Newest message' });
        app.saveCurrentChat('owner');
        const newerVersion = app.historyDirty.get('owner').version;
        const flushing = app.saveChatHistory();

        assert.ok(newerVersion > olderVersion);
        assert.equal(writes.length, 1);
        releases.shift()();
        await new Promise(resolve => setImmediate(resolve));

        if (failFirst) {
            await flushing;
            assert.equal(writes.length, 1);
            assert.equal(app.historyDirty.get('owner').version, newerVersion);
            assert.equal(app.historyDirty.get('owner').flight, null);
        } else {
            assert.equal(writes.length, 2);
            assert.ok(app.historyDirty.has('owner'));
        }
        const retrying = failFirst ? app.saveChatHistory() : flushing;

        assert.equal(writes.length, 2);
        assert.equal(writes[1].title, 'Newest title');
        assert.equal(writes[1].messages.at(-1).content, 'Newest message');
        releases.shift()();
        await retrying;
        await app.saveChatHistory();

        assert.equal(writes.length, 2);
        assert.equal(app.historyDirty.size, 0);
        assert.equal(app.savedChats.get('owner').messages.at(-1).content, 'Newest message');
    });
}

test('overlapping failed removals release only their own membership', async () => {
    const app = appFixture();
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    const removals = [];
    app.historyStore.delete = () => new Promise((resolve, reject) => removals.push({ resolve, reject }));
    app.reportHistoryError = () => {};
    let writes = 0;
    const save = app.historyStore.save;
    app.historyStore.save = chat => {
        writes++;
        return save(chat);
    };

    const first = app.deleteChat('owner', { skipConfirm: true });
    const second = app.deleteChat('owner', { skipConfirm: true });
    app.chatSessions.owner.messages.push({ id: 'later', role: 'user', content: 'Retained while pending' });
    app.saveCurrentChat('owner');

    assert.equal(app.chatHistory[0].messages.at(-1).content, 'Retained while pending');
    assert.equal(writes, 0);
    removals[0].reject(new Error('First removal failed'));
    await first;

    assert.ok(app.isPendingHistoryRemoval('owner'));
    assert.equal(writes, 0);
    removals[1].reject(new Error('Second removal failed'));
    await second;

    assert.ok(!app.isPendingHistoryRemoval('owner'));
    assert.equal(app.pendingHistoryRemovals.size, 0);
    assert.equal(writes, 1);
    assert.equal(app.savedChats.get('owner').messages.at(-1).content, 'Retained while pending');
});

for (const commitFirst of [false, true]) {
    test(`a committed removal remains terminal when overlap fails ${commitFirst ? 'afterward' : 'first'}`, async () => {
        const app = appFixture();
        app.saveCurrentChat('owner');
        await app.saveChatHistory();
        const removals = [];
        app.historyStore.delete = () => new Promise((resolve, reject) => removals.push({ resolve, reject }));
        app.reportHistoryError = () => {};
        const first = app.deleteChat('owner', { skipConfirm: true });
        const second = app.deleteChat('owner', { skipConfirm: true });
        app.chatSessions.owner.messages.push({ id: 'later', role: 'user', content: 'Pending export only' });
        app.saveCurrentChat('owner');
        const retained = app.chatSessions.owner;

        if (commitFirst) {
            removals[0].resolve();
            await first;
            removals[1].reject(new Error('Second removal failed'));
            await second;
        } else {
            removals[1].reject(new Error('Second removal failed'));
            await second;
            removals[0].resolve();
            await first;
        }
        app.saveCurrentChat('owner', retained.messages);
        await app.saveChatHistory();

        assert.ok(app.historyDeleted.has('owner'));
        assert.ok(!app.isPendingHistoryRemoval('owner'));
        assert.ok(!app.chatSessions.owner);
        assert.ok(!app.historyDirty.has('owner'));
        assert.ok(!app.chatHistory.some(chat => chat.id === 'owner'));
    });
}

test('model settings and pin mutations persist only their owning chat', async () => {
    const app = appFixture();
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    const updatedAt = app.chatHistory[0].updatedAt;
    const settings = { modelSettings: { temperature: 0.25 }, speculativeEngine: null, speculativeDraftModel: null };
    app.captureSessionModelSettings = () => settings;
    app.chatSessions.owner.systemPrompt = 'Changed settings';

    app.persistChatSettingsToHistory('owner');
    await app.saveChatHistory();
    app.togglePinChat('owner');
    await app.saveChatHistory();

    const saved = app.savedChats.get('owner');

    assert.equal(saved.updatedAt, updatedAt);
    assert.equal(saved.systemPrompt, 'Changed settings');
    assert.equal(saved.modelSettings.temperature, 0.25);
    assert.equal(saved.pinned, true);
    assert.equal(app.historyDirty.size, 0);
});

test('history loading writes normalized and early dirty rows but leaves restored image rows clean', async t => {
    const app = appFixture();
    const Store = ChatHistory.Store;
    const writes = [];
    const restored = [
        { id: 'clean', title: 'Unchanged', model: 'text', messages: [app.cloneData(upload)] },
        { id: 'normalized', title: 'chat.untitled.old', model: 'alias', messages: [],
            modelSettingsByModel: { alias: { temperature: 0.25 } } },
    ];
    ChatHistory.Store = class {
        constructor() {
            this.generation = 0;
            this.deleted = new Set();
        }
        async load() { return restored; }
        async save(chat) {
            writes.push(app.cloneData(chat));
            return [];
        }
    };
    t.after(() => { ChatHistory.Store = Store; });
    app.resolveGatewayModelId = model => model === 'alias' ? 'text' : model;
    app.untitledChatTitle = () => 'Untitled';
    app.chatHistoryPending = true;
    app.saveCurrentChat('owner');

    assert.equal(writes.length, 0);
    await app.loadChatHistory();
    await app.saveChatHistory();

    assert.deepEqual(writes.map(chat => chat.id).sort(), ['normalized', 'owner']);
    const normalized = writes.find(chat => chat.id === 'normalized');

    assert.equal(normalized.title, 'Untitled');
    assert.equal(normalized.model, 'text');
    assert.equal(normalized.modelSettingsByModel.text.temperature, 0.25);
    assert.equal(app.chatHistory.find(chat => chat.id === 'clean').messages[0].content[1].image_url.url, png);
    assert.equal(app.historyDirty.size, 0);
});

for (const trigger of ['load', 'save']) {
    test(`failed initialization retries through a later ${trigger} and merges only dirty live rows`, async t => {
        const app = appFixture();
        const Store = ChatHistory.Store;
        const writes = [];
        let attempts = 0;
        app.reportHistoryError = () => {};
        ChatHistory.Store = class {
            constructor() {
                this.generation = 0;
                this.deleted = new Set();
                this.attempt = ++attempts;
            }
            async load() {
                if (this.attempt === 1) throw new Error('Storage upgrade blocked');
                return [
                    { id: 'durable', title: 'Saved image', model: 'text', messages: [structuredClone(upload)] },
                    { id: 'owner', title: 'Older durable owner', model: 'text', messages: [] },
                ];
            }
            async save(chat) {
                writes.push(app.cloneData(chat));
                return [];
            }
        };
        t.after(() => { ChatHistory.Store = Store; });
        app.chatHistoryPending = true;
        await app.loadChatHistory();
        assert.equal(attempts, 1, 'Initialization must not retry itself');
        assert.ok(app.chatHistoryPending);
        app.chatHistory.push({
            id: 'owner', title: 'Early live context', model: 'text',
            messages: [structuredClone(upload)],
        });
        app.historyDirty.set('owner', { version: 1, mode: 'save', flight: null });
        if (trigger === 'load') await app.loadChatHistory();
        else await app.saveChatHistory(['owner']);
        await app.saveChatHistory();

        assert.equal(attempts, 2);
        assert.equal(app.chatHistoryPending, false);
        assert.deepEqual(writes.map(chat => chat.id), ['owner']);
        assert.equal(app.chatHistory.find(chat => chat.id === 'owner').title, 'Early live context');
        assert.equal(app.chatHistory.find(chat => chat.id === 'durable').messages[0].content[1].image_url.url, png);
        assert.equal(app.historyDirty.size, 0);
        await app.loadChatHistory();
        assert.equal(attempts, 2, 'Successful initialization should remain loaded');
    });
}

test('import does not report success while history initialization is still failed', async t => {
    const app = appFixture();
    const archive = archiveFixture(t, app);
    app.chatHistoryPending = true;

    await archive.import([{ id: 'imported', title: 'Image', messages: [structuredClone(upload)] }]);

    assert.ok(!archive.alerts.some(message => message.startsWith('chat.import_success')));
    assert.equal(app.savedChats.size, 0);
    assert.equal(app.chatHistory.length, 0);
});

for (const outcome of ['saved', 'skipped', 'failed']) {
    test(`imported replacements use dirty ownership when storage reports ${outcome}`, async t => {
        const app = appFixture();
        app.saveCurrentChat('owner');
        await app.saveChatHistory();
        const prior = app.cloneData(app.chatHistory[0]);
        const archive = archiveFixture(t, app);
        const imported = {
            ...prior, title: 'Imported', updatedAt: '2099-01-01T00:00:00.000Z',
            messages: [{ id: 'generated', role: 'tool', content: 'Imported image', _images: [png] }],
        };
        const writes = [];
        let failing = outcome === 'failed';
        app.reportHistoryError = () => {};
        app.historyStore.save = async (chat, options) => {
            writes.push({ chat: app.cloneData(chat), replace: options.replace });
            if (failing) throw new Error('Storage unavailable');
            if (outcome === 'skipped') return false;
            app.savedChats.set(chat.id, app.cloneData(chat));
            return [];
        };

        await archive.import([imported]);

        assert.equal(writes.length, 1);
        assert.equal(writes[0].replace, true);
        if (outcome === 'skipped') {
            assert.deepEqual(app.cloneData(app.chatHistory[0]), prior);
            assert.equal(app.historyDirty.size, 0);
            assert.equal(archive.alerts.length, 1);
        } else {
            assert.equal(app.chatHistory[0].title, 'Imported');
            assert.equal(app.chatHistory[0].messages[0]._images[0], png);
            if (outcome === 'failed') {
                assert.equal(archive.alerts.length, 0);
                assert.equal(app.historyDirty.get('owner').mode, 'replace');
                failing = false;
                await app.saveChatHistory();

                assert.equal(writes.length, 2);
                assert.equal(writes[1].replace, true);
            }
            assert.equal(app.savedChats.get('owner').messages[0]._images[0], png);
            assert.equal(app.historyDirty.size, 0);
        }
    });
}

test('a skipped import cannot restore its prior row over a newer mutation', async t => {
    const app = appFixture();
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    const archive = archiveFixture(t, app);
    const imported = { ...app.cloneData(app.chatHistory[0]), title: 'Imported', updatedAt: '2099-01-01T00:00:00.000Z' };
    let finishImport;
    let started;
    const saving = new Promise(resolve => { started = resolve; });
    app.historyStore.save = () => new Promise(resolve => {
        finishImport = () => resolve(false);
        started();
    });
    const importing = archive.import([imported]);
    await saving;
    app.saveCurrentChat('owner', [{ id: 'later', role: 'user', content: 'Newer local context' }]);
    const version = app.historyDirty.get('owner').version;
    finishImport();
    await importing;

    assert.equal(app.chatHistory[0].messages[0].content, 'Newer local context');
    assert.equal(app.historyDirty.get('owner').version, version);
    assert.equal(app.historyDirty.get('owner').mode, 'save');
    app.historyStore.save = async chat => {
        app.savedChats.set(chat.id, app.cloneData(chat));
        return [];
    };
    await app.saveChatHistory();

    assert.equal(app.savedChats.get('owner').messages[0].content, 'Newer local context');
    assert.equal(app.historyDirty.size, 0);
});

test('import skips lifetimes still owned by a pending removal', async t => {
    const app = appFixture();
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    const archive = archiveFixture(t, app);
    const prior = app.cloneData(app.chatHistory[0]);
    const removal = app.beginHistoryRemoval(['owner']);

    await archive.import([{ ...prior, title: 'Do not import', updatedAt: '2099-01-01T00:00:00.000Z' }]);
    await app.finishHistoryRemoval(removal, false);

    assert.deepEqual(app.cloneData(app.chatHistory[0]), prior);
    assert.deepEqual(app.savedChats.get('owner'), prior);
    assert.equal(app.historyDirty.size, 0);
});

test('conflicted saves keep subsequent messages available in exported history', async () => {
    const app = appFixture();
    const failure = new Error('Chat history changed in another tab');
    let reported;

    app.historyStore.save = async () => { throw failure; };

    app.reportHistoryError = error => { reported = error; };
    app.saveCurrentChat('owner');
    await app.saveChatHistory();

    app.chatSessions.owner.messages.push({ id: 'later', role: 'user', content: 'Keep my local changes' });
    app.saveCurrentChat('owner');
    await app.saveChatHistory();

    assert.equal(reported, failure);
    assert.equal(JSON.parse(JSON.stringify(app.chatHistory))[0].messages.at(-1).content, 'Keep my local changes');
});

for (const tool of ['generate_image', 'mcp-image']) {
    test(`making an empty chat temporary preserves ${tool} results and the follow-up answer`, async () => {
        const app = appFixture([]);
        app.currentChatId = 'owner';
        app.messages = app.chatSessions.owner.messages;
        app.saveCurrentChat('owner');
        await app.toggleTemporaryChat();
        assert.ok(app.historyStore.deleted.has('owner'));
        let rounds = 0;
        global.fetch = async url => {
            if (url === '/v1/chat/completions') {
                return rounds++ ? sse(null) : sse([call(tool, { model: 'image', prompt: 'Temporary' })]);
            }

            if (url === '/v1/images/generations') return json({ data: [{ b64_json: 'c291cmNl' }] });

            assert.equal(url, '/v1/mcp/execute');

            return json({ content: [{ type: 'image', data: 'c291cmNl', mimeType: 'image/png' }] });
        };

        await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });

        const messages = app.chatSessions.owner.messages;
        assert.equal(rounds, 2);
        assert.equal(messages.find(msg => msg.role === 'tool')._images[0], png);
        assert.equal(messages.at(-1).content, 'Done');
        assert.ok(app.isTemporaryChat('owner'));
        assert.ok(!app.savedChats.has('owner'));
        assert.ok(!app.chatHistory.some(chat => chat.id === 'owner'));
        const finalMessages = [...messages, { id: 'final', role: 'assistant', content: 'Finalized in memory' }];
        app.saveCurrentChat('owner', finalMessages);
        assert.equal(app.chatSessions.owner.messages.at(-1).content, 'Finalized in memory');
        assert.ok(!app.savedChats.has('owner'));
    });
}

test('capacity failures keep generated images and the follow-up answer available for export', async () => {
    const app = appFixture();
    app.currentChatId = 'owner';
    const failure = new Error('Chat history is full');
    let reported;
    let rounds = 0;

    app.historyStore.save = async () => { throw failure; };

    app.reportHistoryError = error => { reported = error; };

    global.fetch = async url => {
        if (url === '/v1/chat/completions') {
            return rounds++ ? sse(null) : sse([call('generate_image', { model: 'image', prompt: 'Keep this' })]);
        }

        assert.equal(url, '/v1/images/generations');

        return json({ data: [{ b64_json: 'c291cmNl' }] });
    };

    await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    await app.saveChatHistory();

    const exported = JSON.parse(JSON.stringify(app.chatHistory))[0];

    assert.equal(reported, failure);
    assert.equal(rounds, 2);
    assert.ok(!app.historyDeleted.has('owner'));
    assert.equal(exported.messages.find(msg => msg.role === 'tool')._images[0], png);
    assert.equal(exported.messages.at(-1).content, 'Done');
});

test('clearing history invalidates an import still reading its file', async t => {
    const app = appFixture();
    const Reader = global.FileReader;
    let reader;
    global.FileReader = class {
        constructor() { reader = this; }
        readAsText() {}
    };
    t.after(() => { global.FileReader = Reader; });
    app.historyStore.generation = 0;
    app.loadChatHistory = async () => {};
    app.importChats({ target: { files: [{}], value: 'archive.json' } });
    app.historyStore.generation++;

    await reader.onload({ target: { result: JSON.stringify([
        { id: 'late-import', messages: [] },
    ]) } });

    assert.equal(app.chatHistory.length, 0);
    assert.equal(app.savedChats.size, 0);
});

test('Markdown image references render as escaped text instead of broken URLs', () => {
    const ref = images.reference('owner', 'result', 'tool', 0);
    const renderer = markdownOptions[0].renderer;
    for (const kind of ['image', 'link']) {
        assert.equal(renderer[kind](ref), `<code>${ref}</code>`);
        assert.equal(renderer[kind]('image:<img src=x onerror=alert(1)>&'), '<code>image:&lt;img src=x onerror=alert(1)&gt;&amp;</code>');
        assert.equal(renderer[kind]('https://example.com/image.png'), false);
        assert.equal(renderer[kind]('/admin/static/image.png'), false);
        assert.equal(renderer[kind]('data:image/png;base64,c291cmNl'), false);
    }
});

test('hidden card bindings render image uploads and non-web tool results without throwing', () => {
    const app = appFixture();
    const bindings = [...template.matchAll(/x-html="(renderMarkdown\([^"]+\))"/g)]
        .map(match => match[1])
        .filter(expression => expression.includes('msg.'));
    const webQuery = template.match(/x-show="msg\._web\?\.query"[\s\S]*?x-text="([^"]+)"/)[1];
    assert.ok(bindings.length > 0);

    for (const msg of [upload, { role: 'tool', content: 'Tool result' }]) {
        const scope = {
            msg,
            renderMarkdown: app.renderMarkdown.bind(app),
            getTextContent: app.getTextContent.bind(app),
        };
        for (const expression of bindings) {
            const html = vm.runInNewContext(expression, scope);
            assert.equal(html, expression.includes('msg.content')
                ? `<p>${app.getTextContent(msg.content)}</p>` : '');
        }
        assert.equal(vm.runInNewContext(webQuery, scope), '“”');
    }
});

test('discovery uses canonical IDs first, deduplicates aliases and never guesses tasks', async () => {
    const requests = [];
    global.fetch = async (url, init) => {
        requests.push(url);
        assert.equal(init.headers.Authorization, 'Bearer test-key');
        return url === '/v1/models'
            ? json({ data: ['a', 'alias', 'alias', 'b', 'unknown', 'text'].map(id => ({ id })) })
            : json({ models: [
                { id: 'a', model_alias: 'b', engine_type: 'image', tasks: ['generation', 'generation'], image_metadata: { default_steps: 28 } },
                { id: 'b', model_alias: 'alias', model_type: 'image', tasks: [], image_metadata: { tasks: ['edit'], default_guidance: 1 } },
                { id: 'unknown', model_type: 'image' },
                { id: 'text', model_type: 'llm' },
            ] });
    };
    const models = await images.discover(controllerScope());
    assert.deepEqual(models, [
        { id: 'a', aliases: [], tasks: ['generation'], defaults: { default_steps: 28 } },
        { id: 'b', aliases: ['alias'], tasks: ['edit'], defaults: { default_guidance: 1 } },
        { id: 'unknown', aliases: [], tasks: [], defaults: {} },
    ]);
    assert.deepEqual(requests, ['/v1/models', '/v1/models/status']);
    global.fetch = async () => new Response('{"detail":"Denied"}', { status: 401 });
    await assert.rejects(images.discover(controllerScope()), /\/v1\/models: Denied/);
});

test('image definitions always win MCP collisions while web toggling stays independent', () => {
    const app = appFixture();
    app.mcpTools = ['generate_image', 'edit_image', 'list_image_models', 'web_search', 'other']
        .map(name => ({ type: 'function', function: { name, description: 'MCP' } }));
    let tools = app.activeTools();
    assert.equal(tools.length, 5);
    assert.notEqual(tools.find(t => t.function.name === 'generate_image').function.description, 'MCP');
    assert.equal(tools.find(t => t.function.name === 'web_search').function.description, 'MCP');
    app.chatSettings.webSearchEnabled = true;
    tools = app.activeTools();
    assert.equal(tools.length, 6);
    assert.notEqual(tools.find(t => t.function.name === 'web_search').function.description, 'MCP');
});

test('image tools follow the actual request model tool support, including aliases and overrides', async () => {
    const app = appFixture();
    global.fetch = async () => json({ models: [
        { id: 'plain', model_alias: 'plain-alias', model_type: 'vlm', supports_tool_calling: false },
        { id: 'capable', model_type: 'vlm', supports_tool_calling: true },
        { id: 'text', model_type: 'llm' },
    ] });
    await app.fetchModelTypes();
    app.aliasToGateway = { 'plain-alias': 'plain' };
    const messages = app.chatSessions.owner.messages;
    const context = { chatId: 'owner', model: 'plain-alias', systemPrompt: '' };
    app.currentModel = 'capable';
    assert.equal(Object.hasOwn(app.buildChatCompletionBody(messages, context), 'tools'), false);
    assert.equal(app.buildChatCompletionBody(messages, { ...context, _modelOverride: 'capable' }).tools.length, 3);
    assert.equal(app.buildChatCompletionBody(messages, { ...context, model: 'text' }).tools.length, 3);
    assert.equal(Object.hasOwn(app.buildChatCompletionBody(messages, {
        ...context, model: 'capable', _modelOverride: 'plain-alias',
    }), 'tools'), false);

    app.mcpTools = [call('existing_mcp_tool', {})];
    assert.deepEqual(app.activeTools('plain'), app.mcpTools);
});

test('new prompts refresh disabled tool support after a diffusion model loads in the same tab', async () => {
    for (const supportsTools of [true, false]) {
        const app = appFixture();
        app.aliasToGateway = { 'diffusion-alias': 'diffusion' };
        app.modelToolSupportMap = { diffusion: false };
        let loaded = false;
        let statusRequests = 0;
        let completions = 0;
        global.fetch = async (url, init) => {
            if (url === '/v1/models/status') {
                statusRequests++;
                return json({ models: [{
                    id: 'diffusion', model_type: 'vlm',
                    supports_tool_calling: loaded && supportsTools,
                }] });
            }
            assert.equal(url, '/v1/chat/completions');
            const body = JSON.parse(init.body);
            assert.equal(body.model, 'diffusion');
            if (completions === 0 || !supportsTools) {
                assert.equal(Object.hasOwn(body, 'tools'), false);
            } else {
                assert.deepEqual(body.tools.map(tool => tool.function.name), [
                    'list_image_models', 'generate_image', 'edit_image',
                ]);
            }
            completions++;
            loaded = true;
            return sse(null);
        };
        const context = { chatId: 'owner', model: 'text', _modelOverride: 'diffusion-alias', systemPrompt: '' };
        await app.streamResponse(context);
        app.chatSessions.owner.messages.push({ id: 'follow-up', role: 'user', content: 'Which image models can I use?' });
        await app.streamResponse(context);
        assert.equal(completions, 2);
        assert.equal(statusRequests, 2);
    }
});

test('Stop during a tool-support refresh prevents the chat completion request', async () => {
    const app = appFixture();
    app.modelToolSupportMap = { text: false };
    const requests = [];
    global.fetch = async (url, init) => {
        requests.push(url);
        assert.equal(url, '/v1/models/status');
        return {
            ok: true,
            json: async () => {
                app.stopChatStreaming('owner');
                assert.equal(init.signal.aborted, true);
                return { models: [{ id: 'text', supports_tool_calling: true }] };
            },
        };
    };
    await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    assert.deepEqual(requests, ['/v1/models/status']);
    assert.equal(app.getStreamSession('owner').isStreaming, false);
});

test('branching rebases copied image references without changing the parent or accepting foreign sources', async () => {
    const uploadRef = images.reference('owner', upload.id, 'upload', 1);
    const toolRef = images.reference('owner', 'result', 'tool', 0);
    const foreignRef = images.reference('foreign', 'result', 'tool', 0);
    const excludedRef = images.reference('owner', 'later', 'tool', 0);
    const messages = [
        structuredClone(upload),
        { id: 'call', role: 'assistant', content: `Editing ${uploadRef}`,
            tool_calls: [call('edit_image', { model: 'image', prompt: 'Edit', image_refs: [uploadRef] })] },
        { id: 'result', role: 'tool', content: JSON.stringify({ outputs: [{ image_ref: toolRef }] }),
            _images: ['data:image/png;base64,bmV3'] },
        { id: 'reuse', role: 'user', content: [
            { type: 'text', text: `Reuse ${toolRef}. Keep ${foreignRef} and ${excludedRef}.` },
        ] },
        { id: 'later', role: 'tool', content: 'Not copied', _images: [png] },
    ];
    const original = JSON.stringify(messages);
    const app = appFixture(messages);
    app.currentChatId = 'owner';
    app.loadChat = chatId => { app.currentChatId = chatId; };
    app.branchChat(3);

    const branchId = app.currentChatId;
    const branch = app.chatSessions[branchId].messages;
    const branchUpload = images.reference(branchId, upload.id, 'upload', 1);
    const branchTool = images.reference(branchId, 'result', 'tool', 0);
    assert.equal(JSON.stringify(messages), original);
    assert.equal(branch.length, 4);
    assert.equal(branch[1].content, `Editing ${branchUpload}`);
    assert.deepEqual(JSON.parse(branch[1].tool_calls[0].function.arguments).image_refs, [branchUpload]);
    assert.equal(JSON.parse(branch[2].content).outputs[0].image_ref, branchTool);
    assert.equal(branch[3].content[0].text, `Reuse ${branchTool}. Keep ${foreignRef} and ${excludedRef}.`);
    app.reuseToolImage(branch[2], 0);
    assert.equal(app.inputMessage, branchTool);
    assert.ok(JSON.stringify(app.chatHistory[0]).includes('base64'));

    global.fetch = async (url, init) => {
        assert.equal(url, '/v1/images/edits');
        assert.deepEqual(await Promise.all(init.body.getAll('image').map(file => file.text())), ['source', 'new']);
        return json({ data: [{ b64_json: 'ZWRpdA==' }] });
    };
    const scope = { ...controllerScope(branch), chatId: branchId };
    await images.execute('edit_image', {
        model: 'image', prompt: 'Edit copied sources', image_refs: [branchUpload, branchTool],
    }, scope);
    for (const ref of [uploadRef, toolRef, foreignRef, excludedRef]) {
        await assert.rejects(images.execute('edit_image', {
            model: 'image', prompt: 'Unavailable source', image_refs: [ref],
        }, scope), /not in this chat/);
    }
});

test('generation preserves unspecified defaults and edit resolves multipart bytes without altering sources', async () => {
    const scope = controllerScope();
    global.fetch = async (url, init) => {
        assert.equal(url, '/v1/images/generations');
        assert.deepEqual(JSON.parse(init.body), { model: 'image-model', prompt: 'A cat', response_format: 'b64_json' });
        return json({ data: [{ b64_json: 'bmV3', output_format: 'webp', steps: 8, guidance: 1 }] });
    };
    const generated = await images.execute('generate_image', { model: 'image-model', prompt: 'A cat' }, scope);
    assert.equal(generated.images[0], 'data:image/webp;base64,bmV3');
    const ref = JSON.parse(generated.content).outputs[0].image_ref;
    assert.equal(ref, images.reference('owner', 'reserved', 'tool', 0));
    scope.messages.push({ id: 'reserved', role: 'tool', content: generated.content, _images: generated.images });
    global.fetch = async (url, init) => {
        assert.equal(url, '/v1/images/edits');
        assert.equal(init.headers['Content-Type'], undefined);
        assert.deepEqual([...init.body.keys()], ['model', 'prompt', 'response_format', 'steps', 'image_strength', 'image', 'image']);
        assert.equal(init.body.get('image_strength'), '0.5');
        const files = init.body.getAll('image');
        assert.deepEqual(await Promise.all(files.map(file => file.text())), ['source', 'new']);
        assert.deepEqual(files.map(file => file.type), ['image/png', 'image/webp']);
        return json({ data: [{ b64_json: 'ZWRpdA==' }] });
    };
    const edited = await images.execute('edit_image', {
        model: 'image-model', prompt: 'Make blue', steps: 4, image_strength: 0.5,
        image_refs: [images.reference('owner', 'upload-1', 'upload', 1), ref],
    }, { ...scope, resultMessageId: 'edit-result' });
    assert.equal(JSON.parse(edited.content).outputs[0].image_ref, images.reference('owner', 'edit-result', 'tool', 0));
    assert.equal(upload.content[1].image_url.url, png);
    assert.equal(scope.messages[1]._images[0], generated.images[0]);
});

test('request projection follows the actual model override and preserves VLM inputs only', () => {
    const app = appFixture();
    app.aliasToGateway = { 'vision-alias': 'vision', 'text-alias': 'text' };
    const messages = app.chatSessions.owner.messages;
    messages.push({ id: 'mcp-image', role: 'tool', content: 'Image from MCP', _ui: false, _images: [png] });
    const context = { chatId: 'owner', model: 'vision-alias', systemPrompt: '', _modelOverride: 'text-alias' };
    const text = app.buildChatCompletionBody(messages, context);
    assert.equal(text.model, 'text');
    assert.ok(!JSON.stringify(text.messages).includes('base64'));
    assert.ok(JSON.stringify(text.messages).includes(images.reference('owner', 'upload-1', 'upload', 1)));
    assert.ok(JSON.stringify(text.messages).includes(images.reference('owner', 'mcp-image', 'tool', 0)));
    const vision = app.buildChatCompletionBody(messages, { ...context, _modelOverride: 'vision-alias' });
    assert.equal(vision.model, 'vision');
    assert.equal(vision.messages[0].content.find(part => part.type === 'image_url').image_url.url, png);
    assert.ok(!JSON.stringify(vision.messages[1]).includes('base64'));
    const saved = messages.map(msg => app.messageForStorage(msg));
    assert.equal(saved[1]._images[0], png);
    assert.equal(saved[0].id, messages[0].id);
    assert.ok(!JSON.stringify(app.buildChatCompletionBody(saved, context).messages).includes('base64'));
});

test('missing, foreign, excluded-variant and stripped image refs fail at the source', async () => {
    const app = appFixture();
    const variants = [
        structuredClone(upload),
        { id: 'round-old', role: 'assistant', tool_calls: [], _ui: false },
        { id: 'old-image', role: 'tool', _images: [png], _ui: false },
        { id: 'old-final', role: 'assistant', content: 'Old' },
        { id: 'round-new', role: 'assistant', tool_calls: [], _ui: false },
        { id: 'new-image', role: 'tool', _images: [png], _ui: false },
        { id: 'new-final', role: 'assistant', content: 'New' },
    ];
    variants[0]._activeVariantId = 'new-final';
    const selected = app.getApiTurnMessages(variants);
    assert.ok(!selected.some(msg => msg.id === 'old-image'));
    const scope = controllerScope(selected);
    let requests = 0;
    global.fetch = async () => { requests++; return json({ data: [] }); };
    for (const ref of [
        images.reference('other-chat', 'upload-1', 'upload', 1),
        images.reference('owner', 'old-image', 'tool', 0),
        images.reference('owner', 'nonexistent', 'tool', 0),
    ]) {
        await assert.rejects(images.execute('edit_image', { model: 'm', prompt: 'p', image_refs: [ref] }, scope), /not in this chat's selected history/);
    }
    const stripped = structuredClone(variants);
    stripped[0].content[1].image_url.url = '';
    for (const msg of stripped.filter(msg => msg._images)) {
        msg._imagesStripped = msg._images.length;
        delete msg._images;
    }
    const ref = images.reference('owner', 'upload-1', 'upload', 1);
    await assert.rejects(images.execute('edit_image', { model: 'm', prompt: 'p', image_refs: [ref] }, controllerScope(stripped)), /unavailable after reload or export/);
    const mcpRef = images.reference('owner', 'new-image', 'tool', 0);
    await assert.rejects(images.execute('edit_image', { model: 'm', prompt: 'p', image_refs: [mcpRef] }, controllerScope(stripped)), /unavailable after reload or export/);
    assert.equal(requests, 0);
});

test('streamResponse performs discovery, generation, edit and final synthesis in the owning background chat', async () => {
    const app = appFixture();
    const requests = [];
    let round = 0;
    let generationRef;
    global.fetch = async (url, init) => {
        requests.push(url);
        if (url === '/v1/chat/completions') {
            const body = JSON.parse(init.body);
            assert.equal(body.model, 'text');
            assert.ok(!JSON.stringify(body.messages).includes('base64'));
            if (round++ === 0) return sse([call('list_image_models', {})]);
            if (round === 2) {
                assert.ok(body.messages.some(msg => msg.role === 'tool' && msg.content.includes('"generation"')));
                return sse([call('generate_image', { model: 'image', prompt: 'A cat' })]);
            }
            if (round === 3) {
                const result = app.chatSessions.owner.messages.find(msg => msg._images?.length);
                generationRef = JSON.parse(result.content).outputs[0].image_ref;
                assert.equal(generationRef, images.reference('owner', result.id, 'tool', 0));
                assert.ok(body.messages.some(msg => msg.role === 'tool' && msg.content.includes(generationRef)));
                return sse([call('edit_image', { model: 'image', prompt: 'Blue cat', image_refs: [generationRef] })]);
            }
            return sse(null);
        }
        if (url === '/v1/models') return json({ data: [{ id: 'image' }] });
        if (url === '/v1/models/status') return json({ models: [{ id: 'image', model_type: 'image', tasks: ['generation', 'edit'] }] });
        if (url === '/v1/images/generations') return json({ data: [{ b64_json: 'Z2Vu' }] });
        assert.equal(url, '/v1/images/edits');
        assert.equal(await init.body.get('image').text(), 'gen');
        return json({ data: [{ b64_json: 'ZWRpdA==' }] });
    };
    await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    assert.equal(round, 4);
    assert.equal(requests.filter(url => url === '/v1/mcp/execute').length, 0);
    assert.equal(app.currentChatId, 'foreground');
    assert.equal(app.messages.length, 0);
    const messages = app.chatSessions.owner.messages;
    assert.equal(messages.filter(msg => msg._images?.length).length, 2);
    assert.equal(messages.at(-1).content, 'Done');
    assert.equal(new Set(messages.map(msg => msg.id)).size, messages.length);
    assert.ok(JSON.stringify(app.savedChats.get('owner')).includes('base64'));
    assert.equal(app.getStreamSession('owner').isStreaming, false);
});

test('parallel result IDs remain distinct and unfinished sibling outputs are unavailable', async () => {
    const app = appFixture();
    let round = 0;
    let generations = 0;
    global.fetch = async (url, init) => {
        if (url === '/v1/chat/completions') {
            if (round++) return sse(null);
            return sse([
                call('generate_image', { model: 'image', prompt: 'One' }, 'g1'),
                call('generate_image', { model: 'image', prompt: 'Two' }, 'g2'),
                call('edit_image', { model: 'image', prompt: 'No source', image_refs: ['image:owner:not-finished:tool:0'] }, 'bad'),
            ]);
        }
        assert.equal(url, '/v1/images/generations');
        generations++;
        return json({ data: [{ b64_json: 'YQ==' }] });
    };
    await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    const results = app.chatSessions.owner.messages.filter(msg => msg.role === 'tool');
    assert.equal(generations, 2);
    assert.notEqual(results[0].id, results[1].id);
    results.slice(0, 2).forEach(msg => assert.equal(
        JSON.parse(msg.content).outputs[0].image_ref, images.reference('owner', msg.id, 'tool', 0)
    ));
    assert.match(results[2].content, /not in this chat's selected history/);
});

test('Stop remains active through delayed response bodies and blocks registration and recursion', async () => {
    const app = appFixture();
    let round = 0;
    let bodyStarted;
    const started = new Promise(resolve => { bodyStarted = resolve; });
    let releaseBody;
    let bodySignal;
    global.fetch = async (url, init) => {
        if (url === '/v1/chat/completions') {
            round++;
            return sse([call('generate_image', { model: 'image', prompt: 'Delayed' })]);
        }
        assert.equal(url, '/v1/images/generations');
        bodySignal = init.signal;
        return {
            ok: true,
            json: () => new Promise(resolve => {
                releaseBody = () => resolve({ data: [{ b64_json: 'bGF0ZQ==' }] });
                bodyStarted();
            }),
        };
    };
    const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    await started;
    assert.equal(bodySignal.aborted, false);
    app.stopChatStreaming('owner');
    assert.equal(bodySignal.aborted, true);
    releaseBody();
    await pending;
    assert.equal(round, 1);
    const messages = app.chatSessions.owner.messages;
    const toolResults = messages.filter(msg => msg.role === 'tool');
    assert.equal(toolResults.length, 1);
    assert.equal(toolResults[0].tool_call_id, 'generate_image');
    assert.match(toolResults[0].content, /stopped by user/);
    assert.equal(toolResults[0]._images, undefined);
    const nextRequest = app.buildChatCompletionBody(messages, { chatId: 'owner', model: 'text', systemPrompt: '' });
    const calls = nextRequest.messages.flatMap(msg => msg.tool_calls || []);
    assert.deepEqual(calls.map(tc => tc.id), nextRequest.messages.flatMap(msg => msg.role === 'tool' ? [msg.tool_call_id] : []));
    assert.ok(!JSON.stringify(app.savedChats.get('owner')).includes('bGF0ZQ=='));
    assert.equal(app.getStreamSession('owner').isStreaming, false);
});

for (const action of ['delete', 'clear']) {
    test(`${action} prevents delayed tool finalizers from resurrecting chat history`, async () => {
        const app = appFixture();
        let bodyStarted;
        const started = new Promise(resolve => { bodyStarted = resolve; });
        let releaseBody;
        global.fetch = async url => {
            if (url === '/v1/chat/completions') {
                return sse([call('generate_image', { model: 'image', prompt: 'Delayed' })]);
            }
            assert.equal(url, '/v1/images/generations');
            return {
                ok: true,
                json: () => new Promise(resolve => {
                    releaseBody = () => resolve({ data: [{ b64_json: 'bGF0ZQ==' }] });
                    bodyStarted();
                }),
            };
        };
        const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
        await started;
        const retained = app.chatSessions.owner;
        if (action === 'delete') {
            await app.deleteChat('owner', { skipConfirm: true });
        } else {
            global.confirm = () => true;
            app.startNewChat = async () => {};
            await app.clearAllHistory();
        }
        releaseBody();
        await pending;
        app.saveCurrentChat('owner', retained.messages, retained.model, retained.systemPrompt);
        assert.ok(!app.chatHistory.some(chat => chat.id === 'owner'));
        assert.ok(!app.savedChats.has('owner'));
    });
}

for (const action of ['delete', 'clear']) {
    test(`failed ${action} keeps new messages exportable without reviving cancelled image work`, async t => {
        const app = appFixture();
        const confirm = global.confirm;
        global.confirm = () => true;
        t.after(() => { global.confirm = confirm; });
        app.historyStore[action] = async () => { throw new Error('Storage unavailable'); };
        app.reportHistoryError = () => {};

        let bodyStarted;
        const started = new Promise(resolve => { bodyStarted = resolve; });
        let releaseBody;
        let rounds = 0;
        global.fetch = async url => {
            if (url === '/v1/chat/completions') {
                rounds++;

                return sse([call('generate_image', { model: 'image', prompt: 'Delayed' })]);
            }

            assert.equal(url, '/v1/images/generations');

            return {
                ok: true,
                json: () => new Promise(resolve => {
                    releaseBody = () => resolve({ data: [{ b64_json: 'bGF0ZQ==' }] });
                    bodyStarted();
                }),
            };
        };
        const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
        await started;

        if (action === 'delete') await app.deleteChat('owner', { skipConfirm: true });
        else await app.clearAllHistory();

        releaseBody();
        await pending;

        assert.equal(rounds, 1);
        const messages = app.chatSessions.owner.messages;
        assert.ok(messages.filter(msg => msg.role === 'tool').every(msg => !msg._images?.length));

        messages.push({ id: 'later', role: 'user', content: 'Continue after failure' });
        app.saveCurrentChat('owner');
        assert.equal(JSON.parse(JSON.stringify(app.chatHistory))[0].messages.at(-1).content, 'Continue after failure');
        assert.ok(!JSON.stringify(app.chatHistory).includes('bGF0ZQ=='));
    });
}

for (const action of ['delete', 'clear']) {
    test(`pending ${action} rejects image results but retains matched cancellation context on failure`, async t => {
        const app = appFixture();
        const confirm = global.confirm;
        global.confirm = () => true;
        t.after(() => { global.confirm = confirm; });
        let bodyStarted;
        const started = new Promise(resolve => { bodyStarted = resolve; });
        let releaseBody;
        let rounds = 0;
        global.fetch = async url => {
            if (url === '/v1/chat/completions') {
                rounds++;
                return sse([call('generate_image', { model: 'image', prompt: 'Canceled' })]);
            }
            return {
                ok: true,
                json: () => new Promise(resolve => {
                    releaseBody = () => resolve({ data: [{ b64_json: 'bGF0ZQ==' }] });
                    bodyStarted();
                }),
            };
        };
        const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
        await started;
        let rejectRemoval;
        app.historyStore[action] = () => new Promise((resolve, reject) => { rejectRemoval = reject; });
        app.reportHistoryError = () => {};
        const removing = action === 'delete'
            ? app.deleteChat('owner', { skipConfirm: true })
            : app.clearAllHistory();
        releaseBody();
        await pending;

        assert.ok(app.isPendingHistoryRemoval('owner'));
        rejectRemoval(new Error('Storage unavailable'));
        await removing;
        const exported = app.chatHistory.find(chat => chat.id === 'owner');
        const body = app.buildChatCompletionBody(exported.messages, { chatId: 'owner', model: 'text', systemPrompt: '' });
        const calls = body.messages.flatMap(message => message.tool_calls || []);
        const results = body.messages.filter(message => message.role === 'tool');

        assert.equal(rounds, 1);
        assert.equal(calls.length, 1);
        assert.deepEqual(calls.map(tc => tc.id), results.map(result => result.tool_call_id));
        assert.match(results[0].content, /stopped by user/);
        assert.ok(!JSON.stringify(exported).includes('bGF0ZQ=='));
        assert.equal(exported.messages[0].content[1].image_url.url, png);
        assert.deepEqual(app.cloneData(exported), app.savedChats.get('owner'));
    });
}

for (const action of ['delete', 'clear']) {
    test(`failed ${action} retains finalized partial text and reasoning in exports`, async t => {
        const app = appFixture();
        app.saveCurrentChat('owner');
        await app.saveChatHistory();
        let tokensSent;
        const sent = new Promise(resolve => { tokensSent = resolve; });
        const confirm = global.confirm;
        global.confirm = () => true;
        t.after(() => { global.confirm = confirm; });
        global.fetch = async () => new Response(new ReadableStream({
            start(controller) {
                controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({
                    choices: [{ delta: { content: 'Partial answer', reasoning_content: 'Partial reasoning' } }],
                })}\n\n`));
                tokensSent();
            },
        }));
        const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
        await sent;
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(app.getStreamSession('owner').streamingContent, 'Partial answer');
        let rejectRemoval;

        app.historyStore[action] = () => new Promise((resolve, reject) => { rejectRemoval = reject; });

        app.reportHistoryError = () => {};

        const removing = action === 'delete'
            ? app.deleteChat('owner', { skipConfirm: true })
            : app.clearAllHistory();

        await pending;
        rejectRemoval(new Error('Storage unavailable'));
        await removing;

        const live = app.chatSessions.owner.messages.at(-1);
        const exported = JSON.parse(JSON.stringify(app.chatHistory)).find(chat => chat.id === 'owner');
        assert.equal(live.content, 'Partial answer');
        assert.equal(live.reasoning_content, 'Partial reasoning');
        assert.equal(exported.messages.at(-1).content, 'Partial answer');
        assert.equal(exported.messages.at(-1).reasoning_content, 'Partial reasoning');
        assert.equal(exported.messages[0].content[1].image_url.url, png);
    });
}

test('successful clearing preserves a conversation sent while storage is pending', async t => {
    const app = appFixture();
    app.currentChatId = 'owner';
    app.messages = app.chatSessions.owner.messages;
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    const oldIds = new Set(app.savedChats.keys());
    let finishClearing;
    const confirm = global.confirm;
    global.confirm = () => true;
    t.after(() => { global.confirm = confirm; });

    app.historyStore.clear = () => new Promise(resolve => {
        finishClearing = () => {
            for (const id of oldIds) app.savedChats.delete(id);

            resolve();
        };
    });

    app.ensureSessionModelSettings = async () => {};

    app.saveModelSettingsForModel = () => {};

    app.forceScrollToBottom = () => {};
    const clearing = app.clearAllHistory();
    await app.startNewChat();
    const newId = app.currentChatId;
    app.inputMessage = 'Sent during pending clear';
    global.fetch = async () => sse(null);
    await app.sendMessage();

    finishClearing();
    await clearing;
    await app.saveChatHistory();

    assert.equal(app.currentChatId, newId);
    assert.equal(app.chatSessions[newId].messages[0].content, 'Sent during pending clear');
    assert.equal(app.chatHistory.find(chat => chat.id === newId).messages.at(-1).content, 'Done');
    assert.equal(app.savedChats.get(newId).messages.at(-1).content, 'Done');
    assert.ok(!app.savedChats.has('owner'));
    assert.ok(!app.chatSessions.owner);
});

test('successful clearing preserves a newer composer, uploads, and active stream', async t => {
    const app = appFixture();
    app.currentChatId = 'owner';
    app.messages = app.chatSessions.owner.messages;
    app.saveCurrentChat('owner');
    await app.saveChatHistory();
    const confirm = global.confirm;
    global.confirm = () => true;
    t.after(() => { global.confirm = confirm; });
    let finishClearing;
    app.historyStore.clear = () => new Promise(resolve => { finishClearing = resolve; });
    const clearing = app.clearAllHistory();
    await app.startNewChat();
    const id = app.currentChatId;
    const stream = app.getStreamSession(id, true);
    stream.isStreaming = true;
    stream.streamingContent = 'New response in progress';
    stream.abortController = new AbortController();
    app.inputMessage = 'Unsent draft';
    app.uploadImages = [{ id: 'new-upload', base64: png }];
    app.uploadDocuments = [{ id: 'new-document', base64: 'document' }];
    const messages = app.messages;

    finishClearing();
    await clearing;

    assert.equal(app.currentChatId, id);
    assert.equal(app.messages, messages);
    assert.equal(app.inputMessage, 'Unsent draft');
    assert.equal(app.uploadImages[0].base64, png);
    assert.equal(app.uploadDocuments[0].base64, 'document');
    assert.equal(app.getStreamSession(id), stream);
    assert.equal(stream.isStreaming, true);
    assert.equal(stream.streamingContent, 'New response in progress');
    assert.equal(stream.abortController.signal.aborted, false);
    assert.ok(!app.chatSessions.owner);
});

for (const action of ['delete', 'clear']) {
    test(`failed ${action} persists partial context even when storage fails before stream finalization`, async t => {
        const app = appFixture();
        app.saveCurrentChat('owner');
        await app.saveChatHistory();
        const confirm = global.confirm;
        global.confirm = () => true;
        t.after(() => { global.confirm = confirm; });
        const chunk = new TextEncoder().encode(`data: ${JSON.stringify({
            choices: [{ delta: { content: 'Retained answer', reasoning_content: 'Retained reasoning' } }],
        })}\n\n`);
        let finishReading;
        let waiting;
        const reading = new Promise(resolve => { waiting = resolve; });
        let first = true;
        const reader = {
            read() {
                if (first) {
                    first = false;
                    return Promise.resolve({ value: chunk, done: false });
                }
                return new Promise((resolve, reject) => {
                    finishReading = () => reject(new DOMException('Stopped', 'AbortError'));
                    waiting();
                });
            },
            cancel: () => Promise.resolve(),
        };
        global.fetch = async () => ({ ok: true, body: { getReader: () => reader } });
        const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
        await reading;
        app.historyStore[action] = async () => { throw new Error('Storage unavailable'); };
        app.reportHistoryError = () => {};
        const stream = app.getStreamSession('owner');
        const controller = stream.abortController;

        if (action === 'delete') await app.deleteChat('owner', { skipConfirm: true });
        else await app.clearAllHistory();

        assert.equal(controller.signal.aborted, true);
        assert.equal(stream.abortController, controller);
        assert.equal(stream.streamingContent, 'Retained answer');
        assert.equal(stream.streamingThinking, 'Retained reasoning');
        assert.equal(stream.isStreaming, true);
        finishReading();
        await pending;
        await app.saveChatHistory();
        const saved = app.savedChats.get('owner');
        const exported = app.chatHistory.find(chat => chat.id === 'owner');

        assert.equal(saved.messages.at(-1).content, 'Retained answer');
        assert.equal(saved.messages.at(-1).reasoning_content, 'Retained reasoning');
        assert.deepEqual(app.cloneData(exported), saved);
        assert.equal(stream.isStreaming, false);
    });
}

test('renaming a chat does not serialize unchanged image histories', async () => {
    const app = appFixture();
    app.chatSessions.unchanged = app.cloneData(app.chatSessions.owner);
    app.saveCurrentChat('owner');
    app.saveCurrentChat('unchanged');
    await app.saveChatHistory();
    const unchanged = app.chatHistory.find(chat => chat.id === 'unchanged');
    const stringify = JSON.stringify;
    let serialized = 0;

    JSON.stringify = (value, ...args) => {
        if (value === unchanged) serialized++;

        return stringify(value, ...args);
    };
    try {
        app.renameDraft = 'Renamed';
        app.saveRenamedChat('owner');
        await app.saveChatHistory();
    } finally {
        JSON.stringify = stringify;
    }

    assert.equal(serialized, 0);
    assert.equal(app.savedChats.get('owner').title, 'Renamed');
    assert.equal(app.savedChats.get('unchanged').messages[0].content[1].image_url.url, png);
});

test('Stop preserves text already streamed before any tool call', async () => {
    const app = appFixture();
    let tokensSent;
    const sent = new Promise(resolve => { tokensSent = resolve; });
    global.fetch = async () => new Response(new ReadableStream({
        start(controller) {
            controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({
                choices: [{ delta: { content: 'Partial answer' } }],
            })}\n\n`));
            tokensSent();
        },
    }));
    const pending = app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    await sent;
    await new Promise(resolve => setImmediate(resolve));
    app.stopChatStreaming('owner');
    await pending;
    assert.equal(app.chatSessions.owner.messages.at(-1).content, 'Partial answer');
    assert.equal(app.getStreamSession('owner').isStreaming, false);
});

test('regenerated streamed tool chains use their own image results, not the excluded variant', async () => {
    const messages = [
        structuredClone(upload),
        { id: 'old-round', role: 'assistant', tool_calls: [], _ui: false },
        { id: 'old-image', role: 'tool', content: 'Old image', _images: [png], _ui: false },
        { id: 'old-final', role: 'assistant', content: 'Old answer' },
    ];
    messages[0]._activeVariantId = 'old-final';
    const app = appFixture(messages);
    let round = 0;
    let ref;
    global.fetch = async (url, init) => {
        if (url === '/v1/chat/completions') {
            const body = JSON.parse(init.body);
            assert.ok(!JSON.stringify(body.messages).includes('old-image'));
            if (round++ === 0) return sse([call('generate_image', { model: 'image', prompt: 'New variant' })]);
            if (round === 2) {
                const result = app.chatSessions.owner.messages.filter(msg => msg._images?.length).at(-1);
                ref = images.reference('owner', result.id, 'tool', 0);
                assert.ok(body.messages.some(msg => msg.role === 'tool' && msg.content.includes(ref)));
                return sse([call('edit_image', { model: 'image', prompt: 'Edit new variant', image_refs: [ref] })]);
            }
            return sse(null);
        }
        if (url === '/v1/images/generations') return json({ data: [{ b64_json: 'bmV3' }] });
        assert.equal(url, '/v1/images/edits');
        assert.equal(await init.body.get('image').text(), 'new');
        return json({ data: [{ b64_json: 'ZWRpdA==' }] });
    };
    await app.streamResponse({
        chatId: 'owner', model: 'text', systemPrompt: '',
        _variantUserIndex: 0, _excludeVariantsAtUserIndex: 0,
    });
    assert.equal(round, 3);
    assert.equal(messages.filter(msg => msg.role === 'tool' && msg._images?.length).length, 3);
    assert.equal(messages.at(-1).content, 'Done');
    assert.ok(app.getApiTurnMessages(messages).some(msg => msg.id === messages.at(-2).id));
    assert.ok(!app.getApiTurnMessages(messages).some(msg => msg.id === 'old-image'));
});

test('image drop, paste and select work for text models but not audio-only models', () => {
    const app = appFixture();
    const files = [];
    app.loadImageFile = file => files.push(file);
    const image = { type: 'image/png' };
    app.handleDrop({ dataTransfer: { files: [image] } });
    let prevented = false;
    app.handlePaste({ clipboardData: { items: [{ type: 'image/png', getAsFile: () => image }] }, preventDefault: () => { prevented = true; } });
    const input = { files: [image], value: 'file' };
    app.handleImageSelect({ target: input });
    assert.equal(files.length, 3);
    assert.equal(prevented, true);
    assert.equal(input.value, '');
    app.currentModel = 'asr';
    app.modelTypeMap.asr = 'audio_stt';
    app.handlePaste({ clipboardData: { items: [{ type: 'image/png', getAsFile: () => image }] } });
    app.handleDrop({ dataTransfer: { files: [image] } });
    assert.equal(files.length, 3);
});

test('MCP image results can be edited and reused in the composer', async () => {
    const app = appFixture();
    const split = app.splitToolContent([{ type: 'image', mimeType: 'image/png', data: 'c291cmNl' }]);
    const mcpMessage = { id: 'mcp-source', role: 'tool', content: split.content, _images: split.images };
    global.fetch = async (url, init) => {
        assert.equal(url, '/v1/images/edits');
        assert.equal(await init.body.get('image').text(), 'source');
        return json({ data: [{ b64_json: 'ZWRpdA==' }] });
    };
    await images.execute('edit_image', {
        model: 'm', prompt: 'Edit MCP image',
        image_refs: [images.reference('owner', 'mcp-source', 'tool', 0)],
    }, controllerScope([upload, mcpMessage]));
    app.currentChatId = 'owner';
    app.reuseToolImage(mcpMessage, 0);
    assert.equal(app.inputMessage, images.reference('owner', 'mcp-source', 'tool', 0));
    app.inputMessage = 'Please edit';
    app.reuseToolImage(mcpMessage, 0);
    assert.equal(app.inputMessage, `Please edit ${images.reference('owner', 'mcp-source', 'tool', 0)}`);
});

test('tool sources use the branch sent to the model even if a different variant is selected while streaming', async () => {
    const messages = [
        structuredClone(upload),
        { id: 'old-image', role: 'tool', content: 'Old', _images: ['data:image/png;base64,b2xk'], _ui: false },
        { id: 'old-final', role: 'assistant', content: 'Old answer' },
        { id: 'selected-image', role: 'tool', content: 'Selected', _images: ['data:image/png;base64,c2VsZWN0ZWQ='], _ui: false },
        { id: 'selected-final', role: 'assistant', content: 'Selected answer' },
        { id: 'follow-up', role: 'user', content: 'Edit the selected image' },
    ];
    messages[0]._activeVariantId = 'selected-final';
    const app = appFixture(messages);
    let round = 0;
    global.fetch = async (url, init) => {
        if (url === '/v1/chat/completions') {
            if (round++) return sse(null);
            assert.ok(JSON.parse(init.body).messages.some(msg => msg.content?.includes('selected-image')));
            messages[0]._activeVariantId = 'old-final';
            return sse([call('edit_image', {
                model: 'image', prompt: 'Edit the sent source',
                image_refs: [images.reference('owner', 'selected-image', 'tool', 0)],
            })]);
        }
        assert.equal(url, '/v1/images/edits');
        assert.equal(await init.body.get('image').text(), 'selected');
        return json({ data: [{ b64_json: 'ZWRpdA==' }] });
    };
    await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    assert.equal(round, 2);
    assert.ok(messages.at(-2)._images?.length);
});

test('deadlines cover body consumption, clean up timers and retain web/MCP behavior', async t => {
    const app = appFixture();
    assert.equal(app.IMAGE_TOOL_TIMEOUT_MS, 600000);
    assert.equal(app.TOOL_TIMEOUT_MS, 30000);
    const scope = controllerScope();
    const timers = [];
    const scheduled = t.mock.method(global, 'setTimeout', fn => timers.push(fn));
    const cleared = t.mock.method(global, 'clearTimeout', () => {});

    global.fetch = async () => ({
        ok: true,
        json: async () => {
            timers.at(-1)();
            return { data: [{ b64_json: 'eA==' }] };
        },
    });
    const timeout = await app.executeChatTool(call('generate_image', { model: 'm', prompt: 'p' }), scope);
    assert.equal(timeout.error, true);
    assert.match(timeout.content, /timed out after 600s/);

    global.fetch = async (url, init) => {
        assert.equal(url, '/v1/mcp/execute');
        assert.deepEqual(JSON.parse(init.body), { tool_name: 'mcp-image', arguments: { a: 1 } });
        return json({ content: [{ type: 'image', mimeType: 'image/png', data: 'eA==' }, { type: 'text', text: 'MCP text' }] });
    };
    const mcp = await app.executeChatTool(call('mcp-image', { a: 1 }), scope);
    assert.deepEqual(mcp.images, ['data:image/png;base64,eA==']);
    assert.ok(!mcp.content.includes('eA=='));

    app.chatSettings.webSearchEnabled = true;
    global.fetch = async (url, init) => {
        assert.equal(url, '/v1/web/search');
        assert.deepEqual(JSON.parse(init.body), { query: 'hello' });
        return json({ ok: true, results: [{ title: 'Result', url: 'https://example.com' }] });
    };
    const web = await app.executeChatTool(call('web_search', { query: 'hello' }), scope);
    assert.equal(web.web.ok, true);
    assert.equal(web.web.results[0].title, 'Result');

    global.fetch = async url => json(url === '/v1/models' ? { data: [] } : { models: [] });
    const discovery = await app.executeChatTool(call('list_image_models', {}), scope);
    assert.deepEqual(JSON.parse(discovery.content), { models: [] });
    assert.deepEqual(scheduled.mock.calls.map(call => call.arguments[1]), [600000, 30000, 30000, 30000]);
    assert.deepEqual(cleared.mock.calls.map(call => call.arguments[0]), [1, 2, 3, 4]);
});

test('fallback abort composition disposes listeners after response body consumption', async t => {
    const app = appFixture();
    const scope = controllerScope();
    const nativeAny = AbortSignal.any;
    const added = t.mock.method(scope.signal, 'addEventListener');
    const removed = t.mock.method(scope.signal, 'removeEventListener');
    AbortSignal.any = undefined;
    t.after(() => { AbortSignal.any = nativeAny; });

    global.fetch = async () => ({
        ok: true,
        json: async () => {
            assert.equal(added.mock.callCount(), 1);
            assert.equal(removed.mock.callCount(), 0);
            return { data: [{ b64_json: 'eA==' }] };
        },
    });
    const result = await app.executeChatTool(call('generate_image', { model: 'm', prompt: 'p' }), scope);
    assert.equal(result.error, false);
    assert.equal(removed.mock.callCount(), 1);
});

test('malformed reserved tools never fall through to MCP and API errors retain their source', async () => {
    const app = appFixture();
    let requests = 0;
    global.fetch = async url => {
        requests++;
        assert.equal(url, '/v1/images/generations');
        return new Response('{"detail":"Model lacks generation"}', { status: 400 });
    };
    const scope = controllerScope();
    const invalid = call('edit_image', {});
    invalid.function.arguments = '{bad';
    assert.match((await app.executeChatTool(invalid, scope)).content, /Invalid image tool arguments JSON/);
    assert.match((await app.executeChatTool(call('edit_image', { model: 'm', prompt: 'p', image_refs: [] }), scope)).content, /Invalid image option image_refs/);
    assert.match((await app.executeChatTool(call('generate_image', { model: 'm', prompt: 'p', mask: 'x' }), scope)).content, /Unsupported image option mask/);
    assert.equal(requests, 0);
    const result = await app.executeChatTool(call('generate_image', { model: 'm', prompt: 'p' }), scope);
    assert.match(result.content, /\/v1\/images\/generations: Model lacks generation/);
    assert.equal(requests, 1);
});

for (const [label, content, urls, text] of [
    ['multiple images', [
        { type: 'image_url', image_url: { url: png } },
        { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,bmV3' } },
        { type: 'text', text: 'What are these?' },
    ], [png, 'data:image/jpeg;base64,bmV3'], 'What are these?'],
    ['text only', 'Hello', [], 'Hello'],
    ['image only', [{ type: 'image_url', image_url: { url: png } }], [png], ''],
    ['stripped image', [
        { type: 'image_url', image_url: { url: '' } },
        { type: 'text', text: 'Old text' },
    ], [], 'Old text'],
]) {
    test(`editing ${label} preserves available images and replaces the text`, async () => {
        const app = appFixture([{ id: 'original', role: 'user', content }]);
        app.currentChatId = 'owner';
        app.messages = app.chatSessions.owner.messages;
        const streams = [];
        app.streamResponse = async context => streams.push(context);

        assert.deepEqual(app.getImageUrls(content), urls);
        assert.equal(app.getTextContent(content), text);
        const stored = app.messageForStorage(app.messages[0]);
        assert.deepEqual(app.getImageUrls(stored.content), urls);
        assert.equal(app.getTextContent(stored.content), text);
        assert.deepEqual(app.getImageUrls(content), urls);

        app.startEdit(0);
        assert.deepEqual(app.editImages, urls);
        assert.equal(app.editContent, text);
        app.editContent = '  New description  ';
        await app.saveEdit(0);

        const edited = app.messages[0];
        assert.equal(edited.role, 'user');
        assert.deepEqual(edited.content, urls.length ? [
            ...urls.map(url => ({ type: 'image_url', image_url: { url } })),
            { type: 'text', text: 'New description' },
        ] : 'New description');
        assert.equal(app.editingIndex, null);
        assert.deepEqual(app.editImages, []);
        assert.equal(streams.length, 1);
        assert.equal(streams[0].chatId, 'owner');
    });
}

for (const [value, expected] of [[null, 10], ['bad', 10], [0, 1], [150, 100], [25, 25]]) {
    test(`tool round limit normalizes ${JSON.stringify(value)} to ${expected}`, () => {
        assert.equal(appFixture().normalizeMaxToolRounds(value), expected);
    });
}

for (const finalAnswer of [true, false]) {
    test(`the last permitted tool round ${finalAnswer ? 'allows a final answer' : 'blocks another tool execution'}`, async () => {
        const app = appFixture();
        app.chatSettings.maxToolRounds = 1;
        let completions = 0;
        let executions = 0;
        global.fetch = async url => {
            if (url === '/v1/chat/completions') {
                completions++;
                return completions === 1 || !finalAnswer
                    ? sse([call('search', { query: 'hello' })]) : sse(null);
            }
            assert.equal(url, '/v1/mcp/execute');
            executions++;
            return json({ content: 'Search result' });
        };
        await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
        assert.equal(completions, 2);
        assert.equal(executions, 1);
        const messages = app.chatSessions.owner.messages;
        assert.equal(messages.at(-1).content, finalAnswer ? 'Done' : 'chat.error.max_tool_rounds');
        const chain = app.getApiTurnMessages(messages);
        assert.equal(chain.find(msg => msg.tool_calls)?._ui, false);
        assert.equal(chain.find(msg => msg.role === 'tool').tool_call_id, 'search');
        assert.equal(app.getStreamSession('owner').isStreaming, false);
    });
}

test('streamed parallel tool calls assemble fragmented names and arguments before dispatch', async () => {
    const app = appFixture();
    const executed = [];
    let completions = 0;
    global.fetch = async (url, init) => {
        if (url === '/v1/chat/completions') {
            if (completions++) return sse(null);
            const deltas = [
                [{ id: 'first', function: { name: 'sea' } }],
                [{ index: 1, id: 'second', function: { name: 'extract', arguments: '{"urls":' } }],
                [{ index: 0, function: { name: 'rch', arguments: '{"q":' } }],
                [{ index: 1, function: { arguments: '["https://example.com"]}' } }],
                [{ index: 0, function: { arguments: '"hello"}' } }],
            ];
            return new Response(deltas.map(tool_calls => `data: ${JSON.stringify({
                choices: [{ delta: { tool_calls } }],
            })}\n\n`).join('') + 'data: [DONE]\n\n');
        }
        assert.equal(url, '/v1/mcp/execute');
        executed.push(JSON.parse(init.body));
        return json({ content: 'Result' });
    };
    await app.streamResponse({ chatId: 'owner', model: 'text', systemPrompt: '' });
    assert.deepEqual(executed, [
        { tool_name: 'search', arguments: { q: 'hello' } },
        { tool_name: 'extract', arguments: { urls: ['https://example.com'] } },
    ]);
    assert.deepEqual(app.chatSessions.owner.messages.filter(msg => msg.role === 'tool')
        .map(msg => msg.tool_call_id), ['first', 'second']);
});

for (const [provider, flag] of [
    ['brave', 'braveKeySet'], ['searxng', 'searxngUrlSet'],
    ['ddgs_custom', 'ddgsBackendsSet'], ['ddgs', null], ['duckduckgo', null],
]) {
    for (const enabled of [false, true]) {
        for (const configured of flag ? [false, true] : [true]) {
            test(`web tools: ${provider}, enabled=${enabled}, configured=${configured}`, () => {
                const app = appFixture();
                app.chatSettings.webSearchEnabled = enabled;
                app.webSearchSettings = { provider };

                if (flag) app.webSearchSettings[flag] = configured;

                app.mcpTools = ['web_search', 'fetch_url', 'other']
                    .map(name => ({ type: 'function', function: { name, description: 'MCP' } }));
                const active = enabled && configured;
                assert.equal(app.webSearchToolsActive(), active);
                assert.equal(app.builtinWebRoute('web_search'), active ? '/v1/web/search' : null);
                assert.equal(app.builtinWebRoute('fetch_url'), active ? '/v1/web/fetch' : null);
                assert.equal(app.builtinWebRoute('other'), null);
                const tools = app.activeTools().filter(tool => !images.owns(tool.function.name));
                assert.deepEqual(tools.map(tool => tool.function.name), ['web_search', 'fetch_url', 'other']);
                assert.equal(tools[0].function.description === 'MCP', !active);
            });
        }
    }
}

for (const [label, turn, activeId, expected] of [
    ['hidden tool round', [
        { id: 'round', role: 'assistant', _ui: false, tool_calls: [{}] },
        { id: 'result', role: 'tool', _ui: false },
        { id: 'final', role: 'assistant', content: 'Answer' },
    ], 'final', ['round', 'result', 'final']],
    ['visible segment boundary', [
        { id: 'round', role: 'assistant', tool_calls: [{}] },
        { id: 'result', role: 'tool', _ui: false },
        { id: 'final', role: 'assistant', content: 'Answer' },
    ], 'final', ['result', 'final']],
    ['unfinished tool round', [
        { id: 'round', role: 'assistant', _ui: false, tool_calls: [{}] },
        { id: 'result', role: 'tool', _ui: false },
    ], null, ['round', 'result']],
]) {
    test(`variant selection retains the expected ${label} chain`, () => {
        assert.deepEqual(appFixture().getActiveVariantChain(turn, activeId).map(msg => msg.id), expected);
    });
}

for (const [label, content, thinking, visible] of [
    ['null', null, null, false],
    ['empty', '', '', false],
    ['whitespace', ' \n\t ', ' \n ', false],
    ['blank text arrays', [{ type: 'text', text: '' }, { type: 'text', text: '\n ' }], null, false],
    ['plain text', ' \nEditing the hats.\n ', null, true],
    ['literal JSON', '{"action":"edit_image","text":"Swap their hats"}', null, true],
    ['text arrays', [{ type: 'text', text: ' ' }, { type: 'text', text: 'Editing' }], null, true],
    ['reasoning only', null, 'Checking the source image.', true],
    ['reasoning with whitespace', '\n', 'Checking the source image.', true],
    ['reasoning with blank arrays', [{ type: 'text', text: ' \n' }], 'Checking the source image.', true],
]) {
    test(`tool-round visibility derives displayable ${label} without changing history`, () => {
        const app = appFixture();

        const round = {
            id: 'round', role: 'assistant', content, _thinking: thinking,
            _toolRound: true, _ui: false, tool_calls: [call('edit_image', {}, 'tc-1')],
        };

        const messages = [
            { id: 'user', role: 'user', content: 'Swap their hats', _activeVariantId: 'final' },
            round,
            { id: 'result', role: 'tool', content: 'Edited', tool_call_id: 'tc-1', _ui: false, _images: [png] },
            { id: 'final', role: 'assistant', content: 'Done' },
            { id: 'other', role: 'assistant', content: 'Other variant' },
        ];

        app.messages = messages;
        const before = structuredClone(messages);
        const apiBefore = app.buildMessagesForApi(messages, '');
        assert.equal(app.isMessageVisible(round, 1), visible, 'active variant');
        assert.equal(app.isMessageVisible(messages[2], 2), true, 'image result stays visible');
        assert.deepEqual(app.buildMessagesForApi(messages, ''), apiBefore);
        assert.deepEqual(apiBefore.map(msg => msg.role), ['user', 'assistant', 'tool', 'assistant']);
        assert.deepEqual(apiBefore[1].tool_calls, round.tool_calls);
        assert.equal(apiBefore[2].tool_call_id, 'tc-1');
        assert.deepEqual(messages, before, 'visibility never mutates protocol messages');

        messages[0]._activeVariantId = 'other';
        assert.equal(app.isMessageVisible(round, 1), false, 'inactive variant');
        assert.equal(app.isMessageVisible(messages[2], 2), false, 'inactive image result');

        app.messages = messages.slice(0, 3);
        assert.equal(app.isMessageVisible(round, 1), visible, 'unfinished flow');
        assert.equal(app.isMessageVisible(messages[2], 2), true, 'unfinished image result');
    });
}

test('tool-round visibility preserves separate tool cards and ordinary assistants', () => {
    const app = appFixture();
    app.messages = [
        { id: 'user', role: 'user', content: 'Edit' },
        { role: 'tool', content: null, _ui: false, _web: { query: 'hats' } },
        { role: 'tool', content: null, _ui: false, _imagesStripped: true },
        { role: 'tool', content: null, _ui: false },
        { id: 'final', role: 'assistant', content: null },
    ];
    assert.deepEqual(app.messages.map((msg, index) => app.isMessageVisible(msg, index)),
        [true, true, true, false, true]);
});

test('tool-round body binding hides blank padding while preserving reasoning and content bytes', () => {
    const app = appFixture();
    const roundTemplate = template.split('x-show="msg.role === \'assistant\' && msg._toolRound"')[1]
        .split('<!-- Built-in web tool activity card')[0];
    const bodyShow = roundTemplate.match(/<div x-show="([^"]+)" class="message-body/)[1];
    const bodyHtml = roundTemplate.match(/x-html="(renderMarkdown\(getTextContent\(msg.content\)\))"/)[1];
    const thinkingShow = roundTemplate.match(/<template x-if="([^"]+)"/)[1];

    for (const [content, expectedText] of [
        [null, ''], [' \n\t ', ' \n\t '],
        [[{ type: 'text', text: '' }, { type: 'text', text: ' ' }], '\n '],
        [' \nEditing the hats.\n ', ' \nEditing the hats.\n '],
        ['{"action":"edit_image","text":"Swap their hats"}', '{"action":"edit_image","text":"Swap their hats"}'],
        [[{ type: 'text', text: 'First' }, { type: 'text', text: 'Second' }], 'First\nSecond'],
    ]) {
        const scope = {
            msg: { content, _thinking: 'Checking the source image.' },
            getTextContent: app.getTextContent.bind(app),
            hasVisibleThinking: app.hasVisibleThinking.bind(app),
            renderMarkdown: text => text,
        };

        assert.equal(Boolean(vm.runInNewContext(bodyShow, scope)), expectedText.trim().length > 0,
            `body padding for ${JSON.stringify(content)}`);
        assert.equal(vm.runInNewContext(thinkingShow, scope), true);
        assert.equal(vm.runInNewContext(bodyHtml, scope), expectedText);
    }
});

test('API message projection omits UI indicators but keeps tool calls and results', () => {
    const app = appFixture();
    const toolCalls = [call('search', { q: 'hello' }, 'tc-1')];
    const messages = [
        { id: 'user', role: 'user', content: 'Search for X' },
        { role: 'tool_call', content: 'Searching', _ui: true },
        { id: 'round', role: 'assistant', content: null, tool_calls: toolCalls, _ui: false },
        { id: 'result', role: 'tool', tool_call_id: 'tc-1', content: 'Result', _ui: false },
        { role: 'tool_call', content: 'Failed', _error: true, _ui: true },
        { id: 'final', role: 'assistant', content: 'Answer' },
    ];
    const body = app.buildChatCompletionBody(messages, { chatId: 'owner', model: 'text', systemPrompt: '' });

    assert.deepEqual(body.messages, [
        { role: 'user', content: 'Search for X' },
        { role: 'assistant', content: null, tool_calls: toolCalls },
        { role: 'tool', content: 'Result', tool_call_id: 'tc-1' },
        { role: 'assistant', content: 'Answer' },
    ]);
});

for (const status of [200, 503]) {
    test(`MCP HTTP ${status} produces a named tool result`, async () => {
        global.fetch = async () => status === 200
            ? json({ content: 'Search result' }) : new Response('', { status });
        const result = await appFixture().executeChatTool(call('search', {}), controllerScope());
        assert.equal(result.toolName, 'search');
        assert.equal(result.error, status !== 200);
        assert.equal(result.content, status === 200 ? 'Search result' : 'Error: search: HTTP 503');
    });
}
