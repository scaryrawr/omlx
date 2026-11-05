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
const script = template.split('{% block scripts %}')[1].match(/<script>([\s\S]*?)<\/script>/)[1]
    .replace('{{ api_key | tojson }}', '""');
const chatApp = vm.runInThisContext(script + '\nchatApp;');
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
    const app = chatApp();
    app.currentChatId = 'foreground';
    app.currentModel = 'text';
    app.chatHistoryPending = false;
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
    assert.ok(!JSON.stringify(app.chatHistory[0]).includes('base64'));

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
    const stripped = messages.map(msg => app.messageForStorage(msg));
    assert.ok(!JSON.stringify(stripped).includes('base64'));
    assert.equal(stripped[1]._imagesStripped, 1);
    assert.equal(stripped[0].id, messages[0].id);
    assert.ok(!JSON.stringify(app.buildChatCompletionBody(stripped, context).messages).includes('base64'));
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
    const stripped = variants.map(msg => app.messageForStorage(msg));
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
    assert.ok(!localStorage.getItem('omlx_chat_history').includes('base64'));
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
    assert.ok(!localStorage.getItem('omlx_chat_history').includes('bGF0ZQ=='));
    assert.equal(app.getStreamSession('owner').isStreaming, false);
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
