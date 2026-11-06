// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const history = vm.runInThisContext(fs.readFileSync(
    path.join(__dirname, '../omlx_web/static/js/chat-history.js'), 'utf8',
) + '\nChatHistory;');
const png = 'data:image/png;base64,c291cmNl';
const fixture = () => ({
    id: 'owner', messages: [
        { id: 'upload', role: 'user', content: [
            { type: 'text', text: 'Edit' },
            { type: 'image_url', image_url: { url: png, detail: 'high' } },
            { type: 'file', file: { filename: 'private.txt', data: 'not persisted', file_data: 'also not persisted' } },
        ] },
        { id: 'variant-one', role: 'tool', _images: [png], _ui: false },
        { id: 'variant-two', role: 'tool', _images: [png], _ui: false },
        { id: 'legacy', role: 'tool', _imagesStripped: 2 },
    ],
});

test('packed records use owned binary buffers and hydrate every retained variant', async () => {
    const chat = fixture();
    const { snapshot, attachments } = history.pack(chat);
    assert.equal(attachments.length, 3);
    assert.ok(attachments.every(image => image.bytes instanceof ArrayBuffer));
    assert.ok(!JSON.stringify(snapshot).includes('base64'));
    assert.ok(!JSON.stringify(snapshot).includes('not persisted'));
    assert.deepEqual(attachments[0].key, ['owner', 'upload', 'upload', 1]);
    assert.equal(new TextDecoder().decode(attachments[0].bytes), 'source');
    assert.equal(attachments[0].mimeType, 'image/png');
    const restored = await history.hydrate(snapshot, attachments);
    assert.equal(restored.messages[0].content[1].image_url.url, png);
    assert.equal(restored.messages[0].content[1].image_url.detail, 'high');
    assert.equal(restored.messages[1]._images[0], png);
    assert.equal(restored.messages[2]._images[0], png);
    assert.equal(restored.messages[3]._imagesStripped, 2);
    assert.equal(chat.messages[0].content[2].file.data, 'not persisted');
});

test('snapshots are captured before asynchronous writes or live mutation', () => {
    const chat = fixture();
    const packed = history.pack(chat);
    chat.messages[0].content[1].image_url.url = '';
    chat.messages.splice(1);
    assert.equal(packed.snapshot.messages.length, 4);
    assert.equal(packed.attachments.length, 3);
});

test('branches have independent attachment ownership and archives carry bytes', async () => {
    const source = fixture();
    const branch = structuredClone(source);
    branch.id = 'branch';
    const packed = history.pack(branch);
    assert.ok(packed.attachments.every(image => image.key[0] === 'branch'));
    const restored = await history.hydrate(packed.snapshot, packed.attachments);
    const archive = history.parseArchive(JSON.parse(JSON.stringify([restored])));
    assert.equal(archive[0].messages[1]._images[0], png);
    assert.equal(history.pack(archive[0]).attachments.length, 3);
});

test('legacy missing-image placeholders are not converted into fabricated bytes', async () => {
    const legacy = { id: 'old', messages: [
        { role: 'user', content: [{ type: 'image_url', image_url: { url: '' } }] },
        { role: 'tool', _imagesStripped: 1 },
    ] };
    const packed = history.pack(history.parseArchive([legacy])[0]);
    assert.equal(packed.attachments.length, 0);
    assert.deepEqual(await history.hydrate(packed.snapshot, []), legacy);
});

test('archive and packing reject message IDs that cannot own IndexedDB attachments', () => {
    for (const id of [true, false, {}, [], NaN, Infinity, -Infinity]) {
        const chat = fixture();
        chat.messages[0].id = id;
        assert.throws(() => history.parseArchive([chat]), /Invalid chat history archive/);
        assert.throws(() => history.pack(chat), /Invalid chat history archive/);
    }
});

test('legacy missing IDs and valid scalar IDs retain their attachment keys', async () => {
    for (const id of [undefined, null, '', 'message', 0, 42, -1.5]) {
        const chat = fixture();
        chat.messages[0].id = id;
        chat.messages[0].content.pop();
        const archived = history.parseArchive([chat])[0];
        const packed = history.pack(archived);
        const messageId = id == null || id === '' ? 'legacy-0' : id;

        assert.deepEqual(packed.attachments[0].key, ['owner', messageId, 'upload', 1]);
        assert.deepEqual(await history.hydrate(packed.snapshot, packed.attachments), archived);
    }
});

test('invalid archive and missing attachment failures are explicit', async () => {
    assert.throws(() => history.parseArchive({ chats: [] }), /Invalid chat history archive/);
    assert.throws(() => history.parseArchive([{ id: '', messages: [] }]), /Invalid chat history archive/);
    assert.throws(() => history.parseArchive([{ id: 'x', messages: [null] }]), /Invalid chat history archive/);
    const packed = history.pack(fixture());
    await assert.rejects(history.hydrate(packed.snapshot, []), /saved chat image is missing/);
    const unavailable = new history.Store({ indexedDB: null, storage: {} });
    await assert.rejects(unavailable.load(), /IndexedDB is unavailable/);
});
