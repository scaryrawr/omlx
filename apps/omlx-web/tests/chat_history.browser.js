// SPDX-License-Identifier: Apache-2.0
async function runChatHistoryBrowserTests() {
    const results = [];
    const stores = [];
    const databases = new Set();
    const image = 'data:image/png;base64,c291cmNl';
    const assert = (condition, message) => {
        if (!condition) throw new Error(message);
    };
    const storage = () => {
        const values = new Map();
        return {
            getItem: key => values.get(key) || null,
            setItem: (key, value) => values.set(key, value),
            removeItem: key => values.delete(key),
        };
    };
    const open = (options = {}) => {
        const name = options.name || `omlx-history-test-${crypto.randomUUID()}`;
        databases.add(name);
        const store = new ChatHistory.Store({ storage: storage(), ...options, name });
        stores.push(store);
        return store;
    };
    const chat = (id = 'owner', time = 1) => ({
        id, updatedAt: new Date(time).toISOString(),
        messages: [
            { id: 'user', role: 'user', content: [{ type: 'image_url', image_url: { url: image } }] },
            { id: 'tool-old', role: 'tool', _images: [image], _ui: false },
            { id: 'tool-new', role: 'tool', _images: [image], _ui: false },
        ],
    });
    const images = async store => {
        const tx = store.db.transaction('attachments');
        return new Promise((resolve, reject) => {
            const read = tx.objectStore('attachments').getAll();
            read.onsuccess = () => resolve(read.result);
            read.onerror = () => reject(read.error);
        });
    };
    const check = async (name, action) => {
        try {
            await action();
            results.push(name);
        } catch (error) {
            throw new Error(`${name}: ${error.message}`, { cause: error });
        }
    };
    try {
        await check('reload and binary storage retain uploads and all variants', async () => {
            const store = open();
            await store.load();
            await store.save(chat());
            const attachments = await images(store);
            assert(attachments.length === 3 && attachments.every(item => item.bytes instanceof ArrayBuffer), 'Expected three owned binary buffers');
            const reload = open({ name: store.name });
            const loaded = await reload.load();
            assert(loaded[0].messages[0].content[0].image_url.url === image, 'Upload did not hydrate');
            assert(loaded[0].messages[1]._images[0] === image, 'Hidden variant did not hydrate');
            await reload.save({ ...loaded[0], messages: loaded[0].messages.slice(0, 1) });
            assert((await images(reload)).length === 1, 'Truncation did not reconcile attachments');
            await reload.delete('owner');
            assert((await images(reload)).length === 0, 'Deletion left attachments');
            assert(await reload.save(chat()) === false, 'Late save resurrected deleted lifetime');
        });
        await check('branch survives parent deletion and export/import preserves bytes', async () => {
            const store = open();
            await store.load();
            await store.save(chat());
            const branch = (await store.load())[0];
            branch.id = 'branch';
            await store.save(branch);
            await store.delete('owner');
            const exported = JSON.stringify(await store.load());
            assert((await images(store)).length === 3, 'Branch lost owned images');
            const target = open();
            await target.load();
            for (const entry of ChatHistory.parseArchive(JSON.parse(exported))) await target.save(entry);
            assert((await target.load())[0].messages[2]._images[0] === image, 'Archive lost image bytes');
        });
        await check('legacy migration commits once and clear cannot reimport leftovers', async () => {
            const legacy = storage();
            legacy.setItem('omlx_chat_history', JSON.stringify([chat('legacy')]));
            const store = open({ storage: legacy });
            assert((await store.load())[0].id === 'legacy', 'Legacy history not migrated');
            assert(!legacy.getItem('omlx_chat_history'), 'Legacy key not removed');
            await store.clear();
            legacy.setItem('omlx_chat_history', 'malformed leftover');
            const reload = open({ name: store.name, storage: legacy });
            assert((await reload.load()).length === 0, 'Migration marker allowed resurrection');
        });
        await check('migration failure preserves the legacy source', async () => {
            const legacy = storage();
            legacy.setItem('omlx_chat_history', 'invalid');
            const store = open({ storage: legacy });
            let failed = false;
            try { await store.load(); } catch { failed = true; }
            assert(failed && legacy.getItem('omlx_chat_history') === 'invalid', 'Migration failure lost original history');
        });
        await check('stale tabs cannot overwrite, delete-resurrect, or clear-resurrect', async () => {
            const first = open();
            await first.load();
            await first.save(chat());
            const second = open({ name: first.name });
            const stale = (await second.load())[0];
            await first.save(chat('owner', 2));
            let failed = false;
            try { await second.save(stale); } catch { failed = true; }
            assert(failed, 'Stale revision replaced a newer chat');
            const third = open({ name: first.name });
            await third.load();
            await first.delete('owner');
            failed = false;
            try { await third.save(stale); } catch { failed = true; }
            assert(failed, 'Stale tab resurrected deleted chat');
            const fourth = open({ name: first.name });
            await fourth.load();
            await first.clear();
            failed = false;
            try { await fourth.save(chat('new-from-stale')); } catch { failed = true; }
            assert(failed, 'Stale epoch created a chat after clear');
            await first.save(chat('fresh'));
            assert((await first.load())[0].id === 'fresh', 'Fresh lifetime after clear failed');
        });
        await check('unrelated tab writes and bounded eviction reconcile independently', async () => {
            const first = open({ limit: 2 });
            await first.load();
            const second = open({ name: first.name, limit: 2 });
            await second.load();
            await first.save(chat('a', 1));
            await second.save(chat('b', 2));
            assert((await first.load()).length === 2, 'Tab write replaced unrelated history');
            const evicted = await first.save(chat('c', 3));
            assert(evicted.includes('a'), 'Oldest chat not evicted');
            assert((await images(first)).length === 6, 'Eviction left orphaned attachments');
        });
        await check('queued snapshots and clear invalidate pre-clear creates', async () => {
            const store = open();
            await store.load();
            const live = chat();
            const saving = store.save(live);
            live.messages = [];
            await saving;
            assert((await store.load())[0].messages.length === 3, 'Save did not capture invocation snapshot');
            const pending = store.save(chat('pending'));
            const clearing = store.clear();
            await Promise.all([pending, clearing]);
            assert((await store.load()).length === 0, 'Queued create survived clear');
            assert((await images(store)).length === 0, 'Clear left image bytes');
        });
        await check('import replacement removes old owned images atomically', async () => {
            const store = open();
            await store.load();
            await store.save(chat());
            const replacement = chat('owner', 2);
            replacement.messages = replacement.messages.slice(0, 1);
            await store.save(replacement, { replace: true });
            assert((await images(store)).length === 1, 'Import replacement left old variant images');
        });
        await check('aborted transactions retain the previous chat and image bytes', async () => {
            const store = open();
            await store.load();
            await store.save(chat());
            const transaction = store.db.transaction.bind(store.db);
            store.db.transaction = (...args) => {
                const tx = transaction(...args);
                if (args[1] === 'readwrite') {
                    tx.objectStore('metadata').get('epoch').onsuccess = () => tx.abort();
                }
                return tx;
            };
            const replacement = chat('owner', 2);
            replacement.messages = [];
            let failed = false;
            try {
                await store.save(replacement);
            } catch {
                failed = true;
            } finally {
                store.db.transaction = transaction;
            }
            assert(failed, 'Aborted save reported success');
            assert((await store.load())[0].messages.length === 3, 'Aborted save replaced the chat');
            assert((await images(store)).length === 3, 'Aborted save deleted image bytes');
        });
        return results;
    } finally {
        for (const store of stores) store.db?.close();
        for (const name of databases) {
            await new Promise((resolve, reject) => {
                const removal = indexedDB.deleteDatabase(name);
                removal.onsuccess = resolve;
                removal.onerror = () => reject(removal.error);
            });
        }
    }
}
