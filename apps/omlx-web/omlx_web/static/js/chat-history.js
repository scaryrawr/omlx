// SPDX-License-Identifier: Apache-2.0
const ChatHistory = (() => {
    const databaseName = 'omlx-chat-history';
    const legacyKey = 'omlx_chat_history';
    const newId = () => Array.from(
        crypto.getRandomValues(new Uint8Array(16)),
        byte => byte.toString(16).padStart(2, '0'),
    ).join('');
    const copy = value => JSON.parse(JSON.stringify(value));
    const request = operation => new Promise((resolve, reject) => {
        operation.onsuccess = () => resolve(operation.result);
        operation.onerror = () => reject(operation.error);
    });
    const completed = transaction => {
        const result = new Promise((resolve, reject) => {
            transaction.oncomplete = resolve;
            transaction.onabort = () => reject(transaction.error || new Error('Chat history transaction aborted'));
        });
        result.catch(() => {});
        return result;
    };

    function parseArchive(value) {
        if (!Array.isArray(value) || value.some(chat =>
            !chat || typeof chat.id !== 'string' || !chat.id || !Array.isArray(chat.messages)
            || chat.messages.some(message => !message || typeof message.role !== 'string')
        )) throw new Error('Invalid chat history archive');
        return copy(value);
    }

    function pack(chat) {
        const snapshot = copy(chat);
        const attachments = [];
        snapshot.messages.forEach((message, index) => {
            const messageId = message.id || `legacy-${index}`;
            const image = (url, kind, slot) => {
                if (!url || !url.startsWith('data:')) return url;
                const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(url);
                if (!match) throw new Error('Invalid inline chat image');

                const bytes = Uint8Array.from(atob(match[2]), char => char.charCodeAt(0));
                const key = [chat.id, messageId, kind, slot];
                attachments.push({ key, chatId: chat.id, bytes: bytes.buffer, mimeType: match[1] });

                return { attachment: key };
            };
            if (message._images) {
                message._images = message._images.map((url, slot) => image(url, 'tool', slot));
            }
            if (Array.isArray(message.content)) {
                message.content = message.content.map((part, slot) => {
                    if (part.type === 'image_url') {
                        return { ...part, image_url: { ...part.image_url, url: image(part.image_url?.url || '', 'upload', slot) } };
                    }
                    if (part.type === 'file') {
                        return {
                            ...part,
                            file: {
                                filename: part.file?.filename || '',
                                mime_type: part.file?.mime_type || '',
                                data: '',
                            },
                        };
                    }

                    return part;
                });
            }
        });
        return { snapshot, attachments };
    }

    function dataUrl(attachment) {
        const bytes = new Uint8Array(attachment.bytes);
        let binary = '';
        for (let offset = 0; offset < bytes.length; offset += 8192) {
            binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
        }

        return `data:${attachment.mimeType};base64,${btoa(binary)}`;
    }

    async function hydrate(chat, attachments) {
        const snapshot = copy(chat);
        const urls = new Map(attachments.map(attachment =>
            [JSON.stringify(attachment.key), dataUrl(attachment)]
        ));
        const image = value => {
            if (typeof value === 'string') return value;

            const url = urls.get(JSON.stringify(value.attachment));
            if (!url) throw new Error('A saved chat image is missing');

            return url;
        };
        for (const message of snapshot.messages) {
            if (message._images) message._images = message._images.map(image);
            if (Array.isArray(message.content)) {
                for (const part of message.content) {
                    if (part.type === 'image_url') part.image_url.url = image(part.image_url.url);
                }
            }
        }
        return snapshot;
    }

    class Store {
        constructor({ indexedDB = globalThis.indexedDB, storage = globalThis.localStorage, onError = () => {}, limit = 100, name = databaseName } = {}) {
            this.indexedDB = indexedDB;
            this.storage = storage;
            this.onError = onError;
            this.limit = limit;
            this.name = name;
            this.revisions = new Map();
            this.seen = new Set();
            this.deleted = new Set();
            this.generation = 0;
            this.queue = Promise.resolve();
            this.ready = this.open();
        }

        async open() {
            if (!this.indexedDB) throw new Error('IndexedDB is unavailable');
            this.db = await new Promise((resolve, reject) => {
                let blocked = false;
                const opening = this.indexedDB.open(this.name, 1);
                opening.onupgradeneeded = () => {
                    const db = opening.result;
                    db.createObjectStore('chats', { keyPath: 'id' });
                    db.createObjectStore('attachments', { keyPath: 'key' }).createIndex('chatId', 'chatId');
                    db.createObjectStore('metadata');
                };
                opening.onsuccess = () => {
                    if (blocked) opening.result.close();
                    else resolve(opening.result);
                };
                opening.onerror = () => reject(opening.error);
                opening.onblocked = () => {
                    blocked = true;
                    reject(new Error('Chat history upgrade is blocked by another tab'));
                };
            });
            this.db.onversionchange = () => {
                this.db.close();
                this.onError(new Error('Chat history changed in another tab. Reload this page.'));
            };
            this.db.onclose = () => this.onError(new Error('Chat history storage connection closed unexpectedly.'));
            const legacy = this.storage.getItem(legacyKey);
            const tx = this.db.transaction(['chats', 'attachments', 'metadata'], 'readwrite');
            const done = completed(tx);
            const metadata = tx.objectStore('metadata');
            const marker = await request(metadata.get('migrated'));
            this.epoch = await request(metadata.get('epoch')) || newId();
            metadata.put(this.epoch, 'epoch');
            if (!marker) {
                const migrated = legacy ? parseArchive(JSON.parse(legacy)).map(pack) : [];
                for (const entry of migrated) {
                    const chats = tx.objectStore('chats');
                    if (!await request(chats.get(entry.snapshot.id))) {
                        chats.put({ id: entry.snapshot.id, revision: 1, chat: entry.snapshot });
                        for (const attachment of entry.attachments) tx.objectStore('attachments').put(attachment);
                    }
                }
                metadata.put(true, 'migrated');
            }
            await done;
            if (legacy) this.storage.removeItem(legacyKey);
        }

        enqueue(action) {
            const operation = this.queue.then(() => this.ready).then(action);
            this.queue = operation.catch(() => {});
            return operation;
        }

        load() {
            return this.enqueue(async () => {
                const tx = this.db.transaction(['chats', 'attachments', 'metadata']);
                const done = completed(tx);
                const records = request(tx.objectStore('chats').getAll());
                const attachments = request(tx.objectStore('attachments').getAll());
                const epoch = request(tx.objectStore('metadata').get('epoch'));
                const [saved, images, currentEpoch] = await Promise.all([records, attachments, epoch]);
                await done;
                this.epoch = currentEpoch;
                const result = [];
                for (const record of saved) {
                    if (!record.chat) {
                        this.deleted.add(record.id);
                        continue;
                    }
                    this.revisions.set(record.id, record.revision);
                    result.push(await hydrate(record.chat, images.filter(image => image.chatId === record.id)));
                }
                return result;
            });
        }

        async removeAttachments(tx, chatId) {
            const store = tx.objectStore('attachments');
            const keys = await request(store.index('chatId').getAllKeys(chatId));
            for (const key of keys) store.delete(key);
        }

        save(chat, { replace = false } = {}) {
            const entry = pack(chat);
            this.seen.add(entry.snapshot.id);
            const generation = this.generation;
            return this.enqueue(async () => {
                if (generation !== this.generation || this.deleted.has(entry.snapshot.id)) return false;
                const tx = this.db.transaction(['chats', 'attachments', 'metadata'], 'readwrite');
                const done = completed(tx);
                const chats = tx.objectStore('chats');
                const epoch = await request(tx.objectStore('metadata').get('epoch'));
                const existing = await request(chats.get(entry.snapshot.id));
                if (replace && existing?.chat
                    && (Date.parse(existing.chat.updatedAt) || 0) >= (Date.parse(entry.snapshot.updatedAt) || 0)) {
                    await done;
                    return false;
                }
                if (epoch !== this.epoch || (existing && (!existing.chat
                    || (!replace && existing.revision !== this.revisions.get(entry.snapshot.id))))) {
                    await done;
                    this.deleted.add(entry.snapshot.id);
                    throw new Error('Chat history changed in another tab. Reload before saving this chat.');
                }
                const revision = (existing?.revision || 0) + 1;
                await this.removeAttachments(tx, entry.snapshot.id);
                chats.put({ id: entry.snapshot.id, revision, chat: entry.snapshot });
                for (const attachment of entry.attachments) tx.objectStore('attachments').put(attachment);
                const records = await request(chats.getAll());
                const active = records.filter(record => record.chat).sort((a, b) =>
                    Number(!!b.chat.pinned) - Number(!!a.chat.pinned)
                    || (Date.parse(b.chat.updatedAt) || 0) - (Date.parse(a.chat.updatedAt) || 0)
                );
                const evicted = active.slice(this.limit);
                for (const record of evicted) {
                    chats.put({ id: record.id, revision: record.revision + 1 });
                    await this.removeAttachments(tx, record.id);
                }
                await done;
                this.revisions.set(entry.snapshot.id, revision);
                for (const record of evicted) this.deleted.add(record.id);
                return evicted.map(record => record.id);
            });
        }

        delete(chatId) {
            this.deleted.add(chatId);
            return this.enqueue(async () => {
                const tx = this.db.transaction(['chats', 'attachments'], 'readwrite');
                const done = completed(tx);
                tx.objectStore('chats').put({ id: chatId, revision: 0 });
                await this.removeAttachments(tx, chatId);
                await done;
            });
        }

        clear() {
            this.generation++;
            for (const id of [...this.revisions.keys(), ...this.seen]) this.deleted.add(id);
            return this.enqueue(async () => {
                const tx = this.db.transaction(['chats', 'attachments', 'metadata'], 'readwrite');
                const done = completed(tx);
                tx.objectStore('chats').clear();
                tx.objectStore('attachments').clear();
                const epoch = newId();
                tx.objectStore('metadata').put(epoch, 'epoch');
                await done;
                this.epoch = epoch;
                this.revisions.clear();
            });
        }
    }

    return { Store, parseArchive, pack, hydrate, newId };
})();
