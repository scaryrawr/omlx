// SPDX-License-Identifier: Apache-2.0
const ChatImageTools = (() => {
    const options = {
        size: { type: 'string', minLength: 1 },
        seed: { type: 'integer' },
        steps: { type: 'integer', minimum: 1 },
        guidance: { type: 'number' },
        n: { type: 'integer', minimum: 1 },
        output_format: { type: 'string', enum: ['png', 'jpeg', 'webp'] },
        negative_prompt: { type: 'string' },
    };
    const registry = {
        list_image_models: {
            description: 'List available image models, their explicitly supported tasks and defaults. Discover models before generating or editing. Unknown tasks are not supported.',
            properties: {},
            required: [],
            execute: async (_args, scope) => ({
                content: JSON.stringify({ models: await discover(scope) }), images: [],
            }),
        },
        generate_image: {
            description: 'Generate images with an available image model. Omit optional parameters to preserve model defaults. Outputs include image_refs for subsequent edits. Never retry inference automatically.',
            properties: { model: { type: 'string', minLength: 1 }, prompt: { type: 'string', minLength: 1 }, ...options },
            required: ['model', 'prompt'],
            execute: (args, scope) => infer(args, scope, false),
        },
        edit_image: {
            description: 'Edit explicit image_refs from this conversation. Sources may be uploads or prior tool images. Never guess a reference, use a URL/path/base64, or edit an unfinished sibling call. Preserves sources and returns fresh images.',
            properties: {
                model: { type: 'string', minLength: 1 }, prompt: { type: 'string', minLength: 1 }, ...options,
                image_refs: { type: 'array', items: { type: 'string', pattern: '^image:' }, minItems: 1 },
                image_strength: { type: 'number', minimum: 0, maximum: 1 },
            },
            required: ['model', 'prompt', 'image_refs'],
            execute: (args, scope) => infer(args, scope, true),
        },
    };

    function owns(name) {
        return Object.hasOwn(registry, name);
    }

    function definitions() {
        return Object.entries(registry).map(([name, entry]) => ({
            type: 'function',
            function: {
                name, description: entry.description,
                parameters: {
                    type: 'object', properties: entry.properties,
                    required: entry.required, additionalProperties: false,
                },
            },
        }));
    }

    function reference(chatId, messageId, kind, slot) {
        return `image:${encodeURIComponent(chatId)}:${encodeURIComponent(messageId)}:${kind}:${slot}`;
    }

    function renderReference(href) {
        if (!href.startsWith('image:')) return false;
        const escaped = href.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<code>${escaped}</code>`;
    }

    function sources(chatId, messages) {
        const found = new Map();
        for (const msg of messages) {
            if (!msg.id) continue;
            if (msg.role === 'user' && Array.isArray(msg.content)) {
                msg.content.forEach((part, slot) => {
                    if (part.type === 'image_url') {
                        found.set(reference(chatId, msg.id, 'upload', slot), part.image_url?.url || '');
                    }
                });
            }
            if (msg.role === 'tool') {
                (msg._images || []).forEach((url, slot) => {
                    found.set(reference(chatId, msg.id, 'tool', slot), url);
                });
                for (let slot = 0; !msg._images?.length && slot < (msg._imagesStripped || 0); slot++) {
                    found.set(reference(chatId, msg.id, 'tool', slot), '');
                }
            }
        }
        return found;
    }

    function project(msg, chatId, vision) {
        if (msg.role === 'user' && Array.isArray(msg.content)) {
            return msg.content.flatMap((part, slot) => {
                if (part.type !== 'image_url') return [part];
                const ref = reference(chatId, msg.id, 'upload', slot);
                const note = { type: 'text', text: `[Image ${ref}${part.image_url?.url ? '' : ' unavailable after reload'}]` };
                return vision && part.image_url?.url ? [note, part] : [note];
            });
        }
        if (msg.role === 'tool' && (msg._images?.length || msg._imagesStripped)) {
            const count = msg._images?.length || msg._imagesStripped;
            const refs = Array.from({ length: count }, (_, slot) => reference(chatId, msg.id, 'tool', slot));
            return `${msg.content || ''}\nImage references: ${refs.join(', ')}${msg._images?.length ? '' : ' (unavailable after reload)'}`;
        }
        return msg.content;
    }

    async function readJson(url, init, scope) {
        scope.signal.throwIfAborted();
        const response = await fetch(url, { ...init, signal: scope.signal });
        let data;
        try {
            data = await response.json();
        } catch (error) {
            scope.signal.throwIfAborted();
            throw new Error(`${url}: ${error.message}`);
        }
        scope.signal.throwIfAborted();
        if (!response.ok) {
            const detail = data.error?.message || data.detail || `HTTP ${response.status}`;
            throw new Error(`${url}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
        }
        return data;
    }

    async function discover(scope) {
        const [publicData, statusData] = await Promise.all([
            readJson('/v1/models', { headers: scope.headers }, scope),
            readJson('/v1/models/status', { headers: scope.headers }, scope),
        ]);
        if (!Array.isArray(publicData.data) || !Array.isArray(statusData.models)) {
            throw new Error('Invalid image model discovery response');
        }
        const canonical = new Map();
        for (const model of statusData.models) {
            if (typeof model.id === 'string') canonical.set(model.id, model);
        }
        const lookup = new Map(canonical);
        for (const model of canonical.values()) {
            if (model.model_alias && !lookup.has(model.model_alias)) {
                lookup.set(model.model_alias, model);
            }
        }
        const result = new Map();
        for (const publicModel of publicData.data) {
            const model = lookup.get(publicModel.id);
            if (!model || (model.model_type !== 'image' && model.engine_type !== 'image')) continue;
            const metadata = model.image_metadata || {};
            const tasks = Array.isArray(model.tasks) && model.tasks.length ? model.tasks : metadata.tasks;
            let entry = result.get(model.id);
            if (!entry) {
                const defaults = {};
                for (const [key, value] of Object.entries(metadata)) {
                    if (key.startsWith('default_')) defaults[key] = value;
                }
                entry = { id: model.id, aliases: [], tasks: [...new Set((Array.isArray(tasks) ? tasks : []).filter(task => task === 'generation' || task === 'edit'))], defaults };
                result.set(model.id, entry);
            }
            if (publicModel.id !== model.id && !entry.aliases.includes(publicModel.id)) entry.aliases.push(publicModel.id);
        }
        return [...result.values()];
    }

    function sourceBlob(ref, scope) {
        const images = sources(scope.chatId, scope.messages);
        if (!images.has(ref)) throw new Error(`Image source ${ref} is not in this chat's selected history`);
        const url = images.get(ref);
        if (!url) throw new Error(`Image source ${ref} is unavailable after reload or export`);
        const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(url);
        if (!match) throw new Error(`Image source ${ref} has no reusable inline image bytes`);
        let binary;
        try {
            binary = atob(match[2]);
        } catch {
            throw new Error(`Image source ${ref} contains invalid inline image bytes`);
        }
        return new Blob([Uint8Array.from(binary, char => char.charCodeAt(0))], { type: match[1] });
    }

    async function infer(args, scope, edit) {
        const body = { model: args.model, prompt: args.prompt, response_format: 'b64_json' };
        for (const key of Object.keys(options)) {
            if (args[key] !== undefined) body[key] = args[key];
        }
        let init;
        if (edit) {
            const form = new FormData();
            for (const [key, value] of Object.entries(body)) form.append(key, String(value));
            if (args.image_strength !== undefined) form.append('image_strength', String(args.image_strength));
            args.image_refs.forEach((ref, index) => form.append('image', sourceBlob(ref, scope), `source-${index}`));
            init = { method: 'POST', headers: scope.headers, body: form };
        } else {
            init = { method: 'POST', headers: { ...scope.headers, 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
        }
        const data = await readJson(edit ? '/v1/images/edits' : '/v1/images/generations', init, scope);
        if (!Array.isArray(data.data) || !data.data.length) throw new Error('Image inference returned no images');
        const images = data.data.map(item => {
            if (typeof item.b64_json !== 'string' || !item.b64_json) throw new Error('Image inference returned no inline image bytes');
            const format = item.output_format || data.output_format || args.output_format || 'png';
            if (!['png', 'jpeg', 'webp'].includes(format)) throw new Error('Image inference returned an unsupported format');
            return `data:image/${format};base64,${item.b64_json}`;
        });
        const outputs = data.data.map((item, slot) => {
            const output = { image_ref: reference(scope.chatId, scope.resultMessageId, 'tool', slot) };
            for (const key of ['revised_prompt', 'steps', 'guidance', 'size', 'seed', 'output_format']) {
                if (item[key] !== undefined) output[key] = item[key];
            }
            return output;
        });
        return { content: JSON.stringify({ model: args.model, outputs }), images };
    }

    async function execute(name, args, scope) {
        if (!owns(name)) throw new Error(`Unknown image tool ${name}`);
        const entry = registry[name];
        if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Image tool arguments must be an object');
        for (const key of entry.required) {
            if (args[key] === undefined) throw new Error(`Image tool requires ${key}`);
        }
        for (const [key, value] of Object.entries(args)) {
            if (!Object.hasOwn(entry.properties, key)) throw new Error(`Unsupported image option ${key}`);
            const rule = entry.properties[key];
            const valid = rule.type === 'array'
                ? Array.isArray(value) && value.length > 0 && value.every(ref => typeof ref === 'string' && ref.startsWith('image:'))
                : rule.type === 'integer' ? Number.isInteger(value)
                : rule.type === 'number' ? typeof value === 'number' && Number.isFinite(value)
                : typeof value === 'string' && (!rule.minLength || value.trim().length >= rule.minLength);
            if (!valid || (rule.minimum !== undefined && value < rule.minimum)
                || (rule.maximum !== undefined && value > rule.maximum)
                || (rule.enum && !rule.enum.includes(value))) throw new Error(`Invalid image option ${key}`);
        }
        return entry.execute(args, scope);
    }

    return { owns, definitions, reference, renderReference, sources, project, execute, discover };
})();
