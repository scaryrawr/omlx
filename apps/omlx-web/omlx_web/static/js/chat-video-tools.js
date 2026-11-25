// SPDX-License-Identifier: Apache-2.0
const ChatVideoTools = (() => {
    const tasks = ['text-to-video', 'image-to-video'];
    const statuses = ['queued', 'in_progress', 'completed', 'failed'];
    const jobIdPattern = /^video_[0-9a-f]{32}$/;
    const registry = {
        list_video_models: {
            description: 'List available video model IDs and their supported tasks before choosing a model.',
            properties: {},
            required: [],
            execute: (_args, scope) => ({ content: JSON.stringify({ models: scope.videoModels }) }),
        },
        generate_video: {
            description: 'Queue one video job using an available model ID. Omit image_ref for text-to-video. For image-to-video, use an exact image_ref from the selected conversation, never a URL or path. A queued job is not a finished video. The user can check status and download it from the chat card. Do not retry submission automatically.',
            properties: {
                model: { type: 'string' },
                prompt: { type: 'string' },
                image_ref: { type: 'string', pattern: '^image:' },
                seconds: { type: 'string', enum: ['4', '8', '12'] },
                size: { type: 'string', pattern: '^[1-9][0-9]*x[1-9][0-9]*$' },
            },
            required: ['model', 'prompt'],
            execute: submit,
        },
    };

    function owns(name) {
        return Object.hasOwn(registry, name);
    }

    function definitions() {
        return Object.entries(registry).map(([name, { description, properties, required }]) => ({
            type: 'function',
            function: { name, description, parameters: { type: 'object', properties, required, additionalProperties: false } },
        }));
    }

    async function readJson(url, init) {
        const response = await fetch(url, init);
        const data = await response.json();
        if (!response.ok) {
            const detail = data?.error?.message || data?.detail || `HTTP ${response.status}`;
            throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        }
        return data;
    }

    function canDiscover(status) {
        return status?.video_generation_ready === true && Array.isArray(status.models)
            && status.models.some(model => model.engine_type === 'video' && model.tasks?.length);
    }

    async function discover(status, headers, signal) {
        if (!canDiscover(status)) return [];
        const entries = new Map(status.models.map(model => [model.id, model]));
        for (const model of status.models) {
            if (model.model_alias && !entries.has(model.model_alias)) entries.set(model.model_alias, model);
        }

        const visible = await readJson('/v1/models', { headers, signal });
        return visible.data.flatMap(({ id }) => {
            const model = entries.get(id);
            if (model?.engine_type !== 'video' || model.unavailable_reason || !Array.isArray(model.tasks)) return [];
            const supported = tasks.filter(task => model.tasks.includes(task));
            return supported.length ? [{ id, tasks: supported }] : [];
        });
    }

    function parseJob(data) {
        if (data?.object !== 'video' || typeof data.id !== 'string' || !jobIdPattern.test(data.id)
            || !statuses.includes(data.status) || !Number.isInteger(data.progress)
            || data.progress < 0 || data.progress > 100) {
            throw new Error('Invalid video job response');
        }
        const job = { id: data.id, status: data.status, progress: data.progress };
        if (job.status === 'failed' && typeof data.error?.message === 'string') job.error = data.error.message;
        return job;
    }

    async function submit(args, scope) {
        const { image_ref: ref, ...body } = args;
        const model = scope.videoModels.find(model => model.id === body.model);
        const task = ref === undefined ? 'text-to-video' : 'image-to-video';
        if (!model?.tasks.includes(task)) throw new Error(`Select an available model supporting ${task}`);

        let init;
        if (ref === undefined) {
            init = { headers: { ...scope.headers, 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
        } else {
            const form = new FormData();
            for (const [key, value] of Object.entries(body)) form.append(key, value);
            form.append('input_reference', ChatImageTools.sourceBlob(ref, scope), 'source');
            init = { headers: scope.headers, body: form };
        }
        const video = parseJob(await readJson('/v1/videos', { ...init, method: 'POST', signal: scope.signal }));
        return { content: JSON.stringify(video), video };
    }

    async function execute(name, args, scope) {
        const { properties, required, execute: run } = registry[name];
        if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Video tool arguments must be an object');
        for (const key of required) {
            if (!Object.hasOwn(args, key)) throw new Error(`Video tool requires ${key}`);
        }
        for (const [key, value] of Object.entries(args)) {
            if (!Object.hasOwn(properties, key)) throw new Error(`Unsupported video option ${key}`);
            const rule = properties[key];
            if (typeof value !== 'string' || !value.trim()
                || (rule.pattern && !new RegExp(rule.pattern).test(value))
                || (rule.enum && !rule.enum.includes(value))) throw new Error(`Invalid video option ${key}`);
        }
        return run(args, scope);
    }

    function jobRoute(id) {
        if (typeof id !== 'string' || !jobIdPattern.test(id)) throw new Error('Invalid video job ID');
        return `/v1/videos/${id}`;
    }

    async function refresh(id, headers) {
        const job = parseJob(await readJson(jobRoute(id), { headers }));
        if (job.id !== id) throw new Error('Mismatched video job ID');
        return job;
    }

    async function download(id, headers) {
        const response = await fetch(`${jobRoute(id)}/content`, { headers, redirect: 'error' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob());
        try {
            const link = document.createElement('a');
            link.href = url;
            link.download = `${id}.mp4`;
            link.click();
        } finally {
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }
    }

    return { owns, definitions, canDiscover, discover, execute, refresh, download };
})();
