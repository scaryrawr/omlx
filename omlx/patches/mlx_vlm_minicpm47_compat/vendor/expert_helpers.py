GLOBAL_SCALE_SUFFIXES = ("weight_scale_2", "weight_global_scale")

EXPERT_WEIGHT_SUFFIXES = (
    "weight",
    "scales",
    "biases",
    *GLOBAL_SCALE_SUFFIXES,
)

def expert_scale_rows(scale, shape):
    """Broadcast a tensor or per-expert factor over its output rows."""
    if scale.size == 1:
        scale = scale.reshape(())
    else:
        scale = scale.reshape(*shape[:-1], -1)
    return mx.broadcast_to(scale, shape)

def expand_expert_scales(weights, prefixes):
    """Keep independent global factors when concatenating output rows."""
    for prefix in prefixes:
        for suffix in GLOBAL_SCALE_SUFFIXES:
            key = f"{prefix}.{suffix}"
            if key in weights:
                weights[key] = expert_scale_rows(
                    weights[key], weights[f"{prefix}.weight"].shape[:-1]
                )

def stack_expert_projection(weights, source, projection, target, num_experts):
    for suffix in (*EXPERT_WEIGHT_SUFFIXES, "bias"):
        if f"{source}.0.{projection}.{suffix}" in weights:
            weights[f"{target}.{suffix}"] = mx.stack(
                [
                    weights.pop(f"{source}.{i}.{projection}.{suffix}")
                    for i in range(num_experts)
                ]
            )

def split_expert_projection(weights, source, targets):
    """Split fused output rows, including each row's quantization parameters."""
    weight_key = f"{source}.weight"
    if source in weights:
        weights[weight_key] = weights.pop(source)
    if weight_key not in weights:
        return
    if weights[weight_key].ndim != 3:
        raise ValueError(
            f"{source} has shape {weights[weight_key].shape}; expected "
            "[num_experts, output_rows, input_columns]"
        )
    shape = weights[weight_key].shape[:-1]
    for suffix in (*EXPERT_WEIGHT_SUFFIXES, "bias"):
        key = f"{source}.{suffix}"
        if key not in weights:
            continue
        value = weights.pop(key)
        if suffix in GLOBAL_SCALE_SUFFIXES:
            value = expert_scale_rows(value, shape)
        axis = -1 if suffix in (*GLOBAL_SCALE_SUFFIXES, "bias") else -2
        parts = mx.split(value, len(targets), axis=axis)
        for target, part in zip(targets, parts):
            weights[f"{target}.{suffix}"] = mx.contiguous(part)

def move_expert_projection(weights, source, target):
    for suffix in (*EXPERT_WEIGHT_SUFFIXES, "bias"):
        key = (
            source if suffix == "weight" and source in weights else f"{source}.{suffix}"
        )
        if key in weights:
            weights[f"{target}.{suffix}"] = weights.pop(key)
