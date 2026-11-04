# SPDX-License-Identifier: Apache-2.0
"""Native-backed source conversion and associated Nemotron MTP artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx

from ..model_classification import is_nemotron_modelopt_source


def streamed_decoder_layer(args, layer_index):
    from mlx_vlm.models.nemotron_h.language import NemotronHBlock

    return NemotronHBlock(
        args.for_layer(layer_index), args.hybrid_override_pattern[layer_index]
    )


def register_source_tensors(index, config: dict) -> int:
    if not is_nemotron_modelopt_source(config):
        return 0
    from mlx_vlm.quantization.checkpoint import _transform_modelopt_nvfp4_weights

    quant = config["quantization_config"]
    registered = 0
    for key in list(index._index):
        if not key.endswith(".weight_scale"):
            continue
        prefix = key.removesuffix(".weight_scale")
        weight_key = prefix + ".weight"
        if index.source_shape(weight_key) is None:
            raise ValueError(f"Missing ModelOpt weight for {prefix}")
        global_key = prefix + ".weight_scale_2"
        nvfp4 = index.source_shape(global_key) is not None
        shape = index.source_shape(weight_key)
        keys = [weight_key, key] + ([global_key] if nvfp4 else [])
        input_scale = prefix + ".input_scale"
        if index.source_shape(input_scale) is not None:
            keys.append(input_scale)
        if nvfp4:
            if index.source_dtype(weight_key) != "U8":
                raise ValueError(f"Invalid ModelOpt NVFP4 weight storage: {weight_key}")
            shape = (*shape[:-1], shape[-1] * 2)
        elif index.source_dtype(weight_key) not in {"F8_E4M3", "U8"}:
            raise ValueError(f"Invalid ModelOpt FP8 weight storage: {weight_key}")

        def materialize(keys=tuple(keys), prefix=prefix, nvfp4=nvfp4):
            source = {name: index.load_source(name) for name in keys}
            transformed, _ = _transform_modelopt_nvfp4_weights(source, quant)
            weight = transformed[prefix + ".weight"]
            if not nvfp4:
                return weight
            dense = mx.dequantize(
                weight,
                transformed[prefix + ".scales"],
                None,
                group_size=16,
                bits=4,
                mode="nvfp4",
            ).astype(mx.float32)
            scale = transformed[prefix + ".weight_scale_2"]
            if scale.ndim:
                scale = scale.reshape((-1,) + (1,) * (dense.ndim - 1))
            return (dense * scale).astype(mx.bfloat16)

        index.register_virtual(weight_key, shape, "BF16", materialize, hides=keys)
        registered += 1
    for key in index._index:
        if key.endswith((".k_scale", ".v_scale")):
            index._hidden.add(key)
        if (
            key.endswith(".weight")
            and index.source_dtype(key) == "U8"
            and key not in index._virtual
        ):
            raise ValueError(f"Unpaired packed ModelOpt weight: {key}")
    return registered


def _parent_digest(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def native_mtp_sidecar(model_path: str | Path, config: dict) -> Path | None:
    declaration = config.get("omlx_mtp")
    if declaration is None:
        return None
    if declaration != {"format": "mlx-vlm", "path": "mtp"}:
        raise ValueError("Invalid native Nemotron MTP sidecar declaration")
    root = Path(model_path).resolve()
    sidecar = (root / "mtp").resolve()
    if sidecar.parent != root:
        raise ValueError(
            "Nemotron MTP sidecar must remain inside its parent checkpoint"
        )
    head_config = json.loads((sidecar / "config.json").read_text())
    if head_config.get("model_type") != "nemotron_h_mtp" or head_config.get(
        "omlx_parent_config_sha256"
    ) != _parent_digest(config):
        raise ValueError("Native Nemotron MTP sidecar does not match its parent")
    if not any(sidecar.glob("*.safetensors")):
        raise ValueError("Native Nemotron MTP sidecar has no weights")
    return sidecar


def export_native_mtp(source: Path, output: Path, output_config: dict) -> None:
    from mlx_vlm.speculative.drafters.nemotron_h_mtp.split import NemotronHMTPSplitter

    source_config = json.loads((source / "config.json").read_text())
    text = source_config.get("text_config") or source_config.get("llm_config") or {}
    output_config["text_config"]["num_nextn_predict_layers"] = int(
        text.get("num_nextn_predict_layers", 0)
    )
    output_config["omlx_mtp"] = {"format": "mlx-vlm", "path": "mtp"}
    sidecar = NemotronHMTPSplitter().split(str(source), str(output / "mtp"))
    config_path = sidecar / "config.json"
    head_config = json.loads(config_path.read_text())
    head_config["omlx_parent_config_sha256"] = _parent_digest(output_config)
    config_path.write_text(json.dumps(head_config, indent=2) + "\n")
