# SPDX-License-Identifier: Apache-2.0
"""Header-only DSV4 JANG bundle audit tests."""

import json

import pytest

from omlx.utils import model_loading

np = pytest.importorskip("numpy")
safetensors_numpy = pytest.importorskip("safetensors.numpy")
save_file = safetensors_numpy.save_file


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _write_dsv4_config(model_dir, extra=None):
    config = {"model_type": "deepseek_v4"}
    if extra:
        config.update(extra)
    _write_json(model_dir / "config.json", config)


def _save_model_safetensors(model_dir, tensors):
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))


def _routed_tensors(projections_by_layer):
    tensors = {}
    for layer, projections in projections_by_layer.items():
        for projection, bits in projections.items():
            base = f"layers.{layer}.mlp.switch_mlp.{projection}"
            tensors[f"{base}.tq_bits"] = np.array([bits], dtype=np.int32)
            tensors[f"{base}.tq_packed"] = np.zeros((1, 8, 1), dtype=np.uint32)
    return tensors


def test_dsv4_weight_map_reads_index_without_shards(tmp_path):
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                "layers.0.attn.attn_sink": "model-00001-of-00001.safetensors"
            },
        },
    )

    assert model_loading._dsv4_weight_map(tmp_path) == {
        "layers.0.attn.attn_sink": "model-00001-of-00001.safetensors"
    }


def test_dsv4_weight_map_falls_back_to_safetensors_headers(tmp_path):
    _save_model_safetensors(
        tmp_path,
        {"layers.0.attn.attn_sink": np.zeros((1,), dtype=np.float32)},
    )

    assert model_loading._dsv4_weight_map(tmp_path) == {
        "layers.0.attn.attn_sink": "model-00001-of-00001.safetensors"
    }


def test_validate_dsv4_control_tensors_accepts_f32_controls(tmp_path):
    _write_dsv4_config(tmp_path)
    _save_model_safetensors(
        tmp_path,
        {
            "hc_head_fn": np.zeros((1,), dtype=np.float32),
            "layers.0.hc_attn_scale": np.zeros((1,), dtype=np.float32),
            "layers.0.attn.attn_sink": np.zeros((1,), dtype=np.float32),
            "layers.0.ffn.gate.bias": np.zeros((1,), dtype=np.float32),
        },
    )

    model_loading._validate_dsv4_control_tensors(tmp_path)
    report = model_loading._audit_dsv4_control_tensor_dtypes(tmp_path)
    assert report["critical_count"] == 4
    assert report["non_f32_count"] == 0


def test_validate_dsv4_control_tensors_rejects_missing_controls(tmp_path):
    _write_dsv4_config(tmp_path)
    _save_model_safetensors(
        tmp_path,
        {"layers.0.self_attn.q_proj.weight": np.zeros((1,), dtype=np.float32)},
    )

    with pytest.raises(RuntimeError, match="No DSV4 critical control tensors"):
        model_loading._validate_dsv4_control_tensors(tmp_path)


def test_validate_dsv4_control_tensors_rejects_non_f32_with_legacy_hint(tmp_path):
    model_dir = tmp_path / "DeepSeek-V4-Flash-JANGTQ"
    model_dir.mkdir()
    _write_dsv4_config(
        model_dir,
        {"_name_or_path": "JANGQ-AI/DeepSeek-V4-Flash-JANGTQ"},
    )
    _save_model_safetensors(
        model_dir,
        {"layers.0.attn.attn_sink": np.zeros((1,), dtype=np.float16)},
    )

    with pytest.raises(RuntimeError) as exc_info:
        model_loading._validate_dsv4_control_tensors(model_dir)

    message = str(exc_info.value)
    assert "critical control tensors are not F32" in message
    assert "V3-F32" in message
    assert "layers.0.attn.attn_sink=F16" in message


def test_validate_dsv4_control_tensors_ignores_non_dsv4(tmp_path):
    _write_json(tmp_path / "config.json", {"model_type": "llama"})

    model_loading._validate_dsv4_control_tensors(tmp_path)
    assert model_loading._audit_dsv4_control_tensor_dtypes(tmp_path)["checked"] is False


def test_audit_dsv4_artifact_bit_plan_accepts_matching_metadata_and_sidecar(tmp_path):
    _write_dsv4_config(
        tmp_path,
        {
            "quantization": {
                "routed_experts": {
                    "bits": 2,
                    "bit_plan": {"routed_layer_bits": {"1": 4}},
                }
            }
        },
    )
    _write_json(tmp_path / "jang_config.json", {"mxtq_seed": 123})
    _save_model_safetensors(
        tmp_path,
        _routed_tensors(
            {
                "0": {"gate_proj": 2, "up_proj": 2, "down_proj": 2},
                "1": {"gate_proj": 4, "up_proj": 4, "down_proj": 4},
            }
        ),
    )
    save_file(
        {
            "codebook.16.2": np.zeros((1,), dtype=np.float32),
            "codebook.8.4": np.zeros((1,), dtype=np.float32),
            "signs.16.123": np.zeros((1,), dtype=np.float32),
            "signs.8.123": np.zeros((1,), dtype=np.float32),
        },
        str(tmp_path / "jangtq_runtime.safetensors"),
    )

    report = model_loading._audit_dsv4_artifact_bit_plan(tmp_path)

    assert report["checked"] is True
    assert report["routed_layer_bits"] == {"0": 2, "1": 4}
    assert report["routed_bit_counts"] == {"2": 1, "4": 1}
    assert report["metadata_matches_actual"] is True
    assert report["sidecar"]["missing_keys"] == []
    assert report["issues"] == []


def test_audit_dsv4_artifact_bit_plan_detects_mixed_projection_bits(tmp_path):
    _write_dsv4_config(tmp_path)
    _save_model_safetensors(
        tmp_path,
        _routed_tensors({"0": {"gate_proj": 2, "up_proj": 4, "down_proj": 2}}),
    )

    report = model_loading._audit_dsv4_artifact_bit_plan(tmp_path)

    assert report["projection_mismatches"] == [
        {
            "layer": "0",
            "projections": {"down_proj": 2, "gate_proj": 2, "up_proj": 4},
        }
    ]
    assert any("mixed tq_bits" in issue for issue in report["issues"])


def test_audit_dsv4_artifact_bit_plan_accepts_stamped_mixed_projection_bits(tmp_path):
    _write_dsv4_config(
        tmp_path,
        {
            "mxtq_bits_by_role": {
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 4,
                }
            }
        },
    )
    _save_model_safetensors(
        tmp_path,
        _routed_tensors({"0": {"gate_proj": 2, "up_proj": 2, "down_proj": 4}}),
    )
    save_file(
        {
            "codebook.16.2": np.zeros((1,), dtype=np.float32),
            "signs.16.42": np.zeros((1,), dtype=np.float32),
            "codebook.8.4": np.zeros((1,), dtype=np.float32),
            "signs.8.42": np.zeros((1,), dtype=np.float32),
        },
        str(tmp_path / "jangtq_runtime.safetensors"),
    )

    report = model_loading._audit_dsv4_artifact_bit_plan(tmp_path)

    assert report["routed_projection_bits"] == {
        "0": {"down_proj": 4, "gate_proj": 2, "up_proj": 2}
    }
    assert report["metadata_routed_projection_bits"] == {
        "down_proj": 4,
        "gate_proj": 2,
        "up_proj": 2,
    }
    assert report["projection_mismatches"] == []
    assert report["issues"] == []


def test_audit_dsv4_artifact_bit_plan_detects_missing_projection(tmp_path):
    _write_dsv4_config(tmp_path)
    _save_model_safetensors(
        tmp_path,
        _routed_tensors({"0": {"gate_proj": 2, "up_proj": 2}}),
    )

    report = model_loading._audit_dsv4_artifact_bit_plan(tmp_path)

    assert report["missing_projection_layers"] == [
        {"layer": "0", "missing": ["down_proj"]}
    ]
    assert any(
        "missing routed tq_bits projections" in issue for issue in report["issues"]
    )


def test_audit_dsv4_artifact_bit_plan_detects_metadata_mismatch(tmp_path):
    _write_dsv4_config(
        tmp_path,
        {
            "quantization": {
                "routed_experts": {
                    "bits": 2,
                    "bit_plan": {"routed_layer_bits": {"1": 3}},
                }
            }
        },
    )
    _save_model_safetensors(
        tmp_path,
        _routed_tensors(
            {
                "0": {"gate_proj": 2, "up_proj": 2, "down_proj": 2},
                "1": {"gate_proj": 4, "up_proj": 4, "down_proj": 4},
            }
        ),
    )

    report = model_loading._audit_dsv4_artifact_bit_plan(tmp_path)

    assert report["metadata_matches_actual"] is False
    assert report["metadata_layer_value_mismatches"] == {
        "1": {"metadata": 3, "actual": 4}
    }
    assert any("metadata routed_layer_bits" in issue for issue in report["issues"])


def test_audit_dsv4_artifact_bit_plan_detects_missing_sidecar_keys(tmp_path):
    _write_dsv4_config(tmp_path)
    _write_json(tmp_path / "jang_config.json", {"mxtq_seed": 7})
    _save_model_safetensors(
        tmp_path,
        _routed_tensors({"0": {"gate_proj": 2, "up_proj": 2, "down_proj": 2}}),
    )
    save_file(
        {"codebook.16.2": np.zeros((1,), dtype=np.float32)},
        str(tmp_path / "jangtq_runtime.safetensors"),
    )

    report = model_loading._audit_dsv4_artifact_bit_plan(tmp_path)

    assert report["sidecar"]["required_keys"] == ["codebook.16.2", "signs.16.7"]
    assert report["sidecar"]["missing_keys"] == ["signs.16.7"]
    assert any("sidecar missing required" in issue for issue in report["issues"])
