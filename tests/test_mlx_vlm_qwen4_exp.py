# SPDX-License-Identifier: Apache-2.0
"""Compatibility contract for upstream Qwen3.8-Flash-Next support."""

from __future__ import annotations

import importlib


def test_qwen4_exp_upstream_module_exports():
    module = importlib.import_module("mlx_vlm.models.qwen4_exp")

    for name in (
        "LanguageModel",
        "Model",
        "ModelConfig",
        "Qwen3VLProcessor",
        "TextConfig",
        "VisionConfig",
        "VisionModel",
    ):
        assert getattr(module, name) is not None
