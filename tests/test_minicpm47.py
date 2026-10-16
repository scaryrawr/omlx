# SPDX-License-Identifier: Apache-2.0
"""Exercise official MiniCPM models with oMLX serving extensions on CPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omlx.patches.minicpm47 import validate_target_config


def test_non_normalized_routing_remains_unsupported():
    validate_target_config({"text_config": {}})
    with pytest.raises(ValueError, match="normalized top-k"):
        validate_target_config({"text_config": {"norm_topk_prob": False}})


def test_official_minicpm_serving_contracts():
    script = """
import mlx.core as mx
mx.set_default_device(mx.cpu)
import pytest
raise SystemExit(pytest.main(['tests/minicpm47_native_cases.py', '-q', '--tb=short']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
