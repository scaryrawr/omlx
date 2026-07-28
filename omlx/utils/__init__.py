# SPDX-License-Identifier: Apache-2.0
"""
Utility modules for oMLX.

This package contains shared utility functions and helpers used across
the oMLX codebase.
"""

from .formatting import format_bytes as format_bytes_util
from .hardware import (
    DEFAULT_MEMORY_BYTES,
    HardwareInfo,
    detect_hardware,
    format_bytes,
    get_chip_name,
    get_max_working_set_bytes,
    get_mlx_device_name,
    get_mlx_lm_version,
    get_mlx_version,
    get_mlx_vlm_version,
    get_total_memory_bytes,
    get_total_memory_gb,
    is_apple_silicon,
    is_mlx_available,
)
from .install import (
    get_cli_command_prefix,
    get_cli_prefix,
    get_install_method,
    is_app_bundle,
    is_homebrew,
)
from .tokenizer import apply_qwen3_fix, get_tokenizer_config

__all__ = [
    # Tokenizer utilities
    "get_tokenizer_config",
    "apply_qwen3_fix",
    # Hardware utilities
    "HardwareInfo",
    "detect_hardware",
    "get_chip_name",
    "get_total_memory_bytes",
    "get_total_memory_gb",
    "get_max_working_set_bytes",
    "get_mlx_device_name",
    "is_mlx_available",
    "is_apple_silicon",
    "get_mlx_version",
    "get_mlx_lm_version",
    "get_mlx_vlm_version",
    "format_bytes",
    "DEFAULT_MEMORY_BYTES",
    # Install detection
    "get_cli_command_prefix",
    "get_cli_prefix",
    "get_install_method",
    "is_app_bundle",
    "is_homebrew",
]
