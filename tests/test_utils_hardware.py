# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.hardware module."""

import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

from omlx.utils.hardware import (
    DEFAULT_MEMORY_BYTES,
    HardwareInfo,
    format_bytes,
    get_chip_name,
    get_max_working_set_bytes,
    get_total_memory_bytes,
    get_total_memory_gb,
    is_apple_silicon,
    is_mlx_available,
)


class TestHardwareInfo:
    """Test cases for HardwareInfo dataclass."""

    def test_hardware_info_creation(self):
        """Test creating HardwareInfo with all fields."""
        info = HardwareInfo(
            chip_name="Apple M4 Pro",
            total_memory_gb=48.0,
            max_working_set_bytes=36 * 1024**3,
            mlx_device_name="Apple M4 Pro",
        )
        assert info.chip_name == "Apple M4 Pro"
        assert info.total_memory_gb == 48.0
        assert info.max_working_set_bytes == 36 * 1024**3
        assert info.mlx_device_name == "Apple M4 Pro"

    def test_hardware_info_default_mlx_device(self):
        """Test HardwareInfo with default mlx_device_name."""
        info = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
        )
        assert info.mlx_device_name is None

    def test_hardware_info_various_chips(self):
        """Test HardwareInfo with various Apple Silicon chips."""
        chips = [
            ("Apple M1", 8.0),
            ("Apple M1 Pro", 16.0),
            ("Apple M1 Max", 32.0),
            ("Apple M1 Ultra", 64.0),
            ("Apple M2", 8.0),
            ("Apple M2 Pro", 16.0),
            ("Apple M2 Max", 32.0),
            ("Apple M2 Ultra", 128.0),
            ("Apple M3", 8.0),
            ("Apple M3 Pro", 18.0),
            ("Apple M3 Max", 36.0),
            ("Apple M4", 16.0),
            ("Apple M4 Pro", 48.0),
            ("Apple M4 Max", 128.0),
        ]
        for chip_name, memory_gb in chips:
            info = HardwareInfo(
                chip_name=chip_name,
                total_memory_gb=memory_gb,
                max_working_set_bytes=int(memory_gb * 0.75 * 1024**3),
            )
            assert info.chip_name == chip_name
            assert info.total_memory_gb == memory_gb


class TestGetChipName:
    """Test cases for get_chip_name function."""

    def test_get_chip_name_success(self):
        """Test get_chip_name with successful sysctl call."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Apple M4 Pro\n", returncode=0)
            result = get_chip_name()
            assert result == "Apple M4 Pro"
            mock_run.assert_called_once()

    def test_get_chip_name_fallback(self):
        """Test get_chip_name fallback when sysctl fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("sysctl not available")
            result = get_chip_name()
            assert result == "Apple Silicon"


class TestGetTotalMemoryBytes:
    """Test cases for get_total_memory_bytes function."""

    def test_get_total_memory_bytes_sysctl_success(self):
        """Test get_total_memory_bytes with successful sysctl call."""
        expected_bytes = 48 * 1024**3  # 48 GB
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=f"{expected_bytes}\n", returncode=0
            )
            result = get_total_memory_bytes()
            assert result == expected_bytes

    def test_get_total_memory_bytes_default_fallback(self):
        """Test get_total_memory_bytes default fallback."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("sysctl failed")
            # Mock HAS_MLX to False so MLX fallback is skipped
            with patch("omlx.utils.hardware.HAS_MLX", False):
                result = get_total_memory_bytes()
                assert result == DEFAULT_MEMORY_BYTES


class TestGetTotalMemoryGb:
    """Test cases for get_total_memory_gb function."""

    def test_get_total_memory_gb_conversion(self):
        """Test that get_total_memory_gb correctly converts bytes to GB."""
        with patch(
            "omlx.utils.hardware.get_total_memory_bytes", return_value=16 * 1024**3
        ):
            result = get_total_memory_gb()
            assert result == 16.0

    def test_get_total_memory_gb_fractional(self):
        """Test get_total_memory_gb with fractional values."""
        with patch(
            "omlx.utils.hardware.get_total_memory_bytes",
            return_value=int(18.5 * 1024**3),
        ):
            result = get_total_memory_gb()
            assert abs(result - 18.5) < 0.01


class TestGetMaxWorkingSetBytes:
    """Test cases for get_max_working_set_bytes function."""

    def test_uses_mlx_max_working_set_when_available(self):
        with patch("omlx.utils.hardware.HAS_MLX", True), patch(
            "omlx.utils.hardware.mx"
        ) as mock_mx:
            mock_mx.metal.is_available.return_value = True
            mock_mx.device_info.return_value = {
                "max_recommended_working_set_size": 36 * 1024**3
            }

            assert get_max_working_set_bytes() == 36 * 1024**3

    def test_falls_back_to_total_memory_without_psutil(self):
        with patch("omlx.utils.hardware.HAS_MLX", False), patch(
            "omlx.utils.hardware.get_total_memory_bytes",
            return_value=64 * 1024**3,
        ):
            assert get_max_working_set_bytes() == 48 * 1024**3


class TestIsAppleSilicon:
    """Test cases for is_apple_silicon function."""

    @pytest.mark.parametrize(
        "os_name, architecture, expected",
        [
            pytest.param("darwin", "arm64", True, id="apple-silicon"),
            pytest.param("linux", "arm64", False, id="wrong-platform"),
            pytest.param("darwin", "x86_64", False, id="wrong-architecture"),
        ],
    )
    def test_is_apple_silicon(self, os_name, architecture, expected):
        with patch.object(sys, "platform", os_name), patch.object(
            platform, "machine", return_value=architecture
        ):
            assert is_apple_silicon() is expected


class TestIsMlxAvailable:
    """Test cases for is_mlx_available function."""

    def test_is_mlx_available_not_apple_silicon(self):
        """Test is_mlx_available returns False on non-Apple Silicon."""
        with patch("omlx.utils.hardware.is_apple_silicon", return_value=False):
            assert is_mlx_available() is False

    def test_is_mlx_available_without_mlx(self):
        with patch("omlx.utils.hardware.is_apple_silicon", return_value=True), patch(
            "omlx.utils.hardware.HAS_MLX", False
        ):
            assert is_mlx_available() is False

    @pytest.mark.parametrize(
        "array_error, expected",
        [
            pytest.param(None, True, id="working-device"),
            pytest.param(RuntimeError("device unavailable"), False, id="device-error"),
        ],
    )
    def test_is_mlx_available_checks_device(self, array_error, expected):
        with patch("omlx.utils.hardware.is_apple_silicon", return_value=True), patch(
            "omlx.utils.hardware.HAS_MLX", True
        ), patch("omlx.utils.hardware.mx") as mock_mx:
            mock_mx.array.side_effect = array_error
            assert is_mlx_available() is expected
            mock_mx.array.assert_called_once_with([1.0, 2.0, 3.0])


class TestFormatBytesHardware:
    """Test cases for format_bytes function in hardware module."""

    @pytest.mark.parametrize(
        "size, expected",
        [
            (1024**3, "1.00 GB"),
            (16 * 1024**3, "16.00 GB"),
            (48 * 1024**3, "48.00 GB"),
            (1024**2, "1.00 MB"),
            (512 * 1024**2, "512.00 MB"),
            (1024, "1.00 KB"),
            (512 * 1024, "512.00 KB"),
            (0, "0 B"),
            (512, "512 B"),
            (1023, "1023 B"),
        ],
    )
    def test_format_bytes(self, size, expected):
        assert format_bytes(size) == expected


class TestDefaultMemoryBytes:
    """Test cases for DEFAULT_MEMORY_BYTES constant."""

    def test_default_memory_bytes_value(self):
        """Test that DEFAULT_MEMORY_BYTES is 8 GB."""
        assert DEFAULT_MEMORY_BYTES == 8 * 1024**3
