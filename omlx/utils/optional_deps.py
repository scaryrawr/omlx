# SPDX-License-Identifier: Apache-2.0
"""Helpers for optional runtime dependencies."""

from __future__ import annotations

from importlib.util import find_spec

MFLUX_PACKAGE = "mflux"
IMAGE_EXTRA_INSTALL_COMMAND = "pip install 'omlx[image]'"
IMAGE_EXTRA_SOURCE_INSTALL_COMMAND = "pip install -e '.[image]'"
IMAGE_EXTRA_INSTALL_HINT = (
    f"Install image support with: {IMAGE_EXTRA_INSTALL_COMMAND}. "
    f"If you are running from a source checkout, use: {IMAGE_EXTRA_SOURCE_INSTALL_COMMAND}."
)
MFLUX_MISSING_MESSAGE = f"mflux is required for image inference. {IMAGE_EXTRA_INSTALL_HINT}"


def is_mflux_available() -> bool:
    """Return True when mflux can be imported, without importing it."""
    return find_spec(MFLUX_PACKAGE) is not None


def require_mflux_available() -> None:
    """Raise ImportError with install guidance when mflux is unavailable."""
    if not is_mflux_available():
        raise ImportError(MFLUX_MISSING_MESSAGE)
