# SPDX-License-Identifier: Apache-2.0
"""Localization coverage for chat image uploads; behavior runs in the JS suite."""

import json
from pathlib import Path

import pytest

I18N_DIR = Path(__file__).parents[1] / "omlx_web" / "i18n"
REQUIRED_IMAGE_KEYS = {
    "chat.upload_image",
    "chat.remove_image",
    "chat.image_not_available",
    "chat.image_preview",
    "chat.error.invalid_image_type",
    "chat.error.image_too_large",
    "chat.error.image_load_failed",
}


@pytest.mark.parametrize("locale", sorted(I18N_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_i18n_image_keys_present(locale):
    translations = json.loads(locale.read_text(encoding="utf-8"))
    for key in REQUIRED_IMAGE_KEYS:
        assert translations.get(key), f"Missing or empty key {key!r} in {locale.name}"
