# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).parents[1] / "omlx_web"


def test_chat_image_tools_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise production chat JavaScript")
    result = subprocess.run(
        [
            node, "--test",
            str(Path(__file__).with_name("chat_image_tools.test.cjs")),
            str(Path(__file__).with_name("chat_history.test.cjs")),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_image_tools_replace_imagine_navigation_and_locales():
    assert not (WEB_ROOT / "templates/imagine.html").exists()
    for template in ["chat.html", "dashboard/_navbar.html"]:
        source = (WEB_ROOT / "templates" / template).read_text()
        assert "/admin/imagine" not in source
        assert "navbar.tab.imagine" not in source
    for locale in (WEB_ROOT / "i18n").glob("*.json"):
        translations = json.loads(locale.read_text())
        assert translations["chat.image_download"]
        assert translations["chat.image_reuse"]
        assert translations["chat.error.history_storage"]
        assert not any(key.startswith("imagine.") for key in translations)
        assert "navbar.tab.imagine" not in translations
