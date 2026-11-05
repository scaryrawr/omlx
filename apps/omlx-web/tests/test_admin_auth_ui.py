# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin login, dashboard, and chat page handlers."""

from unittest.mock import MagicMock

import pytest
from omlx_web import routes as webui

from omlx.admin import routes as admin_routes
from omlx.settings import GlobalSettings


@pytest.fixture
def page_host(monkeypatch, tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.auth.api_key = "test-key"
    settings.server.host = "127.0.0.1"
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(admin_routes, "verify_session", lambda request: False)
    monkeypatch.setattr(
        webui,
        "_host",
        webui.WebUIHost(
            version="test",
            require_admin=admin_routes.require_admin,
            is_admin=admin_routes.is_admin_request,
            ui_language=admin_routes.configured_ui_language,
            main_api_key=admin_routes.configured_api_key,
        ),
    )
    render = MagicMock()
    monkeypatch.setattr(webui.templates, "TemplateResponse", render)
    return settings, render


async def test_login_page_uses_request_first_template_signature(page_host):
    _, render = page_host
    request = MagicMock()

    result = await webui.login_page(request=request)

    assert result is render.return_value
    render.assert_called_once_with(
        request, "login.html", {"api_key_configured": True}
    )


async def test_dashboard_page_uses_request_first_template_signature(page_host):
    _, render = page_host
    request = MagicMock()

    result = await webui.dashboard_page(request=request, is_admin=True)

    assert result is render.return_value
    render.assert_called_once_with(request, "dashboard.html", {})


@pytest.mark.parametrize(
    "api_key, settings_available",
    [("test-chat-key", True), (None, True), (None, False)],
    ids=["configured-key", "no-key", "no-settings"],
)
async def test_chat_page_injects_main_api_key(
    page_host, monkeypatch, api_key, settings_available
):
    settings, render = page_host
    settings.auth.api_key = api_key
    if not settings_available:
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: None)
    request = MagicMock()

    result = await webui.chat_page(request=request, is_admin=True)

    assert result is render.return_value
    render.assert_called_once_with(request, "chat.html", {"api_key": api_key or ""})


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"], ids=["loopback", "network"])
async def test_login_bypass_is_restricted_to_loopback(page_host, host):
    settings, render = page_host
    settings.auth.skip_api_key_verification = True
    settings.server.host = host
    request = MagicMock()

    result = await webui.login_page(request=request)

    if host == "127.0.0.1":
        assert result.status_code == 302
        assert result.headers["location"] == "/admin/dashboard"
        render.assert_not_called()
    else:
        assert result is render.return_value
        render.assert_called_once_with(
            request, "login.html", {"api_key_configured": True}
        )
