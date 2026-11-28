# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from omlx.admin import routes
from omlx.admin.hf_downloader import HFDownloader
from omlx.lora_adapter import (
    AdapterBinding,
    AdapterIssue,
    BaseCandidate,
    BaseRef,
    IssueKind,
    bind_adapter,
    parse_adapter,
)

pytestmark = pytest.mark.asyncio


@dataclass
class Entry:
    model_id: str
    model_path: str
    adapter: AdapterBinding | None = None
    model_format: str = "mlx"
    engine: object | None = None
    is_loading: bool = False


class Settings:
    def __init__(self):
        self.bases = {}
        self.deleted = []

    def get_adapter_bases(self):
        return self.bases.copy()

    def set_adapter_base(self, adapter_id, base_id):
        if base_id is None:
            self.bases.pop(adapter_id, None)
        else:
            self.bases[adapter_id] = base_id

    def delete_settings(self, model_id):
        self.deleted.append(model_id)
        self.bases.pop(model_id, None)

    def get_pinned_model_ids(self):
        return ["z-base"]

    def get_all_settings(self):
        return {}

    def list_profiles(self, model_id):
        return []


class Pool:
    def __init__(self, settings):
        self.settings = settings
        self._entries = {}
        self.adapter_issues = ()
        self.candidates = ()
        self.discoveries = []
        self.applied_overrides = None
        self.unresolved = None

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def get_loaded_model_ids(self):
        return [
            entry.model_id
            for entry in self._entries.values()
            if entry.engine is not None
        ]

    def get_status(self):
        return {
            "models": [
                {
                    "id": entry.model_id,
                    "model_path": entry.model_path,
                    "adapter": entry.adapter.as_status() if entry.adapter else None,
                }
                for entry in self._entries.values()
            ]
        }

    def adapter_base_candidates(self):
        return self.candidates

    def discover_models(self, dirs, pinned_ids):
        self.discoveries.append((dirs, pinned_ids))
        if self.unresolved is None:
            return
        source, issue = self.unresolved
        base_id = self.settings.get_adapter_bases().get(source.public_id)
        if base_id is None:
            self._entries.pop(source.public_id, None)
            self.adapter_issues = (issue,)
            return
        base = next(
            candidate for candidate in self.candidates if candidate.model_id == base_id
        )
        self._entries[source.public_id] = Entry(
            source.public_id,
            str(base.path),
            bind_adapter(source, BaseRef(base_id, base.path)),
        )
        self.adapter_issues = ()

    def apply_settings_overrides(self, settings):
        self.applied_overrides = settings.get_adapter_bases()


@pytest.fixture
def admin(monkeypatch, tmp_path):
    settings = Settings()
    pool = Pool(settings)
    roots = [tmp_path]
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: settings)
    monkeypatch.setattr(
        routes,
        "_get_global_settings",
        lambda: SimpleNamespace(get_effective_model_dirs=lambda: roots),
    )
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(routes, "_hf_downloader", None)
    return pool, settings, roots


@pytest.fixture
def adapter(admin, tmp_path, write_lora_adapter):
    pool, _, _ = admin
    source = parse_adapter(write_lora_adapter(tmp_path / "hub" / "tuned"))
    pool.candidates = (
        BaseCandidate("z-base", tmp_path / "z-base", frozenset({"org/base"})),
        BaseCandidate("a-base", tmp_path / "a-base", frozenset({"org/base"})),
        BaseCandidate("Base-4bit", tmp_path / "Base-4bit", frozenset()),
    )
    base = pool.candidates[0]
    entry = Entry(
        "tuned", str(base.path), bind_adapter(source, BaseRef(base.model_id, base.path))
    )
    pool._entries[entry.model_id] = entry
    return entry


def ambiguous_issue(path):
    return AdapterIssue(
        "tuned", path, IssueKind.BASE_AMBIGUOUS, "Choose a base.", ("a-base", "z-base")
    )


def issue_row(path):
    return {
        "id": "tuned",
        "kind": "base_ambiguous",
        "reason": "Choose a base.",
        "candidates": ["a-base", "z-base"],
        "path": str(path),
    }


async def test_import_mtplx_rejects_adapter_without_invoking_import(
    adapter, monkeypatch
):
    from omlx import oq

    def unexpected_import(model_path):
        pytest.fail(f"MTPLX importer was called with {model_path}")

    monkeypatch.setattr(oq, "import_mtplx_sidecar", unexpected_import)

    with pytest.raises(HTTPException) as exc:
        await routes.import_mtplx("tuned", is_admin=True)

    assert exc.value.status_code == 400
    assert "Import against the base model ID 'z-base' instead." in exc.value.detail


def adapter_row(adapter, base_id):
    return {
        "base_model_id": base_id,
        "rank": 4,
        "alpha": 8.0,
        "weight_bytes": adapter.source.weights_path.stat().st_size,
    }


async def set_base(base_id, adapter_id="tuned"):
    return await routes.set_adapter_base(
        adapter_id, routes.AdapterBaseRequest(base_model_id=base_id), is_admin=True
    )


@pytest.mark.parametrize(
    "state", ["automatic", "pinned", "unresolved", "loaded", "loading"]
)
async def test_get_base_lists_explicit_choices_and_binding(admin, adapter, state):
    pool, settings, _ = admin
    if state == "pinned":
        settings.set_adapter_base("tuned", "z-base")
    elif state == "unresolved":
        pool.adapter_issues = (ambiguous_issue(adapter.adapter.source.path),)
        del pool._entries["tuned"]
    elif state == "loaded":
        adapter.engine = object()
    elif state == "loading":
        adapter.is_loading = True

    assert await routes.get_adapter_base("tuned", is_admin=True) == {
        "base_model_id": "z-base" if state == "pinned" else None,
        "resolved_base_model_id": None if state == "unresolved" else "z-base",
        "candidates": ["Base-4bit", "a-base", "z-base"],
        "can_change": state not in {"loaded", "loading"},
    }


async def test_get_base_rejects_plain_models(admin, adapter):
    pool, _, _ = admin
    pool._entries["z-base"] = Entry("z-base", adapter.model_path)
    with pytest.raises(HTTPException) as exc:
        await routes.get_adapter_base("z-base", is_admin=True)
    assert exc.value.status_code == 404


async def test_list_models_includes_binding_and_all_issues(
    admin, adapter, tmp_path, monkeypatch
):
    pool, _, _ = admin
    monkeypatch.setattr(routes, "_get_global_settings", lambda: None)
    monkeypatch.setattr(routes, "stage_adapter_for", lambda *_: object())
    pool._entries["z-base"] = Entry("z-base", adapter.model_path)
    paths = [tmp_path / "one" / "tuned", tmp_path / "two" / "tuned"]
    pool.adapter_issues = tuple(ambiguous_issue(path) for path in paths)

    result = await routes.list_models(is_admin=True)

    rows = {row["id"]: row for row in result["models"]}
    assert rows["tuned"]["adapter"] == adapter_row(adapter.adapter, "z-base")
    assert rows["z-base"]["adapter"] is None
    assert rows["z-base"]["prefill_helper_supported"] is True
    assert rows["tuned"]["prefill_helper_supported"] is False
    assert result["adapter_issues"] == [issue_row(path) for path in paths]


async def test_adapter_display_name_is_distinct_from_organized_base(
    admin, adapter, monkeypatch
):
    pool, _, roots = admin
    monkeypatch.setattr(routes, "_get_global_settings", lambda: None)
    monkeypatch.setattr(routes, "_model_dirs_for_display", lambda _: roots)
    adapter.model_path = str(roots[0] / "org" / "z-base")
    pool._entries["z-base"] = Entry("z-base", adapter.model_path)

    result = await routes.list_models(is_admin=True)

    rows = {row["id"]: row for row in result["models"]}
    assert rows["tuned"]["display_name"] == "tuned"
    assert rows["z-base"]["display_name"] == "org/z-base"
    assert rows["tuned"]["model_path"] == rows["z-base"]["model_path"]


async def test_admin_rejects_remote_prefill_on_loaded_adapter(admin, adapter):
    from unittest.mock import Mock

    adapter.engine = Mock()
    request = routes.ModelSettingsRequest(prefill_helper_node_id="peer")
    with pytest.raises(HTTPException, match="do not support remote prefill") as caught:
        await routes.update_model_settings("tuned", request, is_admin=True)
    assert caught.value.status_code == 400
    adapter.engine.configure_prefill_helper.assert_not_called()


@pytest.mark.parametrize("dependency", ["_get_engine_pool", "_get_settings_manager"])
async def test_set_base_requires_initialized_services(admin, monkeypatch, dependency):
    monkeypatch.setattr(routes, dependency, lambda: None)
    with pytest.raises(HTTPException) as exc:
        await set_base(None)
    assert exc.value.status_code == 503


async def test_set_base_rejects_unknown_adapter(admin):
    with pytest.raises(HTTPException) as exc:
        await set_base("unknown")
    assert exc.value.status_code == 404


@pytest.mark.parametrize("base_id", ["helper", "tuned", "unknown"])
async def test_set_base_checks_chat_candidates(admin, adapter, base_id):
    pool, settings, _ = admin
    pool._entries["helper"] = Entry("helper", adapter.model_path)
    with pytest.raises(HTTPException) as exc:
        await set_base(base_id)
    assert exc.value.status_code == 400
    assert base_id in exc.value.detail
    assert settings.get_adapter_bases() == {}


@pytest.mark.parametrize("state", ["loaded", "loading"])
@pytest.mark.parametrize("base_id", ["a-base", None])
async def test_set_base_cannot_change_active_binding(admin, adapter, state, base_id):
    pool, settings, _ = admin
    if state == "loaded":
        adapter.engine = object()
    else:
        adapter.is_loading = True
    settings.set_adapter_base("tuned", "z-base")

    with pytest.raises(HTTPException) as exc:
        await set_base(base_id)

    assert exc.value.status_code == 409
    assert settings.get_adapter_bases() == {"tuned": "z-base"}
    assert not pool.discoveries


@pytest.mark.parametrize("collision", ["bound", "base", "unresolved", "reserved"])
async def test_set_base_refuses_colliding_ids(admin, adapter, tmp_path, collision):
    pool, settings, _ = admin
    source_path = adapter.adapter.source.path
    other_path = tmp_path / "other" / "tuned"
    collision_issue = AdapterIssue(
        "tuned", other_path, IssueKind.ID_COLLISION, "Rename the adapter."
    )
    pool.adapter_issues = (collision_issue,)
    if collision == "base":
        pool._entries["tuned"] = Entry("tuned", str(tmp_path / "tuned"))
    elif collision in {"unresolved", "reserved"}:
        del pool._entries["tuned"]
        if collision == "unresolved":
            pool.adapter_issues = (
                ambiguous_issue(source_path),
                ambiguous_issue(other_path),
            )

    with pytest.raises(HTTPException) as exc:
        await set_base("a-base")

    assert exc.value.status_code == 409
    assert settings.get_adapter_bases() == {}
    assert not pool.discoveries


@pytest.mark.parametrize("already_bound", [True, False])
async def test_set_base_pins_then_clears_and_reports_fresh_discovery(
    admin, adapter, already_bound
):
    pool, settings, roots = admin
    source = adapter.adapter.source
    issue = ambiguous_issue(source.path)
    pool.unresolved = (source, issue)
    if not already_bound:
        del pool._entries["tuned"]
        pool.adapter_issues = (issue,)

    assert await set_base("a-base") == {
        "success": True,
        "id": "tuned",
        "base_model_id": "a-base",
        "adapter": adapter_row(adapter.adapter, "a-base"),
        "issue": None,
    }
    assert settings.get_adapter_bases() == pool.applied_overrides == {"tuned": "a-base"}
    discovery = ([str(roots[0])], ["z-base"])
    assert pool.discoveries == [discovery]

    assert await set_base(None) == {
        "success": True,
        "id": "tuned",
        "base_model_id": None,
        "adapter": None,
        "issue": issue_row(source.path),
    }
    assert settings.get_adapter_bases() == pool.applied_overrides == {}
    assert pool.discoveries == [discovery, discovery]


@pytest.mark.parametrize("pool_state", ["ready", "none", "uninitialized", "plain"])
async def test_hf_model_info_reports_matching_local_bases(
    admin, adapter, monkeypatch, pool_state
):
    payload = {"adapter": None, "unsupported_reason": None}
    if pool_state != "plain":
        payload["adapter"] = {"base_repo_id": "Org/Base", "rank": 4}

    async def get_info(**kwargs):
        return payload

    monkeypatch.setattr(HFDownloader, "get_model_info", get_info)
    if pool_state == "none":
        monkeypatch.setattr(routes, "_get_engine_pool", lambda: None)
    elif pool_state == "uninitialized":

        def no_pool():
            raise HTTPException(status_code=503)

        monkeypatch.setattr(routes, "_get_engine_pool", no_pool)

    result = await routes.get_hf_model_info(
        Request({"type": "http", "headers": []}), repo_id="hub/tuned", is_admin=True
    )
    if pool_state == "plain":
        assert result == {"adapter": None, "unsupported_reason": None}
    else:
        assert result["adapter"] == {
            "base_repo_id": "Org/Base",
            "rank": 4,
            "local_base_ids": ["a-base", "z-base"] if pool_state == "ready" else [],
        }


async def test_hf_listing_includes_flat_and_nested_adapters(
    admin, tmp_path, write_lora_adapter
):
    flat = write_lora_adapter(tmp_path / "flat")
    nested = write_lora_adapter(tmp_path / "hub" / "nested")

    result = await routes.list_hf_models(is_admin=True)

    assert {row["name"]: (row["path"], row["repo_id"]) for row in result["models"]} == {
        "flat": (str(flat), None),
        "nested": (str(nested), "hub/nested"),
    }


def write_base(path: Path):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (path / "model.safetensors").write_bytes(b"base weights")
    return path


async def test_delete_adapter_keeps_base_and_releases_state(admin, adapter):
    pool, settings, roots = admin
    base = write_base(Path(adapter.model_path))
    path = adapter.adapter.source.path
    settings.set_adapter_base("tuned", "z-base")

    result = await routes.delete_hf_model("tuned", is_admin=True)

    assert result["success"] is True
    assert not path.exists()
    assert (base / "model.safetensors").read_bytes() == b"base weights"
    assert (base / "config.json").is_file()
    assert pool.get_entry("tuned") is None
    assert settings.deleted == ["tuned"]
    assert settings.get_adapter_bases() == pool.applied_overrides == {}
    assert pool.discoveries == [([str(roots[0])], ["z-base"])]


@pytest.mark.parametrize("delete_adapter", [False, True])
async def test_delete_same_basename_matches_registered_kind(
    admin, adapter, tmp_path, write_lora_adapter, delete_adapter
):
    pool, _, roots = admin
    other_root = tmp_path / "other"
    checkpoint = write_base(other_root / "tuned")
    path = adapter.adapter.source.path
    if delete_adapter:
        duplicate = write_lora_adapter(other_root / "hub" / "tuned")
        roots.insert(0, other_root)
    else:
        pool._entries["tuned"] = Entry("tuned", str(checkpoint))
        roots.append(other_root)

    result = await routes.delete_hf_model("tuned", is_admin=True)

    assert result["success"] is True
    if delete_adapter:
        assert (checkpoint / "model.safetensors").read_bytes() == b"base weights"
        assert duplicate.is_dir()
        assert not path.exists()
    else:
        assert not checkpoint.exists()
        assert path.is_dir()
