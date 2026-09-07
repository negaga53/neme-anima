"""Tests for /api/projects routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app
from neme_anima.storage.project import Project


@pytest.fixture
def app(tmp_path: Path):
    return create_app(state_dir=tmp_path / "state")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_empty(client):
    resp = await client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_project(client, tmp_path: Path):
    target = tmp_path / "newproj"
    resp = await client.post("/api/projects", json={
        "name": "newproj",
        "folder": str(target),
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["slug"] == "newproj"
    assert (target / "project.json").exists()


async def test_create_rejects_existing_folder(client, tmp_path: Path):
    target = tmp_path / "exists"
    target.mkdir()
    resp = await client.post("/api/projects", json={
        "name": "x", "folder": str(target),
    })
    assert resp.status_code == 409


async def test_get_project_returns_full_state(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.get("/api/projects/p")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "p"
    assert body["sources"] == []
    assert body["refs"] == []


async def test_get_missing_returns_404(client):
    resp = await client.get("/api/projects/nope")
    assert resp.status_code == 404


async def test_source_view_includes_extraction_cache_state(
    client, tmp_path: Path,
):
    """The Sources tab needs ``extraction_cache`` per source to drive the
    smart Extract / Re-process button states. A freshly-added video with
    no detection cache must report 'none' so the UI keeps Re-process
    disabled and Extract primary."""
    project = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    project.add_source(vid)
    await client.post(
        "/api/projects/register", json={"folder": str(tmp_path / "p")},
    )
    body = (await client.get("/api/projects/p")).json()
    assert body["sources"][0]["extraction_cache"] == "none"


async def test_source_view_reports_current_after_stamp(
    client, tmp_path: Path,
):
    """Stamping the cache snapshot externally (as run_extract does at
    the end of the track stage) must flip the state to 'current' on the
    next project view fetch — that's what unmutes Re-process and mutes
    Extract."""
    from neme_anima.config import Thresholds
    from neme_anima.extraction_cache import stamp_meta

    project = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    project.add_source(vid)
    # Simulate a completed extract by writing a stub parquet + the meta.
    cache_dir = project.cache_dir_for("ep01")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tracklets.parquet").write_bytes(b"stub")
    stamp_meta(project, "ep01", Thresholds())

    await client.post(
        "/api/projects/register", json={"folder": str(tmp_path / "p")},
    )
    body = (await client.get("/api/projects/p")).json()
    assert body["sources"][0]["extraction_cache"] == "current"


async def test_source_view_reports_stale_after_threshold_drift(
    client, tmp_path: Path,
):
    """If the user changes a scan-affecting threshold (here: detect's
    frame stride) AFTER the cache was stamped, the project view must
    surface 'stale' so the UI can warn the user that Re-process would
    silently use the old detections."""
    from neme_anima.config import Thresholds
    from neme_anima.extraction_cache import stamp_meta

    project = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    project.add_source(vid)
    cache_dir = project.cache_dir_for("ep01")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tracklets.parquet").write_bytes(b"stub")
    # Stamp with the dataclass defaults, then bump frame_stride in the
    # project's overrides — the next view fetch resolves thresholds with
    # the new stride and sees the snapshot diverge.
    stamp_meta(project, "ep01", Thresholds())
    project.thresholds_overrides = {"detect": {"frame_stride": 8}}
    project.save()

    await client.post(
        "/api/projects/register", json={"folder": str(tmp_path / "p")},
    )
    body = (await client.get("/api/projects/p")).json()
    assert body["sources"][0]["extraction_cache"] == "stale"


async def test_get_registered_but_files_deleted_returns_404(client, tmp_path: Path):
    """Registry entry survives but project folder/files are gone — must 404, not 500."""
    import shutil
    folder = tmp_path / "p"
    Project.create(folder, name="p")
    await client.post("/api/projects/register", json={"folder": str(folder)})
    shutil.rmtree(folder)
    resp = await client.get("/api/projects/p")
    assert resp.status_code == 404, resp.text
    assert "p" in resp.json()["detail"]


async def test_patch_thresholds_overrides(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.patch("/api/projects/p", json={
        "thresholds_overrides": {"identify": {"body_max_distance_loose": 0.22}}
    })
    assert resp.status_code == 200
    reloaded = Project.load(tmp_path / "p")
    assert reloaded.thresholds_overrides["identify"]["body_max_distance_loose"] == 0.22


async def test_patch_llm_config_persists_and_returns_in_view(
    client, tmp_path: Path,
):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})

    # Default view: disabled, default endpoint, no model selected.
    resp = await client.get("/api/projects/p")
    assert resp.json()["llm"] == {
        "enabled": False,
        "endpoint": "http://localhost:1234",
        "model": "",
        "prompt": "",
        "api_key": "",
        "disable_thinking": False,
    }

    # Patch only some fields — others stay untouched.
    resp = await client.patch("/api/projects/p", json={
        "llm": {"enabled": True, "model": "qwen2-vl-7b"}
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm"]["enabled"] is True
    assert body["llm"]["model"] == "qwen2-vl-7b"
    # Endpoint preserved at default since the patch didn't touch it.
    assert body["llm"]["endpoint"] == "http://localhost:1234"

    # Persisted on disk.
    reloaded = Project.load(tmp_path / "p")
    assert reloaded.llm.enabled is True
    assert reloaded.llm.model == "qwen2-vl-7b"


async def test_delete_project_unregisters_only(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.request("DELETE", "/api/projects/p", json={"delete_files": False})
    assert resp.status_code == 204
    # Files still on disk.
    assert (tmp_path / "p" / "project.json").exists()
    # But registry is empty.
    list_resp = await client.get("/api/projects")
    assert list_resp.json() == []


async def test_delete_project_with_files(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.request("DELETE", "/api/projects/p", json={"delete_files": True})
    assert resp.status_code == 204
    assert not (tmp_path / "p").exists()


async def test_patch_auto_delete_rejected_roundtrips(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})

    # Default-off surfaces in the view.
    body = (await client.get("/api/projects/p")).json()
    assert body["auto_delete_rejected"] is False

    # PATCH true and confirm it persists.
    resp = await client.patch(
        "/api/projects/p", json={"auto_delete_rejected": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["auto_delete_rejected"] is True

    # Re-fetch to confirm it survived the round trip.
    body = (await client.get("/api/projects/p")).json()
    assert body["auto_delete_rejected"] is True


async def test_delete_with_delete_files_true_removes_folder(
    client, tmp_path: Path,
):
    """The bin-icon UX in the badge always passes delete_files=True.
    Confirm the endpoint actually nukes the folder so the UI's promise
    matches reality."""
    folder = tmp_path / "doomed"
    Project.create(folder, name="doomed")
    await client.post("/api/projects/register", json={"folder": str(folder)})

    assert folder.exists()
    resp = await client.request(
        "DELETE", "/api/projects/doomed",
        json={"delete_files": True},
    )
    assert resp.status_code == 204, resp.text
    assert not folder.exists(), "delete_files=True must rmtree the folder"

    # Registry entry is also gone.
    list_resp = await client.get("/api/projects")
    assert all(r["slug"] != "doomed" for r in list_resp.json())


async def test_delete_with_delete_files_false_keeps_folder(
    client, tmp_path: Path,
):
    folder = tmp_path / "spared"
    Project.create(folder, name="spared")
    await client.post("/api/projects/register", json={"folder": str(folder)})

    resp = await client.request(
        "DELETE", "/api/projects/spared",
        json={"delete_files": False},
    )
    assert resp.status_code == 204
    assert folder.exists(), "delete_files=False must leave the folder alone"


async def test_patch_renames_display_name(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="Old Name")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.patch("/api/projects/p", json={"name": "New Name"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "New Name"
    assert body["slug"] == "p"  # slug + folder unchanged
    reloaded = Project.load(tmp_path / "p")
    assert reloaded.name == "New Name"


async def test_patch_rejects_blank_name(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="Keep")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.patch("/api/projects/p", json={"name": "   "})
    assert resp.status_code == 400
    assert Project.load(tmp_path / "p").name == "Keep"


async def test_project_view_reports_rejected_count(client, tmp_path: Path):
    project = Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    project.rejected_dir.mkdir(parents=True, exist_ok=True)
    (project.rejected_dir / "a__1.png").write_bytes(b"x")
    (project.rejected_dir / "b__2.png").write_bytes(b"y")
    resp = await client.get("/api/projects/p")
    assert resp.json()["rejected_count"] == 2


async def test_delete_rejected_removes_files(client, tmp_path: Path):
    project = Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    project.rejected_dir.mkdir(parents=True, exist_ok=True)
    (project.rejected_dir / "a__1.png").write_bytes(b"x")
    (project.rejected_dir / "b__2.png").write_bytes(b"y")
    resp = await client.delete("/api/projects/p/output/rejected")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert list(project.rejected_dir.iterdir()) == []


async def test_delete_rejected_idempotent_when_empty(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.delete("/api/projects/p/output/rejected")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 0


async def test_tag_autocomplete_defaults_on_and_round_trips(client, tmp_path: Path):
    Project.create(tmp_path / "tac", name="tac")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "tac")})

    # Defaults to True for a freshly created project.
    resp = await client.get("/api/projects/tac")
    assert resp.json()["tag_autocomplete"] is True

    # PATCH it off and confirm it sticks across a reload.
    resp = await client.patch("/api/projects/tac", json={"tag_autocomplete": False})
    assert resp.status_code == 200
    assert resp.json()["tag_autocomplete"] is False
    assert Project.load(tmp_path / "tac").tag_autocomplete is False
