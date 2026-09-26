"""Tests for /api/projects/{slug}/training/* routes (config + path checks).

The actual ``start`` endpoint launches a subprocess and is therefore not
exercised here — those tests would require a real diffusion-pipe install.
We only assert that ``start`` refuses to launch when paths are missing
(the validate_for_run gate).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app
from neme_anima.storage.project import Project


def _write_tiny_safetensors(path: Path, metadata: dict[str, str] | None = None) -> None:
    header = {
        "__metadata__": metadata or {"format": "pt"},
        "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"DATA")


def _read_safetensors_metadata(data: bytes) -> dict[str, str]:
    header_len = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + header_len])
    return header["__metadata__"]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p", name="p")


@pytest.fixture
def app(tmp_path: Path, project: Project):
    a = create_app(state_dir=tmp_path / "state")
    a.state.registry.register(project)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_get_config_returns_defaults_and_problems(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/config")
    assert resp.status_code == 200
    body = resp.json()
    cfg = body["config"]
    assert cfg["preset"] == "style"
    assert cfg["keep_last_n_checkpoints"] == 0
    assert cfg["llm_adapter_lr"] == 0.0
    # Empty paths should produce path_check errors and surface in problems.
    pc = body["path_checks"]
    assert all(pc[k]["error"] for k in (
        "diffusion_pipe_dir", "dit_path", "vae_path", "llm_path",
    ))
    assert len(body["problems"]) >= 4


async def test_patch_config_persists(client, project: Project, tmp_path: Path):
    resp = await client.patch(
        f"/api/projects/{project.slug}/training/config",
        json={
            "preset": "character",
            "learning_rate": 5e-5,
            "keep_last_n_checkpoints": 3,
            "trigger_token": "mychar",
            "resolutions": [768, 1024],
        },
    )
    assert resp.status_code == 200, resp.text
    cfg = resp.json()["config"]
    assert cfg["preset"] == "character"
    assert cfg["learning_rate"] == 5e-5
    assert cfg["keep_last_n_checkpoints"] == 3
    assert cfg["trigger_token"] == "mychar"
    assert cfg["resolutions"] == [768, 1024]
    # Re-read from disk to confirm persistence.
    reloaded = Project.load(project.root)
    assert reloaded.training.preset == "character"
    assert reloaded.training.learning_rate == 5e-5


async def test_check_path_missing(client, project: Project, tmp_path: Path):
    resp = await client.post(
        f"/api/projects/{project.slug}/training/check-path",
        json={"path": str(tmp_path / "nope"), "expect": "file"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert not body["exists"]
    assert "no such" in body["error"].lower()


async def test_check_path_existing_file(client, project: Project, tmp_path: Path):
    f = tmp_path / "some.bin"
    f.write_bytes(b"x")
    resp = await client.post(
        f"/api/projects/{project.slug}/training/check-path",
        json={"path": str(f), "expect": "file"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"]
    assert body["is_file"]
    assert body["error"] is None


async def test_status_when_no_run(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False
    assert body["state"] is None
    assert body["log_lines"] == []


async def test_start_refused_when_paths_missing(client, project: Project):
    # No paths set => validate_for_run returns problems => 409.
    resp = await client.post(f"/api/projects/{project.slug}/training/start")
    assert resp.status_code == 409
    assert "diffusion-pipe" in resp.json()["detail"]


async def test_resume_404s_when_no_runs(client, project: Project):
    resp = await client.post(f"/api/projects/{project.slug}/training/resume")
    assert resp.status_code == 409
    assert "no prior run" in resp.json()["detail"]


async def test_resume_409s_when_already_at_target_epochs(
    client, project: Project,
):
    """Refuse to resume when cfg.epochs is no higher than the highest saved
    epoch — otherwise diffusion-pipe would still grind one more epoch."""
    runs_dir = project.training_runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "20260501-120000-character"
    run_dir.mkdir()
    sub = run_dir / "20260501_12-00-01"
    sub.mkdir()
    (sub / "latest").write_text("global_step100")
    ep = sub / "epoch60"
    ep.mkdir()
    (ep / "adapter_model.safetensors").write_bytes(b"x")
    project.training.epochs = 60
    project.save()
    resp = await client.post(f"/api/projects/{project.slug}/training/resume")
    assert resp.status_code == 409
    detail = resp.json()["detail"].lower()
    assert "already at epoch" in detail and "60" in detail


async def test_resume_409s_when_run_has_no_resumable_state(
    client, project: Project,
):
    """A run wrapper with epoch artifacts but no DeepSpeed ``latest`` file
    is not resumable — the trainer never made it to its first save."""
    runs_dir = project.training_runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "20260501-120000-character"
    run_dir.mkdir()
    sub = run_dir / "20260501_12-00-01"
    sub.mkdir()
    # epoch dir but no latest marker.
    ep = sub / "epoch1"
    ep.mkdir()
    (ep / "adapter_model.safetensors").write_bytes(b"x")
    resp = await client.post(f"/api/projects/{project.slug}/training/resume")
    assert resp.status_code == 409
    assert "no resumable" in resp.json()["detail"].lower()


async def test_dataset_preview_shape(client, project: Project):
    resp = await client.get(
        f"/api/projects/{project.slug}/training/dataset-preview",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_images"] == 0
    assert body["samples"] == []


async def test_run_toml_preview_renders(client, project: Project):
    resp = await client.get(
        f"/api/projects/{project.slug}/training/run-toml-preview",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "[[directory]]" in body["dataset_toml"]
    assert 'type = "anima"' in body["run_toml"]
    assert body["launcher_argv"][0] == "deepspeed"


async def test_runs_list_empty(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


async def test_delete_unknown_run_404s(client, project: Project):
    resp = await client.delete(
        f"/api/projects/{project.slug}/training/runs/no-such-run",
    )
    assert resp.status_code == 404


async def test_export_checkpoint_streams_named_file(app, tmp_path: Path):
    transport = ASGITransport(app=app)
    project = Project.create(tmp_path / "exp", name="My Project!")
    app.state.registry.register(project)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ckpt_dir = project.training_runs_dir / "run1" / "epoch20"
        ckpt_dir.mkdir(parents=True)
        (project.training_runs_dir / "run1" / "run.toml").write_text(
            "# Auto-generated by neme-anima\n",
        )
        _write_tiny_safetensors(ckpt_dir / "adapter_model.safetensors")

        resp = await client.get(
            f"/api/projects/{project.slug}/training/runs/run1/checkpoints/epoch20/export"
        )
        assert resp.status_code == 200, resp.text
        assert resp.content.endswith(b"DATA")
        meta = _read_safetensors_metadata(resp.content)
        assert meta["neme_anima_generated_by"] == "neme-anima"
        assert meta["neme_anima_project_name"] == "My Project!"
        assert meta["neme_anima_checkpoint"] == "epoch20"
        cd = resp.headers["content-disposition"]
        assert "My_Project-epoch20.safetensors" in cd


async def test_export_checkpoint_404_when_missing(client, project: Project):
    resp = await client.get(
        f"/api/projects/{project.slug}/training/runs/nope/checkpoints/epoch1/export"
    )
    assert resp.status_code == 404


async def test_export_checkpoint_404_when_checkpoint_missing(app, tmp_path: Path):
    transport = ASGITransport(app=app)
    project = Project.create(tmp_path / "p3", name="p3")
    app.state.registry.register(project)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # run1 exists but epochX does not.
        project.training_runs_dir.joinpath("run1").mkdir(parents=True)

        resp = await client.get(
            f"/api/projects/{project.slug}/training/runs/run1/checkpoints/epochX/export"
        )
        assert resp.status_code == 404


async def test_export_checkpoint_glob_fallback(app, tmp_path: Path):
    transport = ASGITransport(app=app)
    project = Project.create(tmp_path / "p4", name="p4")
    app.state.registry.register(project)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ckpt_dir = project.training_runs_dir / "run1" / "epoch5"
        ckpt_dir.mkdir(parents=True)
        _write_tiny_safetensors(ckpt_dir / "model.safetensors")

        resp = await client.get(
            f"/api/projects/{project.slug}/training/runs/run1/checkpoints/epoch5/export"
        )
        assert resp.status_code == 200
        assert _read_safetensors_metadata(resp.content)["neme_anima_checkpoint"] == "epoch5"


async def test_resume_calls_start_with_latest_resumable_subdir(
    client, app, project: Project,
):
    """Happy path: resume finds the newest run, picks its resumable
    DeepSpeed subdir, and reuses the run directory."""
    run_dir = project.training_runs_dir / "20260601-120000-anima"
    sub = run_dir / "20260601_12-00-01"
    sub.mkdir(parents=True)
    (sub / "latest").write_text("global_step10")

    calls: list[tuple] = []

    class RecordingManager:
        active_slug = None

        async def start(self, project, *, resume_from_checkpoint=None,
                        run_dir_name=None):
            calls.append((project.slug, resume_from_checkpoint, run_dir_name))
            return {"running": True}

    app.state.training = RecordingManager()
    resp = await client.post(f"/api/projects/{project.slug}/training/resume")
    assert resp.status_code == 202
    assert calls == [(project.slug, sub.name, run_dir.name)]


async def test_patch_sample_config_and_options(client, project: Project):
    resp = await client.patch(
        f"/api/projects/{project.slug}/training/config",
        json={"sample_prompts": ["1girl, smile"], "sample_defer_to_end": True,
              "sample_cfg": 5.0, "sample_sampler": "euler"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["sample_prompts"] == ["1girl, smile"]
    assert body["config"]["sample_defer_to_end"] is True
    assert "euler_ancestral" in body["sample_options"]["samplers"]
    assert "simple" in body["sample_options"]["schedulers"]
    reloaded = Project.load(project.root)
    assert reloaded.training.sample_cfg == 5.0
    assert reloaded.training.sample_sampler == "euler"


def _seed_samples(project: Project, run_name: str = "r1") -> Path:
    d = project.training_runs_dir / run_name / "samples" / "epoch0010"
    d.mkdir(parents=True)
    (d / "p00.png").write_bytes(b"\x89PNG fake")
    (d / "manifest.json").write_text(json.dumps({
        "prompts": ["1girl"], "negative_prompt": "", "settings": {"steps": 30},
        "images": [{"file": "p00.png", "prompt": "1girl"}], "error": None,
    }))
    return d


async def test_list_samples(client, project: Project):
    _seed_samples(project)
    resp = await client.get(f"/api/projects/{project.slug}/training/runs/r1/samples")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_name"] == "r1"
    [ep] = body["epochs"]
    assert ep["epoch"] == 10
    url = ep["images"][0]["url"]
    prefix = f"/api/projects/{project.slug}/training/runs/r1/samples/epoch0010/p00.png?t="
    assert url.startswith(prefix)
    img = await client.get(url)
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.content == b"\x89PNG fake"


async def test_list_samples_unknown_run_404(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/runs/nope/samples")
    assert resp.status_code == 404


@pytest.mark.parametrize(("sample_dir", "filename", "code"), [
    ("epoch0010", "manifest.json", 400),
    ("samples", "p00.png", 400),
    ("epoch0010", "p09.png", 404),
])
async def test_sample_image_rejects(client, project: Project, sample_dir, filename, code):
    _seed_samples(project)
    resp = await client.get(
        f"/api/projects/{project.slug}/training/runs/r1/samples/{sample_dir}/{filename}",
    )
    assert resp.status_code == code
