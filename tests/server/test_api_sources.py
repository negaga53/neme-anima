"""Tests for /api/projects/{slug}/sources routes."""

from __future__ import annotations

import asyncio as _asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app
from neme_anima.server.video_ops import _convert_cmd
from neme_anima.storage.project import Project


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


async def test_add_source(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": [str(vid)]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 1


async def test_add_source_skips_duplicates(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    assert resp.status_code == 200
    body = resp.json()
    # Endpoint reports skipped, not error.
    assert "skipped" in body


async def test_remove_source(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.delete(f"/api/projects/{project.slug}/sources/0")
    assert resp.status_code == 204
    reloaded = Project.load(project.root)
    assert reloaded.sources == []


async def test_patch_excluded_refs(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    img = tmp_path / "ref.png"
    img.write_bytes(b"")
    project.add_ref(img)
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.patch(
        f"/api/projects/{project.slug}/sources/0",
        json={"excluded_refs": [str(img.resolve())]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    # excluded_refs is now a per-character map; this PATCH (no character_slug)
    # targets the default character.
    assert reloaded.sources[0].excluded_refs == {"default": [str(img.resolve())]}


async def test_add_source_accepts_file_uri(client, project: Project, tmp_path: Path):
    vid = tmp_path / "Show E01.mkv"
    vid.write_bytes(b"")
    uri = f"file://{vid.as_posix().replace(' ', '%20')}"
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": [uri]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 1
    assert Path(reloaded.sources[0].path) == vid.resolve()


async def test_add_source_skips_browser_vfs_sentinel(client, project: Project):
    # When the browser hides the path, the frontend may still send the vfs:// fallback;
    # the server should reject it cleanly rather than try to open a junk path.
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": ["vfs://Classroom of the Elite - S03E11 [1080p].mkv"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == []
    assert "vfs://" in body["skipped"][0]


async def test_import_folder_adds_all_videos(client, project: Project, tmp_path: Path):
    folder = tmp_path / "season3"
    folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"")
    (folder / "ep02.mp4").write_bytes(b"")
    (folder / "notes.txt").write_text("ignore")
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/import-folder",
        json={"folder": str(folder)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["added"]) == 2
    assert body["source_root"] == str(folder.resolve())
    reloaded = Project.load(project.root)
    assert reloaded.source_root == str(folder.resolve())


async def test_import_folder_rejects_missing_dir(client, project: Project, tmp_path: Path):
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/import-folder",
        json={"folder": str(tmp_path / "nope")},
    )
    assert resp.status_code == 400


async def test_reimport_brings_back_deleted_rows(client, project: Project, tmp_path: Path):
    folder = tmp_path / "vids"
    folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"")
    (folder / "ep02.mkv").write_bytes(b"")
    await client.post(
        f"/api/projects/{project.slug}/sources/import-folder",
        json={"folder": str(folder)},
    )
    # Delete one row.
    await client.delete(f"/api/projects/{project.slug}/sources/0")
    # Reimport — the deleted row should come back.
    resp = await client.post(f"/api/projects/{project.slug}/sources/reimport")
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 2


async def test_reimport_400_when_no_source_root(client, project: Project):
    resp = await client.post(f"/api/projects/{project.slug}/sources/reimport")
    assert resp.status_code == 400


async def test_thumbnail_extracts_from_real_video(
    client, project: Project, tmp_path: Path,
):
    """End-to-end: ffmpeg generates a sample mp4, the endpoint produces a JPEG."""
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    vid = tmp_path / "ep.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=4:size=160x120:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)],
        capture_output=True, text=True,
    )
    if res.returncode != 0 or not vid.exists():
        pytest.skip(f"ffmpeg sample generation failed: {res.stderr}")

    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/thumbnail")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8")
    # Cached file lives where we expect.
    assert (project.root / ".thumbs" / "ep.jpg").is_file()


async def test_thumbnail_serves_cached_jpeg(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    # Pre-populate the thumbnail cache so we don't have to actually decode video.
    thumbs_dir = project.root / ".thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    (thumbs_dir / "ep01.jpg").write_bytes(b"\xff\xd8\xff\xe0FAKEJPEG")
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8")


async def test_thumbnail_404_when_video_missing(client, project: Project, tmp_path: Path):
    vid = tmp_path / "gone.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    vid.unlink()
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/thumbnail")
    assert resp.status_code == 404


async def test_thumbnail_404_when_idx_out_of_range(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/sources/99/thumbnail")
    assert resp.status_code == 404


async def test_extract_enqueues_job(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    img = tmp_path / "ref.png"
    img.write_bytes(b"")
    project.add_ref(img)
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(f"/api/projects/{project.slug}/sources/0/extract")
    assert resp.status_code == 202
    assert "job_id" in resp.json()


# ---------------- segments + duration + preview/stream ----------------


def _make_synth_video(dest: Path, duration: int = 4) -> bool:
    """Generate a tiny H.264 MP4 via ffmpeg for tests that need a real video.
    Returns False so callers can ``pytest.skip`` cleanly if the local
    ffmpeg isn't available or fails."""
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        return False
    res = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=160x120:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
        capture_output=True, text=True,
    )
    return res.returncode == 0 and dest.exists()


async def test_duration_returns_and_caches(
    client, project: Project, tmp_path: Path,
):
    """``/duration`` probes ffprobe on first call and caches on the Source."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    resp = await client.get(f"/api/projects/{project.slug}/sources/0/duration")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["duration_seconds"] > 0
    assert body["fps"] > 0

    # Cached on the source.
    reloaded = Project.load(project.root)
    assert reloaded.sources[0].duration_seconds == body["duration_seconds"]
    assert reloaded.sources[0].fps == body["fps"]


async def test_duration_reports_and_caches_vcodec(
    client, project: Project, tmp_path: Path,
):
    """``/duration`` returns the source's video codec and caches it so the
    segment editor can tell up-front whether the browser can play the original."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):  # libx264 -> codec_name "h264"
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    resp = await client.get(f"/api/projects/{project.slug}/sources/0/duration")
    assert resp.status_code == 200, resp.text
    assert resp.json()["vcodec"] == "h264"

    # Cached on the source and survives reload.
    reloaded = Project.load(project.root)
    assert reloaded.sources[0].vcodec == "h264"


async def test_put_segments_persists_and_merges(
    client, project: Project, tmp_path: Path,
):
    """Server merges overlapping ranges and sorts them ascending."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid, duration=10):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    resp = await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": [
            {"start_seconds": 5.0, "end_seconds": 7.0},
            {"start_seconds": 1.0, "end_seconds": 4.0},
            {"start_seconds": 3.0, "end_seconds": 6.0},  # overlaps both
        ]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Three ranges merge into one continuous [1, 7].
    assert body["segments"] == [{"start_seconds": 1.0, "end_seconds": 7.0}]

    reloaded = Project.load(project.root)
    assert len(reloaded.sources[0].segments) == 1
    assert reloaded.sources[0].segments[0].start_seconds == 1.0
    assert reloaded.sources[0].segments[0].end_seconds == 7.0


async def test_put_segments_rejects_invalid(
    client, project: Project, tmp_path: Path,
):
    """end <= start and end > duration both return 400."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid, duration=4):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    # Inverted range.
    bad = await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": [{"start_seconds": 3.0, "end_seconds": 1.0}]},
    )
    assert bad.status_code == 400

    # Past the end (4s clip, asking for 30s).
    past = await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": [{"start_seconds": 1.0, "end_seconds": 30.0}]},
    )
    assert past.status_code == 400


async def test_put_segments_clears_with_empty_list(
    client, project: Project, tmp_path: Path,
):
    """An empty segments list reverts to whole-video behaviour."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": [{"start_seconds": 1.0, "end_seconds": 2.0}]},
    )
    resp = await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": []},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"] == []
    reloaded = Project.load(project.root)
    assert reloaded.sources[0].segments == []


async def test_segments_flip_extraction_cache_to_stale(
    client, project: Project, tmp_path: Path,
):
    """Stamping a fresh cache then editing segments should mark cache stale."""
    from neme_anima.config import Thresholds
    from neme_anima.extraction_cache import (
        cache_state_for_source,
        stamp_meta,
    )

    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    # Manually seed the cache state to "current" by writing a parquet stub
    # + stamping the meta. cache_state() needs tracklets.parquet to exist
    # before it even consults the meta — without it the result is "none".
    cache_dir = project.cache_dir_for("ep")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tracklets.parquet").write_bytes(b"stub")
    project_for_stamp = Project.load(project.root)
    stamp_meta(project_for_stamp, "ep", Thresholds())

    fresh = Project.load(project.root)
    assert cache_state_for_source(fresh, 0, Thresholds()) == "current"

    # Edit segments via the API and re-check.
    await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": [{"start_seconds": 1.0, "end_seconds": 2.0}]},
    )
    after = Project.load(project.root)
    assert cache_state_for_source(after, 0, Thresholds()) == "stale"


async def test_stream_serves_video_with_range_support(
    client, project: Project, tmp_path: Path,
):
    """``/stream`` returns the raw file with Range support."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    full = await client.get(f"/api/projects/{project.slug}/sources/0/stream")
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert len(full.content) > 0

    ranged = await client.get(
        f"/api/projects/{project.slug}/sources/0/stream",
        headers={"Range": "bytes=0-15"},
    )
    assert ranged.status_code == 206
    assert len(ranged.content) == 16


async def test_preview_serves_cached_mode_file_with_range(
    client, project: Project, tmp_path: Path,
):
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    previews = project.root / ".previews"
    previews.mkdir(parents=True, exist_ok=True)
    (previews / "ep.h264.mp4").write_bytes(b"0123456789ABCDEF" * 4)

    full = await client.get(f"/api/projects/{project.slug}/sources/0/preview?mode=h264")
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"

    ranged = await client.get(
        f"/api/projects/{project.slug}/sources/0/preview?mode=h264",
        headers={"Range": "bytes=0-7"},
    )
    assert ranged.status_code == 206
    assert len(ranged.content) == 8


async def test_preview_404_when_not_converted(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/preview?mode=remux")
    assert resp.status_code == 404


async def test_delete_preview_removes_cached_files(
    client, project: Project, tmp_path: Path,
):
    """DELETE /preview removes the cached conversion(s) so they can be re-made."""
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    previews = project.root / ".previews"
    previews.mkdir(parents=True, exist_ok=True)
    (previews / "ep.h264.mp4").write_bytes(b"FAKE")
    (previews / "ep.remux.mp4").write_bytes(b"FAKE")

    resp = await client.delete(f"/api/projects/{project.slug}/sources/0/preview")
    assert resp.status_code == 200
    assert sorted(resp.json()["removed"]) == ["h264", "remux"]
    assert not (previews / "ep.h264.mp4").exists()
    assert not (previews / "ep.remux.mp4").exists()

    # Idempotent: deleting again removes nothing and doesn't error.
    resp2 = await client.delete(f"/api/projects/{project.slug}/sources/0/preview")
    assert resp2.status_code == 200
    assert resp2.json()["removed"] == []


async def test_delete_preview_single_mode(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    previews = project.root / ".previews"
    previews.mkdir(parents=True, exist_ok=True)
    (previews / "ep.h264.mp4").write_bytes(b"FAKE")
    (previews / "ep.remux.mp4").write_bytes(b"FAKE")

    resp = await client.delete(f"/api/projects/{project.slug}/sources/0/preview?mode=h264")
    assert resp.status_code == 200
    assert resp.json()["removed"] == ["h264"]
    assert not (previews / "ep.h264.mp4").exists()
    assert (previews / "ep.remux.mp4").exists()  # the other mode is untouched


# ---------------- frame capture ----------------


class _FakeTagger:
    """Stand-in for the WD14 model so capture tests don't load it."""

    def tag(self, arr):  # noqa: ANN001, ARG002
        class _R:
            text = "1girl, smile"

        return _R()


async def test_capture_frame_saves_and_tags(
    client, app, project: Project, tmp_path: Path,
):
    """End-to-end: ffmpeg grabs the frame, the shared ingest path WD14-tags it,
    and it lands as a kept custom_uploads frame at full source resolution."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid, duration=4):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    app.state._tagger = _FakeTagger()
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/capture-frame",
        json={"time_seconds": 1.0},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    fn = body["frame"]["filename"]
    assert fn.startswith("custom_uploads__")
    assert body["frame"]["character_slug"] == "default"
    assert body["llm_error"] is None
    # Timestamp provenance is carried through to the metadata row.
    assert body["frame"]["timestamp_seconds"] == pytest.approx(1.0, abs=0.06)

    png = project.kept_dir / f"{fn}.png"
    txt = project.kept_dir / f"{fn}.txt"
    assert png.is_file()
    assert txt.read_text(encoding="utf-8").startswith("1girl, smile")

    # Captured at native resolution (the synth clip is 160x120), not a thumbnail.
    from PIL import Image

    with Image.open(png) as im:
        assert im.size == (160, 120)

    # And it surfaces in the frames listing under the custom_uploads stem.
    listing = await client.get(
        f"/api/projects/{project.slug}/frames",
        params={"source": "custom_uploads"},
    )
    assert any(f["filename"] == fn for f in listing.json()["items"])


def _make_gray_sequence_video(dest: Path, *, fps: int = 5, frames: int = 4) -> bool:
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        return False

    from PIL import Image

    seq_dir = dest.parent / f"{dest.stem}_frames"
    seq_dir.mkdir()
    for i in range(frames):
        gray = i * 50
        Image.new("RGB", (64, 64), (gray, gray, gray)).save(seq_dir / f"{i:03d}.png")
    res = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(seq_dir / "%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
        ],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0 and dest.exists()


async def test_capture_frame_uses_frame_idx_when_provided(
    client, app, project: Project, tmp_path: Path,
):
    """Frame-step captures send the intended decoded frame index, not just time."""
    vid = tmp_path / "indexed.mp4"
    if not _make_gray_sequence_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    app.state._tagger = _FakeTagger()
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/capture-frame",
        json={"time_seconds": 0.0, "frame_idx": 2},
    )
    assert resp.status_code == 201, resp.text

    from PIL import Image

    fn = resp.json()["frame"]["filename"]
    with Image.open(project.kept_dir / f"{fn}.png") as im:
        center_gray = im.convert("RGB").getpixel((32, 32))[0]
    assert center_gray == pytest.approx(100, abs=8)


def test_frame_index_seek_plan_uses_short_preroll():
    from neme_anima.server.video_ops import _frame_index_seek_plan

    seek_seconds, local_frame_idx = _frame_index_seek_plan(
        68_784, fps=24.0, t_seconds=None,
    )

    assert seek_seconds == pytest.approx(2865.0)
    assert local_frame_idx == 24


def test_frame_index_seek_plan_estimates_fps_from_playhead_time():
    from neme_anima.server.video_ops import _frame_index_seek_plan

    seek_seconds, local_frame_idx = _frame_index_seek_plan(
        2400, fps=None, t_seconds=100.0,
    )

    assert seek_seconds == pytest.approx(99.0)
    assert local_frame_idx == 24


def test_extract_frame_by_index_uses_input_seek(monkeypatch, tmp_path: Path):
    import subprocess

    from neme_anima.server import video_ops as sources_api

    captured: dict[str, list[str]] = {}
    dest = tmp_path / "capture.png"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        dest.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.run", fake_run)

    sources_api._extract_frame_by_index(
        Path("/video.mkv"), dest, 68_784, fps=24.0, t_seconds=None,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("-ss") + 1] == "2865.000"
    assert cmd.index("-ss") < cmd.index("-i")
    assert "select=eq(n\\,24)" in cmd


async def test_capture_frame_routes_to_named_character(
    client, app, project: Project, tmp_path: Path,
):
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    cresp = await client.post(
        f"/api/projects/{project.slug}/characters", json={"name": "Mio"},
    )
    mio_slug = cresp.json()["slug"]

    app.state._tagger = _FakeTagger()
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/capture-frame",
        json={"time_seconds": 0.5, "character_slug": mio_slug},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["frame"]["character_slug"] == mio_slug


async def test_capture_frame_runs_llm_when_enabled(
    client, app, project: Project, tmp_path: Path, monkeypatch,
):
    """When LLM tagging is configured the caption pass runs and writes line 2."""
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")

    # Persist the LLM config *before* adding the source through the API.
    # The API reloads the project from disk per request; saving the stale
    # fixture object after the source was added would clobber it back off.
    project.llm.enabled = True
    project.llm.model = "fake-model"
    project.llm.endpoint = "http://localhost:1234"
    project.save()

    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    def fake_describe_image(
        *, endpoint, model, image_path, prompt, danbooru_tags, api_key=None,
        disable_thinking=False,
    ):
        return "A test pattern frame."

    monkeypatch.setattr("neme_anima.llm.describe_image", fake_describe_image)

    app.state._tagger = _FakeTagger()
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/capture-frame",
        json={"time_seconds": 1.0},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["llm_error"] is None
    fn = body["frame"]["filename"]
    txt = (project.kept_dir / f"{fn}.txt").read_text(encoding="utf-8")
    assert "A test pattern frame." in txt


async def test_capture_frame_unknown_character_404(
    client, project: Project, tmp_path: Path,
):
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/capture-frame",
        json={"time_seconds": 1.0, "character_slug": "nope"},
    )
    assert resp.status_code == 404


async def test_capture_frame_404_for_bad_idx(client, project: Project):
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/99/capture-frame",
        json={"time_seconds": 1.0},
    )
    assert resp.status_code == 404


async def test_capture_frame_404_for_missing_video(
    client, project: Project, tmp_path: Path,
):
    vid = tmp_path / "gone.mp4"
    vid.write_bytes(b"x")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    vid.unlink()
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/capture-frame",
        json={"time_seconds": 1.0},
    )
    assert resp.status_code == 404


async def test_segments_404_for_bad_idx(client, project: Project):
    resp = await client.put(
        f"/api/projects/{project.slug}/sources/99/segments",
        json={"segments": []},
    )
    assert resp.status_code == 404


async def test_segments_404_for_missing_video(
    client, project: Project, tmp_path: Path,
):
    vid = tmp_path / "gone.mp4"
    vid.write_bytes(b"x")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    vid.unlink()
    resp = await client.put(
        f"/api/projects/{project.slug}/sources/0/segments",
        json={"segments": [{"start_seconds": 1.0, "end_seconds": 2.0}]},
    )
    assert resp.status_code == 404


# ---------------- convert command builder ----------------


def test_convert_cmd_remux_copies_video_no_reencode():
    cmd = _convert_cmd(Path("/in.mkv"), Path("/out.mp4"), "remux", with_audio=True)
    assert "copy" in cmd                      # video stream copied, not re-encoded
    assert "hvc1" in cmd                      # tagged so Apple/Safari recognise it
    assert "libx264" not in cmd
    assert "-progress" in cmd and "pipe:1" in cmd


def test_convert_cmd_h264_scales_to_480p():
    cmd = _convert_cmd(Path("/in.mkv"), Path("/out.mp4"), "h264", with_audio=True)
    assert "libx264" in cmd
    assert any("scale='min(854,iw)':-2" in tok for tok in cmd)
    assert "copy" not in cmd
    assert "-progress" in cmd and "pipe:1" in cmd


def test_convert_cmd_without_audio_uses_an():
    cmd = _convert_cmd(Path("/in.mkv"), Path("/out.mp4"), "h264", with_audio=False)
    assert "-an" in cmd
    assert "aac" not in cmd


def test_convert_cmd_remux_without_audio_keeps_video_copy():
    cmd = _convert_cmd(Path("/in.mkv"), Path("/out.mp4"), "remux", with_audio=False)
    assert "-an" in cmd                       # audio suppressed on the no-audio retry
    assert "copy" in cmd and "hvc1" in cmd    # video still copied losslessly
    assert "aac" not in cmd                   # no audio encoder when -an
    assert "-map" in cmd and "0:v:0" in cmd   # video stream still explicitly mapped


# ---------------- convert endpoints ----------------


async def _wait_convert_ready(client, slug, mode, *, tries=100):
    """Poll /convert/status until ready/failed. Returns the final body."""
    for _ in range(tries):
        s = await client.get(
            f"/api/projects/{slug}/sources/0/convert/status?mode={mode}",
        )
        body = s.json()
        if body["state"] in ("ready", "failed"):
            return body
        await _asyncio.sleep(0.1)
    return {"state": "timeout", "pct": 0.0}


async def test_convert_h264_produces_cached_mp4_and_reaches_ready(
    client, project: Project, tmp_path: Path,
):
    vid = tmp_path / "ep.mp4"
    if not _make_synth_video(vid):
        pytest.skip("ffmpeg unavailable for sample generation")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})

    started = await client.post(
        f"/api/projects/{project.slug}/sources/0/convert?mode=h264",
    )
    assert started.status_code == 200, started.text
    assert started.json()["state"] in ("running", "ready")

    final = await _wait_convert_ready(client, project.slug, "h264")
    assert final["state"] == "ready", final
    assert 0.0 <= final["pct"] <= 100.0
    assert (project.root / ".previews" / "ep.h264.mp4").is_file()


async def test_convert_rejects_bad_mode(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/convert?mode=av1",
    )
    assert resp.status_code == 400


async def test_convert_404_when_file_missing(client, project: Project, tmp_path: Path):
    vid = tmp_path / "gone.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    vid.unlink()
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/convert?mode=h264",
    )
    assert resp.status_code == 404


async def test_convert_status_idle_before_start(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.get(
        f"/api/projects/{project.slug}/sources/0/convert/status?mode=h264",
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "idle"


async def test_convert_cached_file_short_circuits(
    client, project: Project, tmp_path: Path,
):
    """A pre-existing cache file means /convert returns ready without ffmpeg."""
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    previews = project.root / ".previews"
    previews.mkdir(parents=True, exist_ok=True)
    (previews / "ep.h264.mp4").write_bytes(b"FAKEMP4DATA")
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/0/convert?mode=h264",
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "ready"


# ---------------- conversion robustness ----------------


def test_run_one_ffmpeg_kill_timer_bounds_a_hung_process():
    """The wall-clock killer must terminate a process that never finishes;
    without it the progress loop blocks on stdout EOF forever."""
    import time as _time

    from neme_anima.server.video_ops import _run_one_ffmpeg

    t0 = _time.monotonic()
    rc, _stderr = _run_one_ffmpeg(
        ["sleep", "5"], ("test", "h264"), 0.0, timeout=0.3,
    )
    assert rc != 0
    assert _time.monotonic() - t0 < 5.0


def test_transcode_cleans_tmp_on_failure(tmp_path: Path, monkeypatch):
    from neme_anima.server import video_ops as sources_mod

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    def always_fail(cmd, key, duration, *, timeout=900.0):
        dest = Path(cmd[-1])
        dest.write_bytes(b"")  # simulate a partial, empty output
        return 1, "simulated encoder explosion"

    monkeypatch.setattr(sources_mod, "_run_one_ffmpeg", always_fail)
    src = tmp_path / "in.mkv"
    src.write_bytes(b"x")
    dest = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        sources_mod._transcode_for_playback(src, dest, "h264", 0.0, ("k", "h264"))
    assert not dest.exists()
    assert not list(tmp_path.glob("*.tmp*"))
