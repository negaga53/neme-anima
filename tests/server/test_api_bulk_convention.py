"""Bulk endpoints report per-item skips and never abort mid-loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from neme_anima.server.app import create_app
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import Project


@pytest.fixture
def project_with_frames(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="p")
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for stem, fi in [("ep01", 10), ("ep02", 20)]:
        name = f"{stem}__s000_t001_f{fi:06}"
        Image.fromarray(img).save(p.kept_dir / f"{name}.png")
        (p.kept_dir / f"{name}.txt").write_text("1girl, smile\n")
        MetadataLog(p.metadata_path).append(
            FrameRecord(
                filename=name,
                kept=True,
                scene_idx=0,
                tracklet_id=1,
                frame_idx=fi,
                timestamp_seconds=fi / 24.0,
                bbox=(0, 0, 16, 16),
                ccip_distance=0.1,
                sharpness=10.0,
                visibility=1.0,
                aspect=0.95,
                score=0.9,
                video_stem=stem,
            )
        )
    return p


@pytest.fixture
def app(tmp_path: Path, project_with_frames: Project):
    a = create_app(state_dir=tmp_path / "state")
    a.state.registry.register(project_with_frames)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


FRAME1 = "ep01__s000_t001_f000010"
FRAME2 = "ep02__s000_t001_f000020"


async def test_bulk_delete_reports_skipped(client, project_with_frames: Project):
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-delete",
        json={"filenames": [FRAME1, "ghost"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == 1  # existing key preserved
    assert body["total"] == 2
    assert body["skipped"] == [{"filename": "ghost", "reason": "not found on disk"}]


async def test_bulk_tags_replace_reports_skipped(client, project_with_frames: Project):
    (project_with_frames.kept_dir / f"{FRAME2}.txt").unlink()
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={"filenames": [FRAME1, FRAME2], "pattern": "smile", "replacement": "frown"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["changed"] == 1
    assert body["total"] == 2
    assert body["skipped"] == [{"filename": FRAME2, "reason": "no sidecar"}]


async def test_bulk_move_reports_skipped_with_reason(
    client,
    project_with_frames: Project,
):
    target = project_with_frames.characters[0].slug
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-move",
        json={"filenames": [FRAME1, "ghost"], "character_slug": target},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["moved"] == 1
    assert body["missing"] == ["ghost"]  # existing key preserved
    assert body["total"] == 2
    assert body["skipped"] == [
        {"filename": "ghost", "reason": "frame metadata not found"},
    ]


async def test_bulk_duplicate_skips_diskless_frame_instead_of_aborting(
    client,
    project_with_frames: Project,
):
    """A frame with metadata but no PNG must become a skip, not a 404 that
    aborts the batch after partial work."""
    (project_with_frames.kept_dir / f"{FRAME1}.png").unlink()
    target = project_with_frames.characters[0].slug
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-duplicate",
        json={"filenames": [FRAME1, FRAME2], "character_slug": target},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["duplicated"]) == 1
    assert body["duplicated"][0].startswith(FRAME2)
    assert body["total"] == 2
    assert [s["filename"] for s in body["skipped"]] == [FRAME1]
    assert "missing on disk" in body["skipped"][0]["reason"]


async def test_single_duplicate_of_diskless_frame_still_404s(
    client,
    project_with_frames: Project,
):
    (project_with_frames.kept_dir / f"{FRAME1}.png").unlink()
    target = project_with_frames.characters[0].slug
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{FRAME1}/duplicate",
        json={"character_slug": target},
    )
    assert resp.status_code == 404


async def test_bulk_retag_danbooru_skips_ghost_frame(
    client,
    app,
    project_with_frames: Project,
):
    """A ghost filename (no PNG on disk) becomes a skip entry with reason
    'frame not found on disk'; the real frame still gets retagged."""

    class _FakeTagger:
        def tag(self, arr):  # noqa: ANN001, ARG002
            class _R:
                text = "retagged"

            return _R()

    app.state._tagger = _FakeTagger()

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-danbooru",
        json={"filenames": [FRAME1, "ghost"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retagged"] == 1
    assert body["total"] == 2
    assert body["skipped"] == [{"filename": "ghost", "reason": "frame not found on disk"}]
    # effective_filenames is parallel to input; ghost resolves to None.
    assert len(body["effective_filenames"]) == 2
    assert body["effective_filenames"][0] == FRAME1
    assert body["effective_filenames"][1] is None


async def test_bulk_retag_danbooru_skips_on_tagger_exception(
    client,
    app,
    project_with_frames: Project,
):
    """When the WD14 tagger raises for a frame the batch must not abort:
    the frame is added to skipped with the exception class + message as
    the reason string, and the loop continues to the next frame."""

    class _ExplodingTagger:
        def tag(self, arr):  # noqa: ANN001, ARG002
            raise RuntimeError("boom")

    app.state._tagger = _ExplodingTagger()

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-danbooru",
        json={"filenames": [FRAME1]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retagged"] == 0
    assert body["total"] == 1
    assert body["skipped"] == [{"filename": FRAME1, "reason": "RuntimeError: boom"}]


async def test_bulk_retag_llm_skips_ghost_frame(
    client,
    project_with_frames: Project,
    monkeypatch,
):
    """A ghost filename on the LLM retag path becomes a skip entry with
    reason 'frame not found on disk'; an existing real frame is still described."""
    project_with_frames.llm.enabled = True
    project_with_frames.llm.model = "fake-model"
    project_with_frames.llm.endpoint = "http://localhost:1234"
    project_with_frames.save()

    def fake_describe_image(
        *,
        endpoint,
        model,
        image_path,
        prompt,
        danbooru_tags,
        api_key=None,
        disable_thinking=False,
    ):
        return "A described frame."

    monkeypatch.setattr("neme_anima.llm.describe_image", fake_describe_image)

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-llm",
        json={"filenames": [FRAME1, "ghost"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["described"] == 1
    assert body["total"] == 2
    assert body["skipped"] == [{"filename": "ghost", "reason": "frame not found on disk"}]
    assert body["effective_filenames"][0] == FRAME1
    assert body["effective_filenames"][1] is None
