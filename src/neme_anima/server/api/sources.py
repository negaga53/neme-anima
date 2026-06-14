"""REST routes for /api/projects/{slug}/sources."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_anima.server import video_ops
from neme_anima.server.api import deps
from neme_anima.server.paths import normalize_input_path
from neme_anima.storage.project import Project, Segment, Source

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["sources"])


class AddSourcesBody(BaseModel):
    paths: list[str]


class ImportFolderBody(BaseModel):
    folder: str


class PatchSourceBody(BaseModel):
    excluded_refs: list[str] | None = None


class SegmentBody(BaseModel):
    start_seconds: float
    end_seconds: float


class PutSegmentsBody(BaseModel):
    segments: list[SegmentBody]


class CaptureFrameBody(BaseModel):
    """Body for the segment-editor's "grab this exact frame" button.

    ``time_seconds`` is the playhead position the user picked in the browser
    <video> element and is stored as timestamp provenance. ``frame_idx`` is the
    optional zero-based decoded frame number; the frontend sends it when source
    fps is known so frame-step captures don't depend on ffmpeg timestamp seek
    rounding. ``character_slug`` (optional) routes the captured frame to that
    character's bucket; omitted = the project's first character, the same
    default the drag-and-drop upload route uses.
    """

    time_seconds: float
    frame_idx: int | None = None
    character_slug: str | None = None


@router.post("/{slug}/sources")
async def add_sources(
    body: AddSourcesBody,
    project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    added: list[str] = []
    skipped: list[str] = []
    for p in body.paths:
        try:
            normalized = normalize_input_path(p)
        except ValueError:
            skipped.append(p)
            continue
        try:
            s = project.add_source(normalized)
            added.append(s.path)
        except ValueError:
            skipped.append(str(normalized.resolve()))
    return {"added": added, "skipped": skipped}


@router.post("/{slug}/sources/import-folder")
async def import_folder(
    body: ImportFolderBody,
    project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    try:
        folder = normalize_input_path(body.folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {folder}")
    added, skipped = project.import_videos_from_folder(folder)
    return {
        "added": [s.path for s in added],
        "skipped": skipped,
        "source_root": project.source_root,
    }


@router.post("/{slug}/sources/reimport")
async def reimport(project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    if not project.source_root:
        raise HTTPException(
            status_code=400,
            detail="no source folder has been imported yet — pick a folder first",
        )
    folder = Path(project.source_root)
    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"source folder is missing: {folder}",
        )
    added, skipped = project.import_videos_from_folder(folder)
    return {
        "added": [s.path for s in added],
        "skipped": skipped,
        "source_root": project.source_root,
    }


@router.delete("/{slug}/sources/{idx}", status_code=204)
async def remove_source(
    idx: int,
    project: Project = Depends(deps.get_project),  # noqa: B008
    _source: Source = Depends(deps.get_source),  # noqa: B008
) -> Response:
    project.remove_source(idx)
    return Response(status_code=204)


@router.patch("/{slug}/sources/{idx}")
async def patch_source(
    idx: int,
    body: PatchSourceBody,
    character_slug: str | None = None,
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    cslug = deps.optional_character_slug(project, character_slug)
    if body.excluded_refs is not None:
        project.set_excluded_refs(
            idx, body.excluded_refs, character_slug=cslug,
        )
    return {"excluded_refs": source.excluded_refs}


@router.get("/{slug}/sources/{idx}/wipe-preview")
async def wipe_preview(
    idx: int,
    project: Project = Depends(deps.get_project),  # noqa: B008
    _source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Return what an Extract / Re-process on this source would wipe.

    The Sources tab calls this BEFORE firing extract or rerun and shows
    a confirmation modal when ``to_wipe.total > 0`` so the user is never
    surprised by lost work. The breakdown by character lets the user
    decide whether to opt-out a character's refs (and so preserve its
    frames) before clicking through.

    Implementation note: we re-build the same per-character refs map
    that the pipeline does, then walk the metadata log to attribute
    every kept frame matching the stem. Rejected files are listed in
    ``to_wipe`` regardless of attribution because they always wipe.
    """
    from collections import Counter

    from neme_anima.pipeline import (
        _kept_frame_owners,
        _preserve_set_from_refs_by_slug,
        _refs_by_character,
    )

    video_stem = project.video_stem(idx)
    refs_by_slug = _refs_by_character(project, idx)
    preserve = _preserve_set_from_refs_by_slug(project, refs_by_slug)
    active_slugs = sorted(s for s, refs in refs_by_slug.items() if refs)

    # Walk kept_dir for files actually present and attribute each via
    # metadata. Iterating the metadata log instead would surface phantom
    # rows for frames whose physical file is gone (dedup demoted to
    # rejected/, user manually deleted, prior wipe ran without metadata
    # cleanup) — and the modal would then warn about wiping frames that
    # don't exist. The wipe itself only touches files on disk, so the
    # preview must too.
    owners = _kept_frame_owners(project, video_stem)
    to_wipe_kept: Counter = Counter()
    to_preserve: Counter = Counter()
    prefix = f"{video_stem}__"
    if project.kept_dir.exists():
        from neme_anima.storage.project import CROP_SUFFIX
        for f in project.kept_dir.iterdir():
            if not f.is_file() or not f.name.startswith(prefix):
                continue
            if f.suffix != ".png":
                continue
            # Crop derivatives ride with the original's ownership and
            # are wiped/preserved alongside it — count one per logical
            # frame so the totals match the user-facing frames view.
            if f.stem.endswith(CROP_SUFFIX):
                continue
            owner = owners.get(f.stem)
            if owner is None:
                to_preserve["__untracked__"] += 1
            elif owner in preserve:
                to_preserve[owner] += 1
            else:
                to_wipe_kept[owner] += 1

    # Rejected files always wipe — count separately so the UI can
    # surface them as "diagnostic samples" distinct from character data.
    rejected_total = 0
    if project.rejected_dir.exists():
        for f in project.rejected_dir.iterdir():
            if f.is_file() and f.name.startswith(prefix):
                rejected_total += 1

    return {
        "video_stem": video_stem,
        "active_slugs": active_slugs,
        "preserve_slugs": sorted(preserve),
        "to_wipe": {
            "by_character": dict(to_wipe_kept),
            "rejected_samples": rejected_total,
            "total": int(sum(to_wipe_kept.values()) + rejected_total),
        },
        "to_preserve": {
            "by_character": dict(to_preserve),
            "total": int(sum(to_preserve.values())),
        },
    }


@router.post("/{slug}/sources/{idx}/extract", status_code=202)
async def extract(
    idx: int,
    request: Request,
    project: Project = Depends(deps.get_project),  # noqa: B008
    _source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    job_id = await request.app.state.queue.submit({
        "kind": "extract",
        "project_folder": str(project.root.resolve()),
        "project_slug": project.slug,
        "source_idx": idx,
    })
    return {"job_id": job_id}


@router.post("/{slug}/sources/{idx}/rerun", status_code=202)
async def rerun(
    idx: int,
    request: Request,
    project: Project = Depends(deps.get_project),  # noqa: B008
    _source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    video_stem = project.video_stem(idx)
    job_id = await request.app.state.queue.submit({
        "kind": "rerun",
        "project_folder": str(project.root.resolve()),
        "project_slug": project.slug,
        "source_idx": idx,
        "video_stem": video_stem,
    })
    return {"job_id": job_id}


@router.get("/{slug}/sources/{idx}/thumbnail")
async def get_thumbnail(
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> FileResponse:
    """Return a cached JPEG thumbnail for the source's video.

    The first request grabs one frame near 10 % of the video's duration via
    OpenCV, saves it under ``<project>/.thumbs/<stem>.jpg``, and serves it.
    Subsequent requests serve the cached file directly.
    """
    video_path = Path(source.path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")

    thumbs_dir = project.root / ".thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    cache_path = thumbs_dir / f"{video_path.stem}.jpg"
    if not cache_path.exists():
        try:
            await asyncio.to_thread(video_ops._extract_thumbnail, video_path, cache_path)
        except Exception as e:
            # Surface the real error to the server log, then to the client.
            logger.exception("thumbnail extraction failed for %s", video_path)
            raise HTTPException(
                status_code=500,
                detail=f"thumbnail extraction failed: {type(e).__name__}: {e}",
            ) from e
    return FileResponse(cache_path, media_type="image/jpeg")


@router.get("/{slug}/sources/{idx}/duration")
async def get_duration(
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Return ``{duration_seconds, fps, vcodec}`` for the source's video.

    ``vcodec`` lets the segment editor tell up-front whether the browser can
    decode the original (an undecodable codec like HEVC plays black-with-audio
    and never fires a ``<video>`` error). Cached on the Source after first probe
    so the segment
    editor doesn't pay a fresh ffprobe round-trip on every open. The probe
    falls through to legacy values (0.0) if ffprobe fails — better than a
    500 for a transient ffmpeg issue when the user just wants to look at
    a video they already extracted from.
    """
    if (
        source.duration_seconds is not None
        and source.fps is not None
        and source.vcodec is not None
    ):
        return {
            "duration_seconds": source.duration_seconds,
            "fps": source.fps,
            "vcodec": source.vcodec,
        }
    video_path = Path(source.path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")
    duration = await asyncio.to_thread(video_ops._probe_duration_seconds, video_path)
    fps = await asyncio.to_thread(video_ops._probe_fps, video_path)
    vcodec = await asyncio.to_thread(video_ops._probe_vcodec, video_path)
    source.duration_seconds = duration
    source.fps = fps
    source.vcodec = vcodec
    project.save()
    return {"duration_seconds": duration, "fps": fps, "vcodec": vcodec}


@router.post("/{slug}/sources/{idx}/capture-frame", status_code=201)
async def capture_frame(
    request: Request,
    body: CaptureFrameBody,
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Grab the requested frame and register it as a kept frame.

    The frame is WD14-tagged, and LLM-described when LLM tagging is enabled,
    then routed to ``character_slug`` (default: the project's first character).

    It reuses the same ingest path as the drag-and-drop upload route, so a
    captured frame is indistinguishable from a manually-added one: it lands
    under the ``custom_uploads`` stem and therefore survives a later
    Extract / Re-process of this source — the user hand-picked it, so a re-scan
    must not wipe it.

    The frame is read from the *original* file at full resolution regardless of
    whether the browser was playing the original stream or a converted preview,
    so HEVC/MKV sources the browser can't decode still yield a crisp
    training frame.
    """
    import tempfile

    from neme_anima.server.api.frames import (
        _get_or_make_tagger,
        ingest_kept_image,
    )

    video_path = Path(source.path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")

    cslug = deps.optional_character_slug(project, body.character_slug)
    target_slug = (
        cslug if cslug
        else (project.characters[0].slug if project.characters else "default")
    )

    # Clamp the requested time into the clip. Use the cached duration when we
    # have it, pulling back a hair from the very end so ffmpeg always has a
    # frame to decode rather than landing past the last presentation timestamp.
    t = max(0.0, float(body.time_seconds))
    if source.duration_seconds and source.duration_seconds > 0:
        t = min(t, max(0.0, source.duration_seconds - 0.05))
    if body.frame_idx is not None and body.frame_idx < 0:
        raise HTTPException(status_code=400, detail="frame_idx must be non-negative")

    project.kept_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_png = Path(td) / "capture.png"
        try:
            if body.frame_idx is None:
                await asyncio.to_thread(video_ops._extract_frame_at, video_path, tmp_png, t)
            else:
                await asyncio.to_thread(
                    video_ops._extract_frame_by_index,
                    video_path,
                    tmp_png,
                    body.frame_idx,
                    fps=source.fps,
                    t_seconds=t,
                )
        except Exception as e:
            logger.exception("frame capture failed for %s @ %.3fs", video_path, t)
            raise HTTPException(
                status_code=500,
                detail=f"frame capture failed: {type(e).__name__}: {e}",
            ) from e
        data = tmp_png.read_bytes()

    tagger = _get_or_make_tagger(request, project)
    rec_dict, llm_error = await ingest_kept_image(
        project,
        data=data,
        filename_hint=f"{video_path.stem}_capture",
        target_slug=target_slug,
        tagger=tagger,
        timestamp_seconds=t,
    )
    if rec_dict is None:
        raise HTTPException(
            status_code=500, detail="could not process captured frame",
        )
    return {"frame": rec_dict, "llm_error": llm_error}


@router.get("/{slug}/sources/{idx}/stream")
async def stream_source(
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> FileResponse:
    """Serve the original video file with HTTP Range support.

    Starlette's :class:`FileResponse` honours the ``Range`` header
    out-of-the-box, so we just point at the file and set a reasonable
    ``Content-Type`` derived from the suffix. The frontend tries this
    endpoint first; if the browser can't decode the container/codec
    (HEVC in MKV being the common case for anime sources) the frontend
    triggers :func:`convert_source` (POST /convert) to produce a
    web-playable copy and then serves it via :func:`get_preview`.
    """
    video_path = Path(source.path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")
    media_type = video_ops._VIDEO_MIME_BY_SUFFIX.get(
        video_path.suffix.lower(), "application/octet-stream",
    )
    return FileResponse(
        video_path, media_type=media_type, filename=video_path.name,
    )


@router.post("/{slug}/sources/{idx}/convert")
async def convert_source(
    mode: str = "h264",
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Kick off (or no-op join) a playback conversion for the given mode.

    Returns immediately with the job state; the browser polls
    :func:`convert_status` for progress and then loads :func:`get_preview`.
    Concurrent identical calls are idempotent — a running job is left alone and
    a finished cache file short-circuits.
    """
    if mode not in ("remux", "h264"):
        raise HTTPException(status_code=400, detail="mode must be 'remux' or 'h264'")
    video_path = Path(source.path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")

    cache_path = video_ops._preview_cache_path(project, video_path, mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = (str(video_path.resolve()), mode)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        video_ops._CONVERT_JOBS[key] = {"state": "ready", "pct": 100.0, "mode": mode, "error": ""}
        return video_ops._CONVERT_JOBS[key]

    job = video_ops._CONVERT_JOBS.get(key)
    if job and job["state"] == "running":
        return job

    video_ops._CONVERT_JOBS[key] = {"state": "running", "pct": 0.0, "mode": mode, "error": ""}
    duration = await asyncio.to_thread(video_ops._probe_duration_seconds, video_path)
    task = asyncio.create_task(
        asyncio.to_thread(video_ops._run_convert_job, key, video_path, cache_path, mode, duration),
    )
    video_ops._CONVERT_TASKS.add(task)
    task.add_done_callback(video_ops._CONVERT_TASKS.discard)
    return video_ops._CONVERT_JOBS[key]


@router.get("/{slug}/sources/{idx}/convert/status")
async def convert_status(
    mode: str = "h264",
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Report conversion progress: {state, pct, mode, error}.

    A present cache file always reads as ``ready`` (covers a server restart
    that dropped the in-memory job state). Otherwise the live job state, or
    ``idle`` if nothing was ever started for this (source, mode).
    """
    if mode not in ("remux", "h264"):
        raise HTTPException(status_code=400, detail="mode must be 'remux' or 'h264'")
    video_path = Path(source.path)
    cache_path = video_ops._preview_cache_path(project, video_path, mode)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {"state": "ready", "pct": 100.0, "mode": mode, "error": ""}
    key = (str(video_path.resolve()), mode)
    return video_ops._CONVERT_JOBS.get(
        key, {"state": "idle", "pct": 0.0, "mode": mode, "error": ""},
    )


@router.get("/{slug}/sources/{idx}/preview")
async def get_preview(
    mode: str = "h264",
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> FileResponse:
    """Serve a previously-converted, web-playable MP4 for the given mode.

    Pure file server with Range support — the actual transcode is owned by
    :func:`convert_source`. Returns 404 if the conversion hasn't been run yet,
    so the frontend knows to POST /convert first.
    """
    if mode not in ("remux", "h264"):
        raise HTTPException(status_code=400, detail="mode must be 'remux' or 'h264'")
    video_path = Path(source.path)
    cache_path = video_ops._preview_cache_path(project, video_path, mode)
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        raise HTTPException(
            status_code=404, detail="preview not generated; POST /convert first",
        )
    return FileResponse(cache_path, media_type="video/mp4")


@router.delete("/{slug}/sources/{idx}/preview")
async def delete_preview(
    mode: str | None = None,
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Delete cached converted preview file(s) for a source.

    Without ``mode`` both the ``remux`` and ``h264`` caches are removed; pass a
    ``mode`` to drop just one. Also clears the in-memory job state so a later
    ``/convert`` re-runs from scratch. Idempotent — returns the modes whose
    files were actually present and removed.
    """
    if mode is not None and mode not in ("remux", "h264"):
        raise HTTPException(status_code=400, detail="mode must be 'remux' or 'h264'")
    video_path = Path(source.path)
    modes = [mode] if mode else ["remux", "h264"]
    removed: list[str] = []
    for m in modes:
        cache_path = video_ops._preview_cache_path(project, video_path, m)
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
            removed.append(m)
        video_ops._CONVERT_JOBS.pop((str(video_path.resolve()), m), None)
    return {"removed": removed}


@router.put("/{slug}/sources/{idx}/segments")
async def put_segments(
    body: PutSegmentsBody,
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: Source = Depends(deps.get_source),  # noqa: B008
) -> dict:
    """Replace the source's segment list. Validates, sorts, merges, persists.

    Defensive validation: even though the UI prevents most of these, the
    server is the source of truth.
      * ``start < end`` with both non-negative.
      * Both within the video's duration (probed if not already cached).
      * Overlapping or touching ranges are merged so the persisted shape
        is canonical (sorted, disjoint, non-empty).

    Returns ``{segments: [...]}`` so the client can confirm exactly what
    landed without a follow-up GET.
    """
    video_path = Path(source.path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")

    duration = source.duration_seconds
    if duration is None or duration <= 0:
        duration = await asyncio.to_thread(video_ops._probe_duration_seconds, video_path)
        if duration > 0:
            source.duration_seconds = duration
    # If duration is still unknown (probe failed), accept the segments as
    # given without an upper-bound check rather than rejecting everything.
    upper = duration if duration > 0 else None

    raw_ranges: list[tuple[float, float]] = []
    for seg in body.segments:
        start = float(seg.start_seconds)
        end = float(seg.end_seconds)
        if start < 0 or end <= start:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid segment [{start}, {end}): need 0 <= start < end"
                ),
            )
        if upper is not None and end > upper + 0.05:  # 50 ms tolerance
            raise HTTPException(
                status_code=400,
                detail=(
                    f"segment end {end:.3f}s exceeds video duration "
                    f"{upper:.3f}s"
                ),
            )
        raw_ranges.append((start, min(end, upper) if upper is not None else end))

    merged = video_ops._merge_ranges(raw_ranges)
    source.segments = [
        Segment(start_seconds=round(a, 3), end_seconds=round(b, 3))
        for (a, b) in merged
    ]
    project.save()
    return {
        "segments": [
            {"start_seconds": s.start_seconds, "end_seconds": s.end_seconds}
            for s in source.segments
        ],
    }


