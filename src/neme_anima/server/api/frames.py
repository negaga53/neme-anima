"""REST routes for /api/projects/{slug}/frames."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_anima.server.api import deps
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import CROP_SUFFIX, Project
from neme_anima.storage.sidecar import read_sidecar, update_sidecar
from neme_anima.tag import split_sidecar
from neme_anima.tag_vocabulary import tag_vocabulary_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["frames"])

# All drag-and-drop uploaded frames live under this synthetic video stem so
# they can be filtered/grouped just like extracted frames.
CUSTOM_VIDEO_STEM = "custom_uploads"

# Largest longest-side we keep on disk for uploaded images. The trainer
# handles bucketing/resizing itself, so cropping is wasteful, but a hard
# downscale ceiling keeps storage and tagging latency bounded.
MAX_UPLOAD_LONGEST_SIDE = 2048


class PutTagsBody(BaseModel):
    text: str


class PutDescriptionBody(BaseModel):
    text: str


class BulkDeleteBody(BaseModel):
    filenames: list[str]


class BulkReplaceBody(BaseModel):
    filenames: list[str]
    pattern: str
    replacement: str
    case_insensitive: bool = False


class BulkRetagBody(BaseModel):
    filenames: list[str]


class CropBody(BaseModel):
    x: int
    y: int
    width: int
    height: int


def _frame_paths(project: Project, filename: str) -> tuple[Path, Path]:
    return (project.kept_dir / f"{filename}.png",
            project.kept_dir / f"{filename}.txt")


def _resolve_tag_target(
    project: Project, filename: str,
) -> tuple[Path, Path, str]:
    """Pick the (image, sidecar, effective_filename) the WD14/LLM taggers
    should operate on.

    Why this exists: the user may have cropped a frame to focus the trained-
    against region. The crop derivative lives at ``{filename}_crop.png`` and
    is what the dataset actually carries forward. Tagging the original wide
    shot would mis-describe pixels that the model never sees, so when a
    derivative is on disk for an original filename we transparently retarget
    the *image* to it. The sidecar always stays at the original filename —
    there is only ever one ``.txt`` per kept frame, and training pairs the
    crop's pixels with that original sidecar at staging time. The effective
    filename returned is also the original's so the UI updates the row the
    user actually sees (crop derivatives are hidden from the grid).

    If ``filename`` is itself a ``_crop`` derivative we strip the suffix and
    treat it as the original — there's no ``_crop_crop`` ghost and the .txt
    still belongs to the original.
    """
    base = filename[: -len(CROP_SUFFIX)] if filename.endswith(CROP_SUFFIX) else filename
    orig_png, txt = _frame_paths(project, base)
    deriv_png, _deriv_txt, _spec = _crop_paths(project, base)
    img = deriv_png if deriv_png.is_file() else orig_png
    return img, txt, base


# Sentinel character_slug filter: matches frames whose slug isn't in the
# project's current character set (e.g. legacy "default" rows in a project
# that has since renamed/deleted the default character). Surfaced to the
# user in the Frames tab as the "unsorted" chip so misrouted frames don't
# silently disappear.
UNSORTED_FILTER_SENTINEL = "__unsorted__"


@router.get("/{slug}/frames")
async def list_frames(
    project: Project = Depends(deps.get_project),  # noqa: B008
    source: str | None = Query(None),
    kept_only: bool = Query(True),
    query: str | None = Query(None),
    character_slug: str | None = Query(None),
    offset: int = Query(0),
    limit: int = Query(500),
) -> dict:
    log = MetadataLog(project.metadata_path)
    known_slugs = {c.slug for c in project.characters}

    # The metadata log is append-only — a delete or rerun leaves orphan rows
    # behind. We dedupe by filename (keep the most recent record) and filter
    # to entries whose image still exists on disk so the UI never shows a
    # row with a broken thumbnail. Crop derivatives (`*_crop`) are hidden
    # from the grid: they're an internal training-target replacement for
    # their original, not a separate frame the user picked.
    by_filename: dict[str, FrameRecord] = {}
    for rec in log.iter_records(video_stem=source):
        if kept_only and not rec.kept:
            continue
        if rec.filename.endswith(CROP_SUFFIX):
            continue
        by_filename[rec.filename] = rec

    kept_dir = project.kept_dir
    rejected_dir = project.rejected_dir
    tokens = _parse_tag_query(query) if query else []
    items = []
    total_in_view = 0
    # Count of on-disk kept frames (in the current source/kept view) whose
    # slug isn't a current character — computed independently of the active
    # character filter and tag query so the top bar can decide whether to
    # surface the "Unsorted" chip at all.
    unsorted_total = 0
    for rec in by_filename.values():
        on_disk = (
            kept_dir / f"{rec.filename}.png" if rec.kept else rejected_dir / f"{rec.filename}.png"
        )
        if not on_disk.is_file():
            continue
        if rec.character_slug not in known_slugs:
            unsorted_total += 1
        # Character-slug filter is applied here (after the on-disk check) so
        # the per-character total reflects what the user can actually see.
        # The "unsorted" sentinel matches any record whose slug isn't in the
        # project's current character list — that way frames orphaned by a
        # rename/delete still surface to the user instead of vanishing.
        if character_slug == UNSORTED_FILTER_SENTINEL:
            if rec.character_slug in known_slugs:
                continue
        elif character_slug is not None and rec.character_slug != character_slug:
            continue
        # `total` is the count in the current source/character/kept_only view
        # *before* the tag query is applied, so the UI can render "X / Y"
        # when a search is active without firing a second listFrames call.
        total_in_view += 1
        if tokens and not _frame_matches_tag_query(
            kept_dir / f"{rec.filename}.txt", tokens,
        ):
            continue
        sidecar_flags = _sidecar_flags(kept_dir / f"{rec.filename}.txt")
        items.append({
            "filename": rec.filename,
            "kept": rec.kept,
            "video_stem": rec.video_stem,
            "scene_idx": rec.scene_idx,
            "tracklet_id": rec.tracklet_id,
            "frame_idx": rec.frame_idx,
            "timestamp_seconds": rec.timestamp_seconds,
            "ccip_distance": rec.ccip_distance,
            "score": rec.score,
            "character_slug": rec.character_slug,
            # The grid swaps in the crop derivative's pixels when one exists.
            "has_crop": (kept_dir / f"{rec.filename}{CROP_SUFFIX}.png").is_file(),
            **sidecar_flags,
        })
    return {
        "count": len(items),
        "total": total_in_view,
        "unsorted_total": unsorted_total,
        "items": items[offset: offset + limit],
    }


def _parse_tag_query(query: str) -> list[tuple[bool, str]]:
    """Split a search query into ``(is_negation, lowercased_substring)`` tokens.

    Whitespace separates tokens; a leading ``~`` flips a token to negation.
    Empty tokens are dropped. Substring (not exact) match against the
    danbooru tag line — typing ``red`` matches ``red eyes`` and ``red hair``,
    and ``~hat`` filters out anything whose tag line contains ``hat``.
    """
    out: list[tuple[bool, str]] = []
    for raw in query.split():
        if raw.startswith("~"):
            t = raw[1:].strip().lower()
            if t:
                out.append((True, t))
        else:
            t = raw.strip().lower()
            if t:
                out.append((False, t))
    return out


def _frame_matches_tag_query(
    txt_path: Path, tokens: list[tuple[bool, str]],
) -> bool:
    """Return True iff every positive token is present and every negation is absent."""
    if not tokens:
        return True
    haystack = read_sidecar(txt_path)[0].lower()
    for is_neg, tok in tokens:
        present = tok in haystack
        if is_neg and present:
            return False
        if not is_neg and not present:
            return False
    return True


def _sidecar_flags(txt_path: Path) -> dict[str, bool]:
    """Return booleans for populated sidecar sections.

    Reads the file rather than just stat'ing — a stale .txt with whitespace
    in either section should still count as empty so overwrite warnings and
    grid badges stay honest. The files are tiny (a few hundred bytes) so this
    is cheap.
    """
    danbooru, description = read_sidecar(txt_path)
    return {"has_tags": bool(danbooru), "has_description": bool(description)}


@router.get("/{slug}/frames/{filename}/image")
async def get_frame_image(
    filename: str, project: Project = Depends(deps.get_project),  # noqa: B008
) -> FileResponse:
    png, _ = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    return FileResponse(png, media_type="image/png")


@router.get("/{slug}/frames/{filename}/tags")
async def get_tags(filename: str, project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    _, txt = _frame_paths(project, filename)
    if not txt.exists():
        return {"text": ""}
    return {"text": txt.read_text(encoding="utf-8").rstrip("\n")}


@router.put("/{slug}/frames/{filename}/tags")
async def put_tags(
    filename: str, body: PutTagsBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Replace the sidecar with the body's text.

    The request can be a single line (danbooru only) or two lines
    (danbooru + description). Either shape goes through ``split_sidecar``
    + ``join_sidecar`` so the danbooru line gets the standard dedupe and
    whitespace normalization — manual edits that introduce a duplicate tag
    don't make it to disk.

    When the body has no description line, the existing on-disk description
    is preserved. The frontend's tag-pill add/remove path can fire this
    PUT before its lazy hover-fetch of the sidecar resolves; in that race
    the body legitimately has no description to send, and we must not
    interpret silence as "delete the description". Sending an explicit
    two-line body still overwrites whatever's on disk — that's the
    description-modal path.
    """
    png, txt = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    danbooru, description = split_sidecar(body.text)
    # No description line in the body means "preserve what's on disk" —
    # see the docstring; an explicit two-line body still overwrites.
    final = update_sidecar(txt, tags=danbooru, description=description or None)
    # Return what was actually saved (post-dedupe), without the trailing
    # newline join_sidecar adds — the frontend uses this to refresh its
    # local tag cache, so any normalization the route applied stays
    # consistent on the client side too.
    return {"text": final.rstrip("\n")}


@router.get("/{slug}/frames/{filename}/description")
async def get_description(filename: str, project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    """Return only the LLM description (line 2 of the sidecar)."""
    _, txt = _frame_paths(project, filename)
    _, description = read_sidecar(txt)
    return {"text": description}


@router.put("/{slug}/frames/{filename}/description")
async def put_description(
    filename: str, body: PutDescriptionBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Replace only the description line; the danbooru tag line is preserved."""
    png, txt = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    update_sidecar(txt, description=body.text)
    return {"text": body.text}


def _cleanup_crop_artifacts(project: Project, filename: str) -> None:
    """Remove crop-related siblings for a frame being deleted.

    Two cases:
      * Original deleted → drop its derivative png and the .crop.json
        sidecar so the next time a same-named frame is added it starts fresh.
        Also removes a legacy ``<name>_crop.txt`` if one is on disk from
        before the image-only derivative layout.
      * Derivative deleted → drop the parent's .crop.json sidecar so the
        next "open original" doesn't show a phantom rectangle for a crop
        whose result has been thrown away.
    """
    if filename.endswith(CROP_SUFFIX):
        parent = filename[: -len(CROP_SUFFIX)]
        spec = project.kept_dir / f"{parent}.crop.json"
        if spec.is_file():
            spec.unlink()
        return
    deriv_png, deriv_txt, spec = _crop_paths(project, filename)
    for p in (deriv_png, deriv_txt, spec):
        if p.is_file():
            p.unlink()


@router.delete("/{slug}/frames/{filename}", status_code=204)
async def delete_frame(filename: str, project: Project = Depends(deps.get_project)) -> Response:  # noqa: B008
    png, txt = _frame_paths(project, filename)
    if png.exists():
        png.unlink()
    if txt.exists():
        txt.unlink()
    _cleanup_crop_artifacts(project, filename)
    return Response(status_code=204)


@router.post("/{slug}/frames/bulk-delete")
async def bulk_delete(body: BulkDeleteBody, project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    deleted = 0
    skipped: list[dict] = []
    for filename in body.filenames:
        png, txt = _frame_paths(project, filename)
        if png.exists():
            png.unlink()
            deleted += 1
        else:
            skipped.append({"filename": filename, "reason": "not found on disk"})
        if txt.exists():
            txt.unlink()
        _cleanup_crop_artifacts(project, filename)
    return {"deleted": deleted, "total": len(body.filenames), "skipped": skipped}


@router.post("/{slug}/frames/bulk-tags-replace")
async def bulk_tags_replace(
    body: BulkReplaceBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Run a regex over the *danbooru* line only — the LLM description on row
    two stays untouched so the user can rewrite tags without disturbing
    captions written by a separate model.

    Tip for adding tags: use a pattern like ``^`` with replacement ``new_tag, ``
    to prepend, or ``$`` with replacement ``, new_tag`` to append. To replace
    the whole tag set, use ``.*`` with the new comma-separated string.
    """
    flags = re.IGNORECASE if body.case_insensitive else 0
    try:
        regex = re.compile(body.pattern, flags)
    except re.error as e:
        raise HTTPException(status_code=422, detail=f"invalid regex: {e}") from e
    changed = 0
    skipped: list[dict] = []
    for filename in body.filenames:
        _, txt = _frame_paths(project, filename)
        if not txt.exists():
            skipped.append({"filename": filename, "reason": "no sidecar"})
            continue
        danbooru, _description = read_sidecar(txt)
        new_danbooru, n = regex.subn(body.replacement, danbooru)
        if n > 0 and new_danbooru != danbooru:
            update_sidecar(txt, tags=new_danbooru)
            changed += n
    return {"changed": changed, "total": len(body.filenames), "skipped": skipped}


@router.post("/{slug}/frames/bulk-retag-danbooru")
async def bulk_retag_danbooru(
    request: Request, body: BulkRetagBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Re-run the WD14 tagger on selected frames; preserves the LLM line."""
    import numpy as np
    from PIL import Image

    tagger = _get_or_make_tagger(request)

    def _tag_one(filename: str) -> tuple[bool, str | None]:
        # Prefer the cropped derivative when one exists for an original.
        # Tags must describe the image we actually train against; tagging
        # a wide shot with a person who isn't in the crop is exactly the
        # bug this avoids.
        png, txt, eff = _resolve_tag_target(project, filename)
        if not png.is_file():
            return False, None
        with Image.open(png) as im:
            arr = np.array(im.convert("RGB"))
        new_danbooru = tagger.tag(arr).text
        update_sidecar(txt, tags=new_danbooru)
        return True, eff

    retagged = 0
    skipped: list[dict] = []
    effective_filenames: list[str | None] = []
    for filename in body.filenames:
        try:
            ok, eff = await asyncio.to_thread(_tag_one, filename)
            err = None if ok else "frame not found on disk"
        except Exception as exc:  # noqa: BLE001 — bulk never throws per-item
            ok, eff, err = False, None, f"{type(exc).__name__}: {exc}"
        if ok:
            retagged += 1
        else:
            skipped.append({"filename": filename, "reason": err})
        effective_filenames.append(eff)
    return {
        "retagged": retagged,
        "total": len(body.filenames),
        "skipped": skipped,
        # Parallel to body.filenames; an entry differs from the input when
        # a `_crop` derivative was tagged in place of the original. The
        # frontend uses this to flip badges on the row that actually got
        # written rather than the row the user clicked.
        "effective_filenames": effective_filenames,
    }


@router.post("/{slug}/frames/bulk-retag-llm")
async def bulk_retag_llm(body: BulkRetagBody, project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    """Re-run the LLM description on selected frames; preserves the danbooru
    line. Returns 422 if the project has no LLM model configured — the
    frontend won't show the button in that state, but a stale tab might still
    fire it.
    """
    from neme_anima.llm import DEFAULT_PROMPT, LLMUnavailable, describe_image

    if not project.llm.model:
        raise HTTPException(
            status_code=422,
            detail="LLM tagging not configured: pick a model in Settings first",
        )
    endpoint = project.llm.endpoint
    model = project.llm.model
    prompt = project.llm.prompt or DEFAULT_PROMPT
    api_key = project.llm.api_key or None

    def _describe_one(filename: str) -> tuple[bool, str | None, str | None]:
        # Same retarget rule as the WD14 path: when a crop exists for an
        # original, describe the crop instead. Otherwise the LLM caption
        # would describe pixels the trainer never sees.
        png, txt, eff = _resolve_tag_target(project, filename)
        if not png.is_file():
            return False, None, None
        danbooru, _ = read_sidecar(txt)
        try:
            description = describe_image(
                endpoint=endpoint, model=model, image_path=png,
                prompt=prompt, danbooru_tags=danbooru or None,
                api_key=api_key,
            )
        except LLMUnavailable as exc:
            return False, str(exc), eff
        update_sidecar(txt, description=description)
        return True, None, eff

    described = 0
    last_error: str | None = None
    skipped: list[dict] = []
    effective_filenames: list[str | None] = []
    for filename in body.filenames:
        ok, err, eff = await asyncio.to_thread(_describe_one, filename)
        if ok:
            described += 1
        else:
            if err:
                last_error = err
            skipped.append({
                "filename": filename,
                "reason": err or "frame not found on disk",
            })
        effective_filenames.append(eff if ok else None)
    return {
        "described": described,
        "total": len(body.filenames),
        "error": last_error,
        "skipped": skipped,
        # Parallel to body.filenames; entry is None when that filename
        # failed. When the crop took priority over the original, the entry
        # is the crop's filename so the frontend pops the right row.
        "effective_filenames": effective_filenames,
    }


def _parse_tag_line(line: str) -> list[str]:
    """Split a danbooru tag line into a deduped, order-preserving list."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in line.split(","):
        t = raw.strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _reconcile_review(raw: dict, existing: list[str], index) -> dict:
    """Turn the model's raw verdict into a trustworthy, applyable diff.

    The model is helpful but imperfect — it occasionally proposes tags that
    aren't real danbooru tags, proposes a non-canonical alias of a tag that's
    already present, or loses track of which tags existed. So we don't trust its
    bookkeeping: an existing tag is KEPT unless it's explicitly removed, every
    proposed addition is canonicalized + validated against the vocabulary, and
    anything that collapses onto an existing/removed tag is dropped (with a note
    explaining why). The result is deterministic regardless of model sloppiness.
    """
    def norm(s: str) -> str:
        return s.replace("_", " ").strip().lower()

    existing_norm = {norm(t) for t in existing}
    notes: list[str] = []

    removed_norm: set[str] = set()
    remove: list[dict] = []
    for r in raw.get("remove", []):
        if not isinstance(r, dict):
            continue
        tag = str(r.get("tag", "")).strip()
        if not tag:
            continue
        n = norm(tag)
        if n not in existing_norm:
            notes.append(f"ignored removal of '{tag}' (not a current tag)")
            continue
        if n in removed_norm:
            continue
        removed_norm.add(n)
        # Report the tag exactly as it appears on disk.
        on_disk = next((t for t in existing if norm(t) == n), tag)
        remove.append({"tag": on_disk, "reason": str(r.get("reason", "")).strip()})

    add: list[dict] = []
    added_norm: set[str] = set()
    for a in raw.get("add", []):
        if not isinstance(a, dict):
            continue
        tag = str(a.get("tag", "")).strip()
        if not tag:
            continue
        canon, real = index.canonicalize(tag)
        cn = norm(canon)
        if not real:
            notes.append(f"dropped proposed '{tag}' (not a danbooru tag)")
            continue
        if cn in existing_norm and cn not in removed_norm:
            notes.append(f"dropped proposed '{tag}' ('{canon}' is already present)")
            continue
        if cn in added_norm:
            continue
        added_norm.add(cn)
        if norm(tag) != cn:
            notes.append(f"'{tag}' normalized to '{canon}'")
        add.append({"tag": canon, "reason": str(a.get("reason", "")).strip()})

    keep = [t for t in existing if norm(t) not in removed_norm]
    proposed_final = keep + [a["tag"] for a in add]
    return {"keep": keep, "remove": remove, "add": add,
            "proposed_final": proposed_final, "notes": notes}


@router.post("/{slug}/frames/{filename}/review-tags")
async def review_frame_tags(
    request: Request, filename: str, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """LLM-assisted tag review for one frame (vision + danbooru tool calling).

    Read-only: returns a proposed diff (keep / remove-with-reason /
    add-with-reason) for the user to accept or reject in the editor — it does
    NOT touch the sidecar. Like the WD14/LLM retag paths, the *image* reviewed
    is the crop derivative when one exists, while tags come from the original's
    sidecar. Returns 422 when no LLM model is configured or the danbooru
    vocabulary hasn't been downloaded.
    """
    from neme_anima.llm import LLMUnavailable, review_tags
    from neme_anima.tag_vocabulary import load_index

    if not project.llm.model:
        raise HTTPException(
            status_code=422,
            detail="LLM tagging not configured: pick a model in Settings first",
        )
    csv_path = tag_vocabulary_path(request.app.state.state_dir)
    if not csv_path.exists():
        raise HTTPException(
            status_code=422,
            detail="tag vocabulary not downloaded; run `neme-anima tags fetch` first",
        )
    png, txt, eff = _resolve_tag_target(project, filename)
    if not png.is_file():
        raise HTTPException(status_code=404, detail="frame not found")
    danbooru, _ = read_sidecar(txt)
    existing = _parse_tag_line(danbooru)

    endpoint = project.llm.endpoint
    model = project.llm.model
    api_key = project.llm.api_key or None

    def _run() -> dict:
        index = load_index(csv_path)
        raw = review_tags(
            endpoint=endpoint, model=model, image_path=png,
            existing_tags=existing, search_fn=index.search, api_key=api_key,
        )
        return _reconcile_review(raw, existing, index)

    try:
        result = await asyncio.to_thread(_run)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["effective_filename"] = eff
    return result


def _record_to_dict(rec: FrameRecord, project: Project | None = None) -> dict:
    flags = (
        _sidecar_flags(project.kept_dir / f"{rec.filename}.txt")
        if project is not None
        else {"has_tags": False, "has_description": False}
    )
    has_crop = (
        (project.kept_dir / f"{rec.filename}{CROP_SUFFIX}.png").is_file()
        if project is not None
        else False
    )
    return {
        "filename": rec.filename,
        "kept": rec.kept,
        "video_stem": rec.video_stem,
        "scene_idx": rec.scene_idx,
        "tracklet_id": rec.tracklet_id,
        "frame_idx": rec.frame_idx,
        "timestamp_seconds": rec.timestamp_seconds,
        "ccip_distance": rec.ccip_distance,
        "score": rec.score,
        "character_slug": rec.character_slug,
        "has_crop": has_crop,
        **flags,
    }


def _find_record(project: Project, filename: str) -> FrameRecord | None:
    """Walk the metadata log and return the most recent record for ``filename``."""
    found: FrameRecord | None = None
    log = MetadataLog(project.metadata_path)
    for rec in log.iter_records():
        if rec.filename == filename:
            found = rec
    return found


# Each original gets at most one crop derivative; re-cropping overwrites it.
# The fixed suffix (no numeric counter) is what makes the round-trip work —
# without it we'd accumulate _crop1/_crop2/... and have no way to find
# "the" derivative when re-opening the original. The suffix itself lives in
# storage/project.py because both the API and the trainer key off it.


def _crop_paths(project: Project, original_filename: str) -> tuple[Path, Path, Path]:
    """Return ``(derivative_png, derivative_txt, crop_spec_json)`` paths.

    The spec sidecar lives next to the *original* (not the derivative) so the
    "load the existing rect when reopening the original" lookup is a single
    file existence check. Naming it with a dot prefix to ``crop`` keeps it
    clearly separate from any tag .txt sidecar.
    """
    crop_base = f"{original_filename}{CROP_SUFFIX}"
    return (
        project.kept_dir / f"{crop_base}.png",
        project.kept_dir / f"{crop_base}.txt",
        project.kept_dir / f"{original_filename}.crop.json",
    )


@router.get("/{slug}/frames/{filename}/crop")
async def get_crop_rect(filename: str, project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    """Return the saved crop rectangle for ``filename`` if one exists.

    404 when no crop has been confirmed for this frame yet — the modal uses
    that as the "no overlay, start full-image" signal.
    """
    _, _, spec = _crop_paths(project, filename)
    if not spec.is_file():
        raise HTTPException(status_code=404, detail="no saved crop")
    try:
        data = json.loads(spec.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"corrupt crop spec: {exc}") from exc
    # Only echo the four fields the client cares about; ignores anything
    # extra we may add later for forwards compat.
    return {
        "x": int(data.get("x", 0)),
        "y": int(data.get("y", 0)),
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
    }


@router.post("/{slug}/frames/{filename}/crop")
async def crop_frame_endpoint(
    filename: str, body: CropBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Save (or overwrite) the cropped derivative for an original frame.

    Behavior: the original is never touched. The derivative lives at a fixed
    ``<filename>_crop`` filename — re-cropping overwrites it. A
    ``<filename>.crop.json`` sidecar records the rectangle so the modal can
    repaint the overlay when the original is reopened.
    """
    from PIL import Image

    src_png, _ = _frame_paths(project, filename)
    if not src_png.exists():
        raise HTTPException(status_code=404, detail="frame not found")

    original = _find_record(project, filename)
    if original is None:
        raise HTTPException(status_code=404, detail="frame metadata not found")

    out_png, out_txt, spec_path = _crop_paths(project, filename)
    crop_base = out_png.stem

    with Image.open(src_png) as im:
        im_w, im_h = im.size
        x = max(0, min(int(body.x), im_w))
        y = max(0, min(int(body.y), im_h))
        w = max(1, min(int(body.width), im_w - x))
        h = max(1, min(int(body.height), im_h - y))
        cropped = im.crop((x, y, x + w, y + h))
        cropped.save(out_png)  # overwrites prior derivative if present

    # The crop derivative is image-only; tags stay on the original sidecar.
    # Remove any legacy `<name>_crop.txt` left over from earlier versions
    # so the trainer-staging step doesn't see a phantom paired sidecar.
    if out_txt.exists():
        out_txt.unlink()

    spec_path.write_text(
        json.dumps({"x": x, "y": y, "width": w, "height": h}),
        encoding="utf-8",
    )

    new_rec = FrameRecord(
        filename=crop_base,
        kept=True,
        scene_idx=original.scene_idx,
        tracklet_id=original.tracklet_id,
        frame_idx=original.frame_idx,
        timestamp_seconds=original.timestamp_seconds,
        bbox=(x, y, x + w, y + h),
        ccip_distance=original.ccip_distance,
        sharpness=original.sharpness,
        visibility=original.visibility,
        aspect=(w / h) if h else 1.0,
        score=original.score,
        video_stem=original.video_stem,
    )
    # The metadata log is append-only; list_frames dedupes by filename and
    # keeps the latest record, so a re-crop "replaces" the visible bbox
    # without us having to rewrite the log.
    MetadataLog(project.metadata_path).append(new_rec)
    return _record_to_dict(new_rec, project)


def _get_or_make_tagger(request: Request):
    """Cache a single Tagger instance on app.state — WD14 model load is slow."""
    cached = getattr(request.app.state, "_tagger", None)
    if cached is not None:
        return cached
    from neme_anima.tag import Tagger
    cached = Tagger()
    request.app.state._tagger = cached
    return cached


def _process_uploaded_image(
    project: Project, data: bytes, filename_hint: str,
) -> tuple[Path, int, int, str]:
    """Decode, downscale-if-huge, save PNG with a unique name. No tagging here.

    Returns (png_path, width, height, base_filename).
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if max(im.width, im.height) > MAX_UPLOAD_LONGEST_SIDE:
            scale = MAX_UPLOAD_LONGEST_SIDE / max(im.width, im.height)
            new_size = (max(1, int(im.width * scale)),
                        max(1, int(im.height * scale)))
            im = im.resize(new_size, Image.LANCZOS)
        # 8-char random suffix is plenty to avoid collisions with concurrent drops.
        token = secrets.token_hex(4)
        base = f"{CUSTOM_VIDEO_STEM}__{token}"
        png_path = project.kept_dir / f"{base}.png"
        # Vanishingly unlikely, but keep the loop tight just in case.
        while png_path.exists():
            token = secrets.token_hex(4)
            base = f"{CUSTOM_VIDEO_STEM}__{token}"
            png_path = project.kept_dir / f"{base}.png"
        im.save(png_path)
        return png_path, im.width, im.height, base


async def ingest_kept_image(
    project: Project,
    *,
    data: bytes,
    filename_hint: str,
    target_slug: str,
    tagger,
    timestamp_seconds: float = 0.0,
) -> tuple[dict | None, str | None]:
    """Decode, store, WD14-tag, optionally LLM-describe, and register one image
    as a kept frame.

    Shared by the drag-and-drop upload route and the segment-editor
    frame-capture route so an image dropped from disk and a frame grabbed from
    a video player land in the dataset by exactly the same path.

    WD14 tagging always runs — that's the "tag it normally" behaviour the
    extraction pipeline applies. The LLM description is a second pass gated on
    the project having LLM tagging enabled with a model selected, matching the
    pipeline and bulk-retag routes.

    Returns ``(record_dict, llm_error)``. ``record_dict`` is ``None`` when the
    bytes couldn't be decoded/saved — the caller treats that as a skip.
    ``llm_error`` carries the human-readable reason the description was left
    blank (so the UI can surface a one-line warning), or ``None``.
    """
    import numpy as np
    from PIL import Image

    try:
        png_path, w, h, base = await asyncio.to_thread(
            _process_uploaded_image, project, data, filename_hint,
        )
    except Exception:
        return None, None

    def _do_tag(p: Path) -> str:
        with Image.open(p) as pim:
            arr = np.array(pim.convert("RGB"))
        return tagger.tag(arr).text

    try:
        tag_text = await asyncio.to_thread(_do_tag, png_path)
    except Exception:
        tag_text = ""

    description = ""
    llm_error: str | None = None
    if project.llm.enabled and project.llm.model:
        from neme_anima.llm import DEFAULT_PROMPT, LLMUnavailable, describe_image

        def _do_describe() -> str:
            return describe_image(
                endpoint=project.llm.endpoint,
                model=project.llm.model,
                image_path=png_path,
                prompt=project.llm.prompt or DEFAULT_PROMPT,
                danbooru_tags=tag_text or None,
                api_key=project.llm.api_key or None,
            )

        try:
            description = await asyncio.to_thread(_do_describe)
        except LLMUnavailable as exc:
            # Endpoint reachable but unhappy (timeout, bad model id, auth
            # refused). Log + remember so the user knows *why* line 2 is blank.
            logger.warning("ingest.describe_failed file=%s: %s", filename_hint, exc)
            llm_error = str(exc)
            description = ""
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingest.describe_crashed file=%s", filename_hint)
            llm_error = f"{type(exc).__name__}: {exc}"
            description = ""

    update_sidecar(
        png_path.with_suffix(".txt"), tags=tag_text, description=description,
    )

    rec = FrameRecord(
        filename=base,
        kept=True,
        scene_idx=0,
        tracklet_id=0,
        frame_idx=0,
        timestamp_seconds=float(timestamp_seconds),
        bbox=(0, 0, w, h),
        ccip_distance=0.0,
        sharpness=0.0,
        visibility=0.0,
        aspect=(w / h) if h else 1.0,
        score=0.0,
        video_stem=CUSTOM_VIDEO_STEM,
        character_slug=target_slug,
    )
    MetadataLog(project.metadata_path).append(rec)
    return _record_to_dict(rec, project), llm_error


@router.post("/{slug}/frames/upload")
async def upload_frames(
    request: Request,
    files: list[UploadFile],
    character_slug: str | None = Query(None),
    project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Accept dropped image files, store + auto-tag them as custom-source frames.

    ``character_slug`` (when provided) routes every dropped image to that
    character's bucket. Omitting it defaults to the project's first
    character — what the mono-character UI does today, and what dropping
    in the "All" view should do until CCIP-routing is wired in.
    """
    cslug = deps.optional_character_slug(project, character_slug)
    target_slug = (
        cslug if cslug
        else (project.characters[0].slug if project.characters else "default")
    )
    project.kept_dir.mkdir(parents=True, exist_ok=True)

    added: list[dict] = []
    skipped: list[str] = []
    # Last LLM-describe error (if any) is bubbled to the response so the
    # UI can surface a one-line warning instead of silently writing the
    # sidecar without line 2. We only keep the most recent message — the
    # first failure usually identifies the underlying issue (timeout,
    # auth, wrong model name) and the rest are echoes.
    llm_error: str | None = None

    tagger = _get_or_make_tagger(request)

    for f in files:
        try:
            data = await f.read()
            if not data:
                skipped.append(f.filename or "<empty>")
                continue
            rec_dict, err = await ingest_kept_image(
                project,
                data=data,
                filename_hint=f.filename or "drop",
                target_slug=target_slug,
                tagger=tagger,
            )
            if rec_dict is None:
                skipped.append(f.filename or "<unknown>")
                continue
            if err:
                llm_error = err
            added.append(rec_dict)
        finally:
            await f.close()

    return {"added": added, "skipped": skipped, "llm_error": llm_error}


# ---------------------------------------------------------------------------
# Multi-character: move + duplicate frame endpoints
# ---------------------------------------------------------------------------


class MoveFrameBody(BaseModel):
    """Body for the move/also-assign endpoints."""
    character_slug: str


class BulkMoveBody(BaseModel):
    filenames: list[str]
    character_slug: str


def _append_moved_record(project: Project, rec: FrameRecord, new_slug: str) -> FrameRecord:
    """Append a copy of ``rec`` with ``character_slug`` swapped.

    The metadata log is append-only and ``list_frames`` is last-write-wins
    per filename, so emitting a fresh record with the new slug is enough to
    flip the frame's character without rewriting older rows. We also pin
    ``kept=True`` here — the move action only makes sense on currently-kept
    frames, and a stale rejected row could otherwise resurrect a deleted
    frame's slug into the active view.
    """
    moved = FrameRecord(
        filename=rec.filename,
        kept=True,
        scene_idx=rec.scene_idx,
        tracklet_id=rec.tracklet_id,
        frame_idx=rec.frame_idx,
        timestamp_seconds=rec.timestamp_seconds,
        bbox=rec.bbox,
        ccip_distance=rec.ccip_distance,
        sharpness=rec.sharpness,
        visibility=rec.visibility,
        aspect=rec.aspect,
        score=rec.score,
        video_stem=rec.video_stem,
        character_slug=new_slug,
    )
    MetadataLog(project.metadata_path).append(moved)
    return moved


@router.post("/{slug}/frames/{filename}/character")
async def move_frame_to_character(
    filename: str, body: MoveFrameBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Reassign a frame to a different character.

    Implementation: append a fresh metadata row with the new slug. The
    on-disk PNG and sidecar stay where they are — this is a metadata-only
    operation, which is what makes per-character "moves" cheap.
    """
    deps.require_character(project, body.character_slug)
    rec = _find_record(project, filename)
    if rec is None:
        raise HTTPException(status_code=404, detail="frame metadata not found")
    moved = _append_moved_record(project, rec, body.character_slug)
    return _record_to_dict(moved, project)


@router.post("/{slug}/frames/bulk-move")
async def bulk_move_to_character(
    body: BulkMoveBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Reassign many frames to a target character in one round-trip."""
    deps.require_character(project, body.character_slug)
    moved = 0
    missing: list[str] = []
    skipped: list[dict] = []
    for fn in body.filenames:
        rec = _find_record(project, fn)
        if rec is None:
            missing.append(fn)
            skipped.append({"filename": fn, "reason": "frame metadata not found"})
            continue
        _append_moved_record(project, rec, body.character_slug)
        moved += 1
    return {"moved": moved, "missing": missing,
            "total": len(body.filenames), "skipped": skipped}


def _duplicate_for_character(
    project: Project, rec: FrameRecord, new_slug: str,
) -> FrameRecord:
    """Copy a frame's PNG + sidecar to a fresh filename and append a row.

    Each character carries an independent caption — different core tags get
    pruned per character — so a true copy (not a hard-link) is the right
    semantics. The new filename keeps the original's recognisable prefix
    plus a short random suffix so two duplicates of the same frame, e.g.
    sent to two different characters, never collide.
    """
    src_png, src_txt = _frame_paths(project, rec.filename)
    if not src_png.is_file():
        raise FileNotFoundError(f"frame missing on disk: {rec.filename}")

    token = secrets.token_hex(4)
    base = f"{rec.filename}_dup_{token}"
    while (project.kept_dir / f"{base}.png").exists():
        token = secrets.token_hex(4)
        base = f"{rec.filename}_dup_{token}"

    dst_png = project.kept_dir / f"{base}.png"
    dst_txt = project.kept_dir / f"{base}.txt"
    dst_png.write_bytes(src_png.read_bytes())
    if src_txt.is_file():
        dst_txt.write_text(src_txt.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dst_txt.write_text("\n", encoding="utf-8")

    new_rec = FrameRecord(
        filename=base,
        kept=True,
        scene_idx=rec.scene_idx,
        tracklet_id=rec.tracklet_id,
        frame_idx=rec.frame_idx,
        timestamp_seconds=rec.timestamp_seconds,
        bbox=rec.bbox,
        ccip_distance=rec.ccip_distance,
        sharpness=rec.sharpness,
        visibility=rec.visibility,
        aspect=rec.aspect,
        score=rec.score,
        video_stem=rec.video_stem,
        character_slug=new_slug,
    )
    MetadataLog(project.metadata_path).append(new_rec)
    return new_rec


@router.post("/{slug}/frames/{filename}/duplicate")
async def duplicate_frame_for_character(
    filename: str, body: MoveFrameBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Also-assign a frame to a second character via a physical copy."""
    deps.require_character(project, body.character_slug)
    rec = _find_record(project, filename)
    if rec is None:
        raise HTTPException(status_code=404, detail="frame metadata not found")
    try:
        new_rec = _duplicate_for_character(project, rec, body.character_slug)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _record_to_dict(new_rec, project)


@router.post("/{slug}/frames/bulk-duplicate")
async def bulk_duplicate_for_character(
    body: BulkMoveBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    """Also-assign many frames to a target character. Returns the new
    filenames so the caller can refresh the view to show them inline."""
    deps.require_character(project, body.character_slug)
    duplicated: list[str] = []
    missing: list[str] = []
    skipped: list[dict] = []
    for fn in body.filenames:
        rec = _find_record(project, fn)
        if rec is None:
            missing.append(fn)
            skipped.append({"filename": fn, "reason": "frame metadata not found"})
            continue
        try:
            new_rec = _duplicate_for_character(project, rec, body.character_slug)
        except FileNotFoundError as e:
            missing.append(fn)
            skipped.append({"filename": fn, "reason": str(e)})
            continue
        duplicated.append(new_rec.filename)
    return {"duplicated": duplicated, "missing": missing,
            "total": len(body.filenames), "skipped": skipped}
