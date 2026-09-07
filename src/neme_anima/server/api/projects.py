"""REST routes for /api/projects."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from neme_anima.config import Thresholds
from neme_anima.extraction_cache import cache_state_for_source
from neme_anima.server.api import deps
from neme_anima.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectBody(BaseModel):
    name: str
    folder: str


class RegisterBody(BaseModel):
    folder: str


class LLMConfigBody(BaseModel):
    enabled: bool | None = None
    endpoint: str | None = None
    model: str | None = None
    prompt: str | None = None
    api_key: str | None = None
    disable_thinking: bool | None = None


class PatchProjectBody(BaseModel):
    name: str | None = None
    thresholds_overrides: dict | None = None
    pause_before_tag: bool | None = None
    auto_delete_rejected: bool | None = None
    tag_autocomplete: bool | None = None
    llm: LLMConfigBody | None = None


class DeleteProjectBody(BaseModel):
    delete_files: bool = False


def _resolve_current_thresholds(project: Project) -> Thresholds:
    """Reproduce pipeline._resolve_thresholds without importing the heavy
    pipeline module here — projects.py loads on every request and we want
    to keep that cheap. Mirrors the override-merge contract exactly."""
    base = Thresholds()
    for section_name, section_overrides in (project.thresholds_overrides or {}).items():
        section = getattr(base, section_name, None)
        if section is None:
            continue
        for k, v in section_overrides.items():
            if hasattr(section, k):
                setattr(section, k, v)
    return base


def _project_view(project: Project) -> dict:
    extracted_stems = _stems_with_kept_frames(project)
    current_thresholds = _resolve_current_thresholds(project)
    sources = []
    for i, s in enumerate(project.sources):
        d = asdict(s)
        # Drop the legacy "extraction_runs" field — it was never populated and
        # the UI doesn't use it. Replace with a persistent "extracted" flag
        # derived from on-disk kept frames so it survives restarts.
        d.pop("extraction_runs", None)
        d["extracted"] = Path(s.path).stem in extracted_stems
        # ``extraction_cache`` drives the smart Extract / Re-process button
        # states in the Sources tab:
        #   - "none": no cache → Extract enabled, Re-process disabled
        #   - "current": cache fresh + scan thresholds match → Extract muted
        #   - "stale": cache exists but scene/detect/track thresholds drifted
        #     → both buttons enabled, UI flags Extract as recommended
        d["extraction_cache"] = cache_state_for_source(
            project, i, current_thresholds,
        )
        sources.append(d)
    return {
        "slug": project.slug,
        "name": project.name,
        "folder": str(project.root.resolve()),
        "created_at": project.created_at.isoformat(),
        "sources": sources,
        # Backwards-compat top-level alias for the default character's refs.
        # The mono-character UI reads this; new character-aware clients should
        # prefer ``characters[*].refs``.
        "refs": [asdict(r) for r in project.refs],
        # Full multi-character listing — exposed so future UI can switch to
        # rendering per-character ref strips without a separate endpoint.
        "characters": [
            {
                "slug": c.slug, "name": c.name,
                "trigger_token": c.trigger_token,
                "refs": [asdict(r) for r in c.refs],
                "ref_count": len(c.refs),
                "core_tags": list(c.core_tags),
                "core_tags_freq_threshold": c.core_tags_freq_threshold,
                "core_tags_enabled": c.core_tags_enabled,
                "multiply": c.multiply,
            }
            for c in project.characters
        ],
        "thresholds_overrides": project.thresholds_overrides,
        "source_root": project.source_root,
        "pause_before_tag": project.pause_before_tag,
        "auto_delete_rejected": project.auto_delete_rejected,
        "tag_autocomplete": project.tag_autocomplete,
        "rejected_count": _count_rejected(project),
        "llm": asdict(project.llm),
    }


def _stems_with_kept_frames(project: Project) -> set[str]:
    """Return the set of video stems that have at least one file written to
    the project's kept/ folder. Files are named ``<video_stem>__...png`` so we
    can recover the stem by splitting on the double underscore.
    """
    kept = project.kept_dir
    if not kept.exists():
        return set()
    stems: set[str] = set()
    try:
        for entry in kept.iterdir():
            if not entry.is_file():
                continue
            stem, sep, _ = entry.name.partition("__")
            if sep:
                stems.add(stem)
    except OSError:
        return set()
    return stems


def _count_rejected(project: Project) -> int:
    """Number of files currently sitting in the project's rejected/ folder."""
    rejected = project.rejected_dir
    if not rejected.exists():
        return 0
    try:
        return sum(1 for e in rejected.iterdir() if e.is_file())
    except OSError:
        return 0


@router.get("")
async def list_projects(request: Request) -> list[dict]:
    rows = request.app.state.registry.list()
    out: list[dict] = []
    for r in rows:
        try:
            project = Project.load(Path(r.folder))
            out.append({
                "slug": r.slug,
                "name": r.name,
                "folder": r.folder,
                "missing": False,
                "source_count": len(project.sources),
                "ref_count": len(project.refs),
                "last_opened_at": r.last_opened_at,
            })
        except FileNotFoundError:
            out.append({
                "slug": r.slug, "name": r.name, "folder": r.folder,
                "missing": True, "source_count": 0, "ref_count": 0,
                "last_opened_at": r.last_opened_at,
            })
    return out


@router.post("", status_code=201)
async def create_project(request: Request, body: CreateProjectBody) -> dict:
    target = Path(body.folder).expanduser()
    try:
        project = Project.create(target, name=body.name)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    request.app.state.registry.register(project)
    return _project_view(project)


@router.post("/register")
async def register_existing(request: Request, body: RegisterBody) -> dict:
    folder = Path(body.folder).expanduser()
    try:
        project = Project.load(folder)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    request.app.state.registry.register(project)
    return _project_view(project)


@router.get("/{slug}")
async def get_project(
    slug: str, request: Request, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    request.app.state.registry.touch(slug)
    return _project_view(project)


@router.patch("/{slug}")
async def patch_project(
    request: Request, body: PatchProjectBody, project: Project = Depends(deps.get_project),  # noqa: B008
) -> dict:
    if body.name is not None:
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="project name cannot be empty")
        project.name = new_name
    if body.thresholds_overrides is not None:
        project.thresholds_overrides = body.thresholds_overrides
    if body.pause_before_tag is not None:
        project.pause_before_tag = body.pause_before_tag
    if body.auto_delete_rejected is not None:
        project.auto_delete_rejected = body.auto_delete_rejected
    if body.tag_autocomplete is not None:
        project.tag_autocomplete = body.tag_autocomplete
    if body.llm is not None:
        if body.llm.enabled is not None:
            project.llm.enabled = body.llm.enabled
        if body.llm.endpoint is not None:
            project.llm.endpoint = body.llm.endpoint
        if body.llm.model is not None:
            project.llm.model = body.llm.model
        if body.llm.prompt is not None:
            project.llm.prompt = body.llm.prompt
        if body.llm.api_key is not None:
            project.llm.api_key = body.llm.api_key
        if body.llm.disable_thinking is not None:
            project.llm.disable_thinking = body.llm.disable_thinking
    project.save()
    request.app.state.registry.register(project)  # refresh name
    return _project_view(project)


@router.delete("/{slug}", status_code=204)
async def delete_project(
    request: Request, slug: str, body: DeleteProjectBody = DeleteProjectBody(),  # noqa: B008
) -> Response:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    request.app.state.registry.unregister(slug)
    if body.delete_files:
        shutil.rmtree(entry.folder, ignore_errors=True)
    return Response(status_code=204)


@router.delete("/{slug}/output/rejected")
async def delete_rejected_frames(project: Project = Depends(deps.get_project)) -> dict:  # noqa: B008
    """Delete every file in the project's rejected/ folder. Idempotent."""
    rejected = project.rejected_dir
    deleted = 0
    if rejected.exists():
        for entry in list(rejected.iterdir()):
            if entry.is_file():
                try:
                    entry.unlink()
                    deleted += 1
                except OSError:
                    pass
    return {"deleted": deleted}
