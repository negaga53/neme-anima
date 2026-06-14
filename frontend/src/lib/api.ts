import type {
  CharacterCopyReport, CharacterView, ConvertMode, ConvertStatus,
  CoreTagsReport, FrameRecord, FramesPage, LLMConfig,
  ProjectListEntry, ProjectView, QueueItem, Segment,
  TagReview, TagReviewItem, TrainingConfig, TrainingConfigResponse,
  TrainingStatus, TrainingRun, TrainingCheckpoint, TrainingDatasetPreview,
  TrainingPathCheck, TrainingLogResponse, TrainingTomlPreview, WipePreview,
} from "./types";

/** Server-side sentinel for the Frames-tab "unsorted" filter. Mirrors the
 *  Python constant in src/neme_anima/server/api/frames.py. Kept in sync
 *  manually since changing it breaks the wire protocol either way. */
export const UNSORTED_FILTER_SENTINEL = "__unsorted__";

/** Suffix of a frame's crop derivative on disk (`<name>_crop.png`). Mirrors
 *  ``CROP_SUFFIX`` in src/neme_anima/storage/project.py. The grid appends this
 *  to a frame's filename to request the cropped pixels in place of the
 *  original when ``FrameRecord.has_crop`` is set. */
export const CROP_SUFFIX = "_crop";

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`API ${status}: ${JSON.stringify(detail)}`);
  }
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const headers = { "Content-Type": "application/json", ...(init.headers ?? {}) };
  const resp = await fetch(url, { ...init, method: init.method ?? "GET", headers });
  if (!resp.ok) {
    let detail: unknown = null;
    try { detail = await resp.json(); } catch { /* body not JSON */ }
    throw new ApiError(resp.status, detail);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

// ---- projects ----

export const listProjects = () => request<ProjectListEntry[]>("/api/projects");

export const createProject = (body: { name: string; folder: string }) =>
  request<ProjectView>("/api/projects", { method: "POST", body: JSON.stringify(body) });

export const registerProject = (folder: string) =>
  request<ProjectView>("/api/projects/register", {
    method: "POST", body: JSON.stringify({ folder }),
  });

export const getProject = (slug: string) =>
  request<ProjectView>(`/api/projects/${encodeURIComponent(slug)}`);

export const patchProject = (
  slug: string,
  body: {
    name?: string;
    thresholds_overrides?: Record<string, Record<string, unknown>>;
    pause_before_tag?: boolean;
    auto_delete_rejected?: boolean;
    tag_autocomplete?: boolean;
    llm?: Partial<LLMConfig>;
  },
) => request<ProjectView>(`/api/projects/${encodeURIComponent(slug)}`, {
  method: "PATCH", body: JSON.stringify(body),
});

export const deleteProject = (slug: string, deleteFiles: boolean) =>
  request<void>(`/api/projects/${encodeURIComponent(slug)}`, {
    method: "DELETE", body: JSON.stringify({ delete_files: deleteFiles }),
  });

export const deleteRejectedFrames = (slug: string) =>
  request<{ deleted: number }>(
    `/api/projects/${encodeURIComponent(slug)}/output/rejected`,
    { method: "DELETE" },
  );

export const getVersion = () => request<{ version: string }>("/api/version");

// ---- sources ----

export const addSources = (slug: string, paths: string[]) =>
  request<{ added: string[]; skipped: string[] }>(
    `/api/projects/${encodeURIComponent(slug)}/sources`,
    { method: "POST", body: JSON.stringify({ paths }) },
  );

export const importSourcesFolder = (slug: string, folder: string) =>
  request<{ added: string[]; skipped: string[]; source_root: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/import-folder`,
    { method: "POST", body: JSON.stringify({ folder }) },
  );

export const reimportSources = (slug: string) =>
  request<{ added: string[]; skipped: string[]; source_root: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/reimport`,
    { method: "POST" },
  );

export const sourceThumbnailUrl = (slug: string, idx: number) =>
  `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/thumbnail`;

/** URL of the raw video file for the segment-editor's <video> element.
 *  The browser will refuse to decode unsupported codecs (HEVC, AV1 in
 *  containers it can't play) — the modal listens for `error` events and
 *  falls back to {@link sourcePreviewUrl} when that happens. */
export const sourceStreamUrl = (slug: string, idx: number) =>
  `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/stream`;

/** URL of a converted, web-playable MP4 for the given mode. The file is
 *  produced by {@link convertSource}; this URL only serves it (404 until
 *  conversion has run). `remux` is a lossless HEVC rewrap, `h264` a 480p
 *  transcode. */
export const sourcePreviewUrl = (slug: string, idx: number, mode: ConvertMode) =>
  `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/preview?mode=${mode}`;

/** Kick off a playback conversion. Returns the initial job state; poll
 *  {@link getConvertStatus} for progress. */
export const convertSource = (slug: string, idx: number, mode: ConvertMode) =>
  request<ConvertStatus>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/convert?mode=${mode}`,
    { method: "POST" },
  );

export const getConvertStatus = (slug: string, idx: number, mode: ConvertMode) =>
  request<ConvertStatus>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/convert/status?mode=${mode}`,
  );

/** Delete the cached converted preview(s) for a source. Omit `mode` to remove
 *  both the remux and h264 caches. */
export const deleteSourcePreview = (slug: string, idx: number, mode?: ConvertMode) =>
  request<{ removed: ConvertMode[] }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/preview`
      + (mode ? `?mode=${mode}` : ""),
    { method: "DELETE" },
  );

export const getSourceDuration = (slug: string, idx: number) =>
  request<{ duration_seconds: number; fps: number; vcodec: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/duration`,
  );

export const saveSourceSegments = (
  slug: string, idx: number, segments: Segment[],
) => request<{ segments: Segment[] }>(
  `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/segments`,
  { method: "PUT", body: JSON.stringify({ segments }) },
);

/** Grab the exact frame from the source video and register it as a kept frame.
 *  When ``frame_idx`` is present the server extracts that decoded frame number;
 *  otherwise it falls back to ``time_seconds``. The frame is WD14-tagged, plus
 *  an LLM caption when enabled, and routed to ``character_slug`` (default: the
 *  project's first character). It is read from the original file at full
 *  resolution, so it stays crisp even when the player uses a 480p preview. */
export const captureSourceFrame = (
  slug: string,
  idx: number,
  body: { time_seconds: number; frame_idx?: number; character_slug?: string },
) => request<{ frame: FrameRecord; llm_error: string | null }>(
  `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/capture-frame`,
  { method: "POST", body: JSON.stringify(body) },
);

export const removeSource = (slug: string, idx: number) =>
  request<void>(`/api/projects/${encodeURIComponent(slug)}/sources/${idx}`, { method: "DELETE" });

export const setExcludedRefs = (
  slug: string,
  idx: number,
  excluded: string[],
  characterSlug?: string,
) => {
  const url = characterSlug
    ? `/api/projects/${encodeURIComponent(slug)}/sources/${idx}` +
      `?character_slug=${encodeURIComponent(characterSlug)}`
    : `/api/projects/${encodeURIComponent(slug)}/sources/${idx}`;
  return request<{ excluded_refs: Record<string, string[]> }>(url, {
    method: "PATCH",
    body: JSON.stringify({ excluded_refs: excluded }),
  });
};

export const extractSource = (slug: string, idx: number) =>
  request<{ job_id: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/extract`,
    { method: "POST" },
  );

export const rerunSource = (slug: string, idx: number) =>
  request<{ job_id: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/rerun`,
    { method: "POST" },
  );

export const sourceWipePreview = (slug: string, idx: number) =>
  request<WipePreview>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/wipe-preview`,
  );

// ---- refs ----

export const addRefs = (slug: string, paths: string[], characterSlug?: string) => {
  const url = characterSlug
    ? `/api/projects/${encodeURIComponent(slug)}/refs` +
      `?character_slug=${encodeURIComponent(characterSlug)}`
    : `/api/projects/${encodeURIComponent(slug)}/refs`;
  return request<{ added: string[]; skipped: string[] }>(url, {
    method: "POST",
    body: JSON.stringify({ paths }),
  });
};

export const refImageUrl = (slug: string, refPath: string): string => {
  const name = refPath.split("/").pop() ?? refPath;
  return `/api/projects/${encodeURIComponent(slug)}/refs/${encodeURIComponent(name)}/image`;
};

export const uploadRefs = async (
  slug: string, files: File[], characterSlug?: string,
) => {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const url = characterSlug
    ? `/api/projects/${encodeURIComponent(slug)}/refs/upload` +
      `?character_slug=${encodeURIComponent(characterSlug)}`
    : `/api/projects/${encodeURIComponent(slug)}/refs/upload`;
  const resp = await fetch(url, { method: "POST", body: fd });
  if (!resp.ok) {
    let detail: unknown = null;
    try { detail = await resp.json(); } catch { /* body not JSON */ }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<{ added: string[]; skipped: string[] }>;
};

// ---- characters ----

export const listCharacters = (slug: string) =>
  request<CharacterView[]>(
    `/api/projects/${encodeURIComponent(slug)}/characters`,
  );

export const createCharacter = (
  slug: string, body: { name: string; slug?: string },
) =>
  request<CharacterView>(
    `/api/projects/${encodeURIComponent(slug)}/characters`,
    { method: "POST", body: JSON.stringify(body) },
  );

export const updateCharacter = (
  slug: string,
  characterSlug: string,
  body: {
    name?: string;
    trigger_token?: string;
    core_tags?: string[];
    core_tags_freq_threshold?: number;
    core_tags_enabled?: boolean;
    multiply?: number;
  },
) =>
  request<CharacterView>(
    `/api/projects/${encodeURIComponent(slug)}/characters/${encodeURIComponent(characterSlug)}`,
    { method: "PATCH", body: JSON.stringify(body) },
  );

export const computeCharacterCoreTags = (
  slug: string,
  characterSlug: string,
  body: { threshold?: number; blacklist?: string[] } = {},
) =>
  request<CoreTagsReport>(
    `/api/projects/${encodeURIComponent(slug)}/characters/${encodeURIComponent(characterSlug)}/core-tags/compute`,
    { method: "POST", body: JSON.stringify(body) },
  );

export const copyCharacterToProject = (
  slug: string,
  characterSlug: string,
  body: { destination_slug: string; dry_run?: boolean },
) =>
  request<CharacterCopyReport>(
    `/api/projects/${encodeURIComponent(slug)}/characters/${encodeURIComponent(characterSlug)}/copy-to`,
    { method: "POST", body: JSON.stringify(body) },
  );

export const deleteCharacter = (slug: string, characterSlug: string) =>
  request<void>(
    `/api/projects/${encodeURIComponent(slug)}/characters/${encodeURIComponent(characterSlug)}`,
    { method: "DELETE" },
  );

export const removeRef = (slug: string, path: string) =>
  request<void>(`/api/projects/${encodeURIComponent(slug)}/refs`, {
    method: "DELETE", body: JSON.stringify({ path }),
  });

// ---- frames ----

export const listFrames = (
  slug: string,
  opts: {
    source?: string;
    /** Whitespace-separated tag substrings; tokens prefixed with `~` negate.
     *  Server-side filter so large grids stay paginated by what matched. */
    query?: string;
    /** Filter by character; pass UNSORTED_FILTER_SENTINEL for orphan rows
     *  whose slug isn't in the project's current character set. */
    characterSlug?: string;
    offset?: number;
    limit?: number;
  } = {},
) => {
  const q = new URLSearchParams();
  if (opts.source) q.set("source", opts.source);
  if (opts.query) q.set("query", opts.query);
  if (opts.characterSlug) q.set("character_slug", opts.characterSlug);
  if (opts.offset !== undefined) q.set("offset", String(opts.offset));
  if (opts.limit !== undefined) q.set("limit", String(opts.limit));
  const qs = q.toString();
  return request<FramesPage>(
    `/api/projects/${encodeURIComponent(slug)}/frames${qs ? `?${qs}` : ""}`,
  );
};

export const moveFrameToCharacter = (
  slug: string, filename: string, characterSlug: string,
) =>
  request<FrameRecord>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/character`,
    { method: "POST", body: JSON.stringify({ character_slug: characterSlug }) },
  );

export const bulkMoveFrames = (
  slug: string, filenames: string[], characterSlug: string,
) =>
  request<{ moved: number; missing: string[] }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/bulk-move`,
    {
      method: "POST",
      body: JSON.stringify({ filenames, character_slug: characterSlug }),
    },
  );

export const duplicateFrameForCharacter = (
  slug: string, filename: string, characterSlug: string,
) =>
  request<FrameRecord>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/duplicate`,
    { method: "POST", body: JSON.stringify({ character_slug: characterSlug }) },
  );

export const bulkDuplicateFrames = (
  slug: string, filenames: string[], characterSlug: string,
) =>
  request<{ duplicated: string[]; missing: string[] }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/bulk-duplicate`,
    {
      method: "POST",
      body: JSON.stringify({ filenames, character_slug: characterSlug }),
    },
  );

export const getTags = (slug: string, filename: string) =>
  request<{ text: string }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/tags`,
  );

export const putTags = (slug: string, filename: string, text: string) =>
  request<{ text: string }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/tags`,
    { method: "PUT", body: JSON.stringify({ text }) },
  );

export const getDescription = (slug: string, filename: string) =>
  request<{ text: string }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/description`,
  );

export const putDescription = (slug: string, filename: string, text: string) =>
  request<{ text: string }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/description`,
    { method: "PUT", body: JSON.stringify({ text }) },
  );

export const deleteFrame = (slug: string, filename: string) =>
  request<void>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );

export const bulkDeleteFrames = (slug: string, filenames: string[]) =>
  request<{ deleted: number }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/bulk-delete`,
    { method: "POST", body: JSON.stringify({ filenames }) },
  );

export const bulkTagsReplace = (
  slug: string,
  body: { filenames: string[]; pattern: string; replacement: string; case_insensitive?: boolean },
) => request<{ changed: number }>(
  `/api/projects/${encodeURIComponent(slug)}/frames/bulk-tags-replace`,
  { method: "POST", body: JSON.stringify(body) },
);

export const bulkRetagDanbooru = (slug: string, filenames: string[]) =>
  request<{
    retagged: number;
    total: number;
    /** Per-item failures: {filename, reason}. Empty when all succeeded. */
    skipped: { filename: string; reason: string }[];
    /** Parallel to the request `filenames`. Entry differs from the input
     *  when a `_crop` derivative was tagged in place of the original. */
    effective_filenames: (string | null)[];
  }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/bulk-retag-danbooru`,
    { method: "POST", body: JSON.stringify({ filenames }) },
  );

export const bulkRetagLLM = (slug: string, filenames: string[]) =>
  request<{
    described: number;
    total: number;
    error: string | null;
    /** Per-item failures: {filename, reason}. Empty when all succeeded. */
    skipped: { filename: string; reason: string }[];
    /** Parallel to the request `filenames`. Entry is null when that
     *  filename failed; differs from the input when a `_crop` derivative
     *  was described in place of the original. */
    effective_filenames: (string | null)[];
  }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/bulk-retag-llm`,
    { method: "POST", body: JSON.stringify({ filenames }) },
  );

/** Ask the LLM to review one frame's tags against its (cropped) image. */
export const reviewTags = (slug: string, filename: string) =>
  request<TagReview>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/review-tags`,
    { method: "POST" },
  );

// ---- llm ----

export const discoverLLMModels = (endpoint: string, apiKey: string = "") =>
  request<{ models: string[] }>(`/api/llm/discover-models`, {
    method: "POST",
    body: JSON.stringify({ endpoint, api_key: apiKey }),
  });

export const frameImageUrl = (slug: string, filename: string) =>
  `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/image`;

export const cropFrame = (
  slug: string, filename: string,
  rect: { x: number; y: number; width: number; height: number },
) => request<FrameRecord>(
  `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/crop`,
  { method: "POST", body: JSON.stringify(rect) },
);

/** Returns the saved crop rectangle for a frame, or null if none has been
 *  confirmed yet (server returns 404). */
export const getCropRect = async (
  slug: string, filename: string,
): Promise<{ x: number; y: number; width: number; height: number } | null> => {
  try {
    return await request(
      `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/crop`,
    );
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
};

export const uploadFrames = async (
  slug: string, files: File[], characterSlug?: string,
) => {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const url = characterSlug
    ? `/api/projects/${encodeURIComponent(slug)}/frames/upload` +
      `?character_slug=${encodeURIComponent(characterSlug)}`
    : `/api/projects/${encodeURIComponent(slug)}/frames/upload`;
  const resp = await fetch(url, { method: "POST", body: fd });
  if (!resp.ok) {
    let detail: unknown = null;
    try { detail = await resp.json(); } catch { /* body not JSON */ }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<{
    added: FrameRecord[];
    skipped: string[];
    /** Set when LLM tagging is enabled but `describe_image` failed for at
     *  least one file — the upload itself succeeded, only line 2 is
     *  missing. UI surfaces this as a one-shot warning. */
    llm_error: string | null;
  }>;
};

// ---- queue ----

export const listQueue = () => request<QueueItem[]>("/api/queue");
export const cancelJob = (jobId: string) =>
  request<void>(`/api/queue/${encodeURIComponent(jobId)}`, { method: "DELETE" });
export const resumeJob = (jobId: string) =>
  request<void>(`/api/queue/${encodeURIComponent(jobId)}/resume`, { method: "POST" });

// ---- training ----

export const getTrainingConfig = (slug: string) =>
  request<TrainingConfigResponse>(
    `/api/projects/${encodeURIComponent(slug)}/training/config`,
  );

export const patchTrainingConfig = (
  slug: string, body: Partial<TrainingConfig>,
) => request<TrainingConfigResponse>(
  `/api/projects/${encodeURIComponent(slug)}/training/config`,
  { method: "PATCH", body: JSON.stringify(body) },
);

export const checkTrainingPath = (
  slug: string, path: string, expect: "any" | "file" | "dir" = "any",
) => request<TrainingPathCheck>(
  `/api/projects/${encodeURIComponent(slug)}/training/check-path`,
  { method: "POST", body: JSON.stringify({ path, expect }) },
);

export const getTrainingStatus = (slug: string) =>
  request<TrainingStatus>(
    `/api/projects/${encodeURIComponent(slug)}/training/status`,
  );

export const getTrainingLog = (slug: string, tail = 1000) =>
  request<TrainingLogResponse>(
    `/api/projects/${encodeURIComponent(slug)}/training/log?tail=${tail}`,
  );

export const startTraining = (
  slug: string,
  body: { resume_from_checkpoint?: string; run_dir_name?: string } = {},
) => request<TrainingStatus>(
  `/api/projects/${encodeURIComponent(slug)}/training/start`,
  { method: "POST", body: JSON.stringify(body) },
);

export const stopTraining = (slug: string) =>
  request<TrainingStatus>(
    `/api/projects/${encodeURIComponent(slug)}/training/stop`,
    { method: "POST" },
  );

export const resumeTraining = (
  slug: string,
  body: { resume_from_checkpoint?: string; run_dir_name?: string } = {},
) => request<TrainingStatus>(
  `/api/projects/${encodeURIComponent(slug)}/training/resume`,
  { method: "POST", body: JSON.stringify(body) },
);

export const listTrainingRuns = (slug: string) =>
  request<{ runs: TrainingRun[] }>(
    `/api/projects/${encodeURIComponent(slug)}/training/runs`,
  );

export const listTrainingCheckpoints = (slug: string, runName: string) =>
  request<{ run_name: string; run_dir: string; checkpoints: TrainingCheckpoint[] }>(
    `/api/projects/${encodeURIComponent(slug)}/training/runs/${encodeURIComponent(runName)}/checkpoints`,
  );

export const deleteTrainingCheckpoint = (
  slug: string, runName: string, ckptName: string, subdir: string = "",
) => {
  const url = `/api/projects/${encodeURIComponent(slug)}/training/runs/${encodeURIComponent(runName)}/checkpoints/${encodeURIComponent(ckptName)}`;
  const qs = subdir ? `?subdir=${encodeURIComponent(subdir)}` : "";
  return request<void>(url + qs, { method: "DELETE" });
};

export const exportCheckpointUrl = (
  slug: string, runName: string, ckptName: string, subdir: string = "",
) => {
  const url = `/api/projects/${encodeURIComponent(slug)}/training/runs/${encodeURIComponent(runName)}/checkpoints/${encodeURIComponent(ckptName)}/export`;
  return subdir ? `${url}?subdir=${encodeURIComponent(subdir)}` : url;
};

export const deleteTrainingRun = (slug: string, runName: string) =>
  request<void>(
    `/api/projects/${encodeURIComponent(slug)}/training/runs/${encodeURIComponent(runName)}`,
    { method: "DELETE" },
  );

export const getTrainingDatasetPreview = (slug: string) =>
  request<TrainingDatasetPreview>(
    `/api/projects/${encodeURIComponent(slug)}/training/dataset-preview`,
  );

export const getTrainingTomlPreview = (slug: string) =>
  request<TrainingTomlPreview>(
    `/api/projects/${encodeURIComponent(slug)}/training/run-toml-preview`,
  );
