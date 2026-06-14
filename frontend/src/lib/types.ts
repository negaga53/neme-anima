// Mirrors the Pydantic models in src/neme_anima/server/api/*.py.

/** A time-range the user wants the extraction pipeline to restrict itself
 *  to. Stored in seconds (not frames) so it stays meaningful if the source
 *  file is re-encoded at a different framerate. Empty list on a Source =
 *  legacy "process the whole video" behaviour. */
export interface Segment {
  start_seconds: number;
  end_seconds: number;
}

export interface Source {
  path: string;
  added_at: string;
  /** Per-character opt-out map: ``{character_slug: [ref_path, ...]}``.
   *  Replaces the pre-multi-character flat list. Old projects auto-migrate
   *  to ``{default: [...]}`` server-side, so the frontend only ever sees
   *  the dict shape. */
  excluded_refs: Record<string, string[]>;
  /** True when the project has at least one kept frame on disk for this
   *  video stem — survives server restarts. */
  extracted: boolean;
  /** Detection-cache freshness for this source's video. Drives the smart
   *  Extract / Re-process button states.
   *   - "none": no cache yet — Extract is the only path forward.
   *   - "current": cache is fresh and scan-affecting (scene/detect/track)
   *     thresholds haven't changed since it was stamped — Re-process is
   *     enough; Extract is disabled to steer the user away from a wasted
   *     re-scan.
   *   - "stale": cache exists but at least one scan-affecting threshold
   *     differs — Re-process would silently use stale detections, so the
   *     UI flags Extract with a warning. */
  extraction_cache: "none" | "current" | "stale";
  /** Per-source time-range restrictions. Empty list = process the whole
   *  video. Editing the list flips ``extraction_cache`` to ``"stale"``
   *  on the next project load. */
  segments: Segment[];
  /** ffprobe'd video duration, cached on the source after the first
   *  segment-editor open so we don't repay the probe cost. ``null`` until
   *  the modal has been opened at least once. */
  duration_seconds: number | null;
  /** Average framerate from ffprobe, cached alongside ``duration_seconds``. */
  fps: number | null;
}

export interface RefImage {
  path: string;
  added_at: string;
}

export interface CharacterView {
  slug: string;
  name: string;
  trigger_token: string;
  refs: RefImage[];
  ref_count: number;
  core_tags: string[];
  core_tags_freq_threshold: number;
  core_tags_enabled: boolean;
  multiply: number;
}

export interface LLMConfig {
  enabled: boolean;
  endpoint: string;
  model: string;
  prompt: string;
  /** Bearer token for providers that gate /v1/* (OpenAI, OpenRouter, hosted
   *  vLLM, …). Empty for LMStudio and other unauthenticated local servers. */
  api_key: string;
}

export interface ProjectView {
  slug: string;
  name: string;
  folder: string;
  created_at: string;
  sources: Source[];
  /** Backwards-compat alias for ``characters[0].refs`` — kept on the wire
   *  while older code paths still read it. New code should use
   *  ``characters`` directly. */
  refs: RefImage[];
  characters: CharacterView[];
  thresholds_overrides: Record<string, Record<string, unknown>>;
  source_root: string | null;
  pause_before_tag: boolean;
  auto_delete_rejected: boolean;
  tag_autocomplete: boolean;
  rejected_count: number;
  llm: LLMConfig;
}

export interface ProjectListEntry {
  slug: string;
  name: string;
  folder: string;
  missing: boolean;
  source_count: number;
  ref_count: number;
  last_opened_at: string;
}

export interface FrameRecord {
  filename: string;
  kept: boolean;
  video_stem: string;
  scene_idx: number;
  tracklet_id: number;
  frame_idx: number;
  timestamp_seconds: number;
  ccip_distance: number;
  score: number;
  /** Slug of the character the frame is currently routed to. Defaults to
   *  ``"default"`` for legacy single-character extractions. */
  character_slug: string;
  /** True when the .txt sidecar has a non-empty second line (an LLM
   *  description). Drives the at-a-glance "described" badge in the grid. */
  has_description: boolean;
  /** True when the .txt sidecar has a non-empty first line of danbooru tags.
   *  Used to warn before bulk re-tagging overwrites curated tag lines. */
  has_tags: boolean;
  /** True when a crop derivative (`<name>_crop.png`) exists on disk. The grid
   *  shows the cropped pixels in its place; selection/sidecar stay on the
   *  original filename. */
  has_crop: boolean;
}

export interface FramesPage {
  /** Number of items matching the current filter (source + tag query). */
  count: number;
  /** Count in the current source/kept_only view before the tag query is
   *  applied — drives the "X / total" badge when a search is active. */
  total: number;
  /** On-disk kept frames in the current source view whose slug isn't a
   *  current character — independent of the active character filter. Drives
   *  whether the top bar shows the "Unsorted" chip. */
  unsorted_total: number;
  items: FrameRecord[];
}

export interface QueueItem {
  job_id: string;
  status: "pending" | "running" | "done" | "cancelled" | "failed";
  payload: Record<string, unknown>;
  error: string | null;
}

export type EventType =
  | "queue.update"
  | "job.progress"
  | "job.stages"
  | "job.frame"
  | "job.log"
  | "job.done";

export type StageStatus = "pending" | "running" | "done" | "failed";

export interface PipelineStage {
  key: string;
  label: string;
  status: StageStatus;
  current: number;
  total: number;
  pct: number;
  message: string;
}

export interface JobStages {
  job_id: string;
  project: string;
  source_idx: number | null;
  kind: "extract" | "rerun" | string;
  stages: PipelineStage[];
  summary: { kept?: number; rejected?: number } | null;
  updated_at: number;
  /** True when the runner is parked at wait_for_resume() and a click on the
   *  yellow pause indicator will release it. */
  paused?: boolean;
  pause_message?: string;
}

export interface ServerEvent {
  type: EventType | "training.status" | "training.log";
  payload: Record<string, unknown>;
}

// ---- training ----

export interface TrainingConfig {
  preset: "style" | "character" | string;
  diffusion_pipe_dir: string;
  dit_path: string;
  vae_path: string;
  llm_path: string;
  launcher_override: string;

  rank: number;
  alpha: number;

  learning_rate: number;
  optimizer_betas: number[];
  weight_decay: number;
  eps: number;
  warmup_steps: number;
  gradient_clipping: number;

  micro_batch_size: number;
  gradient_accumulation_steps: number;

  transformer_dtype: "bfloat16" | "float8" | string;
  blocks_to_swap: number;
  optimizer_type: "adamw_optimi" | "AdamW8bitKahan" | string;
  activation_checkpointing_mode: "default" | "unsloth" | string;

  resolutions: number[];
  enable_ar_bucket: boolean;
  min_ar: number;
  max_ar: number;
  num_ar_buckets: number;

  epochs: number;
  eval_every_n_epochs: number;
  save_every_n_epochs: number;

  sigmoid_scale: number;
  llm_adapter_lr: number;

  caption_mode: "tags" | "nl" | "mixed" | string;
  tag_dropout_pct: number;
  trigger_token: string;

  keep_last_n_checkpoints: number;
}

export interface TrainingPathCheck {
  path: string;
  exists: boolean;
  is_file: boolean;
  is_dir: boolean;
  error: string | null;
}

export interface TrainingConfigResponse {
  config: TrainingConfig;
  path_checks: {
    diffusion_pipe_dir: TrainingPathCheck;
    dit_path: TrainingPathCheck;
    vae_path: TrainingPathCheck;
    llm_path: TrainingPathCheck;
  };
  problems: string[];
}

export interface TrainingRunState {
  project_slug: string;
  run_dir: string;
  run_name: string;
  status:
    | "starting" | "running" | "stopping" | "stopped" | "finished" | "failed";
  started_at: string;
  finished_at: string | null;
  pid: number | null;
  exit_code: number | null;
  error: string | null;
  epoch: number | null;
  step: number | null;
  loss: number | null;
  last_log_line: string;
  resumed_from: string | null;
  stop_requested: boolean;
  total_epochs: number | null;
}

export interface TrainingLogLine {
  t: number;
  stream: "stdout" | "stderr" | "disk" | string;
  line: string;
}

export interface TrainingStatus {
  slug: string;
  running: boolean;
  global_active_slug: string | null;
  state: TrainingRunState | null;
  log_lines: TrainingLogLine[];
}

export interface TrainingCheckpoint {
  name: string;
  path: string;
  epoch: number | null;
  step: number | null;
  size_bytes: number;
  modified_at: string;
  /** Path of the parent directory relative to the run dir, "" if direct. */
  subdir: string;
}

export interface TrainingRun {
  name: string;
  path: string;
  /** Count of epoch-style LoRA outputs (excludes DeepSpeed plumbing). */
  checkpoints: number;
  latest_checkpoint: string | null;
  /** Highest epoch saved in this run, if any. */
  latest_epoch: number | null;
  /** Diffusion-pipe sub-run-dir name we can resume from, if any. */
  resumable_subdir: string | null;
  total_size_bytes: number;
  modified_at: string;
}

export interface TrainingDatasetSample {
  filename: string;
  tags: string;
  nl: string;
  rendered: string;
}

export interface TrainingDatasetPreview {
  total_images: number;
  with_tags: number;
  with_descriptions: number;
  samples: TrainingDatasetSample[];
  kept_dir: string;
}

export interface TrainingLogResponse {
  source: "live" | "disk" | "none";
  run_name?: string;
  lines: TrainingLogLine[];
}

export interface TrainingTomlPreview {
  dataset_toml: string;
  run_toml: string;
  launcher_argv: string[];
}

export type ConvertMode = "remux" | "h264";

export interface ConvertStatus {
  state: "idle" | "running" | "ready" | "failed";
  pct: number;
  mode: ConvertMode;
  error: string;
}

export interface WipePreview {
  video_stem: string;
  active_slugs: string[];
  preserve_slugs: string[];
  to_wipe: {
    by_character: Record<string, number>;
    rejected_samples: number;
    total: number;
  };
  to_preserve: {
    by_character: Record<string, number>;
    total: number;
  };
}

export type CoreTagsReport = {
  character_slug: string;
  corpus_size: number;
  threshold: number;
  tags: { tag: string; freq: number }[];
  blacklisted: string[];
};

export type CharacterCopyReport = {
  character_slug: string;
  sources_added: string[];
  sources_skipped: string[];
  refs_added: string[];
  refs_renamed: Record<string, string>;
  frames_added: string[];
  frames_skipped: string[];
  custom_uploads_added: number;
  crops_copied: number;
  metadata_rows_appended: number;
  dry_run: boolean;
};

/** A {tag, reason} suggestion from the LLM tag-review pass. */
export interface TagReviewItem {
  tag: string;
  reason: string;
}

/** The reconciled, applyable diff the review endpoint returns. */
export interface TagReview {
  /** Existing tags retained (existing minus accepted removals). */
  keep: string[];
  /** Existing tags the model flagged for removal, with reasons. */
  remove: TagReviewItem[];
  /** New, canonicalized danbooru tags to add, with reasons. */
  add: TagReviewItem[];
  /** keep + add — the list if every suggestion is accepted. */
  proposed_final: string[];
  /** Human-readable notes on what reconciliation dropped/normalized. */
  notes: string[];
  /** The filename the review actually targeted (the crop, when one exists). */
  effective_filename: string;
}
