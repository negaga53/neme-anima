import * as api from "$lib/api";
import type {
  ServerEvent, TrainingConfig, TrainingConfigResponse, TrainingLogLine,
  TrainingRun, TrainingStatus, TrainingDatasetPreview, TrainingPathCheck,
} from "$lib/types";

// Cap the live log buffer client-side; the server keeps a longer history
// on disk and the user can hit "Reload log" if they need older lines.
const LOG_MAX = 2000;

class TrainingStore {
  configResp = $state<TrainingConfigResponse | null>(null);
  status = $state<TrainingStatus | null>(null);
  runs = $state<TrainingRun[]>([]);
  preview = $state<TrainingDatasetPreview | null>(null);
  log = $state<TrainingLogLine[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  /** Bumped when new sample images land (or a run ends) so open sample grids refetch. */
  samplesVersion = $state(0);

  // Slug we're currently subscribed to. The Training tab calls
  // ``setProject`` on mount and on project switch so we know which slug
  // owns incoming WebSocket events.
  slug = $state<string | null>(null);

  async setProject(slug: string) {
    this.slug = slug;
    this.error = null;
    this.loading = true;
    try {
      const [cfg, st, runs, pv, log] = await Promise.all([
        api.getTrainingConfig(slug),
        api.getTrainingStatus(slug),
        api.listTrainingRuns(slug),
        api.getTrainingDatasetPreview(slug),
        api.getTrainingLog(slug, 1000),
      ]);
      this.configResp = cfg;
      this.status = st;
      this.runs = runs.runs;
      this.preview = pv;
      this.log = log.lines.slice(-LOG_MAX);
    } catch (e) {
      this.error = String(e);
    } finally {
      this.loading = false;
    }
  }

  ingest(ev: ServerEvent) {
    if (!this.slug) return;
    if (ev.type === "training.status") {
      const slug = ev.payload.slug as string | undefined;
      if (!slug || slug !== this.slug) return;
      this.status = {
        slug,
        running: !!ev.payload.running,
        global_active_slug: this.status?.global_active_slug ?? null,
        state: (ev.payload.state ?? null) as TrainingStatus["state"],
        log_lines: this.status?.log_lines ?? [],
      };
      // Reload the run list opportunistically when a run finishes — new
      // checkpoints may have been written.
      const finalStatus = (ev.payload.state as { status?: string })?.status;
      if (finalStatus && (finalStatus === "finished" || finalStatus === "failed" || finalStatus === "stopped")) {
        this.refreshRuns();
        this.samplesVersion++;
      }
    } else if (ev.type === "training.log") {
      const slug = ev.payload.slug as string | undefined;
      if (!slug || slug !== this.slug) return;
      const line: TrainingLogLine = {
        t: (ev.payload.t as number) ?? Date.now() / 1000,
        stream: (ev.payload.stream as string) ?? "stdout",
        line: (ev.payload.line as string) ?? "",
      };
      // Mutate in-place via a fresh array to keep $state happy without
      // copying the entire log on every line.
      const next = this.log.length >= LOG_MAX
        ? this.log.slice(this.log.length - LOG_MAX + 1)
        : this.log.slice();
      next.push(line);
      this.log = next;
    } else if (ev.type === "training.sample") {
      const slug = ev.payload.slug as string | undefined;
      if (!slug || slug !== this.slug) return;
      this.samplesVersion++;
    }
  }

  async patch(body: Partial<TrainingConfig>) {
    if (!this.slug) return;
    this.configResp = await api.patchTrainingConfig(this.slug, body);
  }

  async checkPath(
    field: "diffusion_pipe_dir" | "dit_path" | "vae_path" | "llm_path",
    raw: string,
  ): Promise<TrainingPathCheck> {
    if (!this.slug) throw new Error("no project");
    const expect: "dir" | "file" = field === "diffusion_pipe_dir" ? "dir" : "file";
    return api.checkTrainingPath(this.slug, raw, expect);
  }

  async start(opts: { resume_from_checkpoint?: string; run_dir_name?: string } = {}) {
    if (!this.slug) return;
    this.status = await api.startTraining(this.slug, opts);
    await this.refreshRuns();
  }

  async stop() {
    if (!this.slug) return;
    this.status = await api.stopTraining(this.slug);
  }

  async resume(opts: { resume_from_checkpoint?: string; run_dir_name?: string } = {}) {
    if (!this.slug) return;
    this.status = await api.resumeTraining(this.slug, opts);
    await this.refreshRuns();
  }

  async refreshRuns() {
    if (!this.slug) return;
    const r = await api.listTrainingRuns(this.slug);
    this.runs = r.runs;
  }

  async refreshPreview() {
    if (!this.slug) return;
    this.preview = await api.getTrainingDatasetPreview(this.slug);
  }

  async refreshLog() {
    if (!this.slug) return;
    const log = await api.getTrainingLog(this.slug, 2000);
    this.log = log.lines.slice(-LOG_MAX);
  }

  async deleteCheckpoint(runName: string, ckptName: string, subdir: string = "") {
    if (!this.slug) return;
    await api.deleteTrainingCheckpoint(this.slug, runName, ckptName, subdir);
    await this.refreshRuns();
  }

  async deleteRun(runName: string) {
    if (!this.slug) return;
    await api.deleteTrainingRun(this.slug, runName);
    await this.refreshRuns();
  }
}

export const trainingStore = new TrainingStore();
