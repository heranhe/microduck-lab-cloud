export type CloudProvider = "colab" | "hf";
export type ComputeSource = "local" | CloudProvider;

export type CloudAccount = {
  installed?: boolean;
  connected: boolean;
  /** Why it is not connected; re-authorizing cannot fix a network outage. */
  reason?: "ok" | "not_installed" | "auth_expired" | "network" | "unknown";
  balance?: number | null;
  rate?: number | null;
  /** Runtimes Google is billing this account for, whoever started them. */
  active_assignments?: number | null;
  username?: string;
  email?: string | null;
  message?: string;
};

export type CloudJobProgress = {
  iteration?: number; iterations?: number; reward?: string; iter_time?: string;
  elapsed?: string; eta?: string; log_age?: number;
  gpu_util?: number; gpu_mem_used?: number; gpu_mem_total?: number;
};

export type CloudJob = {
  id: string; platform: CloudProvider; task: string; gpu: string; device?: string; vram?: string;
  flavor?: string; state: string; remote_state?: string; started?: number; allocated_at?: number | null;
  released?: boolean; released_at?: number; message?: string; url?: string;
  progress?: CloudJobProgress; last_heartbeat_at?: number; poll_failures?: number;
  max_minutes?: number; max_ccu?: number | null; ccu_spent?: number; rate?: number;
  artifacts?: string[]; stop_reason?: string;
};

export type ColabServerView = {
  ok: boolean | null; checked_at: number | null; count: number | null;
  orphans: { endpoint: string; hardware: string; name: string }[];
  others: { endpoint: string; hardware: string; name: string }[];
};

export const CLOUD_ACTIVE_STATES = new Set(["allocating", "starting", "running", "saving", "finalizing", "detached", "stopping"]);

/** Anything Google may still be billing: "released" is the only all-clear. */
export function needsRelease(job: Pick<CloudJob, "state" | "released">): boolean {
  return job.released !== true && (CLOUD_ACTIVE_STATES.has(job.state) || job.released === false);
}

type Phase = { en: string; zh: string };
const PHASES: Record<string, Phase> = {
  allocating: { en: "Allocating GPU", zh: "正在分配 GPU" },
  starting: { en: "Starting", zh: "正在启动" },
  saving: { en: "Saving progress", zh: "正在保存进度" },
  finalizing: { en: "Downloading results", zh: "下载结果中" },
  stopping: { en: "Disconnecting", zh: "正在断开" },
  stopped: { en: "Stopped", zh: "已停止" },
  done: { en: "Completed", zh: "已完成" },
  failed: { en: "Failed", zh: "失败" },
  detached: { en: "Needs attention", zh: "需要检查" },
};
const REMOTE_PHASES: Record<string, Phase> = {
  setup: { en: "Installing dependencies", zh: "正在安装依赖" },
  missing: { en: "Starting", zh: "正在启动" },
  training: { en: "Training", zh: "训练中" },
  saving: { en: "Saving progress", zh: "正在保存进度" },
  exporting: { en: "Exporting model", zh: "正在导出模型" },
  rescued: { en: "Saving progress", zh: "正在保存进度" },
  done: { en: "Downloading results", zh: "下载结果中" },
  failed: { en: "Collecting logs", zh: "正在取回日志" },
};

/**
 * What is really happening. `state` stays "running" from launch to download, so
 * showing it directly says "Training" during four paid minutes of `uv sync`.
 */
export function cloudPhaseText(job: Pick<CloudJob, "state" | "remote_state">, locale: "en" | "zh"): string {
  const phase = job.state === "running"
    ? REMOTE_PHASES[job.remote_state ?? "missing"] ?? REMOTE_PHASES.missing
    : PHASES[job.state];
  return phase ? phase[locale] : job.state;
}

/** Null when there is no real number yet; never a made-up 0%. */
export function cloudProgressPercent(job: Pick<CloudJob, "progress">): number | null {
  const iteration = job.progress?.iteration;
  const total = job.progress?.iterations;
  if (iteration == null || !total) return null;
  return Math.min(100, 100 * (iteration + 1) / total);
}

/** Seconds until the job's time limit; null before the runtime was assigned. */
export function cloudBudgetLeft(job: Pick<CloudJob, "allocated_at" | "max_minutes">, nowMs: number): number | null {
  if (!job.allocated_at || !job.max_minutes) return null;
  return Math.max(0, job.allocated_at + job.max_minutes * 60 - nowMs / 1000);
}

/** Saving needs a live worker and a trainer that has something to save. */
export function canSaveAndDisconnect(job: Pick<CloudJob, "platform" | "state" | "remote_state" | "released">): boolean {
  return job.platform === "colab" && job.released !== true && job.state === "running" && job.remote_state === "training";
}

/** The last answer from the runtime is old enough to stop trusting the numbers. */
export function cloudHeartbeatStale(job: Pick<CloudJob, "state" | "last_heartbeat_at">, nowMs: number, limitS = 120): boolean {
  return job.state === "running" && !!job.last_heartbeat_at && nowMs / 1000 - job.last_heartbeat_at > limitS;
}

export type CloudHardware = {
  id: string;
  label: string;
  gpu: string;
  quantity: string;
  vram: string;
  cost: number;
  unit: string;
};

export type HfCloudAccount = CloudAccount & { hardware: CloudHardware[] };

/**
 * Local teaching recipes and the official microduck_rl tasks are different
 * training stacks. Keep this mapping deliberately small: a cloud launch is
 * offered only when the explicitly selected official task has a registered upstream implementation.
 */
export function cloudTaskFor(robotId: string, behaviorId?: string): string | null {
  if (robotId !== "microduck") return null;
  if (behaviorId === "official_velstand") return "Mjlab-VelStand-Flat-MicroDuck";
  if (behaviorId === "official_velocity") return "Mjlab-Velocity-Flat-MicroDuck";
  return null;
}
