"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { LAB_HTTP } from "@/lib/lab";
import { useI18n, type Locale } from "@/lib/i18n";
import styles from "./cloud.module.css";

type Account = { installed: boolean; connected: boolean; balance?: number | null; rate?: number | null; message?: string };
type Job = {
  id: string;
  session?: string;
  task: string;
  gpu: string;
  state: string;
  remote_state?: string;
  started?: number;
  allocated_at?: number;
  message?: string;
  released?: boolean;
};

const ACTIVE_STATES = new Set(["allocating", "starting", "running", "finalizing", "detached", "stopping"]);

function needsRelease(job: Job): boolean {
  return ACTIVE_STATES.has(job.state) || job.released === false;
}

function runtimeStart(job: Job): number | undefined {
  // New jobs have an allocation timestamp. Older saved jobs can only use
  // the request time as a fallback; never count a pending allocation.
  return job.allocated_at ?? (job.state === "allocating" || job.state === "stopping" ? undefined : job.started);
}

function elapsed(started: number | undefined, now: number): string {
  if (!started || !now) return "00:00:00";
  const seconds = Math.max(0, Math.floor(now / 1000 - started));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds % 3600 / 60);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function stateText(state: string, locale: Locale): string {
  if (locale === "en") {
    return ({ allocating: "Allocating GPU", starting: "Starting", running: "Training", setup: "Setting up",
      training: "Training", exporting: "Exporting", finalizing: "Downloading results", detached: "Reconnected session",
      stopping: "Disconnecting", stopped: "Stopped", done: "Completed", failed: "Failed" } as Record<string, string>)[state] ?? state;
  }
  return ({ allocating: "正在分配 GPU", starting: "正在启动", running: "训练中", setup: "配置环境中",
    training: "训练中", exporting: "导出模型中", finalizing: "下载结果中", detached: "重启后待处理",
    stopping: "正在断开", stopped: "已停止", done: "已完成", failed: "失败" } as Record<string, string>)[state] ?? state;
}

function messageText(message: string, locale: Locale): string {
  if (locale === "en") return message;
  const fixed: Record<string, string> = {
    "Lab restarted; check and stop this Colab session.": "实验室已重启，请检查并断开此 Colab 会话。",
    "Training or ONNX export did not complete": "训练或 ONNX 导出未完成。",
    "Run `colab usage` in Terminal to connect your Google account.": "请在终端运行 `colab usage`，连接您的 Google 账号。",
    "Install google-colab-cli": "请先安装 Google Colab CLI。",
  };
  if (fixed[message]) return fixed[message];
  if (message.includes("No positive Colab compute-unit balance")) return "未检测到可用 CCU 余额，请检查订阅和登录账号。";
  if (message.includes("Connect your Google account")) return "请先在终端运行 `colab usage` 完成 Google 授权。";
  if (message.includes("Unsupported task or GPU")) return "不支持该训练任务或 GPU 型号。";
  if (message.includes("Training limits are out of range")) return "训练参数超出允许范围。";
  if (message.includes("Origin not allowed")) return "当前网页来源不被允许，请通过本机地址打开实验室。";
  if (message.includes("Unknown Colab job")) return "找不到该 Colab 任务。";
  if (message.includes("Colab CLI is not installed")) return "Colab CLI 尚未安装，请先运行 `uv sync`。";
  if (message.includes("Failed to fetch") || message.includes("NetworkError")) return "无法连接 duck-lab；若云端会话可能仍在运行，请在终端执行 `uv run colab sessions` 检查。";
  return message;
}

export default function CloudPage() {
  const { tr, locale, toggle } = useI18n();
  const [account, setAccount] = useState<Account | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [gpu, setGpu] = useState("T4");
  const [task, setTask] = useState("Mjlab-Velocity-Flat-MicroDuck");
  const [iterations, setIterations] = useState(1000);
  const [envs, setEnvs] = useState(64);
  const [now, setNow] = useState(0);
  const [online, setOnline] = useState(true);
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");
  const [starting, setStarting] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);
  const [stoppingId, setStoppingId] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [accountResponse, jobsResponse] = await Promise.all([
        fetch(`${LAB_HTTP}/cloud/colab/account`, { cache: "no-store" }),
        fetch(`${LAB_HTTP}/cloud/colab/jobs`, { cache: "no-store" }),
      ]);
      if (!accountResponse.ok || !jobsResponse.ok) throw new Error("Lab unavailable");
      const [nextAccount, nextJobs] = await Promise.all([accountResponse.json(), jobsResponse.json()]);
      setAccount(nextAccount);
      setJobs(nextJobs.jobs ?? []);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => {
    const initialTimer = window.setTimeout(() => { void refresh(); setNow(Date.now()); }, 0);
    const refreshTimer = window.setInterval(() => void refresh(), 15000);
    const clockTimer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => { window.clearTimeout(initialTimer); window.clearInterval(refreshTimer); window.clearInterval(clockTimer); };
  }, [refresh]);

  const request = async (url: string, body?: object) => {
    const response = await fetch(`${LAB_HTTP}${url}`, {
      method: "POST",
      ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { detail = (await response.json()).detail ?? detail; } catch { /* Keep HTTP status. */ }
      throw new Error(detail);
    }
    return response.json();
  };

  const start = async () => {
    setStarting(true); setActionError(""); setNotice("");
    try {
      await request("/cloud/colab/jobs", { task, gpu, iterations, envs });
      await refresh();
      setNotice(tr("Training request sent. The timer starts after GPU allocation succeeds.", "训练请求已提交；GPU 分配成功后开始计时。"));
    } catch (error) { setActionError(messageText(String(error), locale)); }
    finally { setStarting(false); }
  };

  const disconnectAll = async () => {
    setDisconnecting(true); setActionError(""); setNotice("");
    try {
      const result = await request("/cloud/colab/stop-all");
      await refresh();
      const pending = (result.stopped as Job[]).some(job => job.released !== true);
      setNotice(pending
        ? tr("Disconnect requested. Some sessions are still releasing; keep this page open and check the status.", "已提交断开请求；部分会话仍在释放，请留在此页查看状态。")
        : tr("All MicroDuck Colab sessions were released.", "本项目的 Colab 会话已全部释放。"));
    } catch (error) {
      setActionError(messageText(String(error), locale));
    } finally { setDisconnecting(false); }
  };

  const stopOne = async (id: string) => {
    setStoppingId(id); setActionError(""); setNotice("");
    try {
      const result = await request(`/cloud/colab/jobs/${id}/stop`) as Job;
      await refresh();
      if (result.released !== true) {
        setNotice(tr("Release is pending. Check the job status again shortly.", "正在释放会话，请稍后再次检查任务状态。"));
      }
    } catch (error) { setActionError(messageText(String(error), locale)); }
    finally { setStoppingId(""); }
  };

  const pendingJobs = jobs.filter(needsRelease);
  const started = pendingJobs.map(runtimeStart).filter((value): value is number => typeof value === "number");
  const firstStart = started.length ? Math.min(...started) : undefined;
  const sortedJobs = [...jobs].sort((a, b) => (b.started ?? 0) - (a.started ?? 0));
  const hasBalance = account?.connected && typeof account.balance === "number" && account.balance > 0;
  const hasReleaseFailure = pendingJobs.some(job => !ACTIVE_STATES.has(job.state));

  return <div className={styles.page}>
    <header className={styles.commandBar}>
      <div className={styles.barInner}>
        <nav className={styles.nav} aria-label={tr("Main navigation", "主导航")}>
          <Link href="/">{tr("Lab", "实验室")}</Link>
          <Link href="/sim">{tr("World", "模拟世界")}</Link>
          <Link href="/train">{tr("Training", "训练分析")}</Link>
          <button type="button" className={styles.languageButton} onClick={toggle}
            aria-label={locale === "en" ? "切换到中文" : "Switch to English"}>
            {locale === "en" ? "中文" : "EN"}
          </button>
        </nav>
        <div className={styles.safety}>
          <span className={`${styles.signal} ${pendingJobs.length ? styles.signalActive : ""}`} role="status">
            {pendingJobs.length === 0 ? tr("● No cloud GPU running", "● 云算力未运行") : hasReleaseFailure
              ? tr("● Release unconfirmed", "● 释放未确认")
              : tr(`● ${pendingJobs.length} cloud session${pendingJobs.length > 1 ? "s" : ""} active`, `● ${pendingJobs.length} 个云端会话进行中`)}
          </span>
          <span className={styles.elapsed}
            title={tr("Time since the earliest active GPU allocation succeeded; this is not Google's billable time.", "从当前最早的 GPU 分配成功时计时，不等于 Google 实际计费时长。")}
          >
            {tr("Elapsed", "已使用时长")} <strong>{elapsed(firstStart, now)}</strong>
          </span>
          <button type="button" className={styles.disconnect} disabled={!pendingJobs.length || disconnecting}
            onClick={() => void disconnectAll()} title={tr("Stop only the Colab sessions started by this lab", "仅停止本项目启动的 Colab 会话")}>
            {disconnecting ? tr("Disconnecting…", "正在断开…") : tr("Disconnect cloud GPU", "一键断开云算力")}
          </button>
        </div>
      </div>
    </header>

    <main className={styles.content}>
      {!online && <div className={styles.error} role="alert">
        {tr("Cannot reach duck-lab. Start the local server and refresh. If a GPU session may still be active, run `uv run colab sessions` in Terminal to check it.",
          "无法连接 duck-lab。请启动本地服务并刷新；如果云端会话可能仍在运行，请在终端执行 `uv run colab sessions` 检查。")}
      </div>}
      {actionError && <div className={styles.error} role="alert">{actionError}</div>}
      {notice && <div className={styles.notice} role="status">{notice}</div>}

      <section className={styles.hero}>
        <div className={styles.eyebrow}>MICRODUCK / COLAB GPU</div>
        <h1>{tr("Train MicroDuck on Colab", "用 Colab 训练 MicroDuck")}</h1>
        <p>{tr("Use your own Google account and compute-unit balance to run the official MicroDuck GPU training stack. You can disconnect this lab's sessions from the pinned control above at any time.",
          "使用您自己的 Google 账号与算力余额，运行 MicroDuck 官方 GPU 训练栈。页面顶部的断开按钮始终可见，随时可停止本项目创建的会话。")}</p>
        <p className={styles.benefit}>{tr(
          "🚀 Eligible paid Google AI Pro members receive 200 Colab CCUs monthly. At an illustrative 13.3 CCUs/hour, that is about 15 cumulative A100 hours. Actual rates and GPU availability vary.",
          "🚀 符合条件的付费 Google AI Pro 会员每月可获得 200 Colab CCU。按示例费率 13.3 CCU/小时估算，约为累计 15 小时 A100；实际扣费率和 GPU 可用性会变化。"
        )}</p>
      </section>

      <div className={styles.grid}>
        <section className={styles.card}>
          <h2>{tr("Account & balance", "账号与余额")}</h2>
          {account?.connected ? <p>✓ {tr("Connected", "已连接")} · {tr("Balance", "余额")} <strong>{account.balance ?? "—"} CCU</strong> · {tr("Current rate", "当前费率")} <strong>{account.rate ?? "—"} CCU/h</strong></p>
            : <>
              <p>{tr("Connect once in Terminal using Google's official Colab CLI. Credentials stay on your computer; this app never asks for a password or authorization code.",
                "首次使用请在终端通过 Google 官方 Colab CLI 授权。凭据保存在您的电脑上；本应用不会索取密码或授权码。")}</p>
              <code className={styles.command}>cd microduck_local && uv run colab usage</code>
              {account?.message && <p className={styles.hint}>{messageText(account.message, locale)}</p>}
            </>}
          {account?.connected && !hasBalance && <p className={styles.warning}>{tr(
            "No positive CCU balance was detected. Verify that Colab and this CLI use the same eligible Google account.",
            "未检测到可用 CCU。请确认 Colab 网页与本机 CLI 登录的是同一个享有权益的 Google 账号。"
          )}</p>}
          <p><a href="https://colab.research.google.com/github/heranhe/microduck-lab-cloud/blob/main/notebooks/microduck_train.ipynb"
            target="_blank" rel="noreferrer">{tr("Open Colab notebook (Windows supported) ↗", "打开 Colab Notebook（支持 Windows）↗")}</a></p>
          <button type="button" className={styles.secondary} onClick={() => void refresh()}>{tr("Refresh account & jobs", "刷新账号与任务")}</button>
        </section>

        <section className={styles.card}>
          <h2>{tr("New GPU training job", "新建 GPU 训练任务")}</h2>
          <div className={styles.fields}>
            <label className={styles.field}>{tr("Training task", "训练任务")}
              <select value={task} onChange={event => setTask(event.target.value)}>
                <option value="Mjlab-Velocity-Flat-MicroDuck">MicroDuck · {tr("Walk", "行走")}</option>
                <option value="Mjlab-VelStand-Flat-MicroDuck">MicroDuck · {tr("Walk & recover", "行走与起身")}</option>
              </select>
            </label>
            <label className={styles.field}>GPU
              <select value={gpu} onChange={event => setGpu(event.target.value)}>
                {["T4", "L4", "A100", "H100"].map(option => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label className={styles.field}>{tr("Iterations", "迭代次数")}
              <input type="number" min={1} max={100000} value={iterations} onChange={event => setIterations(Number(event.target.value))} />
            </label>
            <label className={styles.field}>{tr("Parallel environments", "并行环境数")}
              <input type="number" min={1} max={4096} value={envs} onChange={event => setEnvs(Number(event.target.value))} />
            </label>
          </div>
          <button type="button" className={styles.primary} disabled={!online || !hasBalance || starting} onClick={() => void start()}>
            {starting ? tr("Sending request…", "正在提交…") : tr("Start training on Colab", "在 Colab 启动训练")}
          </button>
          <p className={styles.hint}>{tr("GPU allocation is subject to Colab availability. The timer above starts when allocation succeeds.",
            "GPU 分配取决于 Colab 实际资源。顶部计时从 GPU 分配成功时开始。")}</p>
        </section>
      </div>

      <section className={`${styles.card} ${styles.jobs}`}>
        <h2>{tr("Cloud jobs", "云端任务")}</h2>
        {sortedJobs.length === 0 && <p>{tr("No cloud jobs yet.", "暂无云端任务。")}</p>}
        {sortedJobs.map(job => <div className={styles.job} key={job.id}>
          <div className={styles.jobHead}>
            <strong>{job.id}</strong>
            <span>{job.gpu}</span>
            <span className={styles.status}>{stateText(job.state === "running" ? job.remote_state ?? job.state : job.state, locale)}</span>
            {needsRelease(job) && <span className={styles.jobTime}>{tr("Elapsed", "已使用时长")} {elapsed(runtimeStart(job), now)}</span>}
          </div>
          {job.message && <p>{messageText(job.message, locale)}</p>}
          {job.released === false && !ACTIVE_STATES.has(job.state) && <p className={styles.warning}>
            {tr("GPU release was not confirmed. Use the red button above to retry, or check Colab sessions in Terminal.",
              "未确认 GPU 已释放。请使用顶部红色按钮重试，或在终端检查 Colab 会话。")}
          </p>}
          <div className={styles.jobActions}>
            {needsRelease(job) && <button type="button" className={styles.secondary} disabled={stoppingId === job.id}
              onClick={() => void stopOne(job.id)}>{stoppingId === job.id ? tr("Disconnecting…", "正在断开…") : tr("Disconnect this job", "断开此任务")}</button>}
            {job.state === "done" && <a href={`${LAB_HTTP}/cloud/colab/jobs/${job.id}/policy.onnx`}>
              {tr("Download ONNX", "下载 ONNX 模型")}
            </a>}
          </div>
        </div>)}
      </section>
    </main>
  </div>;
}
