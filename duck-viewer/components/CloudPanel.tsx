"use client";

import { useCallback, useEffect, useMemo, useState, type MutableRefObject } from "react";
import { LAB_HTTP, deleteHfToken, fetchHfSettings, saveHfToken, type HfSettings, type LabClient } from "@/lib/lab";
import type { CloudAccount as Account, CloudHardware as Hardware, HfCloudAccount as HfAccount, CloudProvider as Provider } from "@/lib/cloudCompute";
import { useI18n, type Locale } from "@/lib/i18n";
import { requestCloudOpen, requestTeachOpen } from "@/lib/ui";
import styles from "./CloudPanel.module.css";

type Job = {
  id: string; platform: Provider; task: string; gpu: string; device?: string; vram?: string;
  flavor?: string; state: string; remote_state?: string; started?: number; allocated_at?: number | null;
  released?: boolean; released_at?: number; message?: string; url?: string;
};

const ACTIVE = new Set(["allocating", "starting", "running", "finalizing", "detached", "stopping"]);
function needsRelease(job: Job) { return job.released !== true && (ACTIVE.has(job.state) || job.released === false); }
function clock(start: number | null | undefined, now: number) {
  if (!start) return "00:00:00";
  const s = Math.max(0, Math.floor(now / 1000 - start));
  return `${String(Math.floor(s / 3600)).padStart(2,"0")}:${String(Math.floor(s % 3600 / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
}
function duration(seconds: number | null | undefined) {
  const s = Math.max(0, Math.floor(seconds ?? 0));
  return `${String(Math.floor(s / 3600)).padStart(2,"0")}:${String(Math.floor(s % 3600 / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
}
function stateText(state: string, locale: Locale) {
  const en: Record<string,string> = { allocating:"Allocating GPU", starting:"Starting", running:"Training", finalizing:"Downloading", stopping:"Disconnecting", stopped:"Stopped", done:"Completed", failed:"Failed", detached:"Needs attention" };
  const zh: Record<string,string> = { allocating:"正在分配 GPU", starting:"正在启动", running:"训练中", finalizing:"下载结果中", stopping:"正在断开", stopped:"已停止", done:"已完成", failed:"失败", detached:"需要检查" };
  return (locale === "en" ? en : zh)[state] ?? state;
}
function errorText(value: unknown, locale: Locale) {
  const raw = value instanceof Error ? value.message : String(value);
  if (locale === "en") return raw;
  if (raw.includes("Connect a Hugging Face")) return "请先连接 Hugging Face 账号。";
  if (raw.includes("Connect your Google")) return "请先在终端运行 colab usage 连接 Google 账号。";
  if (raw.includes("paid Colab compute units")) return "当前没有付费 Colab 计算单元。可选择 T4 尝试免费层。";
  if (raw.includes("Stop and confirm all Hugging Face")) return "请先停止全部 Hugging Face 任务，确认云算力已释放后再修改 Token。";
  if (raw.includes("Failed to fetch") || raw.includes("Lab unavailable")) return "无法连接本地 duck-lab。";
  return raw;
}

async function jsonRequest(path: string, init?: RequestInit) {
  const response = await fetch(`${LAB_HTTP}${path}`, { cache:"no-store", ...init });
  if (!response.ok) {
    const detail = await response.json().then((v) => v.detail).catch(() => `HTTP ${response.status}`);
    throw new Error(detail);
  }
  return response.json();
}

export function CloudPanel({ clientRef }: { clientRef: MutableRefObject<LabClient | null> }) {
  const { tr, locale } = useI18n();
  const [open, setOpen] = useState(() => typeof window !== "undefined" && new URLSearchParams(window.location.search).get("cloud") === "1");
  const [provider, setProvider] = useState<Provider>("colab");
  const [colab, setColab] = useState<Account>({ connected:false });
  const [hf, setHf] = useState<HfAccount>({ connected:false, hardware:[] });
  const [hfSettings, setHfSettings] = useState<HfSettings | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [refreshError, setRefreshError] = useState("");
  const [notice, setNotice] = useState("");

  const refresh = useCallback(async () => {
    // Publish each provider immediately: account lookup must never delay
    // showing another provider's billable jobs and emergency stop control.
    const updateJobs = (platform: Provider) => (data: { jobs?: Job[] }) =>
      setJobs((previous) => [
        ...previous.filter((job) => job.platform !== platform),
        ...(data.jobs ?? []).map((job) => ({ ...job, platform })),
      ]);
    const results = await Promise.allSettled([
      jsonRequest("/cloud/colab/account").then(setColab),
      jsonRequest("/cloud/colab/jobs").then(updateJobs("colab")),
      jsonRequest("/cloud/hf/account").then(setHf),
      jsonRequest("/cloud/hf/jobs").then(updateJobs("hf")),
      fetchHfSettings().then(setHfSettings),
    ]);
    const failed = results.find((result) => result.status === "rejected");
    setRefreshError(failed?.status === "rejected" ? errorText(failed.reason, locale) : "");
  }, [locale]);

  useEffect(() => {
    const first = window.setTimeout(() => void refresh(), 0);
    const poll = window.setInterval(() => void refresh(), 15000);
    const tick = window.setInterval(() => setNow(Date.now()), 1000);
    return () => { clearTimeout(first); clearInterval(poll); clearInterval(tick); };
  }, [refresh]);

  useEffect(() => {
    const openPanel = () => setOpen(true);
    const closePanel = () => setOpen(false);
    const refreshPanel = () => void refresh();
    window.addEventListener("microduck:open-cloud", openPanel);
    window.addEventListener("microduck:close-cloud", closePanel);
    window.addEventListener("microduck:cloud-refresh", refreshPanel);
    return () => {
      window.removeEventListener("microduck:open-cloud", openPanel);
      window.removeEventListener("microduck:close-cloud", closePanel);
      window.removeEventListener("microduck:cloud-refresh", refreshPanel);
    };
  }, [refresh]);

  const cloudActive = useMemo(() => jobs.filter(needsRelease), [jobs]);
  const primary = cloudActive.find((j) => j.allocated_at) ?? cloudActive[0];
  const local = clientRef.current?.frame?.training ?? null;
  const localActive = local?.status === "training";
  const hasActiveCompute = localActive || cloudActive.length > 0;
  const elapsed = primary
    ? clock(primary.allocated_at, now)
    : localActive
      ? duration(local.progress.overallElapsed)
      : "00:00:00";
  const device = primary
    ? `${primary.platform === "hf" ? "HF" : "Colab"} · ${primary.device ?? primary.gpu}${primary.vram ? ` · ${primary.vram}` : ""}`
    : localActive
      ? tr(`This computer · CPU · ${local.behavior.title}`, `本机 · CPU · ${local.behavior.title}`)
      : tr("Training compute idle", "训练算力未开启");

  const stopActiveCompute = async () => {
    setBusy("disconnect"); setError(""); setNotice("");
    try {
      const requests: Promise<unknown>[] = [];
      if (localActive) requests.push(jsonRequest("/teach/stop", { method: "POST" }));
      const providers = new Set(cloudActive.map((j) => j.platform));
      requests.push(...[...providers].map((p) => jsonRequest(`/cloud/${p === "hf" ? "hf" : "colab"}/stop-all`, { method:"POST" })));
      const results = await Promise.allSettled(requests);
      const failed = results.find((result) => result.status === "rejected");
      if (failed?.status === "rejected") throw failed.reason;
      const pending = results.some((result) => {
        if (result.status !== "fulfilled") return false;
        const data = result.value as { stopped?: Job[] };
        return data.stopped?.some((job) => job.released !== true);
      });
      if (pending) requestCloudOpen();
      setNotice(primary
        ? pending
          ? tr("Disconnect requested. Release is not yet confirmed; check the job status below.", "已请求断开，但尚未确认资源释放。请查看下方任务状态，必要时重试。")
          : tr("Cloud sessions were released.", "本项目的云算力已释放。")
        : tr("Local training was stopped.", "本地训练已停止。"));
      window.dispatchEvent(new CustomEvent("microduck:cloud-refresh"));
    } catch (e) { setError(errorText(e, locale)); requestCloudOpen(); }
    finally { setBusy(""); }
  };

  const stopOne = async (job: Job) => {
    setBusy(job.id); setError("");
    try {
      await jsonRequest(`/cloud/${job.platform === "hf" ? "hf" : "colab"}/jobs/${job.id}/stop`, { method:"POST" });
      window.dispatchEvent(new CustomEvent("microduck:cloud-refresh"));
    } catch (e) { setError(errorText(e, locale)); }
    finally { setBusy(""); }
  };

  const connectHf = async () => {
    setBusy("token"); setError("");
    try { setHfSettings(await saveHfToken(token)); setToken(""); await refresh(); }
    catch (e) { setError(errorText(e, locale)); }
    finally { setBusy(""); }
  };

  const disconnectHf = async () => {
    setBusy("token"); setError("");
    try { setHfSettings(await deleteHfToken()); await refresh(); }
    catch (e) { setError(errorText(e, locale)); }
    finally { setBusy(""); }
  };

  const account = provider === "colab" ? colab : hf;
  const providerJobs = jobs.filter((j) => j.platform === provider).sort((a,b) => (b.started ?? 0) - (a.started ?? 0));

  return <>
    <div className={styles.toolbar} data-policy-ui>
      <button className={styles.trainButton} onClick={requestTeachOpen}>🎓 {tr("Training", "训练")}</button>
      <button className={styles.toolButton} onClick={() => open ? setOpen(false) : requestCloudOpen()}>☁ {tr("Cloud", "云算力")}</button>
    </div>
    <div className={`${styles.statusBar} ${hasActiveCompute ? styles.statusBarActive : ""}`} data-policy-ui>
      <span className={styles.dot}/><span>{device}</span><span className={styles.timer}>{elapsed}</span>
      <button className={styles.disconnect} disabled={!hasActiveCompute || busy === "disconnect"} onClick={stopActiveCompute} title={tr("Safety stop for the active local or cloud training job", "停止当前本地训练或断开云端算力，防止继续运行或计费") }>
        {busy === "disconnect"
          ? tr("Stopping…", "停止中…")
          : primary
            ? tr("■ Disconnect now", "■ 一键断开")
            : localActive
              ? tr("■ Stop training", "■ 停止训练")
              : tr("Stop / disconnect", "停止 / 断开")}
      </button>
    </div>
    {open && <section className={styles.panel} data-policy-ui>
      <header className={styles.header}><span className={styles.title}>☁ {tr("Cloud compute", "云算力中心")}</span><button className={styles.close} onClick={() => setOpen(false)}>✕</button></header>
      <div className={styles.tabs}>
        <button className={`${styles.tab} ${provider === "colab" ? styles.tabActive : ""}`} onClick={() => setProvider("colab")}>Google Colab</button>
        <button className={`${styles.tab} ${provider === "hf" ? styles.tabActive : ""}`} onClick={() => setProvider("hf")}>🤗 Hugging Face</button>
      </div>
      <div className={styles.body}>
        <div className={styles.account}>
          <span className={`${styles.accountMark} ${account.connected ? styles.accountMarkOn : ""}`}/>
          <div style={{flex:1}}>
            <div className={account.connected ? styles.good : ""}>{account.connected ? tr("Account connected", "账号已连接") : tr("Account not connected", "账号未连接")}</div>
            {provider === "colab" && colab.connected && <div className={styles.muted}>{tr("Paid balance", "付费余额")} {colab.balance ?? "—"} CCU{colab.rate != null ? ` · ${colab.rate} CCU/h` : ""}</div>}
            {provider === "hf" && hfSettings?.configured && <div className={styles.muted}>{hfSettings.username} · {hfSettings.masked}</div>}
            {account.message && <div className={styles.muted}>{account.message}</div>}
          </div>
          {provider === "colab" && <button className={styles.refreshButton} disabled={busy === "refresh"} onClick={async () => { setBusy("refresh"); await refresh(); setBusy(""); }}>{busy === "refresh" ? tr("Checking…", "检查中…") : tr("Refresh", "刷新")}</button>}
          {provider === "hf" && hfSettings?.configured && <button className={styles.dangerLink} disabled={busy === "token"} onClick={disconnectHf}>{tr("remove token", "移除 Token")}</button>}
        </div>

        {provider === "colab" && colab.connected && !(typeof colab.balance === "number" && colab.balance > 0) && <div className={styles.balanceWarning}>
          <strong>{tr("No paid CCU shown · Free T4 can still be requested", "无可用付费 CCU · 仍可尝试免费 T4")}</strong>
          <span>{tr("Choose T4 in Training. Google decides whether a free GPU is available and may end the session without warning. L4, A100 and H100 require paid compute units.", "请在“训练”中选择 T4。免费 GPU 能否分配由 Google 动态决定，运行时也可能随时结束；L4、A100 和 H100 需要付费计算单元。")}</span>
          <a href="https://research.google.com/colaboratory/faq.html" target="_blank" rel="noreferrer">{tr("Colab free-tier limits ↗", "查看 Colab 免费层限制 ↗")}</a>
        </div>}

        {provider === "hf" && !hfSettings?.configured && <>
          <div className={styles.tokenRow}><input className={styles.input} type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} placeholder="hf_…"/><button className={styles.smallButton} disabled={!token || busy === "token"} onClick={connectHf}>{busy === "token" ? tr("Checking…", "验证中…") : tr("Connect", "连接")}</button></div>
          <div className={styles.help}>{tr("Use a fine-grained token with Jobs and model write access.", "请使用具有 Jobs 和模型写入权限的细粒度 Token。")} <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">{tr("Create token", "创建 Token")}</a></div>
        </>}
        {provider === "colab" && !colab.connected && <div className={styles.help}>{tr("Run `uv run colab usage` once in Terminal to authorize your Google account.", "请先在终端运行 `uv run colab usage`，通过 Google 官方流程完成授权。")}</div>}
        <div className={styles.managerNote}>
          <strong>{tr("Training starts in the teaching panel.", "训练统一从教学面板开始。")}</strong>
          <span>{tr("Choose an action first, then select Local, Colab, or Hugging Face beside the Start button.", "先选择动作，再在“开始训练”按钮旁选择本地、Colab 或 Hugging Face。")}</span>
          <button className={styles.smallButton} onClick={() => { setOpen(false); requestTeachOpen(); }}>▶ {tr("Choose action and train", "选择动作并训练")}</button>
        </div>

        {provider === "hf" && hf.hardware.length > 0 && <div className={styles.resources}>
          <div className={styles.formTitle}>{tr("Available hardware", "可用算力卡")}</div>
          {hf.hardware.slice(0, 5).map((h: Hardware) => <div className={styles.resource} key={h.id}>
            <span>{h.label}</span><span>{h.vram} · ${h.cost}/{h.unit}</span>
          </div>)}
        </div>}
        {notice && <div className={styles.notice}>{notice}</div>}
        {(error || refreshError) && <div className={styles.error} role="alert">{error || refreshError}</div>}

        <div className={styles.jobs}>
          <div className={styles.formTitle} style={{marginTop:0}}>{tr("Recent jobs", "最近任务")}</div>
          {!providerJobs.length && <div className={styles.muted}>{tr("No cloud jobs yet.", "暂无云端任务。")}</div>}
          {providerJobs.slice(0,6).map((job) => <div className={styles.job} key={`${job.platform}-${job.id}`}>
            <div><span className={styles.jobState}>{stateText(job.state, locale)}</span> · {job.device ?? job.gpu}{job.vram ? ` · ${job.vram}` : ""}</div>
            <div className={styles.jobMeta}>{clock(job.allocated_at, job.released_at ? job.released_at * 1000 : now)} · {job.id}{job.message ? ` · ${job.message}` : ""}</div>
            <div className={styles.jobActions}>
              {job.url && <a className={styles.jobLink} href={job.url} target="_blank" rel="noreferrer">↗</a>}
              {job.state === "done" && <a className={styles.jobLink} href={`${LAB_HTTP}/cloud/${job.platform === "hf" ? "hf" : "colab"}/jobs/${job.id}/policy.onnx`}>↓ ONNX</a>}
              {needsRelease(job) && <button className={styles.dangerLink} disabled={busy === job.id} onClick={() => stopOne(job)}>{tr("stop", "断开")}</button>}
            </div>
          </div>)}
        </div>
      </div>
    </section>}
  </>;
}
