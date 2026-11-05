"use client";

import { useCallback, useEffect, useMemo, useState, type MutableRefObject } from "react";
import Link from "next/link";
import { plansHref } from "@/lib/plans";
import { displayedTraining, LAB_HTTP, deleteHfToken, fetchHfSettings, saveHfToken, type HfSettings, type LabClient } from "@/lib/lab";
import {
  canSaveAndDisconnect,
  cloudHeartbeatStale,
  cloudPhaseText,
  needsRelease,
  type CloudAccount as Account,
  type CloudHardware as Hardware,
  type CloudJob as Job,
  type ColabServerView,
  type HfCloudAccount as HfAccount,
  type CloudProvider as Provider,
} from "@/lib/cloudCompute";
import { useI18n, type Locale } from "@/lib/i18n";
import { localizeBehavior } from "@/lib/teachLocalization";
import { requestCloudOpen, requestCloudTraining, requestPolicyToggle, requestTeachOpen, usePolicyOpen } from "@/lib/ui";
import { appendCloudScore, cloudProgressView, localProgressView } from "@/lib/trainProgress";
import { TrainProgress } from "./TrainProgress";
import { TrainingScoreChart } from "./TrainingScoreChart";
import { useTrainingScoreHistory } from "./useTrainingScoreHistory";
import styles from "./CloudPanel.module.css";
import { usePanelDrag } from "./usePanelDrag";

import { useSelectedDuck } from "@/lib/select";

function clock(start: number | null | undefined, now: number) {
  if (!start) return "00:00:00";
  const s = Math.max(0, Math.floor(now / 1000 - start));
  return `${String(Math.floor(s / 3600)).padStart(2,"0")}:${String(Math.floor(s % 3600 / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
}
function duration(seconds: number | null | undefined) {
  const s = Math.max(0, Math.floor(seconds ?? 0));
  return `${String(Math.floor(s / 3600)).padStart(2,"0")}:${String(Math.floor(s % 3600 / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
}
function accountReasonText(account: Account, locale: Locale) {
  if (account.reason === "auth_expired") return locale === "en" ? "Google sign-in has expired. Connect again." : "Google 授权已失效，请重新连接。";
  if (account.reason === "network") return locale === "en" ? "Can't reach Google Colab. Check the network or proxy, then refresh." : "无法连接 Google Colab，请检查网络或代理后点击刷新。";
  if (account.reason === "not_installed") return locale === "en" ? "Colab CLI is not installed. Run `uv sync` in microduck_local." : "未安装 Colab CLI，请在 microduck_local 目录运行 `uv sync`。";
  return account.message;
}
function errorText(value: unknown, locale: Locale) {
  const raw = value instanceof Error ? value.message : String(value);
  if (locale === "en") return raw;
  if (raw.includes("Connect a Hugging Face")) return "请先连接 Hugging Face 账号。";
  if (raw.includes("Connect your Google")) return "请先在终端运行 colab usage 连接 Google 账号。";
  if (raw.includes("paid Colab compute units")) return "当前没有付费 Colab 计算单元。可选择 T4 尝试免费层。";
  if (raw.includes("Colab CLI runtime is incompatible") || raw.includes("JupyterSubprotocol") || raw.includes("jupyter_kernel_client")) {
    return "本地 Colab CLI 与内核客户端版本不兼容，请在 microduck_local 目录运行 `uv sync` 后重试。";
  }
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
  const data = await response.json();
  if (data.ok === false) throw new Error(data.message || data.detail || "Request failed");
  return data;
}

export function CloudPanel({ clientRef }: { clientRef: MutableRefObject<LabClient | null> }) {
  const scoreHistory = useTrainingScoreHistory(clientRef);
  const panelDrag = usePanelDrag("cloud");
  const { tr, locale } = useI18n();
  const policyOpen = usePolicyOpen();
  const [open, setOpen] = useState(() => typeof window !== "undefined" && new URLSearchParams(window.location.search).get("cloud") === "1");
  const [provider, setProvider] = useState<Provider>("colab");
  const [colab, setColab] = useState<Account>({ connected:false });
  const [hf, setHf] = useState<HfAccount>({ connected:false, hardware:[] });
  const [hfSettings, setHfSettings] = useState<HfSettings | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [server, setServer] = useState<ColabServerView | null>(null);
  const [authUrl, setAuthUrl] = useState("");
  const [authMode, setAuthMode] = useState<"callback" | "paste">("callback");
  const [authCode, setAuthCode] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [refreshError, setRefreshError] = useState("");
  const [notice, setNotice] = useState("");

  const [cloudScore, setCloudScore] = useState<{ jobId: string; points: { x: number; y: number }[] }>({ jobId: "", points: [] });

  const updateJobs = useCallback((platform: Provider) => (data: { jobs?: Job[]; server?: ColabServerView }) => {
    if (data.server) setServer(data.server);
    if (platform === "colab") {
      const live = (data.jobs ?? []).filter(needsRelease);
      setCloudScore((previous) => appendCloudScore(previous, live.find((j) => j.allocated_at) ?? live[0]));
    }
    setJobs((previous) => [
      ...previous.filter((job) => job.platform !== platform),
      ...(data.jobs ?? []).map((job) => ({ ...job, platform })),
    ]);
  }, []);

  const refresh = useCallback(async (force = false) => {
    // Publish each provider immediately: account lookup must never delay
    // showing another provider's billable jobs and emergency stop control.
    const results = await Promise.allSettled([
      jsonRequest(`/cloud/colab/account${force ? "?refresh=true" : ""}`).then(setColab),
      jsonRequest("/cloud/colab/jobs").then(updateJobs("colab")),
      jsonRequest("/cloud/hf/account").then(setHf),
      jsonRequest("/cloud/hf/jobs").then(updateJobs("hf")),
      fetchHfSettings().then(setHfSettings),
    ]);
    const failed = results.find((result) => result.status === "rejected");
    setRefreshError(failed?.status === "rejected" ? errorText(failed.reason, locale) : "");
  }, [locale, updateJobs]);

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

  useEffect(() => {
    if (open && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("microduck:close-policy"));
    }
  }, [open]);

  const cloudActive = useMemo(() => jobs.filter(needsRelease), [jobs]);
  const hasCloudActive = cloudActive.length > 0;
  // Reading job state is a local dictionary lookup, so a paid run can tick faster
  // than the account check without spawning anything.
  useEffect(() => {
    if (!hasCloudActive) return;
    const poll = window.setInterval(() => {
      void jsonRequest("/cloud/colab/jobs").then(updateJobs("colab")).catch(() => {});
    }, 5000);
    return () => clearInterval(poll);
  }, [hasCloudActive, updateJobs]);
  const primary = cloudActive.find((j) => j.allocated_at) ?? cloudActive[0];
  const selectedDuck = useSelectedDuck();
  const local = displayedTraining(clientRef.current?.frame, selectedDuck);
  const localActive = local?.status === "training" || local?.status === "stopping";
  const saving = local?.status === "stopping";
  const progress = local?.progress;
  const steps = progress?.overallSteps ?? progress?.steps ?? 0;
  const total = progress?.overallTotal ?? progress?.total ?? 0;
  const sps = progress?.sps;
  const nextSnap = progress?.nextSnapStep;
  const snapEta = nextSnap != null && sps && sps > 0 ? Math.max(0, nextSnap - (progress?.steps ?? 0)) / sps : null;
  const hasActiveCompute = localActive || cloudActive.length > 0;
  const elapsed = primary
    ? clock(primary.allocated_at, now)
    : localActive
      ? duration(local.progress.overallElapsed)
      : "00:00:00";
  const device = primary
    ? `${primary.platform === "hf" ? "HF" : "Colab"} · ${primary.device ?? primary.gpu}${primary.vram ? ` · ${primary.vram}` : ""}`
    : localActive
      ? tr(`This computer · CPU · ${local.behavior.title}`, `本机 · CPU · ${localizeBehavior(local.behavior, locale).title}`)
      : tr("Training compute idle", "训练算力未开启");

  const canSave = !!primary && canSaveAndDisconnect(primary);
  const savingNow = primary?.state === "saving";
  const primaryStale = !!primary && cloudHeartbeatStale(primary, now);
  // The same readout for a CPU run and a GPU run.
  const progressView = localActive && progress && local
    ? localProgressView({ status: local.status === "stopping" ? "stopping" : "training", steps, total, etaSeconds: progress.etaSeconds, score: progress.ep_rew }, locale)
    : primary ? cloudProgressView(primary, now, locale) : null;
  const chartPoints = localActive ? scoreHistory : cloudScore.points;
  const showChart = localActive || (!!primary && cloudScore.points.length > 0);

  const saveAndDisconnect = async (job: Job) => {
    setBusy("disconnect"); setError(""); setNotice("");
    try {
      await jsonRequest(`/cloud/colab/jobs/${job.id}/stop?rescue=true`, { method: "POST" });
      setNotice(tr("Saving the latest progress, then releasing the GPU automatically.", "正在保存最新进度，完成后自动释放算力（最多约 7 分钟）。"));
      window.dispatchEvent(new CustomEvent("microduck:cloud-refresh"));
    } catch (e) { setError(errorText(e, locale)); requestCloudOpen(); }
    finally { setBusy(""); }
  };

  const connectColab = async (manual = false) => {
    setBusy("auth"); setError(""); setNotice("");
    // Open during the click: browsers block a tab opened after an await.
    const tab = window.open("", "_blank");
    if (tab) tab.opener = null;
    try {
      const started = await jsonRequest(`/cloud/colab/auth/start${manual ? "?manual=true" : ""}`, { method: "POST" });
      if (started.url) {
        setAuthUrl(started.url);
        setAuthMode(started.mode === "callback" ? "callback" : "paste");
        if (tab) tab.location.href = started.url;
      } else {
        tab?.close();
        setNotice(tr("Google account is already connected.", "Google 账号已连接。"));
        await refresh(true);
      }
    } catch (e) { tab?.close(); setError(errorText(e, locale)); }
    finally { setBusy(""); }
  };

  // In callback mode the redirect lands on the lab, so the panel only has to watch.
  useEffect(() => {
    if (!authUrl || authMode !== "callback") return;
    let stopped = false;
    const check = async () => {
      try {
        const status = await jsonRequest("/cloud/colab/auth/status");
        if (stopped) return;
        if (status.state === "connected") {
          setAuthUrl("");
          setNotice(tr("Google account connected.", "Google 账号已连接。"));
          await refresh(true);
        } else if (status.state === "failed") {
          setAuthUrl("");
          setError(String(status.message || tr("Google sign-in did not finish.", "Google 授权未完成。")));
        } else if (status.state === "idle") {
          setAuthUrl("");
        }
      } catch { /* The next tick retries; the lab may be restarting. */ }
    };
    const timer = window.setInterval(check, 2000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [authUrl, authMode]); // eslint-disable-line react-hooks/exhaustive-deps -- refresh and tr are rebuilt each render.

  const finishConnect = async () => {
    setBusy("auth"); setError("");
    try {
      setColab(await jsonRequest("/cloud/colab/auth/finish", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code: authCode }) }));
      setAuthUrl(""); setAuthCode("");
      setNotice(tr("Google account connected.", "Google 账号已连接。"));
    } catch (e) {
      // Google's codes are one-time, so a refused one needs a fresh sign-in.
      setAuthUrl(""); setAuthCode("");
      setError(errorText(e, locale));
    }
    finally { setBusy(""); }
  };

  const cancelConnect = async () => {
    setAuthUrl(""); setAuthCode("");
    await jsonRequest("/cloud/colab/auth/cancel", { method: "POST" }).catch(() => {});
  };

  const releaseOrphan = async (endpoint: string) => {
    if (!window.confirm(tr(
      "This runtime is not owned by the lab. It may be a notebook you opened on the Colab website, and releasing it ends that work. Release it?",
      "这个会话不属于本实验室，也可能是你在 Colab 网页打开的 Notebook，释放后其中的工作会结束。确定释放吗？",
    ))) return;
    setBusy(endpoint); setError("");
    try {
      const result = await jsonRequest("/cloud/colab/orphans/release", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ endpoint }) });
      setServer(result.server);
      await refresh(true);
    } catch (e) { setError(errorText(e, locale)); }
    finally { setBusy(""); }
  };

  const stopActiveCompute = async () => {
    if (canSave || savingNow) {
      if (!window.confirm(tr(
        "Disconnect now? Training progress on the runtime that has not been downloaded will be lost.",
        "立即断开？云端尚未下载的训练进度都会丢失。",
      ))) return;
    }
    setBusy("disconnect"); setError(""); setNotice("");
    try {
      const requests: Promise<unknown>[] = [];
      if (localActive && !saving) requests.push(jsonRequest("/teach/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run: local.runName }) }));
      const providers = new Set(cloudActive.map((j) => j.platform));
      requests.push(...[...providers].map((p) => jsonRequest(`/cloud/${p === "hf" ? "hf" : "colab"}/stop-all`, { method:"POST" })));
      const results = await Promise.allSettled(requests);
      const failed = results.find((result) => result.status === "rejected");
      if (failed?.status === "rejected") throw failed.reason;
      const pending = results.some((result) => {
        if (result.status !== "fulfilled") return false;
        const data = result.value as { stopped?: Job[] };
        return data.stopped?.some((job) => typeof job === "object" && job.released !== true);
      });
      if (pending) requestCloudOpen();
      setNotice(primary
        ? pending
          ? tr("Disconnect requested. Release is not yet confirmed; check the job status below.", "已请求断开，但尚未确认资源释放。请查看下方任务状态，必要时重试。")
          : tr("Cloud sessions were released.", "本项目的云算力已释放。")
        : tr("Saving the checkpoint before stopping…", "正在保存检查点，完成后停止训练…"));
      window.dispatchEvent(new CustomEvent("microduck:cloud-refresh"));
    } catch (e) { setError(errorText(e, locale)); requestCloudOpen(); }
    finally { setBusy(""); }
  };

  const stopOne = async (job: Job) => {
    if ((canSaveAndDisconnect(job) || job.state === "saving") && !window.confirm(tr(
      "Disconnect now? Training progress on the runtime that has not been downloaded will be lost.",
      "立即断开？云端尚未下载的训练进度都会丢失。",
    ))) return;
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
      <button className={`${styles.toolButton} ${open ? styles.toolButtonActive : ""}`} onClick={() => open ? setOpen(false) : requestCloudOpen()}>☁ {tr("Cloud", "云算力")}</button>
      <button className={`${styles.toolButton} ${policyOpen ? styles.toolButtonActive : ""}`} onClick={requestPolicyToggle}>🧠 {tr("Models", "模型")}</button>
      <Link className={styles.toolButton} href={plansHref()} title={tr("Create or import a training plan", "创建或导入训练方案")}>📋 {tr("Plans", "方案")}</Link>
    </div>
    <div className={`${styles.statusBar} ${hasActiveCompute ? styles.statusBarActive : ""} ${localActive ? styles.trainingBar : ""}`} data-policy-ui>
      <div className={styles.statusContent}>
      <div className={styles.statusHeading}>
        <span className={styles.dot}/>
        <button className={styles.monitorLink} onClick={requestTeachOpen} title={local?.runName}>{device}</button>
        {hasActiveCompute && <span className={styles.timer}>{elapsed}</span>}
        {!hasActiveCompute && <button className={styles.monitorLink} onClick={requestTeachOpen}>{tr("New training", "新建训练")}</button>}
        {canSave && primary && <button className={styles.disconnect} disabled={busy === "disconnect"} onClick={() => void saveAndDisconnect(primary)} title={tr("Keep the latest checkpoint, then release the GPU", "保留最新检查点，然后释放算力")}>
          {tr("⏸ Save and disconnect", "⏸ 保存并断开")}
        </button>}
        {hasActiveCompute && <button className={styles.disconnect} disabled={saving || busy === "disconnect"} onClick={stopActiveCompute}>
          {saving ? tr("Saving…", "保存中…") : busy === "disconnect" ? tr("Stopping…", "停止中…") : (canSave || savingNow) ? tr("■ Disconnect now", "■ 立即断开") : primary ? tr("■ Disconnect", "■ 断开云算力") : tr("⏸ Save and stop", "⏸ 保存并停止")}
        </button>}
        {localActive && <details className={styles.more}>
          <summary aria-label={tr("Training actions", "训练操作")}>⋯</summary>
          <div className={styles.moreMenu}>
            {progress && <div className={styles.snapshotLine}>
              <span>{tr("Latest checkpoint", "最近检查点")}：{progress.lastCheckpointStep != null ? `@${progress.lastCheckpointStep.toLocaleString()}` : tr("None", "暂无")}</span>
              {!saving && nextSnap != null && <span>{tr("Next snapshot", "下次快照")} @{nextSnap.toLocaleString()}{snapEta != null ? ` · ≈${duration(snapEta)}` : ""}</span>}
              {local?.stage && <span>{tr("Stage", "阶段")} {local.stage.idx}/{local.stage.count}</span>}
              <span>{sps && sps > 0 ? `${Math.round(sps).toLocaleString()} ${tr("steps/s", "步/秒")}` : tr("Measuring speed…", "正在测量速度…")}</span>
            </div>}
            <button onClick={requestTeachOpen}>{tr("Open monitor", "查看训练监控")}</button>
            <button onClick={async () => {
              if (!window.confirm(tr("Force terminate? Progress since the last saved checkpoint will be lost.", "强制终止训练？最近检查点之后的进度会丢失。"))) return;
              try { await jsonRequest("/teach/stop?force=true", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run: local?.runName }) }); }
              catch (e) { setError(errorText(e, locale)); }
            }}>{tr("Force terminate", "强制终止")}</button>
          </div>
        </details>}
      </div>
      {progressView && <TrainProgress view={progressView}/>}
      {primary && <>
        <div className={styles.snapshotLine}>{tr("Keep this Mac open: it finishes and releases the job.", "请勿合上电脑：结果下载和算力释放都由本机完成。")}</div>
        {primaryStale && <div className={styles.error} role="alert">{tr("No answer from the runtime for a while. The GPU may still be billing.", "云端已有一段时间没有响应，算力可能仍在计费。")}</div>}
        {primary.message && <div className={styles.error} role="alert">{errorText(primary.message, locale)}</div>}
      </>}
      {error && <div className={styles.error} role="alert">{error}</div>}
      </div>
      {showChart && <div className={styles.scoreChart}>
        <div className={styles.scoreHeading}>
          <span title={tr("Score per practice run; higher is better", "每轮训练得分，越高表示动作完成得越好")}>{tr("Training score · higher is better", "训练得分 · 越高越好")}</span>
          {chartPoints.length > 0 && <strong>{chartPoints[chartPoints.length - 1].y.toFixed(2)}</strong>}
        </div>
        {chartPoints.length > 0
          ? <TrainingScoreChart points={chartPoints} label={tr("Score per practice run; higher is better", "每轮训练得分，越高表示动作完成得越好")} />
          : <div className={styles.scoreEmpty}>{tr("Waiting for the first score…", "等待首轮得分…")}</div>}
      </div>}
    </div>
    {open && <section {...panelDrag.panelProps} className={styles.panel} data-policy-ui style={{ ...(panelDrag.point ? { left: panelDrag.point.x, top: panelDrag.point.y, right: "auto" } : {}), ...panelDrag.resizeStyle }}>
      <header className={styles.header} {...panelDrag.dragHandle} style={{ cursor: "inherit", touchAction: "none", userSelect: "none" }}><span className={styles.title}>☁ {tr("Cloud compute", "云算力中心")}</span><button className={styles.close} onClick={() => setOpen(false)}>✕</button></header>
      <div className={styles.tabs}>
        <button className={`${styles.tab} ${provider === "colab" ? styles.tabActive : ""}`} onClick={() => setProvider("colab")}>Google Colab</button>
        <button className={`${styles.tab} ${provider === "hf" ? styles.tabActive : ""}`} onClick={() => setProvider("hf")}>🤗 Hugging Face</button>
      </div>
      <div className={styles.body}>
        <div className={styles.account}>
          <span className={`${styles.accountMark} ${account.connected ? styles.accountMarkOn : ""}`}/>
          <div style={{flex:1}}>
            <div className={account.connected ? styles.good : ""}>{account.connected ? tr("Account connected", "账号已连接") : tr("Account not connected", "账号未连接")}</div>
            {provider === "colab" && colab.connected && <div className={styles.connectionMethod}>
              {tr("Google account", "Google 账户")}：{colab.email || tr("Email unavailable · refresh to retry", "暂未获取邮箱，点击刷新重试")}
            </div>}
            {account.connected && <div className={styles.connectionMethod}>
              {tr("Connection method", "连接方式")}：{provider === "colab" ? "Google Colab CLI (google-colab-cli)" : tr("Hugging Face access token", "Hugging Face Token")}
            </div>}
            {provider === "colab" && colab.connected && <div className={styles.muted}>{tr("Paid balance", "付费余额")} {colab.balance ?? "—"} CCU{colab.rate != null ? ` · ${colab.rate} CCU/h` : ""}</div>}
            {provider === "hf" && hfSettings?.configured && <div className={styles.muted}>{hfSettings.username} · {hfSettings.masked}</div>}
            {provider === "colab" && colab.connected && colab.active_assignments != null && <div className={colab.active_assignments > 0 ? styles.good : styles.muted}>
              {tr("Active runtimes on Google", "Google 侧活动会话")}：{colab.active_assignments}
            </div>}
            {provider === "colab" ? (!colab.connected && colab.reason ? <div className={styles.muted}>{accountReasonText(colab, locale)}</div> : null) : (account.message && <div className={styles.muted}>{account.message}</div>)}
          </div>
          {provider === "colab" && <button className={styles.refreshButton} disabled={busy === "refresh"} onClick={async () => { setBusy("refresh"); await refresh(true); setBusy(""); }}>{busy === "refresh" ? tr("Checking…", "检查中…") : tr("Refresh", "刷新")}</button>}
          {provider === "hf" && hfSettings?.configured && <button className={styles.dangerLink} disabled={busy === "token"} onClick={disconnectHf}>{tr("remove token", "移除 Token")}</button>}
        </div>

        {account.connected && !hasCloudActive && <div className={styles.tokenRow}>
          <button className={styles.smallButton} onClick={requestCloudTraining}>▶ {tr("Start cloud training", "开始云训练")}</button>
          <span className={styles.help}>{tr("Pick an official task, then press Cloud.", "选一个官方任务，再点“云算力”。")}</span>
        </div>}

        {provider === "colab" && colab.connected && !(typeof colab.balance === "number" && colab.balance > 0) && <div className={styles.balanceWarning}>
          <strong>{tr("No paid CCU shown · Free T4 can still be requested", "无可用付费 CCU · 仍可尝试免费 T4")}</strong>
          <span>{tr("Choose T4 in Training. Google decides whether a free GPU is available and may end the session without warning. L4, A100 and H100 require paid compute units.", "请在“训练”中选择 T4。免费 GPU 能否分配由 Google 动态决定，运行时也可能随时结束；L4、A100 和 H100 需要付费计算单元。")}</span>
          <a href="https://research.google.com/colaboratory/faq.html" target="_blank" rel="noreferrer">{tr("Colab free-tier limits ↗", "查看 Colab 免费层限制 ↗")}</a>
        </div>}

        {provider === "hf" && !hfSettings?.configured && <>
          <div className={styles.tokenRow}><input className={styles.input} type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} placeholder="hf_…"/><button className={styles.smallButton} disabled={!token || busy === "token"} onClick={connectHf}>{busy === "token" ? tr("Checking…", "验证中…") : tr("Connect", "连接")}</button></div>
          <div className={styles.help}>{tr("Use a fine-grained token with Jobs and model write access.", "请使用具有 Jobs 和模型写入权限的细粒度 Token。")} <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">{tr("Create token", "创建 Token")}</a></div>
        </>}
        {provider === "colab" && !colab.connected && colab.reason !== "network" && colab.reason !== "not_installed" && (authUrl && authMode === "callback" ? <>
          <div className={styles.help}>{tr("Choose your account and approve access on the Google page. This panel connects by itself once you're done.", "请在 Google 页面选择账号并同意授权，完成后这里会自动连接。")} <a href={authUrl} target="_blank" rel="noreferrer">{tr("Open the Google page again", "重新打开 Google 页面")}</a></div>
          <div className={styles.tokenRow}>
            <span className={styles.help}>{tr("Waiting for Google…", "正在等待 Google 授权…")}</span>
            <button className={styles.refreshButton} onClick={() => connectColab(true)}>{tr("Use a code instead", "改用授权码")}</button>
            <button className={styles.refreshButton} onClick={cancelConnect}>{tr("Cancel", "取消")}</button>
          </div>
        </> : authUrl ? <>
          <div className={styles.help}>{tr("Approve access on the Google page, then paste the code it shows here.", "请在 Google 页面完成授权，再把页面显示的授权码粘贴到这里。")} <a href={authUrl} target="_blank" rel="noreferrer">{tr("Open the Google page", "打开 Google 授权页面")}</a></div>
          <div className={styles.tokenRow}>
            <input className={styles.input} autoComplete="off" value={authCode} onChange={(e) => setAuthCode(e.target.value)} placeholder={tr("Authorization code", "授权码")}/>
            <button className={styles.smallButton} disabled={!authCode.trim() || busy === "auth"} onClick={finishConnect}>{busy === "auth" ? tr("Checking…", "验证中…") : tr("Finish", "完成")}</button>
            <button className={styles.refreshButton} onClick={cancelConnect}>{tr("Cancel", "取消")}</button>
          </div>
        </> : <>
          <div className={styles.tokenRow}><button className={styles.smallButton} disabled={busy === "auth"} onClick={() => connectColab()}>{busy === "auth" ? tr("Opening…", "正在打开…") : tr("Connect Google account", "连接 Google 账号")}</button></div>
          <div className={styles.help}>{tr("Uses Google's official sign-in. The lab never sees your password; the token stays on this computer, readable only by you.", "使用 Google 官方授权。实验室不接触你的密码，令牌只保存在本机，且仅你的账户可读。")}</div>
        </>)}
        {provider === "colab" && server?.ok === false && cloudActive.some((job) => job.platform === "colab") && <div className={styles.balanceWarning}>
          <strong>{tr("Can't check Google's session list", "暂时无法向 Google 查询会话")}</strong>
          <span>{tr("If this continues, end the runtime from Runtime → Manage sessions on the Colab website.", "如果一直如此，请在 Colab 网页的“代码执行程序 → 管理会话”中手动终止。")}</span>
          <a href="https://colab.research.google.com/" target="_blank" rel="noreferrer">{tr("Open Colab ↗", "打开 Colab ↗")}</a>
        </div>}
        {provider === "colab" && (server?.orphans.length ?? 0) > 0 && <div className={styles.balanceWarning}>
          <strong>{tr("Runtime not owned by any lab job", "发现不属于任何实验室任务的会话")}</strong>
          <span>{tr("It may still be billing. It could also be a notebook you opened yourself.", "它可能仍在计费，也可能是你自己在网页打开的 Notebook。")}</span>
          {server?.orphans.map((orphan) => <span key={orphan.endpoint}>
            {orphan.hardware} · {orphan.endpoint}{" "}
            <button className={styles.dangerLink} disabled={busy === orphan.endpoint} onClick={() => void releaseOrphan(orphan.endpoint)}>{tr("release", "释放")}</button>
          </span>)}
        </div>}
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
            <div><span className={styles.jobState}>{cloudPhaseText(job, locale)}</span> · {job.device ?? job.gpu}{job.vram ? ` · ${job.vram}` : ""}</div>
            <div className={styles.jobMeta}>{clock(job.allocated_at, job.released_at ? job.released_at * 1000 : now)} · {job.id}{job.message ? ` · ${errorText(job.message, locale)}` : ""}</div>
            <div className={styles.jobActions}>
              {job.url && <a className={styles.jobLink} href={job.url} target="_blank" rel="noreferrer">↗</a>}
              {(job.state === "done" || job.artifacts?.includes("policy.onnx")) && <a className={styles.jobLink} href={`${LAB_HTTP}/cloud/${job.platform === "hf" ? "hf" : "colab"}/jobs/${job.id}/policy.onnx`}>↓ ONNX</a>}
              {job.platform === "colab" && job.artifacts?.includes("checkpoint.pt") && <a className={styles.jobLink} href={`${LAB_HTTP}/cloud/colab/jobs/${job.id}/artifacts/checkpoint.pt`} title={tr("Training checkpoint, for resuming; deploy the ONNX", "训练检查点，用于续训；部署请使用 ONNX")}>↓ {tr("checkpoint", "检查点")}</a>}
              {canSaveAndDisconnect(job) && <button className={styles.dangerLink} disabled={busy === "disconnect"} onClick={() => void saveAndDisconnect(job)}>{tr("save & stop", "保存并断开")}</button>}
              {needsRelease(job) && <button className={styles.dangerLink} disabled={busy === job.id} onClick={() => stopOne(job)}>{canSaveAndDisconnect(job) || job.state === "saving" ? tr("stop now", "立即断开") : tr("stop", "断开")}</button>}
            </div>
          </div>)}
        </div>
      </div>
    </section>}
  </>;
}
