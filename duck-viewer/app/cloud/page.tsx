"use client";

import { useCallback, useEffect, useState } from "react";
import { LAB_HTTP } from "@/lib/lab";
import Link from "next/link";
import { useI18n } from "@/lib/i18n";

type Account = { installed: boolean; connected: boolean; balance?: number | null; rate?: number | null; message?: string };
type Job = { id: string; task: string; gpu: string; state: string; remote_state?: string; message?: string; released?: boolean };

export default function CloudPage() {
  const { tr, locale, toggle } = useI18n();
  const [account, setAccount] = useState<Account | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [gpu, setGpu] = useState("T4");
  const [task, setTask] = useState("Mjlab-Velocity-Flat-MicroDuck");
  const [iterations, setIterations] = useState(1000);
  const [envs, setEnvs] = useState(64);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => {
    try {
      const [a, j] = await Promise.all([
        fetch(`${LAB_HTTP}/cloud/colab/account`).then(r => r.json()),
        fetch(`${LAB_HTTP}/cloud/colab/jobs`).then(r => r.json()),
      ]);
      setAccount(a); setJobs(j.jobs ?? []);
    } catch { setError(tr("Start duck-lab first.", "请先启动 duck-lab。")); }
  }, [tr]);
  useEffect(() => { void refresh(); const timer = setInterval(() => void refresh(), 15000); return () => clearInterval(timer); }, [refresh]);
  const start = async () => {
    setBusy(true); setError("");
    try {
      const r = await fetch(`${LAB_HTTP}/cloud/colab/jobs`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, gpu, iterations, envs }) });
      if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
      await refresh();
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };
  const stop = async (id: string) => {
    setError("");
    try {
      const r = await fetch(`${LAB_HTTP}/cloud/colab/jobs/${id}/stop`, { method: "POST" });
      if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
      await refresh();
    } catch (e) { setError(String(e)); }
  };
  const box = { background: "#191e28", border: "1px solid #384252", borderRadius: 12, padding: 20, marginBottom: 16 };
  return <main style={{ maxWidth: 800, margin: "40px auto 90px", padding: "0 20px", color: "#e8edf6", fontFamily: "system-ui" }}>
    <nav style={{ display: "flex", gap: 16, alignItems: "center" }}>
      <Link href="/" style={{ color: "#8cc8ff" }}>{tr("Lab", "实验室")}</Link>
      <Link href="/sim" style={{ color: "#8cc8ff" }}>{tr("World", "世界")}</Link>
      <Link href="/train" style={{ color: "#8cc8ff" }}>{tr("Training", "训练")}</Link>
      <button onClick={toggle}>{locale === "en" ? "中文" : "EN"}</button>
    </nav>
    <h1>{tr("Google Colab training", "Google Colab 云端训练")}</h1>
    <p>{tr("Train with your own Google account and Colab compute units. GPU availability depends on your account and Colab capacity.", "使用您自己的 Google 账号和 Colab 算力单位训练。GPU 是否可用取决于账号权益及 Colab 资源。")}</p>
    <section style={box}>
      <h2>{tr("Account", "账号")}</h2>
      {account?.connected ? <p>✓ {tr("Connected", "已连接")} · {tr("Balance", "余额")}: {account.balance ?? "—"} CCU · {tr("Current rate", "当前费率")}: {account.rate ?? "—"} CCU/h</p> :
        <><p>{tr("Connect once in Terminal. The credentials stay on your computer; this app never asks for your Google password or authorization code.", "首次使用请在终端连接。凭据只保存在您的电脑上；本应用不会索取 Google 密码或授权码。")}</p>
          <code style={{ display: "block", padding: 12, background: "#0d1117", overflowX: "auto" }}>cd microduck_local && uv run colab usage</code>
          <p>{tr("After authorization, click Refresh.", "完成授权后点击刷新。")}</p></>}
      {account?.connected && !(typeof account.balance === "number" && account.balance > 0) && <p style={{ color: "#ffb07d" }}>
        {tr("No positive CCU balance was detected. Check the Google account and Colab subscription before starting GPU training.",
          "未检测到可用的 CCU 余额。启动 GPU 训练前请核对 Google 账号和 Colab 订阅权益。")}
      </p>}
      <p><a href="https://colab.research.google.com/github/heranhe/microduck-lab-cloud/blob/main/notebooks/microduck_train.ipynb"
        target="_blank" rel="noreferrer" style={{ color: "#8cc8ff" }}>
        {tr("Open the notebook in Colab (also works on Windows)", "在 Colab 中打开 Notebook（也支持 Windows）")}
      </a></p>
      <button onClick={() => void refresh()}>{tr("Refresh", "刷新")}</button>
    </section>
    <section style={box}>
      <h2>{tr("New training job", "新建训练任务")}</h2>
      <label>{tr("Task", "任务")} <select value={task} onChange={e => setTask(e.target.value)}>
        <option value="Mjlab-Velocity-Flat-MicroDuck">MicroDuck · {tr("Walking", "行走")}</option>
        <option value="Mjlab-VelStand-Flat-MicroDuck">MicroDuck · {tr("Walk and recover", "行走与起身")}</option>
      </select></label>{" "}
      <label>GPU <select value={gpu} onChange={e => setGpu(e.target.value)}>{["T4", "L4", "A100", "H100"].map(g => <option key={g}>{g}</option>)}</select></label>{" "}
      <label>{tr("Iterations", "迭代次数")} <input type="number" min={1} max={100000} value={iterations} onChange={e => setIterations(Number(e.target.value))} style={{ width: 90 }} /></label>{" "}
      <label>{tr("Environments", "并行环境数")} <input type="number" min={1} max={4096} value={envs} onChange={e => setEnvs(Number(e.target.value))} style={{ width: 75 }} /></label>
      <p><button disabled={!account?.connected || !(typeof account.balance === "number" && account.balance > 0) || busy} onClick={() => void start()}>{tr("Start on Colab", "在 Colab 启动")}</button></p>
    </section>
    <section style={box}>
      <h2>{tr("Jobs", "训练任务")}</h2>
      {jobs.some(j => ["running", "starting", "allocating", "detached"].includes(j.state)) &&
        <button onClick={async () => {
          const r = await fetch(`${LAB_HTTP}/cloud/colab/stop-all`, { method: "POST" });
          if (!r.ok) setError((await r.json()).detail ?? `HTTP ${r.status}`);
          await refresh();
        }} style={{ color: "#ffb07d" }}>
          {tr("Stop all MicroDuck Colab jobs", "停止所有 MicroDuck Colab 任务")}
        </button>}
      {jobs.length === 0 && <p>{tr("No cloud jobs yet.", "暂无云端任务。")}</p>}
      {jobs.map(j => <div key={j.id} style={{ borderTop: "1px solid #384252", padding: "12px 0" }}>
        <strong>{j.id}</strong> · {j.gpu} · {j.remote_state ?? j.state}
        {j.message && <p>{j.message}</p>}
        {["running", "starting", "allocating", "detached"].includes(j.state) ? <button onClick={() => void stop(j.id)}>{tr("Stop and release GPU", "停止并释放 GPU")}</button> : null}{" "}
        {j.state === "done" && <a href={`${LAB_HTTP}/cloud/colab/jobs/${j.id}/policy.onnx`} style={{ color: "#8cc8ff" }}>{tr("Download ONNX", "下载 ONNX")}</a>}
        {j.released === false && <p style={{ color: "#ffb07d" }}>{tr("Release was not confirmed. Check Colab sessions.", "未确认释放，请检查 Colab 会话。")}</p>}
      </div>)}
    </section>
    {error && <p role="alert" style={{ color: "#ff9f89" }}>{error}</p>}
  </main>;
}
