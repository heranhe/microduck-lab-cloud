"use client";

// Describe what to train in words. The model may only choose an existing recipe
// and tune the reward terms that already exist, or choreograph keyframes for the
// imitation recipe; nothing it writes is run, and nothing starts until the
// person has read the draft and pressed start.

import { useEffect, useState } from "react";
import { ServoChoice } from "./ServoChoice";
import { useI18n } from "@/lib/i18n";
import { planRequest, type Catalog, type Plan } from "@/lib/plans";
import { changedWeights, termLabel } from "@/lib/aiPlan";
import type { Servo } from "@/lib/servo";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
type AiSettings = { configured: boolean; model: string; base_url: string };

type Props = {
  robot: string;
  catalog: Catalog | null;
  servo: Servo;
  onServo: (servo: Servo) => void;
  online: boolean;
  /** A cloud job is running: starting a CPU run would be refused anyway. */
  blockedReason: string;
  onStarted: (title: string) => void;
};

export function AiPlanner({ robot, catalog, servo, onServo, online, blockedReason, onStarted }: Props) {
  const { tr } = useI18n();
  const [open, setOpen] = useState(false);
  const [settings, setSettings] = useState<AiSettings | null>(null);
  const [form, setForm] = useState({ base_url: "https://api.openai.com/v1", model: "", api_key: "" });
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [draft, setDraft] = useState<Plan | null>(null);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    if (!open || !online) return;
    let stale = false;
    planRequest<AiSettings>("/plans/ai/settings")
      .then((s) => { if (!stale) { setSettings(s); setForm((f) => ({ ...f, base_url: s.base_url || f.base_url, model: s.model })); } })
      .catch(() => { if (!stale) setSettings(null); });
    return () => { stale = true; };
  }, [open, online]);

  const run = async (name: string, work: () => Promise<void>) => {
    setBusy(name); setError("");
    try { await work(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(""); }
  };

  const saveSettings = () => run("settings", async () => {
    await planRequest("/plans/ai/settings", "PUT", form);
    setForm((f) => ({ ...f, api_key: "" }));
    setSettings(await planRequest<AiSettings>("/plans/ai/settings"));
  });
  const generate = () => run("draft", async () => {
    setDraft(null); setNotice("");
    const r = await planRequest<{ draft: Plan; notice: string }>("/plans/ai/draft", "POST", { prompt, robot });
    setDraft(r.draft); setNotice(r.notice);
  });
  const start = () => run("start", async () => {
    if (!draft?.local) return;
    const saved = await planRequest<Plan>("/plans", "POST", {
      ...draft, local: { ...draft.local, ...(robot === "microduck" ? { servo } : {}) },
    });
    await planRequest(`/plans/${saved.id}/launch`, "POST", { revision: saved.revision, compute: "local" });
    onStarted(draft.title);
    setDraft(null); setPrompt(""); setOpen(false);
  });

  const recipe = catalog?.behaviors.find((b) => b.id === draft?.local?.behavior);
  const changes = draft?.local ? changedWeights(recipe, draft.local.weights) : [];
  const clip = draft?.reference;
  const box: React.CSSProperties = { margin: "0 0 8px", padding: 8, borderRadius: 8, border: "1px solid rgba(255,255,255,0.09)", background: "rgba(10,13,18,0.58)" };
  const input: React.CSSProperties = { width: "100%", boxSizing: "border-box", background: "#12151b", color: "#e8e6e1", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, padding: "4px 7px", font: `11px ${mono}`, outline: "none" };
  const small: React.CSSProperties = { background: "#1c2230", color: "#9fd0e6", border: "1px solid rgba(112,184,216,0.4)", borderRadius: 6, padding: "3px 9px", font: `11px ${mono}`, cursor: "pointer" };

  return (
    <div style={{ margin: "0 0 8px" }}>
      <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)}
        style={{ border: 0, background: "none", color: "#d8c97d", cursor: "pointer", padding: 0, font: `11px ${mono}` }}>
        {open ? "▾" : "▸"} ✨ {tr("Describe what to train with AI", "用 AI 描述想训练的动作")}
      </button>
      {open && (
        <div style={{ ...box, marginTop: 6 }}>
          <div style={{ color: "#747e8d", fontSize: 10, marginBottom: 6 }}>
            {tr("AI picks a recipe and tunes existing reward terms, or choreographs keyframes. It never writes code, and nothing starts until you press start.",
              "AI 只会选择已有配方并调整已有的奖励项，或为新动作编排关键帧；不会写代码，也不会替你开始训练。")}
          </div>
          {!online ? (
            <div style={{ color: "#efb37d", fontSize: 10 }}>{tr("Start duck-lab first.", "请先启动 duck-lab。")}</div>
          ) : settings && !settings.configured ? (
            <>
              <div style={{ color: "#9fb4d8", fontSize: 10, marginBottom: 4 }}>{tr("Connect your own AI service (OpenAI-compatible, or a local one).", "连接你自己的 AI 服务（兼容 OpenAI 接口，或本机服务）。")}</div>
              <input aria-label="base url" style={input} value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} placeholder="https://api.openai.com/v1"/>
              <input aria-label="model" style={{ ...input, marginTop: 4 }} value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} placeholder={tr("Model name", "模型名称")}/>
              <input aria-label="api key" type="password" autoComplete="off" style={{ ...input, marginTop: 4 }} value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} placeholder={tr("API key (kept on this computer)", "API Key（只保存在本机）")}/>
              <button type="button" disabled={!form.model.trim() || busy === "settings"} style={{ ...small, marginTop: 6 }} onClick={saveSettings}>{tr("Save", "保存")}</button>
            </>
          ) : (
            <>
              <textarea aria-label={tr("What should it learn?", "想让它学什么？")} value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3}
                placeholder={tr("e.g. stand on the left leg for 5 seconds, head up, don't wobble", "例如：用左腿单脚站稳 5 秒，头保持抬起，不要乱晃")}
                style={{ ...input, resize: "vertical" }}/>
              <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 6 }}>
                <button type="button" disabled={!prompt.trim() || busy === "draft"} style={small} onClick={generate}>
                  {busy === "draft" ? tr("Thinking…", "生成中…") : tr("Generate plan", "生成方案")}
                </button>
                {settings && <span style={{ color: "#747e8d", fontSize: 10 }}>{settings.model}</span>}
              </div>
            </>
          )}
          {error && <div role="alert" style={{ color: "#ef957d", fontSize: 10, marginTop: 6, whiteSpace: "pre-wrap" }}>{error}</div>}
          {draft && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontWeight: 700 }}>{draft.title}</div>
              <div style={{ color: "#aab3c0", fontSize: 11, margin: "3px 0" }}>{draft.description}</div>
              <div style={{ color: "#8b93a3", fontSize: 10 }}>{tr("Success looks like", "成功的样子")}：{draft.success}</div>
              {clip && (
                <div style={{ color: "#9fd0e6", fontSize: 10, marginTop: 4 }}>
                  🎬 {tr(`AI choreographed ${clip.keys.length} keyframes over ${clip.duration}s; the robot learns to reproduce it (imitation training).`,
                    `AI 编排了 ${clip.keys.length} 个关键帧、时长 ${clip.duration} 秒；机器人会学着复现它（模仿训练）。`)}
                </div>
              )}
              {draft.local && (
                <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 4 }}>
                  {recipe?.emoji} {recipe?.title ?? draft.local.behavior} · {(draft.local.steps / 1e6).toFixed(1)}M {tr("steps", "步")}
                </div>
              )}
              {changes.length > 0 && (
                <div style={{ marginTop: 4, fontSize: 10 }}>
                  <div style={{ color: "#8b93a3" }}>{tr("Reward terms AI set", "AI 设置的奖励项")}</div>
                  {changes.map((c) => (
                    <div key={c.key} style={{ color: "#c5d9e4" }}>
                      {termLabel(recipe, c.key)} <span style={{ color: "#7db8d8" }}>{c.weight}</span>
                      <span style={{ color: "#747e8d" }}>{c.was == null ? tr(" (added)", "（新增）") : tr(` (was ${c.was})`, `（原为 ${c.was}）`)}</span>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ color: "#d8b46f", fontSize: 10, margin: "6px 0" }}>{notice}</div>
              {robot === "microduck" && <ServoChoice value={servo} onChange={onServo}/>}
              {blockedReason && <div style={{ color: "#e6a28f", fontSize: 10 }}>{blockedReason}</div>}
              <div style={{ display: "flex", gap: 6 }}>
                <button type="button" disabled={!!blockedReason || busy === "start"}
                  style={{ ...small, background: "#694318", color: "#ffe0a6", border: "1px solid #d39a48", fontWeight: 700 }} onClick={start}>
                  {busy === "start" ? tr("Starting…", "启动中…") : tr("▶ Save and start on this computer", "▶ 保存并开始本机训练")}
                </button>
                <button type="button" style={small} onClick={() => { setDraft(null); setNotice(""); }}>{tr("Discard", "放弃")}</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
