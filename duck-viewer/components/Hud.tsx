"use client";

// Overlay: connection state, per-duck stats (polled from the frame ref at 4 Hz
// so the 25 Hz stream never causes React re-renders), a system-stats strip,
// helper spawn/remove buttons, and the command bar.

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteColabToken,
  deleteHfToken,
  duckRowKeys,
  fetchColabSettings,
  fetchHfSettings,
  finishColabAuth,
  killAllColabSessions,
  saveHfToken,
  startColabAuth,
  type ColabSettings,
  type DuckFrame,
  type Frame,
  type HfSettings,
  type LabClient,
} from "@/lib/lab";
import { loadJSON, saveJSON } from "@/lib/persist";
import { useI18n } from "@/lib/i18n";
import { setSelectedDuck, useSelectedDuck } from "@/lib/select";
import { setCloudSettingsOpen, setDuckLabels, setHudRight, useCloudSettingsOpen } from "@/lib/ui";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** ⚙ settings: connect a Hugging Face token (BYOK). The token is posted to
 *  the LOCAL lab once, validated against whoami(), stored 0600 on the user's
 *  machine, and only a mask ever comes back — this modal never holds a live
 *  token after Save resolves. Unlocks the coming real-GPU training step
 *  (microduck_rl on HF Jobs, the user's own account and billing). */
function HfSettingsModal({ onClose }: { onClose: () => void }) {
  const { tr } = useI18n();
  const [settings, setSettings] = useState<HfSettings | null>(null);
  const [colabSettings, setColabSettings] = useState<ColabSettings | null>(null);
  const [colabBusy, setColabBusy] = useState(false);
  const [colabError, setColabError] = useState<string | null>(null);
  const [colabCode, setColabCode] = useState("");
  const [authStep, setAuthStep] = useState<"idle" | "awaiting_code">("idle");
  const [authUrl, setAuthUrl] = useState<string | null>(null);
  const [killMsg, setKillMsg] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Only Escape. The rest of the keyboard is claimed by the data-modal marker
  // on the backdrop below: stopPropagation CANNOT do it from here, because the
  // scene's shortcut listener is on window/capture too and registered first,
  // and stopPropagation never affects same-element listeners — the earlier
  // attempt let Backspace delete the duck behind the modal anyway, while
  // blocking React's own onKeyDown (Enter-to-save) on the way down.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    let alive = true;
    fetchHfSettings()
      .then((s) => alive && setSettings(s))
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)));
    fetchColabSettings()
      .then((c) => alive && setColabSettings(c))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      setSettings(await saveHfToken(token));
      setToken("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    setError(null);
    try {
      setSettings(await deleteHfToken());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const startColabLogin = async () => {
    setColabBusy(true);
    setColabError(null);
    try {
      const { authUrl: url } = await startColabAuth();
      setAuthUrl(url);
      setAuthStep("awaiting_code");
      window.open(url, "_blank");
    } catch (e) {
      setColabError(e instanceof Error ? e.message : String(e));
    } finally {
      setColabBusy(false);
    }
  };

  const finishColabLogin = async () => {
    if (!colabCode.trim()) return;
    setColabBusy(true);
    setColabError(null);
    try {
      const res = await finishColabAuth(colabCode);
      setColabSettings(res);
      setAuthStep("idle");
      setColabCode("");
    } catch (e) {
      setColabError(e instanceof Error ? e.message : String(e));
    } finally {
      setColabBusy(false);
    }
  };

  const disconnectColab = async () => {
    setColabBusy(true);
    setColabError(null);
    try {
      await deleteColabToken();
      setColabSettings({ configured: false });
    } catch (e) {
      setColabError(e instanceof Error ? e.message : String(e));
    } finally {
      setColabBusy(false);
    }
  };

  const handleKillAll = async () => {
    if (!confirm(tr("Stop all active Colab cloud VMs immediately to prevent charges?", "确定立即停止所有正在运行的 Colab 云端虚拟机以防扣费？"))) return;
    setColabBusy(true);
    setKillMsg(null);
    setColabError(null);
    try {
      const res = await killAllColabSessions();
      setKillMsg(res.message);
    } catch (e) {
      setColabError(e instanceof Error ? e.message : String(e));
    } finally {
      setColabBusy(false);
    }
  };

  return (
    <div
      onClick={onClose}
      // data-policy-ui: the armed-chip global pointerdown handler skips panel
      // UI. Without it, clicking inside this modal while a chip is armed
      // assigns that policy to whichever duck sits behind the overlay
      // (nearestDuck projects screen positions; it can't see the modal).
      data-policy-ui
      // data-modal: the whole-keyboard gate the scene reads (lib/ui.ts). It has
      // to sit on a node that exists only while this dialog does, so it can
      // never be left armed or disarmed by mistake.
      data-modal
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        zIndex: 1100, // above the drag ghost (100) and the tooltips (1000)
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#161b26",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 10,
          padding: "14px 16px",
          width: 340,
          fontFamily: mono,
          fontSize: 11,
          color: "#aab3c0",
          lineHeight: 1.5,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", marginBottom: 8 }}>
          <span style={{ color: "#dfe5ee", fontSize: 12, fontWeight: 700 }}>☁ {tr("cloud compute accounts", "云算力账户")}</span>
          <span style={{ flex: 1 }} />
          <button
            onClick={onClose}
            style={{ background: "none", border: "none", color: "#8b93a3", cursor: "pointer", fontFamily: mono }}
          >
            ✕
          </button>
        </div>
        <div style={{ color: "#dfe5ee", marginBottom: 4 }}>🤗 Hugging Face</div>
        {settings?.configured ? (
          <>
            <div style={{ marginBottom: 8 }}>
              {tr("connected as", "已连接账号")} <span style={{ color: "#7ab87a" }}>{settings.username}</span>{" "}
              <span style={{ color: "#566072" }}>({settings.masked})</span>
            </div>
            <button
              onClick={disconnect}
              disabled={busy}
              style={{
                background: "#1c2230",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 6,
                color: "#e07a5f",
                cursor: "pointer",
                fontFamily: mono,
                fontSize: 11,
                padding: "3px 10px",
              }}
            >
              {tr("disconnect", "断开连接")}
            </button>
          </>
        ) : (
          <>
            <div style={{ marginBottom: 8 }}>
              {tr(
                "Paste an access token to unlock GPU training on HF Jobs — your own account and billing. Create one at ",
                "粘贴访问令牌以启用 HF Jobs GPU 训练，使用你自己的账号和账单。请在此创建："
              )}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noreferrer"
                style={{ color: "#7db8d8" }}
              >
                hf.co/settings/tokens
              </a>{" "}
              {tr(" (write access).", "（需要写入权限）。")}
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="password"
                // Not a site login: keep password managers from offering to
                // save the token and later autofilling a revoked one.
                autoComplete="off"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && token && !busy && save()}
                placeholder="hf_…"
                autoFocus
                style={{
                  flex: 1,
                  background: "#0f131c",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 6,
                  color: "#dfe5ee",
                  fontFamily: mono,
                  fontSize: 11,
                  padding: "4px 8px",
                  outline: "none",
                }}
              />
              <button
                onClick={save}
                disabled={busy || !token}
                style={{
                  background: "#243247",
                  border: "1px solid #7db8d8",
                  borderRadius: 6,
                  color: "#cfe4f5",
                  cursor: busy || !token ? "default" : "pointer",
                  opacity: busy || !token ? 0.5 : 1,
                  fontFamily: mono,
                  fontSize: 11,
                  padding: "3px 10px",
                }}
              >
                {busy ? tr("checking…", "验证中…") : tr("save", "保存")}
              </button>
            </div>
            <div style={{ color: "#566072", marginTop: 6 }}>
              {tr(
                "stored only on this machine (hf-token.json, gitignored) — never sent anywhere but huggingface.co.",
                "仅保存在本机（hf-token.json，已忽略 Git），除 huggingface.co 外不会发送到其他地方。"
              )}
            </div>
          </>
        )}
        {error && <div style={{ color: "#e07a5f", marginTop: 8 }}>{error}</div>}

        {/* 分隔线 */}
        <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "14px 0 10px 0" }} />

        {/* ☁ Google Colab 区块 */}
        <div style={{ color: "#dfe5ee", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
          <span>☁ Google Colab</span>
          <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 4, background: "rgba(122,184,122,0.18)", color: "#7ab87a" }}>
            T4 / L4 / A100 / H100
          </span>
          <span style={{ fontSize: 9, color: "#6c788d" }}>
            动态计费
          </span>
        </div>

        {colabSettings?.configured ? (
          <div>
            <div style={{ marginBottom: 6, display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
              <div>
                {tr("connected as", "已连接账号")}{" "}
                <span style={{ color: "#7ab87a" }}>{colabSettings.email}</span>
              </div>
              <button
                onClick={disconnectColab}
                disabled={colabBusy}
                style={{
                  background: "#1c2230",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 6,
                  color: "#e07a5f",
                  cursor: "pointer",
                  fontFamily: mono,
                  fontSize: 10,
                  padding: "2px 8px",
                }}
              >
                {tr("disconnect", "断开连接")}
              </button>
            </div>
            <div style={{ color: "#8b93a3", fontSize: 10, lineHeight: 1.4, marginBottom: 8 }}>
              {tr(
                "✓ Cloud compute ready. You can switch between T4/L4/A100 models in Teach panel. Checkpoints will auto-backup to Google Drive.",
                "✓ 云端算力已就绪。训练面板支持自由选择 T4/L4/A100 等显卡型号，训练产物自动备份至 Google Drive 云盘。"
              )}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ marginBottom: 6, color: "#aab3c0", fontSize: 10 }}>
              {tr(
                "Connect your Google account to run fast GPU training using your free or Pro Colab compute units.",
                "连接你的 Google 账号，即可调度云端免费或 Pro 会员的 GPU 算力点数进行加速训练。"
              )}
            </div>

            {authStep === "idle" ? (
              <button
                onClick={startColabLogin}
                disabled={colabBusy}
                style={{
                  background: "#243247",
                  border: "1px solid #7db8d8",
                  borderRadius: 6,
                  color: "#cfe4f5",
                  cursor: colabBusy ? "default" : "pointer",
                  fontFamily: mono,
                  fontSize: 11,
                  padding: "4px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <span>🔗</span>
                <span>{colabBusy ? tr("generating auth link…", "生成授权链接中…") : tr("Sign in with Google", "登录并授权 Google 账号")}</span>
              </button>
            ) : (
              <div style={{ background: "#11141c", border: "1px solid rgba(125,184,216,0.25)", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ color: "#7db8d8", fontSize: 10, fontWeight: 600, marginBottom: 4 }}>
                  {tr("Step 1: Authorize in browser", "步骤 1：在浏览器中完成授权")}
                </div>
                {authUrl && (
                  <a
                    href={authUrl}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      display: "inline-block",
                      color: "#9fb4d8",
                      fontSize: 10,
                      textDecoration: "underline",
                      marginBottom: 8,
                    }}
                  >
                    {tr("Click here if the Google login tab didn't open ↗", "如果授权页面未自动打开，请点击此处 ↗")}
                  </a>
                )}
                <div style={{ color: "#7db8d8", fontSize: 10, fontWeight: 600, marginBottom: 4 }}>
                  {tr("Step 2: Paste the authorization code below", "步骤 2：将页面上显示的授权码粘贴在下方")}
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    type="text"
                    autoComplete="off"
                    value={colabCode}
                    onChange={(e) => setColabCode(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && colabCode && !colabBusy && finishColabLogin()}
                    placeholder="4/0A..."
                    style={{
                      flex: 1,
                      background: "#0a0c10",
                      border: "1px solid rgba(255,255,255,0.15)",
                      borderRadius: 4,
                      color: "#dfe5ee",
                      fontFamily: mono,
                      fontSize: 11,
                      padding: "4px 6px",
                      outline: "none",
                    }}
                  />
                  <button
                    onClick={finishColabLogin}
                    disabled={colabBusy || !colabCode.trim()}
                    style={{
                      background: "#243247",
                      border: "1px solid #7db8d8",
                      borderRadius: 4,
                      color: "#cfe4f5",
                      cursor: colabBusy || !colabCode.trim() ? "default" : "pointer",
                      opacity: colabBusy || !colabCode.trim() ? 0.5 : 1,
                      fontFamily: mono,
                      fontSize: 11,
                      padding: "3px 8px",
                    }}
                  >
                    {colabBusy ? tr("verifying…", "验证中…") : tr("confirm", "完成绑定")}
                  </button>
                </div>
              </div>
            )}
            <div style={{ color: "#566072", fontSize: 10, marginTop: 6 }}>
              {tr(
                "credentials stored only on your machine (~/.config/colab-cli/token.json) — never shared.",
                "凭证仅保存在本机 (~/.config/colab-cli/token.json)，不会发送给任何第三方。"
              )}
            </div>
          </div>
        )}

        {colabError && <div style={{ color: "#e07a5f", marginTop: 6, fontSize: 10 }}>{colabError}</div>}
        {killMsg && <div style={{ color: "#7ab87a", marginTop: 6, fontSize: 10 }}>{killMsg}</div>}

        {/* 🚨 紧急防扣费熔断专区 */}
        <div style={{ height: 1, background: "rgba(255,255,255,0.06)", margin: "10px 0 8px 0" }} />
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(224,122,95,0.08)", border: "1px solid rgba(224,122,95,0.2)", borderRadius: 6, padding: "6px 8px" }}>
          <div>
            <div style={{ color: "#e07a5f", fontSize: 10, fontWeight: 600 }}>
              🛡 {tr("Cost & VM Safety Kill-Switch", "算力防扣费安全熔断")}
            </div>
            <div style={{ color: "#8b93a3", fontSize: 9 }}>
              {tr("Stop all cloud VMs and cut off billing immediately", "强制关停所有云端虚拟机，立即切断计费")}
            </div>
          </div>
          <button
            onClick={handleKillAll}
            disabled={colabBusy}
            style={{
              background: "#321d1d",
              border: "1px solid #e07a5f",
              borderRadius: 4,
              color: "#e07a5f",
              cursor: colabBusy ? "default" : "pointer",
              fontFamily: mono,
              fontSize: 10,
              padding: "3px 8px",
              fontWeight: 600,
              whiteSpace: "nowrap",
            }}
          >
            🛑 {tr("Kill All VMs", "一键关停全部")}
          </button>
        </div>
      </div>
    </div>
  );
}

/** cpu% → strip color: calm → amber >75 → red >90. */
function cpuColor(cpu: number): string {
  if (cpu > 90) return "#e07a5f";
  if (cpu > 75) return "#d8c97d";
  return "#8b93a3";
}

function cpuBar(cpu: number): string {
  const filled = Math.max(0, Math.min(4, Math.ceil(cpu / 25)));
  return "▮".repeat(filled) + "░".repeat(4 - filled);
}

/** Tiny 60-sample cpu history, sampled ~1/s from the 4 Hz poll. */
function CpuSparkline({ samples }: { samples: number[] }) {
  if (samples.length < 2) return null;
  const w = 46, h = 9;
  const pts = samples
    .map((v, i) => {
      const x = (i / (samples.length - 1)) * w;
      const y = h - 1 - (Math.min(100, Math.max(0, v)) / 100) * (h - 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      width={w}
      height={h}
      style={{ display: "inline-block", verticalAlign: "middle", opacity: 0.8 }}
    >
      <polyline points={pts} fill="none" stroke="#57627a" strokeWidth={1} />
    </svg>
  );
}

function trainFpsLabel(fps: number | null, isZh = false): string {
  if (fps == null) return "—";
  if (fps >= 1000) return `${(fps / 1000).toFixed(1)}k ${isZh ? "步/秒" : "steps/s"}`;
  return `${Math.round(fps)} ${isZh ? "步/秒" : "steps/s"}`;
}

/** Finished-run badge for the stats strip's "train …" cell. */
const TRAIN_STATE_BADGE: Record<"done" | "stopped" | "failed", string> = {
  done: "✔ done",
  stopped: "■ stopped",
  failed: "✗ failed",
};

/** 471_552 → "472k", 1_500_000 → "1.5M" — compact steps for the trainee row. */
function abbrevSteps(n: number): string {
  if (n >= 1e6) {
    const m = n / 1e6;
    return `${m >= 10 ? Math.round(m) : Math.round(m * 10) / 10}M`;
  }
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return `${Math.round(n)}`;
}

/** 47 → "47s", 312 → "5m", 5_580 → "1h33m" — how long the run has been
 *  training, for the stats strip. Minutes drop the seconds: at a glance
 *  "5m" answers "how long has this been going", and the strip already
 *  ticks via steps/s. */
function abbrevElapsed(s: number): string {
  if (s < 60) return `${Math.floor(s)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`;
}

/** No frame for this long ⇒ the stream is stalled, however open the socket
 *  looks. ~75 missed frames at the server's 25 Hz. Measured, not guessed: at
 *  2 s this fired constantly while a teach run had the machine at 98% cpu and
 *  the lab loop was merely starved, not dead. 3 s still catches a real
 *  stoppage in a couple of seconds — the one that prompted this ran 9 s. */
const STALL_MS = 3000;

/** The corner badge. "live" has to mean frames are ARRIVING, not merely that
 *  the WebSocket is open — a lab whose duck loop died kept the socket up and
 *  the badge sat green over a frozen, empty scene. */
function linkBadge(connected: boolean, stalled: boolean, isZh = false) {
  if (!connected)
    return { dot: "○", label: isZh ? "离线" : "offline", color: "#e07a5f",
             title: isZh ? "未连接到实验室" : "not connected to the lab" };
  if (stalled)
    return { dot: "●", label: isZh ? "卡顿" : "stalled", color: "#d8c97d",
             title: isZh ? "已连接，但 3 秒未收到画面，仿真循环可能已停止"
               : "connected, but no frames for 3s — the lab is still there, its duck loop may have stopped" };
  return { dot: "●", label: isZh ? "实时" : "live", color: "#7dd87d",
           title: isZh ? "正在接收画面" : "frames arriving" };
}

/** One duck's forward speed: "0.21 / 0.45" — achieved over asked-for, m/s.
 *  The PAIR is the point: policies here reliably deliver about half the speed
 *  they are commanded, and that gap is the most informative number on the
 *  row. Trick policies run a pinned-zero command (the server sends cmdSpeed
 *  null), so they show the achieved figure alone — "0.00 / 0.00" under a
 *  backflip is noise. "—" is the single frame after an episode reset, before
 *  the averaging window has a sample. */
function SpeedCell({ d }: { d: DuckFrame }) {
  if (d.speed == null) return <span style={{ color: "#566072" }}>—</span>;
  return (
    <>
      <span style={{ color: "#7db8d8" }}>{d.speed.toFixed(2)}</span>
      {d.cmdSpeed != null && (
        <span style={{ color: "#8b93a3" }}>{` / ${d.cmdSpeed.toFixed(2)}`}</span>
      )}
    </>
  );
}

/** Trick policies (teach runs, the 🎓 trainee, 🤝 helpers) are scored on
 *  their own recipe — the walking r̄ is meaningless for them. */
function isTrickDuck(d: DuckFrame): boolean {
  return (
    d.name.startsWith("teach-") || d.name.startsWith("🎓") || d.name.startsWith("🤝")
  );
}

function RowButton({
  label,
  title,
  color,
  disabled,
  onClick,
}: {
  label: string;
  title: string;
  color: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      style={{
        width: 18,
        height: 16,
        padding: 0,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(255,255,255,0.05)",
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: 4,
        color,
        fontFamily: mono,
        fontSize: 10,
        lineHeight: 1,
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.35 : 1,
      }}
    >
      {label}
    </button>
  );
}

export function Hud({
  clientRef,
  connected,
  error,
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
  connected: boolean;
  error: string | null;
}) {
  const { isZh, tr } = useI18n();
  const [frame, setFrame] = useState<Frame | null>(null);
  // Collapsed ⇄ open state of the top-left stats panel (the bottom-left cmd
  // bar is unaffected). Persisted like the PolicyPanel/TeachPanel toggles.
  // Hooks below run regardless of `open` so the poll keeps hook order stable.
  const [open, setOpen] = useState(() => loadJSON("hudOpen", true));
  const settingsOpen = useCloudSettingsOpen();
  // Stable: this HUD re-renders ~4x/s, and an inline arrow would make the
  // modal's keyboard-gate effect tear down and re-run on every one of them.
  const closeSettings = useCallback(() => setCloudSettingsOpen(false), []);
  // Same treatment for the bottom-left camera-help bar — it's pure reference
  // text, so folding it away frees corner space (and the Next dev badge sits
  // right under it in dev). Starts collapsed: first sight of the scene should
  // be ducks, not a key list. Unconditional hook: order stays stable.
  const [cmdBarOpen, setCmdBarOpen] = useState(() => loadJSON("cmdBarOpen", false));
  // Floating duck name labels on/off — persisted here, mirrored into the
  // shared ui store so every Duck inside the Canvas reacts live.
  const [labels, setLabels] = useState(() => loadJSON("duckLabels", true));
  // In embedded panes the surrounding app may keep keyboard focus for itself —
  // the page can't fix that, but it can at least SAY so (cmd-bar hint below).
  const [pageFocused, setPageFocused] = useState(true);
  const [stalled, setStalled] = useState(false);
  const cpuHistory = useRef<number[]>([]);
  const lastSample = useRef(0);
  useEffect(() => {
    const id = setInterval(() => {
      const f = clientRef.current?.frame ?? null;
      setFrame(f);
      setPageFocused(document.hasFocus());
      const now = Date.now();
      const seen = clientRef.current?.lastFrameAt ?? 0;
      setStalled(seen > 0 && now - seen > STALL_MS);
      if (f?.stats && now - lastSample.current >= 1000) {
        lastSample.current = now;
        cpuHistory.current.push(f.stats.cpu);
        if (cpuHistory.current.length > 60) cpuHistory.current.shift();
      }
    }, 250);
    return () => clearInterval(id);
  }, [clientRef]);

  useEffect(() => saveJSON("hudOpen", open), [open]);
  useEffect(() => saveJSON("cmdBarOpen", cmdBarOpen), [cmdBarOpen]);
  useEffect(() => {
    saveJSON("duckLabels", labels);
    setDuckLabels(labels);
  }, [labels]);

  const panel: React.CSSProperties = {
    position: "absolute",
    // Above the ducks' floating DOM labels (drei Html, zIndexRange [10, 0]) —
    // matching AnimPanel; labels must never scribble over panel text.
    zIndex: 20,
    background: "rgba(14, 16, 20, 0.82)",
    border: "1px solid rgba(255,255,255,0.09)",
    borderRadius: 10,
    padding: "10px 12px",
    color: "#e8e6e1",
    fontFamily: mono,
    fontSize: 12,
    lineHeight: 1.55,
    backdropFilter: "blur(6px)",
  };

  // Publish this panel's right edge into the shared ui store so the
  // top-center 🎥/📷 capture panel can dodge a wide HUD (long duck names
  // reached right under its buttons). One callback ref serves both the open
  // panel and the collapsed pill — only one is mounted at a time.
  const hudRO = useRef<ResizeObserver | null>(null);
  const hudEdgeRef = useCallback((el: HTMLElement | null) => {
    hudRO.current?.disconnect();
    hudRO.current = null;
    if (!el) {
      setHudRight(0);
      return;
    }
    const publish = () => setHudRight(el.getBoundingClientRect().right);
    publish();
    hudRO.current = new ResizeObserver(publish);
    hudRO.current.observe(el);
  }, []);

  const stats = frame?.stats;
  const training = frame?.training ?? null;
  const restarting = training?.restarting ?? false;
  const rowKeys = frame ? duckRowKeys(frame.ducks) : [];
  const link = linkBadge(connected, stalled, isZh);
  // Stage selection (click a duck / a row): the selected row echoes the amber
  // ring under the duck, and Delete removes it.
  const selectedDuck = useSelectedDuck();

  return (
    <>
      {settingsOpen && <HfSettingsModal onClose={closeSettings} />}
      {open ? (
      <div
        ref={hudEdgeRef}
        // data-policy-ui: the armed-chip pointerdown handler skips panel UI.
        // nearestDuck is pure projection with an 80px radius, so without this
        // a click on a HUD button assigns the armed policy to whatever duck
        // happens to project behind the panel.
        data-policy-ui
        style={{ ...panel, top: 14, left: 14, minWidth: 240 }}
      >
        <div
          style={{
            fontSize: 13,
            fontWeight: 700,
            marginBottom: 4,
            display: "flex",
            alignItems: "center",
          }}
        >
          <span style={{ flex: 1 }}>
            🦆 {tr("duck lab", "鸭子实验室")}{" "}
            {/* 跳转到全新 /sim 世界仿真页面 */}
            <Link
              href="/sim"
              title={tr("the world page — rooms, sensors, brains", "世界模式 — 房间、传感器、视觉大脑与足球")}
              style={{ color: "#7db8d8", fontWeight: 500, textDecoration: "none", marginLeft: 8, fontSize: 11 }}
            >
              {tr("sim world →", "世界模式 →")}
            </Link>
          </span>
          <span style={{ color: link.color }} title={link.title}>
            {link.dot} {link.label}
          </span>
          <button
            onClick={() => setLabels((v) => !v)}
            title={labels ? tr("hide duck name labels", "隐藏鸭子名称") : tr("show duck name labels", "显示鸭子名称")}
            style={{
              background: "none",
              border: "none",
              color: labels ? "#8b93a3" : "#566072",
              cursor: "pointer",
              fontFamily: mono,
              fontSize: 12,
              padding: "0 4px",
              marginLeft: 10,
              // Emoji ignore CSS color — dim + desaturate carries the state.
              opacity: labels ? 1 : 0.45,
              filter: labels ? "none" : "grayscale(1)",
            }}
          >
            🏷
          </button>
          {training?.status === "training" && training?.backend === "colab" && (
            <button
              onClick={async () => {
                if (confirm(tr("Stop Colab training and kill cloud GPU immediately?", "确定立即停止 Colab 训练并释放云端 GPU 虚拟机？"))) {
                  await killAllColabSessions();
                }
              }}
              title={tr("Emergency kill cloud GPU to stop billing", "紧急释放云端 GPU 停止计费")}
              style={{
                background: "rgba(224, 122, 95, 0.2)",
                border: "1px solid #e07a5f",
                color: "#e07a5f",
                borderRadius: 4,
                cursor: "pointer",
                fontFamily: mono,
                fontSize: 10,
                padding: "1px 5px",
                marginLeft: 4,
                fontWeight: 600,
              }}
            >
              🛑 {tr("Kill GPU", "释放GPU")}
            </button>
          )}
          <button
            onClick={() => setCloudSettingsOpen(true)}
            title={tr("Cloud compute accounts — Google Colab & Hugging Face", "云算力账户—Google Colab 与 Hugging Face")}
            style={{
              background: "rgba(125,184,216,0.12)",
              border: "1px solid rgba(125,184,216,0.3)",
              borderRadius: 4,
              color: "#cfe4f5",
              cursor: "pointer",
              fontFamily: mono,
              fontSize: 10,
              padding: "2px 7px",
              marginLeft: 6,
              display: "inline-flex",
              alignItems: "center",
              gap: 3,
              whiteSpace: "nowrap",
            }}
          >
            <span>☁</span>
            <span>{tr("Cloud Accounts", "云算力账户")}</span>
          </button>
          <button
            onClick={() => setOpen(false)}
            title={tr("collapse", "收起")}
            style={{
              background: "none",
              border: "none",
              color: "#8b93a3",
              cursor: "pointer",
              fontFamily: mono,
              fontSize: 12,
              padding: "0 4px",
              marginLeft: 10,
            }}
          >
            —
          </button>
        </div>
        {error && <div style={{ color: "#e07a5f", maxWidth: 260 }}>{error}</div>}
        {frame && (
          <table style={{ borderSpacing: "10px 1px", marginLeft: -10 }}>
            <thead>
              <tr style={{ color: "#8b93a3", textAlign: "left" }}>
                <th>{tr("policy", "策略")}</th>
                <th
                  title="forward speed in metres per second, averaged over the
 last half second — what it manages / what it was asked for"
                >
                  m/s
                </th>
                <th>{tr("t", "时间")}</th>
                <th>{tr("falls", "跌倒")}</th>
                <th title={tr("average reward", "平均奖励")}>r̄</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {frame.ducks.map((d, i) => {
                const trick = isTrickDuck(d);
                // While training runs, surface live progress right in the
                // roster: "🎓 Spin in place · 471k/1.5M". Finished runs
                // render the streamed name verbatim — the server relabels
                // the trainee to "<emoji> <run-short> <mark>" (e.g.
                // "🦩 one_leg-22f079 ✔") the moment a job ends, so 🎓 means
                // "actively training" and nothing else. Counts are OVERALL
                // across a curriculum chain (falling back to per-stage for
                // single runs) — a per-stage counter here looked like the
                // run reset at every stage handoff.
                const name =
                  d.id === "trainee" && training?.status === "training"
                    ? `🎓 ${training.behavior.title} · ${abbrevSteps(
                        training.progress.overallSteps ??
                          training.progress.steps ??
                          0,
                      )}/${abbrevSteps(
                        training.progress.overallTotal ??
                          training.progress.total ??
                          0,
                      )}`
                    : d.name;
                // Trainee row grows a ＋ (spawn helper) while training runs —
                // removal is pointless there (the server refuses it), so no ✕.
                // Every OTHER row gets a ✕ (same remove_duck message for all);
                // helpers keep their restart pause since removing one mid-warm-
                // restart is refused server-side anyway.
                const isHelper = d.id.startsWith("helper");
                // At the helper cap the server refuses the spawn with a toast
                // that's easy to miss — disable the ＋ so it can't read as
                // "button does nothing".
                const helperCap = training?.maxHelpers ?? 6;
                const atHelperCap = (training?.helpers ?? 0) >= helperCap;
                const action =
                  d.id === "trainee" && training?.status === "training" ? (
                    <RowButton
                      label="＋"
                      title={
                        atHelperCap
                          ? tr(`helper cap (${helperCap})`, `辅助鸭上限（${helperCap}）`)
                          : restarting
                            ? tr("loading…", "加载中…")
                            : tr("add a helper — another viewer of the same live policy (does not change training speed)", "添加辅助鸭—展示同一实时策略（不影响训练速度）")
                      }
                      color="#7db8d8"
                      disabled={restarting || atHelperCap}
                      onClick={() => clientRef.current?.sendSpawnHelper()}
                    />
                  ) : (
                    <RowButton
                      label="✕"
                      title={
                        isHelper && restarting
                          ? tr("restarting…", "重启中…")
                          : isHelper
                            ? tr("remove this helper", "移除这只辅助鸭")
                            : tr("remove this duck from the lab", "从实验室移除这只鸭子")
                      }
                      color="#e0a08f"
                      disabled={isHelper && restarting}
                      onClick={() => clientRef.current?.sendRemoveDuck(d.id)}
                    />
                  );
                const isSelected = d.id === selectedDuck;
                return (
                  // keyed by stable id (dedup-qualified) — several ducks can
                  // run (and be named after) the same policy since assignment
                  // landed. Clicking a row (toggle-)selects the duck, same as
                  // clicking it on the stage; amber echoes the floor ring.
                  <tr
                    key={rowKeys[i]}
                    onClick={(e) => {
                      // the ✕/＋ buttons keep their own click
                      if ((e.target as HTMLElement).closest("button")) return;
                      setSelectedDuck(isSelected ? null : d.id);
                    }}
                    title={isSelected ? tr("selected — ⌫ removes it", "已选中—按 ⌫ 移除") : tr("click to select", "点击选中")}
                    style={{ cursor: "pointer" }}
                  >
                    <td
                      style={{
                        whiteSpace: "nowrap",
                        color: isSelected ? "#e8b24a" : undefined,
                        fontWeight: isSelected ? 700 : undefined,
                      }}
                    >
                      {isSelected ? "▸ " : ""}
                      {name}
                      {d.hold && (
                        <div style={{ color: d.hold.success ? "#7dd87d" : "#e8b24a", fontSize: 11 }}>
                          {tr(d.name.includes("single_leg_hop") || d.name.includes("long_jump") ? "two-foot hopping" : "single-leg hold",
                              d.name.includes("single_leg_hop") ? "单脚跳跃" : (d.name.includes("long_jump") ? "双脚连续跳跃" : "单脚保持"))} {d.name.includes("long_jump") ? `${Math.round(d.hold.seconds)} / ${Math.round(d.hold.target)} 次` : `${d.hold.seconds.toFixed(2)} / ${d.hold.target.toFixed(0)}s`}
                          {d.hold.success ? tr(" · achieved this episode ✓", " · 本回合已达标 ✓") : ""}
                        </div>
                      )}
                    </td>
                    <td
                      style={{
                        whiteSpace: "nowrap",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      <SpeedCell d={d} />
                    </td>
                    <td>{(d.step / 50).toFixed(0)}s</td>
                    <td style={{ color: d.falls ? "#e07a5f" : "#7dd87d" }}>{d.falls}</td>
                    <td
                      style={trick ? { color: "#566072" } : undefined}
                      title={
                        trick
                          ? "r̄ scores the WALKING recipe — trick policies score low here by design"
                          : undefined
                      }
                    >
                      {d.rew.toFixed(1)}
                    </td>
                    <td style={{ padding: 0 }}>{action}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {stats && (
          <div
            style={{
              marginTop: 6,
              paddingTop: 6,
              borderTop: "1px solid rgba(255,255,255,0.08)",
              fontSize: 10,
              color: "#8b93a3",
              whiteSpace: "nowrap",
              display: "flex",
              alignItems: "center",
              gap: 5,
            }}
          >
            <span style={{ color: cpuColor(stats.cpu) }}>
              cpu {cpuBar(stats.cpu)} {Math.round(stats.cpu)}%
            </span>
            <CpuSparkline samples={cpuHistory.current} />
            <span>· {tr("mem", "内存")} {Math.round(stats.mem)}%</span>
            {/* live steps/s only while the trainer actually runs — a finished
                job kept showing its last rate, which read as "still going"
                (and the server only nulls trainFps after its next restart).
                The wall clock rides along in both states: overallElapsed
                spans stage handoffs and warm restarts (unlike elapsed_s)
                and freezes at finish, so "✔ done · 1h07m" is the run's
                total training time. */}
            {training &&
              (training.status === "training" || training.restarting ? (
                <span>
                  · {tr("train", "训练")} {trainFpsLabel(stats.trainFps, isZh)}
                  {training.progress.overallElapsed != null &&
                    ` · ${abbrevElapsed(training.progress.overallElapsed)}`}
                </span>
              ) : (
                <span style={{ color: "#566072" }}>
                  · {tr("train", "训练")} {{
                    done: tr("✔ done", "✔ 已完成"),
                    stopped: tr("■ stopped", "■ 已停止"),
                    failed: tr("✗ failed", "✗ 失败"),
                  }[training.status]}
                  {training.progress.overallElapsed != null &&
                    ` · ${abbrevElapsed(training.progress.overallElapsed)}`}
                </span>
              ))}
          </div>
        )}
      </div>
      ) : (
        // Collapsed: the whole stats panel folds into this pill (same pattern
        // as the collapsed 🧠 policies / 🎓 teach buttons), dot still live.
        <button
          ref={hudEdgeRef}
          onClick={() => setOpen(true)}
          style={{
            position: "absolute",
            zIndex: 20,
            top: 14,
            left: 14,
            background: "rgba(14,16,20,0.86)",
            color: "#e8e6e1",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 10,
            padding: "8px 12px",
            fontFamily: mono,
            fontSize: 12,
            cursor: "pointer",
            backdropFilter: "blur(6px)",
          }}
        >
          🦆 {tr("duck lab", "鸭子实验室")}{" "}
          <span style={{ color: link.color }} title={link.title}>
            {link.dot}
          </span>
        </button>
      )}

      {/* Viewport help. The keyboard flies the CAMERA (Maya/Blender-style) —
          ducks are driven by their RL policies alone (walking policies follow
          the server's auto demo script; trick policies do their trick).
          Collapsible like the panels above; the focus hint folds away with it
          (keys still work — the hint is a nicety, not a control). */}
      {cmdBarOpen ? (
        <div style={{ ...panel, bottom: 14, left: 14, maxWidth: 265 }}>
          <div style={{ display: "flex", alignItems: "flex-start" }}>
            <div style={{ color: "#8b93a3", flex: 1 }}>
              {/* Restart leads: it is the only key here that touches the SIM
                  rather than the view, and the panel grows upward from a fixed
                  bottom edge — so the last line is the one the `next dev`
                  badge sits on top of. Camera list keeps the tail. */}
              <div style={{ color: "#a5adbb", marginBottom: 3 }}>
                {tr("↺ R restart sim — every duck's episode from zero", "↺ R 重启仿真—所有鸭子从第 0 步开始")}
              </div>
              <div style={{ color: "#a5adbb", marginBottom: 3 }}>
                {tr("🖱 click a duck to select · ⌫ remove it · esc deselect", "🖱 点击鸭子选中 · ⌫ 移除 · Esc 取消选中")}
              </div>
              {tr(
                "🎥 drag orbit · scroll zoom · 2-finger swipe slide · A/D slide · W/S·↑↓ dolly · ←/→ orbit · Q/E up·down · Shift+R reset view",
                "🎥 拖动旋转 · 滚轮缩放 · 双指滑动平移 · A/D 平移 · W/S·↑↓ 推进 · ←/→ 旋转 · Q/E 升降 · Shift+R 重置视角"
              )}
            </div>
            <button
              onClick={() => setCmdBarOpen(false)}
              title={tr("collapse", "收起")}
              style={{
                background: "none",
                border: "none",
                color: "#8b93a3",
                cursor: "pointer",
                fontFamily: mono,
                fontSize: 12,
                padding: "0 4px",
                marginLeft: 10,
              }}
            >
              —
            </button>
          </div>
          {!pageFocused && (
            <div style={{ color: "#566072", marginTop: 3 }}>
              ⌨ {tr("click the scene to enable keys", "点击场景以启用键盘")}
            </div>
          )}
        </div>
      ) : (
        // Collapsed: compact pill, nudged right of the Next dev badge that
        // squats in the very corner during `next dev`.
        <button
          onClick={() => setCmdBarOpen(true)}
          title={tr("keyboard controls", "键盘操作")}
          style={{
            position: "absolute",
            zIndex: 20,
            bottom: 14,
            left: 56,
            background: "rgba(14,16,20,0.86)",
            color: "#e8e6e1",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 10,
            padding: "8px 12px",
            fontFamily: mono,
            fontSize: 12,
            cursor: "pointer",
            backdropFilter: "blur(6px)",
          }}
        >
          🎥 {tr("controls", "操作说明")}
        </button>
      )}
    </>
  );
}
