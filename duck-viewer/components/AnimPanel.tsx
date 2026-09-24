"use client";

// 🎬 animate (bottom-center): a keyframe animation editor for the robot.
//
// Pose the duck (sliders, or by dragging body parts in the 3D scene), key the
// poses on a timeline, scrub/play them back, save the clip. The saved JSON is
// the handoff to the imitation-RL side, which resamples it at 50 Hz and
// rewards a policy for tracking it — so the editor never invents a pose the
// contract can't express: joints are clamped to the MJCF servo limits, key
// times ascend from t = 0, and interpolation is linear in joint space (what
// the resampler does).
//
// Every pose shown here is forward kinematics from POST /pose on the server's
// scratch model — the lab ducks and their WS stream are untouched.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  animStore,
  animVersion,
  clampJoint,
  clipProblem,
  defaultPose,
  fetchJoints,
  keyAt,
  listClips,
  loadClip,
  newClip,
  NUM_JOINTS,
  PoseStreamer,
  putClip,
  removeClip,
  ROOT_SEL,
  round3,
  sampleClip,
  setAnimMeta,
  setAnimMode,
  setAnimVisible,
  setSelected,
  setSelectedRig,
  subscribeAnim,
  type AnimMode,
  withKey,
  type Clip,
  type JointsMeta,
  type Pose,
  type StoredClip,
} from "@/lib/anim";
import { LAB_HTTP } from "@/lib/lab";
import { loadJSON, saveJSON } from "@/lib/persist";
import { useI18n } from "@/lib/i18n";
import {
  RIG_CONTROLS,
  rigApply,
  rigBodies,
  rigBodyMap,
  rigMeasure,
  rigRange,
  rigVector,
  type RigVector,
} from "@/lib/rig";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
const GROUPS = ["left leg", "head + neck", "right leg"] as const;
const GROUPS_ZH: Record<string, string> = {
  "left leg": "左腿",
  "head + neck": "头部与颈部",
  "right leg": "右腿",
};

const RIG_ZH: Record<string, string> = {
  squat: "蹲起",
  lean: "前后倾",
  "L swing": "左腿摆动",
  "R swing": "右腿摆动",
  sway: "侧摆",
  stance: "站距",
  twist: "扭转",
  toes: "足尖",
  look: "视线",
};
const TRACK_PAD = 10; // px inset of the timeline track inside its box

const btn: React.CSSProperties = {
  background: "#1c2230",
  color: "#9fb4d8",
  border: "1px solid rgba(255,255,255,0.10)",
  borderRadius: 7,
  padding: "3px 8px",
  fontFamily: mono,
  fontSize: 11,
  cursor: "pointer",
  whiteSpace: "nowrap",
};

const field: React.CSSProperties = {
  background: "#12161e",
  color: "#e8e6e1",
  border: "1px solid rgba(255,255,255,0.10)",
  borderRadius: 6,
  padding: "3px 6px",
  fontFamily: mono,
  fontSize: 11,
};

export function AnimPanel() {
  const { isZh, tr } = useI18n();
  const [open, setOpen] = useState(() => loadJSON("animOpen", false));
  const [meta, setMeta] = useState<JointsMeta | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  // Unsaved work survives a refresh — an authored pose is expensive to redo.
  const [clip, setClip] = useState<Clip>(() => loadJSON<Clip | null>("animClip", null) ?? newClip(null));
  const [pose, setPose] = useState<Pose>(() => ({ joints: new Array(NUM_JOINTS).fill(0), rootPitch: 0 }));
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [clips, setClips] = useState<StoredClip[]>([]);
  const [browsing, setBrowsing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [poseErr, setPoseErr] = useState<string | null>(null);

  // 3D selection lives in the shared store (PoseDuck writes it on click).
  useSyncExternalStore(subscribeAnim, animVersion, () => 0);
  const selected = animStore.selected;

  // Latest-value mirrors for the callbacks that outlive a render: the rAF
  // playback loop, the timeline drag, and the 3D pose drag all need "the clip
  // as it is now", not as it was when their closure was made.
  const clipRef = useRef(clip);
  const poseRef = useRef(pose);
  const playheadRef = useRef(playhead);
  useEffect(() => {
    clipRef.current = clip;
    poseRef.current = pose;
    playheadRef.current = playhead;
  });

  // --- joint metadata (limits, defaults, body map) -------------------------
  useEffect(() => {
    if (!open || meta) return;
    let stale = false;
    fetchJoints()
      .then((m) => {
        if (stale) return;
        setMeta(m);
        setAnimMeta(m);
        setMetaErr(null);
        // A clip restored from before we knew the limits (or a fresh one) gets
        // the real DEFAULT_POSE now.
        setClip((c) => (c.keys.length === 1 && c.keys[0].joints.every((v) => v === 0)
          ? newClip(m, c.name)
          : c));
      })
      .catch((e) => !stale && setMetaErr(String(e?.message ?? e)));
    return () => {
      stale = true;
    };
  }, [open, meta]);

  useEffect(() => saveJSON("animOpen", open), [open]);
  useEffect(() => saveJSON("animClip", clip), [clip]);
  // The ghost duck only exists once we know the joint layout — with an
  // unreachable /joints (a lab older than these endpoints) the panel shows
  // its error and the scene stays exactly as it was.
  useEffect(() => {
    setAnimVisible(open && !!meta);
    return () => setAnimVisible(false);
  }, [open, meta]);

  // Land on the clip's first pose once the metadata (and therefore the clip)
  // is settled, so the ghost duck shows something real straight away.
  const seeded = useRef(false);
  useEffect(() => {
    if (!meta || seeded.current) return;
    seeded.current = true;
    setPose(sampleClip(clipRef.current, 0));
    setPlayhead(0);
  }, [meta]);

  // --- preview: every pose change goes to POST /pose -----------------------
  const streamerRef = useRef<PoseStreamer | null>(null);
  useEffect(() => {
    const s = new PoseStreamer(
      (r) => {
        animStore.bodies = r.bodies; // read per-frame by PoseDuck, no re-render
        setPoseErr(null);
      },
      (e) => setPoseErr(e)
    );
    streamerRef.current = s;
    return () => {
      s.close();
      streamerRef.current = null;
    };
  }, []);
  useEffect(() => {
    if (!open || !meta) return;
    streamerRef.current?.request(pose);
  }, [pose, open, meta]);

  // --- editing -------------------------------------------------------------

  /** Set the working pose — and, when the playhead is parked on a key, update
   *  that key with it (auto-key, the behaviour an animator expects). */
  const applyPose = useCallback((next: Pose) => {
    setPose(next);
    const t = playheadRef.current;
    setClip((c) => (keyAt(c, t) >= 0 ? withKey(c, t, next) : c));
  }, []);

  const setJoint = useCallback(
    (idx: number, value: number) => {
      const p = poseRef.current;
      if (idx === ROOT_SEL) {
        const lo = meta?.rootPitchRange[0] ?? -Math.PI;
        const hi = meta?.rootPitchRange[1] ?? Math.PI;
        applyPose({ ...p, rootPitch: Math.min(hi, Math.max(lo, value)) });
      } else {
        const v = clampJoint(meta, idx, value);
        applyPose({ ...p, joints: p.joints.map((x, k) => (k === idx ? v : x)) });
      }
    },
    [meta, applyPose]
  );

  // The 3D drag calls in here; a ref keeps the registered function stable
  // while it always sees the current pose/clip.
  const setJointRef = useRef(setJoint);
  useEffect(() => {
    setJointRef.current = setJoint;
  }, [setJoint]);
  useEffect(() => {
    animStore.applyJointDelta = (idx, delta) => {
      const p = poseRef.current;
      const cur = idx === ROOT_SEL ? p.rootPitch : p.joints[idx];
      setJointRef.current(idx, cur + delta);
    };
    return () => {
      animStore.applyJointDelta = null;
    };
  }, []);

  // --- rig: macro controls over coupled joints (lib/rig.ts) ----------------
  // Directions only depend on the joint metadata, so resolve them once per
  // meta; measure/range are re-read from the live pose every render.
  const rigVectors = useMemo<RigVector[]>(
    () => (meta ? RIG_CONTROLS.map((c) => rigVector(meta, c)).filter((v): v is RigVector => !!v) : []),
    [meta]
  );

  // What a 3D click edits: one servo (joints) or the mapped rig control.
  // Persisted like the other panel toggles; PoseDuck reads it off the store.
  const [mode, setMode] = useState<AnimMode>(() => loadJSON<AnimMode>("animMode", "joints"));
  useEffect(() => {
    saveJSON("animMode", mode);
    setAnimMode(mode);
  }, [mode]);
  // body → rig-control map for rig-mode picking in the scene.
  useEffect(() => {
    animStore.rigForBody = meta ? rigBodyMap(meta, rigVectors) : [];
  }, [meta, rigVectors]);

  /** Select a rig control (row click or 3D pick lands here via the store):
   *  highlights its bodies on the duck and flips the scene to rig mode, so
   *  the next 3D drag drives THIS control. */
  const selectRig = useCallback(
    (v: RigVector) => {
      if (!meta) return;
      setMode("rig");
      setSelectedRig({ id: v.ctrl.id, label: v.ctrl.label, bodies: rigBodies(meta, v) });
    },
    [meta]
  );
  const rigVectorsRef = useRef(rigVectors);
  useEffect(() => {
    rigVectorsRef.current = rigVectors;
  }, [rigVectors]);

  const setRig = useCallback(
    (v: RigVector, value: number) => {
      applyPose(rigApply(v, poseRef.current, value));
    },
    [applyPose]
  );
  const setRigRef = useRef(setRig);
  useEffect(() => {
    setRigRef.current = setRig;
  }, [setRig]);
  // The 3D ⇕ handle drags a rig control by id, pointer-speed, via the store.
  useEffect(() => {
    animStore.applyRigDelta = (rigId, delta) => {
      const v = rigVectorsRef.current.find((x) => x.ctrl.id === rigId);
      if (!v) return;
      setRigRef.current(v, rigMeasure(v, poseRef.current) + delta);
    };
    return () => {
      animStore.applyRigDelta = null;
    };
  }, []);

  const seek = useCallback((t: number) => {
    const c = clipRef.current;
    const clamped = Math.max(0, Math.min(c.duration, t));
    setPlayhead(clamped);
    setPose(sampleClip(c, clamped));
  }, []);

  // --- playback ------------------------------------------------------------
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const c = clipRef.current;
      let t = playheadRef.current + dt;
      if (t >= c.duration) {
        if (c.loop) t = c.duration > 0 ? t % c.duration : 0;
        else {
          t = c.duration;
          setPlaying(false);
        }
      }
      seek(t);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, seek]);

  // --- keys ----------------------------------------------------------------
  const keyIdx = keyAt(clip, playhead);

  const addKey = () => {
    setClip((c) => withKey(c, playhead, poseRef.current));
    pushToast(tr(`◆ key at ${playhead.toFixed(2)}s`, `◆ 已在 ${playhead.toFixed(2)} 秒添加关键帧`));
  };

  const deleteKey = () => {
    if (keyIdx <= 0) return; // key 0 anchors t = 0 (contract)
    setClip((c) => ({ ...c, keys: c.keys.filter((_, i) => i !== keyIdx) }));
  };

  const setDuration = (d: number) => {
    const last = clip.keys[clip.keys.length - 1]?.t ?? 0;
    const dur = round3(Math.max(0.1, Math.min(120, d)));
    if (dur < last) return; // would cut off the last key — the server rejects it
    setClip((c) => ({ ...c, duration: dur }));
    if (playheadRef.current > dur) seek(dur);
  };

  // --- clip storage --------------------------------------------------------
  const refreshClips = useCallback(() => {
    listClips()
      .then(setClips)
      .catch(() => setClips([]));
  }, []);
  useEffect(() => {
    if (browsing) refreshClips();
  }, [browsing, refreshClips]);

  const problem = clipProblem(clip);

  const save = async (announce = true) => {
    if (problem) {
      pushToast(`⚠ ${problem}`);
      return false;
    }
    setSaving(true);
    try {
      await putClip(clip);
      if (announce) pushToast(tr(`💾 saved “${clip.name}” (${clip.keys.length} keys)`, `💾 已保存“${clip.name}”（${clip.keys.length} 个关键帧）`));
      refreshClips();
      return true;
    } catch (e) {
      pushToast(`⚠ save failed: ${String((e as Error)?.message ?? e)}`);
      return false;
    } finally {
      setSaving(false);
    }
  };

  /** Start a training run that tracks a SAVED clip (by name on disk — the
   *  trainer subprocess loads it from clips/, so it must be saved first). */
  const trainClip = async (name: string) => {
    try {
      const res = await fetch(`${LAB_HTTP}/teach`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "copy the animation", clip: name }),
      });
      const data = await res.json();
      if (!data.matched) {
        pushToast(`⚠ ${data.message ?? "the lab wouldn't start that run"}`);
        return;
      }
      setBrowsing(false);
      pushToast(tr(`⚡ training a policy to perform “${name}” — watch the 🎓 duck`, `⚡ 正在训练“${name}”动作—请观看 🎓 鸭子`));
    } catch (e) {
      pushToast(`⚠ ${String((e as Error)?.message ?? e)}`);
    }
  };

  const trainThis = async () => {
    // Save first: an unsaved edit would train the previous version of it.
    if (await save(false)) await trainClip(clip.name);
  };

  const openClip = async (name: string) => {
    try {
      const c = await loadClip(name);
      setClip(c);
      clipRef.current = c;
      setPlayhead(0);
      setPose(sampleClip(c, 0));
      setBrowsing(false);
      pushToast(tr(`📂 loaded “${name}”`, `📂 已加载“${name}”`));
    } catch (e) {
      pushToast(`⚠ ${String((e as Error)?.message ?? e)}`);
    }
  };

  const dropClip = async (name: string) => {
    try {
      await removeClip(name);
      refreshClips();
      pushToast(tr(`🗑 deleted “${name}”`, `🗑 已删除“${name}”`));
    } catch (e) {
      pushToast(`⚠ ${String((e as Error)?.message ?? e)}`);
    }
  };

  // --- timeline gestures ---------------------------------------------------
  const trackRef = useRef<HTMLDivElement | null>(null);
  const dragKey = useRef<number | null>(null);

  const timeAtX = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return 0;
    const r = el.getBoundingClientRect();
    const usable = Math.max(1, r.width - 2 * TRACK_PAD);
    const u = (clientX - r.left - TRACK_PAD) / usable;
    return round3(Math.max(0, Math.min(1, u)) * clip.duration);
  };

  /** Pointer capture keeps a drag alive outside the element — but a synthetic
   *  or already-released pointer id throws, and that must not abort the
   *  gesture (same guard as PolicyPanel's chip drags). */
  const capture = (e: React.PointerEvent<HTMLDivElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // no capture — the gesture still works through normal bubbling
    }
  };

  const scrub = (e: React.PointerEvent<HTMLDivElement>) => {
    if (dragKey.current !== null) return;
    setPlaying(false);
    capture(e);
    seek(timeAtX(e.clientX));
  };

  const scrubMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (dragKey.current !== null || e.buttons === 0) return;
    seek(timeAtX(e.clientX));
  };

  const keyDown = (i: number) => (e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    setPlaying(false);
    seek(clip.keys[i].t);
    if (i === 0) return; // pinned: the contract wants a key at exactly t = 0
    dragKey.current = i;
    capture(e);
  };

  const keyMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const i = dragKey.current;
    if (i === null) return;
    e.stopPropagation();
    const c = clipRef.current;
    // Clamp strictly between neighbours: retiming must never reorder keys or
    // collide two times (both are contract violations downstream).
    const lo = (c.keys[i - 1]?.t ?? 0) + 0.01;
    const hi = (c.keys[i + 1]?.t ?? c.duration) - (c.keys[i + 1] ? 0.01 : 0);
    const t = round3(Math.max(lo, Math.min(hi, timeAtX(e.clientX))));
    setClip((cc) => ({
      ...cc,
      keys: cc.keys.map((k, n) => (n === i ? { ...k, t } : k)),
    }));
    setPlayhead(t);
  };

  const keyUp = () => {
    dragKey.current = null;
  };

  // --- render --------------------------------------------------------------

  if (!open)
    return (
      <button
        onClick={() => setOpen(true)}
        title={tr("keyframe animation editor — pose the duck, key it, save a clip", "关键帧动画编辑器—调整姿态、添加关键帧并保存动作")}
        style={{
          position: "absolute",
          bottom: 14,
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(14,16,20,0.86)",
          color: "#e8e6e1",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 10,
          padding: "8px 12px",
          fontFamily: mono,
          fontSize: 12,
          cursor: "pointer",
          backdropFilter: "blur(6px)",
          zIndex: 20,
        }}
      >
        🎬 {tr("animate", "动画")}
      </button>
    );

  const jointRows = (group: string) =>
    (meta?.joints ?? []).filter((j) => j.group === group);

  return (
    <div
      ref={(el) => {
        // ◎ focus frames the duck above this panel — it needs the real rect.
        animStore.panelEl = el;
      }}
      // Armed-chip guard: nearestDuck projects screen positions with an 80px
      // radius, so a click in this panel would otherwise assign to a duck
      // behind it.
      data-policy-ui
      style={{
        position: "absolute",
        bottom: 14,
        left: "50%",
        transform: "translateX(-50%)",
        // Bottom-centre, capped so the right edge stays clear of the teach
        // panel (right: 14, width 320 → its left edge is 100vw - 334): a
        // centred panel of width W reaches 50vw + W/2, hence the 688px term.
        // The max() floor keeps it usable on a narrow window at the cost of
        // some overlap there — collapse a panel, as the other three expect.
        width: "min(520px, max(340px, calc(100vw - 688px)))",
        // Deliberately short: this is an editor for a 3D scene, and a panel
        // that eats the stage hides the thing being posed. The joint list
        // scrolls inside whatever is left.
        maxHeight: "min(56vh, 470px)",
        display: "flex",
        flexDirection: "column",
        background: "rgba(14, 16, 20, 0.88)",
        border: "1px solid rgba(255,255,255,0.09)",
        borderRadius: 10,
        color: "#e8e6e1",
        fontFamily: mono,
        fontSize: 12,
        lineHeight: 1.5,
        backdropFilter: "blur(6px)",
        zIndex: 20,
        overflow: "hidden",
      }}
    >
      {/* ---- header ---- */}
      <div
        style={{
          padding: "7px 12px",
          fontWeight: 700,
          fontSize: 13,
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexShrink: 0,
        }}
      >
        <span style={{ flex: 1 }}>🎬 {tr("animate", "动画")}</span>
        <button
          style={btn}
          title={tr("frame the preview duck", "将预览鸭子置于画面中心")}
          onClick={() => {
            animStore.focusRequest = 1;
          }}
        >
          ◎ {tr("focus", "聚焦")}
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
          }}
        >
          —
        </button>
      </div>

      {metaErr && (
        <div style={{ color: "#e07a5f", padding: "6px 12px" }}>
          ⚠ {tr("can't reach the lab's /joints on :8788", "无法访问 :8788 实验室的 /joints")} — {metaErr}
        </div>
      )}

      {/* ---- clip bar ---- */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "7px 12px",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          flexShrink: 0,
          flexWrap: "wrap",
        }}
      >
        <input
          value={clip.name}
          onChange={(e) => setClip((c) => ({ ...c, name: e.target.value }))}
          placeholder={tr("clip name", "动作名称")}
          title={tr("saved as clips/<name>.json", "保存为 clips/<名称>.json")}
          style={{ ...field, width: 132 }}
        />
        <label style={{ color: "#8b93a3", fontSize: 10, display: "flex", alignItems: "center", gap: 4 }}>
          {tr("dur", "时长")}
          <input
            type="number"
            min={0.1}
            max={120}
            step={0.05}
            value={clip.duration}
            onChange={(e) => setDuration(Number(e.target.value))}
            style={{ ...field, width: 58 }}
          />
          s
        </label>
        <label
          style={{ color: "#8b93a3", fontSize: 10, display: "flex", alignItems: "center", gap: 3 }}
          title={tr("loop the clip (the RL side reads this flag)", "循环播放动作（RL 训练会读取此标记）")}
        >
          <input
            type="checkbox"
            checked={clip.loop}
            onChange={(e) => setClip((c) => ({ ...c, loop: e.target.checked }))}
          />
          {tr("loop", "循环")}
        </label>
        <div style={{ flex: 1 }} />
        <button style={btn} onClick={() => save()} disabled={saving} title={tr("save to clips/", "保存到 clips/")}>
          {saving ? "…" : `💾 ${tr("save", "保存")}`}
        </button>
        <button style={btn} onClick={() => setBrowsing((b) => !b)} title={tr("saved clips", "已保存动作")}>
          📂
        </button>
        <button
          style={{ ...btn, color: "#e8c87d", borderColor: "rgba(216,198,125,0.4)" }}
          onClick={trainThis}
          title={tr("save the clip so a policy can be trained to track it", "保存动作并训练策略进行跟随")}
        >
          ⚡ {tr("train this", "训练此动作")}
        </button>
      </div>

      {problem && (
        <div style={{ color: "#e8b24a", padding: "4px 12px", fontSize: 10, flexShrink: 0 }}>
          ⚠ {problem}
        </div>
      )}

      {browsing && (
        <div
          style={{
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            padding: "5px 12px",
            maxHeight: 120,
            overflowY: "auto",
            flexShrink: 0,
          }}
        >
          {!clips.length && (
            <div style={{ color: "#8b93a3", fontSize: 10 }}>{tr("no saved clips yet", "暂无已保存动作")}</div>
          )}
          {clips.map((c) => (
            <div
              key={c.name}
              style={{ display: "flex", alignItems: "center", gap: 6, margin: "2px 0" }}
            >
              <button style={{ ...btn, flex: 1, textAlign: "left" }} onClick={() => openClip(c.name)}>
                {c.name}
              </button>
              <span style={{ color: "#8b93a3", fontSize: 9, flexShrink: 0 }}>
                {c.keys?.length ?? 0} {tr("keys", "关键帧")} · {c.duration}s{c.loop ? ` · ${tr("loop", "循环")}` : ""}
              </span>
              <button
                style={{ ...btn, color: "#e8c87d", padding: "3px 6px" }}
                title={`train a policy to perform ${c.name}`}
                onClick={() => trainClip(c.name)}
              >
                ⚡
              </button>
              <button
                style={{ ...btn, color: "#e07a5f", padding: "3px 6px" }}
                title={`delete ${c.name}`}
                onClick={() => dropClip(c.name)}
              >
                🗑
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ---- timeline ---- */}
      <div style={{ padding: "8px 12px", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
          <button style={btn} title={tr("back to start", "回到开始")} onClick={() => { setPlaying(false); seek(0); }}>
            ⏮
          </button>
          <button style={btn} onClick={() => setPlaying((p) => !p)} title={tr("play / pause", "播放 / 暂停")}>
            {playing ? "⏸" : "▶"}
          </button>
          <button style={btn} onClick={addKey} title={tr("key the current pose at the playhead", "在播放头处为当前姿态添加关键帧")}>
            ◆ {tr("key", "关键帧")}
          </button>
          <button
            style={{ ...btn, opacity: keyIdx > 0 ? 1 : 0.4 }}
            onClick={deleteKey}
            disabled={keyIdx <= 0}
            title={
              keyIdx > 0
                ? "delete the key under the playhead"
                : keyIdx === 0
                  ? "the first key anchors t = 0 and can't be deleted"
                  : "no key under the playhead"
            }
          >
            ✕ {tr("key", "关键帧")}
          </button>
          <div style={{ flex: 1 }} />
          <span style={{ color: keyIdx >= 0 ? "#ffd166" : "#8b93a3", fontSize: 10 }}>
            {keyIdx >= 0
              ? tr(`● on key ${keyIdx + 1} — edits auto-key`, `● 当前为第 ${keyIdx + 1} 关键帧—编辑会自动记录`)
              : tr("○ unkeyed pose", "○ 未记录的姿态")}
          </span>
          <span style={{ color: "#a5adbb", fontSize: 11 }}>
            {playhead.toFixed(2)} / {clip.duration.toFixed(2)}s
          </span>
        </div>

        <div
          ref={trackRef}
          onPointerDown={scrub}
          onPointerMove={(e) => {
            scrubMove(e);
            keyMove(e);
          }}
          onPointerUp={keyUp}
          style={{
            position: "relative",
            height: 44,
            background: "#12161e",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 7,
            cursor: "ew-resize",
            touchAction: "none",
            userSelect: "none",
          }}
        >
          {/* second gridlines */}
          {Array.from({ length: Math.floor(clip.duration * 4) + 1 }, (_, i) => i / 4)
            .filter((t) => t > 0 && t < clip.duration)
            .map((t) => (
              <div
                key={t}
                style={{
                  position: "absolute",
                  left: `calc(${TRACK_PAD}px + ${(t / clip.duration) * 100}% - ${
                    (TRACK_PAD * 2 * t) / clip.duration
                  }px)`,
                  top: 6,
                  bottom: 6,
                  width: 1,
                  background:
                    Math.abs(t % 1) < 1e-6 ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.06)",
                }}
              />
            ))}

          {/* key markers */}
          {clip.keys.map((k, i) => {
            const u = clip.duration > 0 ? k.t / clip.duration : 0;
            const isHere = i === keyIdx;
            return (
              <div
                key={i}
                onPointerDown={keyDown(i)}
                onPointerMove={keyMove}
                onPointerUp={keyUp}
                title={
                  i === 0
                    ? "t = 0 — the clip's anchor key (fixed)"
                    : `key ${i + 1} at ${k.t.toFixed(2)}s — drag to retime`
                }
                style={{
                  position: "absolute",
                  left: `calc(${TRACK_PAD}px + ${u * 100}% - ${TRACK_PAD * 2 * u}px)`,
                  top: 7,
                  width: 13,
                  height: 13,
                  marginLeft: -6.5,
                  transform: "rotate(45deg)",
                  background: isHere ? "#ffd166" : "#7db8d8",
                  border: `1px solid ${isHere ? "#fff0c2" : "rgba(255,255,255,0.35)"}`,
                  borderRadius: 2,
                  cursor: i === 0 ? "not-allowed" : "grab",
                  touchAction: "none",
                  zIndex: 3,
                }}
              />
            );
          })}
          {clip.keys.map((k, i) => {
            const u = clip.duration > 0 ? k.t / clip.duration : 0;
            return (
              <div
                key={`t${i}`}
                style={{
                  position: "absolute",
                  left: `calc(${TRACK_PAD}px + ${u * 100}% - ${TRACK_PAD * 2 * u}px)`,
                  bottom: 4,
                  transform: "translateX(-50%)",
                  color: i === keyIdx ? "#ffd166" : "#7f8798",
                  fontSize: 9,
                  pointerEvents: "none",
                  whiteSpace: "nowrap",
                }}
              >
                {k.t.toFixed(2)}
              </div>
            );
          })}

          {/* playhead */}
          <div
            style={{
              position: "absolute",
              left: `calc(${TRACK_PAD}px + ${
                (clip.duration > 0 ? playhead / clip.duration : 0) * 100
              }% - ${(TRACK_PAD * 2 * (clip.duration > 0 ? playhead / clip.duration : 0))}px)`,
              top: 2,
              bottom: 2,
              width: 2,
              marginLeft: -1,
              background: "#7dd87d",
              boxShadow: "0 0 6px rgba(125,216,125,0.6)",
              pointerEvents: "none",
              zIndex: 4,
            }}
          />
        </div>
      </div>

      {/* ---- scene-drag mode: what clicking the duck edits ---- */}
      {meta && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "0 12px 7px",
            flexShrink: 0,
          }}
        >
          <span style={{ color: "#8b93a3", fontSize: 10 }}>{tr("clicking the duck edits", "点击鸭子时编辑")}</span>
          <button
            style={{
              ...btn,
              // Full `border` shorthand, not borderColor: toggling a partial
              // override on and off makes React warn about conflicting styles.
              ...(mode === "joints"
                ? { color: "#ffd166", border: "1px solid rgba(255,209,102,0.55)", background: "#2a2612" }
                : {}),
            }}
            title={tr("a click selects one servo; dragging rotates just that hinge", "点击选择一个舵机，拖动只旋转该关节")}
            onClick={() => setMode("joints")}
          >
            🦴 {tr("joints", "关节")}
          </button>
          <button
            style={{
              ...btn,
              ...(mode === "rig"
                ? { color: "#8ee6d6", border: "1px solid rgba(95,208,189,0.55)", background: "#0e2a26" }
                : {}),
            }}
            title={tr("a click selects the rig control for that part (feet → toes, thigh → swing, shin → squat, trunk → lean, head → look, hip sides → sway/twist); dragging drives the whole coupling", "点击选择该部位的联动控制，拖动会驱动整组关节")}
            onClick={() => setMode("rig")}
          >
            🎮 {tr("rig", "联动")}
          </button>
        </div>
      )}

      {/* ---- joints ---- */}
      <div style={{ overflowY: "auto", padding: "0 12px 8px" }}>
        {!meta && !metaErr && (
          <div style={{ color: "#8b93a3", padding: "8px 0" }}>{tr("loading joint limits…", "正在加载关节限位…")}</div>
        )}
        {meta && (
          <>
            {rigVectors.length > 0 && (
              <>
                <div
                  style={{ color: "#8b93a3", fontSize: 10, margin: "6px 0 2px" }}
                  title="each control drives several servos in a fixed coupling; its range ends where the first servo hits its MJCF limit"
                >
                  🎮 {tr("rig", "联动控制")}
                </div>
                {rigVectors.map((v) => (
                  <RigRow
                    key={v.ctrl.id}
                    v={v}
                    pose={pose}
                    selected={animStore.selectedRig?.id === v.ctrl.id}
                    onSelect={() => selectRig(v)}
                    onChange={(x) => setRig(v, x)}
                  />
                ))}
              </>
            )}
            <div style={{ color: "#8b93a3", fontSize: 10, margin: "6px 0 2px" }}>{tr("trunk", "躯干")}</div>
            <JointRow
              label={tr("root pitch", "躯干俯仰")}
              hint={tr("− lean back", "− 向后倾")}
              min={meta.rootPitchRange[0]}
              max={meta.rootPitchRange[1]}
              value={pose.rootPitch}
              def={0}
              selected={selected === ROOT_SEL}
              onSelect={() => {
                setMode("joints"); // symmetric with rig rows: row click sets the scene mode
                setSelected(ROOT_SEL);
              }}
              onChange={(v) => setJoint(ROOT_SEL, v)}
            />
            {GROUPS.map((g) => (
              <div key={g}>
                <div style={{ color: "#8b93a3", fontSize: 10, margin: "6px 0 2px" }}>{isZh ? GROUPS_ZH[g] : g}</div>
                {jointRows(g).map((j) => (
                  <JointRow
                    key={j.name}
                    label={j.name.replace(/^(left|right)_/, "")}
                    min={j.min}
                    max={j.max}
                    value={pose.joints[j.index] ?? 0}
                    def={j.default}
                    selected={selected === j.index}
                    onSelect={() => {
                      setMode("joints");
                      setSelected(j.index);
                    }}
                    onChange={(v) => setJoint(j.index, v)}
                  />
                ))}
              </div>
            ))}
            <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
              <button
                style={btn}
                title="every joint back to DEFAULT_POSE"
                onClick={() => applyPose(defaultPose(meta))}
              >
                ↺ {tr("default pose", "默认姿态")}
              </button>
              <button
                style={btn}
                title="start over: one key at t = 0, standing"
                onClick={() => {
                  const c = newClip(meta, clip.name);
                  setClip(c);
                  clipRef.current = c;
                  setPlaying(false);
                  setPlayhead(0);
                  setPose(sampleClip(c, 0));
                }}
              >
                ✧ {tr("new clip", "新建动作")}
              </button>
            </div>
          </>
        )}
        <div style={{ color: "#566072", fontSize: 9, marginTop: 8, lineHeight: 1.45 }}>
          {tr(
            "click a body part to edit it — 🦴 drags one servo, 🎮 drags that part's rig control (feet→toes, thigh→swing, shin→squat, trunk→lean, head→look, shift = fine) · the ⇕ handle drags the selected rig control (squat when none) and parks on the part it moves · rig sliders end where a servo hits its limit — hover one to see which · keys interpolate linearly and the RL side resamples the saved clip at 50 Hz",
            "点击身体部位进行编辑—🦴 拖动单个舵机，🎮 拖动联动控制（Shift = 精细调整）· ⇕ 手柄拖动已选联动控制 · 滑块在舵机到达限位时停止 · 关键帧线性插值，RL 以 50 Hz 重采样已保存动作"
          )}
          {poseErr && <span style={{ color: "#e07a5f" }}> · preview: {poseErr}</span>}
        </div>
      </div>
    </div>
  );
}

/** Typed exact-value entry, shared by joint and rig rows: exact values (0
 *  above all) are what an animator reaches for, and nudging a slider onto one
 *  is a fight. Local draft while focused so a half-typed "-" or "0." isn't
 *  parsed and snapped out from under the cursor; commit on Enter or blur,
 *  Escape reverts, and the value is clamped into [min, max] on the way in. */
function ValueField({
  value,
  min,
  max,
  color,
  title,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  color: string;
  title?: string;
  onChange: (v: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commitDraft = () => {
    if (draft == null) return;
    const n = Number(draft.trim());
    if (draft.trim() !== "" && Number.isFinite(n))
      onChange(round3(Math.max(min, Math.min(max, n))));
  };
  return (
    <input
      value={draft ?? value.toFixed(3)}
      onClick={(e) => e.stopPropagation()}
      onFocus={(e) => {
        setDraft(value.toFixed(3));
        e.currentTarget.select();
      }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        commitDraft();
        setDraft(null);
      }}
      onKeyDown={(e) => {
        e.stopPropagation();   // R restarts the sim; don't while typing
        if (e.key === "Enter") {
          commitDraft();
          e.currentTarget.blur();
        } else if (e.key === "Escape") {
          setDraft(null);
          e.currentTarget.blur();
        }
      }}
      title={title ?? `type an exact value (${min.toFixed(2)} … ${max.toFixed(2)} rad)`}
      style={{
        width: 52,
        flexShrink: 0,
        textAlign: "right",
        fontSize: 10,
        fontFamily: mono,
        color,
        background: draft != null ? "rgba(255,255,255,0.08)" : "transparent",
        border: "1px solid",
        borderColor: draft != null ? "rgba(255,209,102,0.5)" : "transparent",
        borderRadius: 4,
        padding: "1px 3px",
        outline: "none",
      }}
    />
  );
}

/** One rig control: a macro slider over coupled joints. The range is computed
 *  from the CURRENT pose — it is exactly how far this control can go before
 *  some servo hits its MJCF limit, and the tooltip names that servo, because
 *  "why won't it squat lower" deserves a real answer. */
function RigRow({
  v,
  pose,
  selected,
  onSelect,
  onChange,
}: {
  v: RigVector;
  pose: Pose;
  selected: boolean;
  onSelect: () => void;
  onChange: (value: number) => void;
}) {
  const { isZh, tr } = useI18n();
  const value = rigMeasure(v, pose);
  // round3(…) || 0 folds the projection's float dust (and −0) into true zero
  // so the readout never says “−0.000”.
  const shown = round3(value) || 0;
  const r = rigRange(v, pose);
  // A degenerate range (some servo already pinned by a raw-joint edit) still
  // renders — the slider just has nowhere to go, which is itself the answer.
  const min = Math.min(r.min, value);
  const max = Math.max(r.max, value);
  const atLimit = value <= r.min + 1e-4 || value >= r.max - 1e-4;
  const joints = v.parts.map((p) => p.name.replace(/^(left|right)_/, (m) => m[0] === "l" ? "L " : "R ")).join(", ");
  return (
    <div
      onPointerDown={onSelect}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        height: 21,
        padding: "0 4px",
        margin: "0 -4px",
        borderRadius: 5,
        background: selected ? "rgba(95,208,189,0.13)" : "transparent",
        cursor: "pointer",
      }}
    >
      <span
        style={{ width: 74, flexShrink: 0, fontSize: 10, color: selected ? "#8ee6d6" : "#6fbfae", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        title={`${v.ctrl.title}\ndrives: ${joints}`}
      >
        {isZh ? (RIG_ZH[v.ctrl.label] ?? v.ctrl.label) : v.ctrl.label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={0.005}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        title={`${shown.toFixed(3)} rad · − end: ${r.minBy} at its limit · + end: ${r.maxBy} at its limit`}
        style={{ flex: 1, minWidth: 60, height: 12, accentColor: selected ? "#8ee6d6" : "#5fd0bd" }}
      />
      <ValueField
        value={shown}
        min={r.min}
        max={r.max}
        color={atLimit ? "#e8b24a" : "#e8e6e1"}
        title={
          atLimit
            ? `at the rig limit — ${value <= r.min + 1e-4 ? r.minBy : r.maxBy} has no travel left`
            : undefined
        }
        onChange={onChange}
      />
      <span
        style={{ width: 78, flexShrink: 0, textAlign: "right", fontSize: 9, color: "#566072" }}
        title={`travel from here: ${r.min.toFixed(2)} … ${r.max.toFixed(2)} rad (ends at ${r.minBy} / ${r.maxBy})`}
      >
        {v.ctrl.hint}
      </span>
      <button
        onClick={() => onChange(0)}
        title={tr("back to the default pose along this control", "将此联动控制恢复到默认姿态")}
        style={{ background: "none", border: "none", color: "#8b93a3", cursor: "pointer", fontFamily: mono, fontSize: 11, padding: "0 2px", flexShrink: 0 }}
      >
        ↺
      </button>
    </div>
  );
}

/** One joint: name, limit-clamped slider, readout, reset. The limits are the
 *  MJCF's own (`model.jnt_range`, served by /joints) — shown, not just
 *  enforced, because "why won't this bend further" is the first question an
 *  animator asks. */
function JointRow({
  label,
  hint,
  min,
  max,
  value,
  def,
  selected,
  onSelect,
  onChange,
}: {
  label: string;
  hint?: string;
  min: number;
  max: number;
  value: number;
  def: number;
  selected: boolean;
  onSelect: () => void;
  onChange: (v: number) => void;
}) {
  const { tr } = useI18n();
  const atLimit = value <= min + 1e-4 || value >= max - 1e-4;
  return (
    <div
      onPointerDown={onSelect}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        height: 21,
        padding: "0 4px",
        margin: "0 -4px",
        borderRadius: 5,
        background: selected ? "rgba(255,209,102,0.12)" : "transparent",
        cursor: "pointer",
      }}
    >
      <span
        style={{
          width: 74,
          flexShrink: 0,
          fontSize: 10,
          color: selected ? "#ffd166" : "#a5adbb",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={0.005}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        title={`${label}: ${min.toFixed(3)} … ${max.toFixed(3)} rad`}
        style={{
          flex: 1,
          minWidth: 60,
          height: 12,
          accentColor: selected ? "#ffd166" : "#7db8d8",
        }}
      />
      <ValueField
        value={value}
        min={min}
        max={max}
        color={atLimit ? "#e8b24a" : "#e8e6e1"}
        onChange={onChange}
      />
      <span
        style={{ width: 78, flexShrink: 0, textAlign: "right", fontSize: 9, color: "#566072" }}
        title={tr("joint limits from the MJCF", "来自 MJCF 的关节限位")}
      >
        {hint ?? `${min.toFixed(2)}…${max.toFixed(2)}`}
      </span>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onChange(def);
        }}
        title={tr(`reset to ${def.toFixed(3)}`, `重置为 ${def.toFixed(3)}`)}
        style={{
          background: "none",
          border: "none",
          color: "#8b93a3",
          cursor: "pointer",
          fontFamily: mono,
          fontSize: 11,
          padding: "0 2px",
          flexShrink: 0,
        }}
      >
        ↺
      </button>
    </div>
  );
}
