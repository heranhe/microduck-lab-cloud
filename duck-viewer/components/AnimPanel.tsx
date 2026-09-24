"use client";

// 🎬 animate (bottom-center): a keyframe animation editor for the robot.
//
// Pose the robot (sliders, rig controls, IK handles, or by dragging body
// parts in the 3D scene), key the poses on a timeline, scrub/play them back,
// save the clip. The saved JSON is the handoff to the imitation-RL side,
// which resamples it at 50 Hz and rewards a policy for tracking it — so the
// editor never invents a pose the contract can't express: joints are clamped
// to the MJCF servo limits, key times ascend from t = 0, and interpolation is
// linear in joint space (what the resampler does).
//
// Every pose shown here is forward kinematics from POST /pose on the server's
// scratch model — the lab ducks and their WS stream are untouched. IK handles
// go through POST /ik, whose answer IS a /pose answer for the solved joints.
//
// Two bodies: the Microduck (14 joints) and the Unitree G1 (29). The robot
// switch re-fetches /joints for that body, and a clip carries its robot so a
// saved G1 clip opens as the G1. The panel's own layout never assumes a joint
// count or a section name — both come from the metadata.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  type Balance,
  animStore,
  animVersion,
  balanceColor,
  balanceLabel,
  clampJoint,
  clipProblem,
  clipRobot,
  defaultPose,
  fetchJoints,
  fetchRobots,
  isIkResult,
  isUntouched,
  keyAt,
  listClips,
  loadClip,
  newClip,
  PoseStreamer,
  putClip,
  removeClip,
  ROOT_SEL,
  round3,
  sameBalance,
  sampleClip,
  setAnimMeta,
  setAnimMode,
  setAnimVisible,
  setSelected,
  setSelectedEffector,
  setSelectedRig,
  setShowBalance,
  subscribeAnim,
  type AnimMode,
  withKey,
  zeroPose,
  type Clip,
  type JointsMeta,
  type Pose,
  type RobotId,
  type RobotInfo,
  type StoredClip,
} from "@/lib/anim";
import { robotChipLabel, setActiveRobot, useActiveRobot } from "@/lib/activeRobot";
import { robotEmoji } from "@/lib/robots";
import { LAB_HTTP } from "@/lib/lab";
import { loadJSON, saveJSON } from "@/lib/persist";
import { useI18n } from "@/lib/i18n";
import {
  rigApply,
  rigBodies,
  rigBodyMap,
  rigControlsFor,
  rigMeasure,
  rigRange,
  rigVector,
  type RigVector,
} from "@/lib/rig";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
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

/** "left_hip_pitch" and "left_hip_pitch_joint" both read as "hip pitch" in
 *  a section that already says which leg. */
const jointLabel = (name: string) => name.replace(/^(left|right)_/, "").replace(/_joint$/, "");

/** A residual the eye should know about: past the millimetre the solver
 *  calls converged, as centimetres. Null below that. */
export function residualLabel(metres: number | undefined): string | null {
  if (metres == null || metres <= 0.001) return null;
  return `${(metres * 100).toFixed(1)} cm short`;
}

/** True when the 🎯 rows would read the same, so an /ik answer that changes
 *  no readout costs no render of its own. */
function sameResidual(a: Record<string, number>, b: Record<string, number>): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => residualLabel(a[k]) === residualLabel(b[k]));
}

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
  // Which body is being posed. Persisted; a loaded clip can switch it.
  const [robot, setRobot] = useState<RobotId>(() => loadJSON<RobotId>("animRobot", "microduck"));
  const [robots, setRobots] = useState<RobotInfo[]>([]);
  const [meta, setMeta] = useState<JointsMeta | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  // Unsaved work survives a refresh — an authored pose is expensive to redo.
  const [clip, setClip] = useState<Clip>(() => loadJSON<Clip | null>("animClip", null) ?? newClip(null));
  const [pose, setPose] = useState<Pose>(() => zeroPose());
  // The solver's last word per effector, for the 🎯 rows. Only moves when
  // an /ik answer lands, so a slider drag never re-renders for it.
  const [ikResidual, setIkResidual] = useState<Record<string, number>>({});
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [clips, setClips] = useState<StoredClip[]>([]);
  const [browsing, setBrowsing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [poseErr, setPoseErr] = useState<string | null>(null);
  // CoM vs the soles for the pose on screen, for the readout row. The 3D
  // marker reads the store per-frame instead; this state only moves while
  // the readout is showing, and only when what it says would change. It
  // starts from the store's last read so a panel that opens with the marker
  // already on has a row to show before the first pose comes back.
  const [balance, setBalance] = useState<Balance | null>(() => animStore.balance);

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

  // --- which bodies this lab can pose --------------------------------------
  useEffect(() => {
    if (!open) return;
    let stale = false;
    fetchRobots()
      .then((rs) => !stale && setRobots(rs))
      .catch(() => !stale && setRobots([])); // an older lab: the duck alone, no switch
    return () => {
      stale = true;
    };
  }, [open]);
  useEffect(() => saveJSON("animRobot", robot), [robot]);

  /** Pose a different body. The metadata is dropped first so nothing is
   *  sent for a pose the wrong length; the effect below fetches the new one
   *  and settles the clip against it. */
  const switchRobot = useCallback((id: RobotId) => {
    setRobot(id);
    setMeta(null);
    setPlaying(false);
    // A body chosen HERE — the switch, or a clip loaded for another body — is
    // the robot the person is working with: the 🧠 palette and 🎓 teach follow.
    setActiveRobot(id);
  }, []);

  // …and this editor follows a robot chosen THERE (palette, teach, a click on
  // the stage), but only with nothing to lose: switching bodies starts a
  // fresh clip, so an authored pose pins the editor to its own body.
  const activeRobot = useActiveRobot();
  useEffect(() => {
    if (!open || activeRobot === robot || meta?.robot !== robot) return;
    // …and only onto a body this editor can actually pose: a MARS selected
    // in the 🧠 palette must not drag the 🎬 editor onto a body with no
    // effectors and no soles, which is a panel of empty sliders.
    if (!robots.some((r) => r.id === activeRobot && r.ready && r.animate !== false)) return;
    if (!isUntouched(clipRef.current, poseRef.current, meta)) return;
    switchRobot(activeRobot as RobotId);
  }, [open, activeRobot, robot, robots, meta, switchRobot]);

  // --- joint metadata (limits, defaults, body map) -------------------------
  // Fetched for the current robot whenever it is not the one the metadata
  // in hand describes. Once it lands the clip is settled against it: a clip
  // for another body (or one restored from before we knew the limits, still
  // all zeros) becomes a fresh clip on this one; a clip already for this
  // body is kept. Either way the ghost lands on the clip's first pose.
  useEffect(() => {
    if (!open || meta?.robot === robot) return;
    let stale = false;
    fetchJoints(robot)
      .then((m) => {
        if (stale) return;
        setMeta(m);
        setAnimMeta(m);
        streamerRef.current?.setRobot(m.robot);
        setMetaErr(null);
        const cur = clipRef.current;
        const fits =
          clipRobot(cur) === m.robot && cur.keys.every((k) => k.joints.length === m.numJoints);
        const blank = cur.keys.length === 1 && cur.keys[0].joints.every((v) => v === 0);
        const next = !fits || blank ? newClip(m, cur.name) : cur;
        if (next !== cur) {
          setClip(next);
          clipRef.current = next;
        }
        setPlayhead(0);
        setPose(sampleClip(next, 0, m.numJoints));
      })
      .catch((e) => !stale && setMetaErr(String(e?.message ?? e)));
    return () => {
      stale = true;
    };
  }, [open, robot, meta]);

  useEffect(() => saveJSON("animOpen", open), [open]);
  useEffect(() => saveJSON("animClip", clip), [clip]);
  // The ghost only exists once we know the joint layout — with an
  // unreachable /joints (a lab older than these endpoints) the panel shows
  // its error and the scene stays exactly as it was.
  useEffect(() => {
    setAnimVisible(open && !!meta);
    return () => setAnimVisible(false);
  }, [open, meta]);

  // --- editing -------------------------------------------------------------

  /** Set the working pose — and, when the playhead is parked on a key, update
   *  that key with it (auto-key, the behaviour an animator expects). The ref
   *  is written here too, not only after the render: a 3D drag fires several
   *  times between renders and each step must build on the last. */
  const applyPose = useCallback((next: Pose) => {
    poseRef.current = next;
    setPose(next);
    const t = playheadRef.current;
    setClip((c) => (keyAt(c, t) >= 0 ? withKey(c, t, next) : c));
  }, []);

  // --- preview: every pose change goes to POST /pose -----------------------
  const streamerRef = useRef<PoseStreamer | null>(null);
  useEffect(() => {
    const s = new PoseStreamer(
      (r) => {
        animStore.bodies = r.bodies; // read per-frame by PoseDuck, no re-render
        animStore.effectors = r.effectors ?? null;
        const bal = r.balance ?? null;
        animStore.balance = bal;
        // A slider drag must not re-render the panel per step (setPoseErr
        // below bails on identity; a fresh object here would not).
        if (animStore.showBalance) setBalance((prev) => (sameBalance(prev, bal) ? prev : bal));
        setPoseErr(null);
        // An /ik answer carries the SOLVED joints: adopt them as an edit
        // (auto-keyed like any other). The pose effect below then re-sends
        // that pose to /pose once — a round trip, not a loop, because a
        // /pose answer never comes back through here as an edit.
        if (isIkResult(r)) {
          animStore.ikResidual = r.ik.residual;
          setIkResidual((prev) => (sameResidual(prev, r.ik.residual) ? prev : r.ik.residual));
          applyPose({ joints: r.joints, rootPitch: poseRef.current.rootPitch });
        }
      },
      (e) => setPoseErr(e)
    );
    streamerRef.current = s;
    return () => {
      s.close();
      streamerRef.current = null;
    };
  }, [applyPose]);
  useEffect(() => {
    // The length guard covers the moment between a robot switch and its
    // metadata: a 14-joint pose must never be sent for a 29-joint body.
    if (!open || !meta || pose.joints.length !== meta.numJoints) return;
    streamerRef.current?.request(pose);
  }, [pose, open, meta]);

  // A 🎯 handle drag (PoseDuck) asks the solver for one effector's position,
  // pointer-speed, from the pose as it is now; the answer lands above.
  useEffect(() => {
    animStore.applyIkDrag = (id, pos) => {
      streamerRef.current?.requestIk(poseRef.current, { [id]: { pos } });
    };
    return () => {
      animStore.applyIkDrag = null;
    };
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
    () =>
      meta ? rigControlsFor(meta).map((c) => rigVector(meta, c)).filter((v): v is RigVector => !!v) : [],
    [meta]
  );

  // What a 3D click edits: one servo (joints) or the mapped rig control.
  // Persisted like the other panel toggles; PoseDuck reads it off the store.
  const [mode, setMode] = useState<AnimMode>(() => loadJSON<AnimMode>("animMode", "joints"));
  useEffect(() => {
    saveJSON("animMode", mode);
    setAnimMode(mode);
  }, [mode]);

  // The ⊕ CoM marker in the scene. Persisted like `mode`, and off by default:
  // the balance read is a question you ask of a pose, not scene furniture.
  const [showBalance, setShowMarker] = useState(() => loadJSON("animBalance", false));
  useEffect(() => {
    saveJSON("animBalance", showBalance);
    setShowBalance(showBalance);
    // The readout ignores pose responses while hidden, so ask for the pose
    // again whenever it becomes visible: the store's last read seeds the row
    // (at mount above, at the toggle below) and this replaces it with a
    // current one.
    if (showBalance && open && meta) streamerRef.current?.request(poseRef.current);
  }, [showBalance, open, meta]);
  // body → rig-control map for rig-mode picking in the scene.
  useEffect(() => {
    animStore.rigForBody = meta ? rigBodyMap(meta, rigVectors) : [];
  }, [meta, rigVectors]);

  /** Select a rig control (row click or 3D pick lands here via the store):
   *  highlights its bodies on the robot and flips the scene to rig mode, so
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
      pushToast(tr(`⚡ training a policy to perform “${name}” — watch the 🎓 trainee`, `⚡ 正在训练“${name}”动作—请观看 🎓 机器人`));
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
      setPlaying(false);
      // A clip for another body switches the editor to it first; the
      // metadata effect then lands on this clip's first pose (it keeps a
      // clip that already fits). For the current body, land on it now.
      if (clipRobot(c) !== robot) switchRobot(clipRobot(c));
      else setPose(sampleClip(c, 0, meta?.numJoints));
      setBrowsing(false);
      const robotSuffix = clipRobot(c) !== robot ? ` (${robotEmoji(clipRobot(c))} ${clipRobot(c)})` : "";
      pushToast(tr(`📂 loaded “${name}”${robotSuffix}`, `📂 已加载“${name}”${robotSuffix}`));
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
        title={tr("keyframe animation editor — pose the robot, key it, save a clip", "关键帧动画编辑器—调整姿态、添加关键帧并保存动作")}
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
  // The switch shows every body this editor can POSE — `animate`, which the
  // lab answers by asking the body (a `PoseScratch` needs effectors, soles
  // and a base link, and `pose_scratch("mars")` died on the last of those
  // the moment a wheeled body was registered). A lab too old to send the
  // flag listed only posable bodies anyway, so an absent one means yes.
  // A lab without /robots lists none and the editor is the duck's, as it
  // always was.
  const robotChips = robots.filter((r) => r.animate !== false);
  const effectors = meta?.effectors ?? [];
  const selectedEffector = animStore.selectedEffector;

  return (
    <div
      ref={(el) => {
        // ◎ focus frames the robot above this panel — it needs the real rect.
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
        {/* the robot switch: one chip per body the lab can pose */}
        {robotChips.map((r) => {
          const active = r.id === robot;
          return (
            <button
              key={r.id}
              style={{
                ...btn,
                padding: "2px 7px",
                opacity: r.ready ? 1 : 0.45,
                cursor: r.ready ? "pointer" : "not-allowed",
                ...(active
                  ? { color: "#8ee6d6", border: "1px solid rgba(95,208,189,0.55)", background: "#0e2a26" }
                  : {}),
              }}
              disabled={!r.ready}
              title={
                r.ready
                  ? `pose the ${r.title} (${r.numJoints} joints)`
                  : `uv run fetch-robot ${r.id}`
              }
              onClick={() => {
                if (r.id !== robot) switchRobot(r.id);
              }}
            >
              {robotChipLabel(r)}
            </button>
          );
        })}
        <button
          style={btn}
          title={tr("frame the preview robot", "将预览机器人置于画面中心")}
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
          ⚠ {tr(`can't reach the lab's /joints${robot !== "microduck" ? `?robot=${robot}` : ""} on :8788`, `无法访问 :8788 实验室的 /joints${robot !== "microduck" ? `?robot=${robot}` : ""}`)} — {metaErr}
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

      {/* ---- scene-drag mode: what clicking the robot edits ---- */}
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
          <span style={{ color: "#8b93a3", fontSize: 10 }}>{tr("clicking the robot edits", "点击机器人时编辑")}</span>
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
          <button
            style={{
              ...btn,
              ...(mode === "ik"
                ? { color: "#ffd166", border: "1px solid rgba(255,209,102,0.55)", background: "#2a2612" }
                : {}),
              opacity: effectors.length ? 1 : 0.45,
            }}
            disabled={!effectors.length}
            title={
              effectors.length
                ? "a click selects the IK handle for that limb (a foot, a hand, the head); dragging a handle asks the solver for the joints that put it there, the other feet held where they are"
                : "this lab's /joints lists no IK handles"
            }
            onClick={() => setMode("ik")}
          >
            🎯 ik
          </button>
          <div style={{ flex: 1 }} />
          <button
            style={{
              ...btn,
              ...(showBalance
                ? { color: "#7dd87d", border: "1px solid rgba(125,216,125,0.5)", background: "#11241a" }
                : {}),
            }}
            title="show where the centre of mass falls: a ball at the CoM, a plumb line, and a crosshair on the floor with the soles outlined under it — green inside a sole, blue inside the two-foot stance, amber outside"
            onClick={() => {
              const next = !showBalance;
              setShowMarker(next);
              // Seed the row from the store's last read, so turning the
              // marker on says something before the pose round-trip lands.
              if (next) setBalance(animStore.balance);
            }}
          >
            ⊕ balance
          </button>
        </div>
      )}

      {/* ---- balance: where the CoM lands relative to the soles ---- */}
      {meta && showBalance && (
        <div
          style={{ padding: "0 12px 7px", fontSize: 10, flexShrink: 0 }}
          title={
            balance
              ? `signed distance from the CoM's ground projection to each sole's footprint (positive = inside): left ${balance.feet.left.marginMm} mm${balance.feet.left.grounded ? "" : " (in the air)"}, right ${balance.feet.right.marginMm} mm${balance.feet.right.grounded ? "" : " (in the air)"}; to the stance (the hull of the grounded soles): ${balance.support.marginMm} mm.\nA STATIC check: no velocity, no momentum, no ankle torque. It says how hard this pose is to hold, not whether the robot stands.`
              : undefined
          }
        >
          {balance ? (
            <span style={{ color: balanceColor(balance) }}>
              ⊕ CoM {balanceLabel(balance)}
              <span style={{ color: "#566072" }}> · static, no momentum</span>
            </span>
          ) : (
            <span style={{ color: "#8b93a3" }}>
              {animStore.bodies ? "⊕ this lab's /pose doesn't report balance" : "⊕ …"}
            </span>
          )}
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
            {mode === "ik" && effectors.length > 0 && (
              <>
                <div
                  style={{ color: "#8b93a3", fontSize: 10, margin: "6px 0 2px" }}
                  title="each handle is a point the solver can be asked to put somewhere: drag it in the scene, and the joints on that limb follow. The readout is how far short the last solve fell."
                >
                  🎯 {tr("ik handles", "IK 手柄")}
                </div>
                {effectors.map((e) => {
                  const sel = selectedEffector === e.id;
                  const short = residualLabel(ikResidual[e.id]);
                  return (
                    <div
                      key={e.id}
                      onPointerDown={() => {
                        setMode("ik");
                        setSelectedEffector(e.id);
                      }}
                      title={`${e.label} — ${e.kind} on ${e.bodyName}; the solver may move ${e.chain.length} joint${e.chain.length === 1 ? "" : "s"}`}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        height: 21,
                        padding: "0 4px",
                        margin: "0 -4px",
                        borderRadius: 5,
                        background: sel ? "rgba(255,209,102,0.12)" : "transparent",
                        cursor: "pointer",
                      }}
                    >
                      <span style={{ width: 74, flexShrink: 0, fontSize: 10, color: sel ? "#ffd166" : "#a5adbb" }}>
                        {e.kind === "foot" ? "🦶" : e.kind === "hand" ? "✋" : "👤"} {e.label}
                      </span>
                      <span style={{ flex: 1, fontSize: 9, color: short ? "#e8b24a" : "#566072" }}>
                        {short ?? (sel ? tr("drag the handle in the scene", "在场景中拖动手柄") : "")}
                      </span>
                    </div>
                  );
                })}
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
            {meta.groups.map((g) => (
              <div key={g}>
                <div style={{ color: "#8b93a3", fontSize: 10, margin: "6px 0 2px" }}>{isZh ? GROUPS_ZH[g] : g}</div>
                {jointRows(g).map((j) => (
                  <JointRow
                    key={j.name}
                    label={jointLabel(j.name)}
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
                  setPose(sampleClip(c, 0, meta.numJoints));
                }}
              >
                ✧ {tr("new clip", "新建动作")}
              </button>
            </div>
          </>
        )}
        <div style={{ color: "#566072", fontSize: 9, marginTop: 8, lineHeight: 1.45 }}>
          {tr(
            "click a body part to edit it — 🦴 drags one servo, 🎮 drags that part's rig control (feet→toes, thigh→swing, shin→squat, trunk→lean, head→look, shift = fine), 🎯 drags that limb's IK handle in the view plane and the solver finds the joints · the ⇕ handle drags the selected rig control (squat when none) and parks on the part it moves · rig sliders end where a servo hits its limit — hover one to see which · keys interpolate linearly and the RL side resamples the saved clip at 50 Hz",
            "点击身体部位进行编辑—🦴 拖动单个舵机，🎮 拖动联动控制（Shift = 精细调整），🎯 在视图平面拖动肢体 IK 手柄由求解器计算关节 · ⇕ 手柄拖动已选联动控制 · 滑块在舵机到达限位时停止 · 关键帧线性插值，RL 以 50 Hz 重采样已保存动作"
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
  const joints = v.parts
    .map((p) => p.name.replace(/_joint$/, "").replace(/^(left|right)_/, (m) => (m[0] === "l" ? "L " : "R ")))
    .join(", ");
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
