"""Where does the ball GO when a duck loses it — the loss-cause audit.

    cd microduck_local
    uv run python scripts/probe_ball_loss.py --seeds 12 --seconds 180 --jobs 12
    uv run python scripts/probe_ball_loss.py --arm "shipped=" --arm "hold=look_hold_s=2.5" --out runs/loss

`probe_search.py` counts how often the ball is out of view and how long each
loss lasts. It cannot say WHY, and every head-tracking knob so far has been
argued from the diff and judged on the pooled numbers. This reads the TRUTH at
every detector frame that does not report the ball and files it under the gate
that dropped it, mirroring `Detector._visible` for a point target:

    behind     the ball is behind the camera plane
    h_out      off the side of the frame (the horizontal frustum) - the case a
               head YAW can fix, and the only one it can
    v_low      below the bottom of the frame - the near ball a head PITCH can fix
    v_high     above the frame (the head is pitched down at something else)
    occluded   inside the frame, behind a duck
    small      inside the frame, unoccluded, but a p_find under 1 rolled a miss
    noise      the same with p_find = 1: the datasheet's own miss rate
    range      beyond `max_range_m`

Every LOSS EVENT (from `Chase.DET_MAX_AGE` after the last sighting to the next
frame with a ball, as `probe_search` counts them) carries the cause of the latest
blind frame when it is declared, the brain's state, what the head was COMMANDED and where it
actually WAS, whether the tracker still had the ball, whether `yaw_clear` was
holding the head on the line, and where the ball truly was (bearing off the
body, range). So the question "would a head that followed the ball have kept
it?" is answered per event from the geometry, not argued.

Two more things the head-tracking law has never had measured:

  * **the servo**: the commanded head yaw against the camera's true yaw off
    the body, cross-correlated over lags, so the gain and the delay of the
    walker's head are numbers;
  * **the coast gap**: how much of the blind time falls between `predict_s`
    (when the head stops following the track) and `coast_s` (when the tracker
    forgets it) - the window where the brain still believes in a bearing and
    the head is looking straight ahead.

Arms (`--arm LABEL=KNOBS`, `MICRODUCK_CHASE` syntax) run paired seeds and are
compared per seed with Student's t (`compare_pitch.paired`), the knobs read
back off the CONSTRUCTED brain (playbook rule 0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.controllers import Chase, ChaseParams, tof_clearance_bearings
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import PitchMetrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_pitch import paired  # noqa: E402

CAUSES = ("behind", "h_out", "v_low", "v_high", "occluded", "small", "noise", "range")
MAX_LAG = 20            # ticks (0.4 s) scanned for the head servo's delay


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def classify(det, data, tgt) -> dict:
    """Why this frame does NOT report the ball, from the truth. Mirrors the
    point-target branch of `Detector._visible` gate for gate, so a cause here
    is the gate that actually fired, not a guess at it."""
    s = det.spec
    origin = np.asarray(data.site_xpos[det.site_id], dtype=np.float64)
    R = data.site_xmat[det.site_id].reshape(3, 3)
    Rb = data.xmat[det.own_root].reshape(3, 3)
    body_yaw = math.atan2(Rb[1, 0], Rb[0, 0])
    cam_yaw = _wrap(math.atan2(R[1, 0], R[0, 0]) - body_yaw)
    cam_pitch = -math.asin(float(np.clip(R[2, 0], -1.0, 1.0)))
    p = np.asarray(data.xpos[tgt.body], dtype=np.float64) - origin
    rng = float(np.linalg.norm(p))
    trunk = data.xpos[det.own_root]
    bearing_body = _wrap(math.atan2(p[1] + origin[1] - trunk[1], p[0] + origin[0] - trunk[0]) - body_yaw)
    out = {"rng": round(rng, 3), "bearingBody": round(bearing_body, 3), "camYaw": round(cam_yaw, 3),
           "camPitch": round(cam_pitch, 3)}
    if rng > s.max_range_m:
        out["cause"] = "range"
        return out
    local = R.T @ p
    bearing = math.atan2(local[1], local[0])
    elev = math.atan2(local[2], math.hypot(local[0], local[1]))
    out["bearingCam"] = round(bearing, 3)
    out["elevCam"] = round(elev, 3)
    if local[0] <= 0:
        out["cause"] = "behind"
        return out
    half_h, half_v = math.radians(s.fov_h_deg) / 2, math.radians(s.fov_v_deg) / 2
    alpha = math.atan(tgt.radius / rng)
    frac_h, _ = det._clip_arc(bearing - alpha, bearing + alpha, half_h)
    if frac_h < s.partial_min:
        out["cause"] = "h_out"
        return out
    frac_v, _ = det._clip_arc(elev - alpha, elev + alpha, half_v)
    if frac_v < s.partial_min:
        out["cause"] = "v_low" if elev < 0 else "v_high"
        return out
    frac_occl = det._unoccluded(data, tgt, origin, p, rng)
    if frac_occl < s.occl_min:
        out["cause"] = "occluded"
        return out
    width = 2.0 * math.atan(tgt.radius / rng)
    p_find = float(np.clip((width - s.w_none) / (s.w_full - s.w_none), 0.0, 1.0))
    seen = frac_h * frac_v * frac_occl
    p_find *= min(seen / s.seen_full, 1.0) if s.seen_full > 0 else 1.0
    out["pFind"] = round(p_find, 3)
    out["cause"] = "small" if p_find < 0.999 else "noise"
    return out


def _hook(det, tgt) -> None:
    """Stamp every captured frame with the truth about the ball, at capture
    time (before the latency), when the frame does not report it."""
    orig = det.capture

    def capture(data, t):
        fr = orig(data, t)
        fr.truth = None if any(x.cls == "ball" for x in fr.detections) else classify(det, data, tgt)
        return fr
    det.capture = capture


def _servo(cmd: np.ndarray, act: np.ndarray) -> dict:
    """The head yaw servo: for each lag, the least-squares gain of the camera's
    true yaw on the command that many ticks earlier, and the fit's residual.
    The best lag is the delay; the gain there is what a command buys."""
    n = len(cmd)
    moving = np.abs(cmd) > 0.05
    if n < 4 * MAX_LAG or moving.sum() < 200:
        return {"lagTicks": None, "gain": None, "rms": None, "movingFrac": float(moving.mean()) if n else 0.0}
    best = None
    for k in range(MAX_LAG + 1):
        c, a = cmd[:n - k], act[k:]
        m = np.abs(c) > 0.05
        if m.sum() < 100:
            continue
        g = float(np.dot(c[m], a[m]) / max(np.dot(c[m], c[m]), 1e-9))
        rms = float(np.sqrt(np.mean((a[m] - g * c[m]) ** 2)))
        if best is None or rms < best[2]:
            best = (k, g, rms)
    if best is None:
        return {"lagTicks": None, "gain": None, "rms": None, "movingFrac": float(moving.mean())}
    return {"lagTicks": best[0], "gain": round(best[1], 3), "rms": round(best[2], 3),
            "movingFrac": round(float(moving.mean()), 4)}


def run(seed: int, seconds: float, per_side: int, knobs: str = "", ball_out_s: float = 5.0) -> dict:
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=ball_out_s)
    teams: dict = {}
    brains: dict[str, Chase] = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    live = {k: getattr(next(iter(brains.values())).p, k) for k in sorted(ChaseParams.env_names())}
    want = ChaseParams.from_env()
    for k, v in live.items():
        assert v == getattr(want, k), f"{k}: live brain has {v!r}, MICRODUCK_CHASE asked {getattr(want, k)!r}"
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    ball_dof = int(w.model.jnt_dofadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    for d in w.ducks.values():
        tgt = next(x for x in d.detector.targets if x.cls == "ball")
        _hook(d.detector, tgt)
    metrics = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    goal_seq = 0
    p0 = next(iter(brains.values())).p
    ticks = seen_ticks = 0
    frames = blind_frames = 0
    cause_frames: Counter = Counter()                 # blind frames by cause
    cause_state: dict = defaultdict(Counter)          # blind frames by cause x brain state
    coast_gap = 0                                     # blind frames with a track older than predict_s and younger than coast_s
    coast_gap_reach = 0                               # ...of those, the ball inside the head's reach (h_out only)
    losses: list[dict] = []
    last_ball_t: dict[str, float] = {d.id: 0.0 for d in sc.ducks}
    last_frame_t: dict[str, float] = {d.id: -1.0 for d in sc.ducks}
    last_truth: dict[str, dict | None] = {d.id: None for d in sc.ducks}
    open_loss: dict[str, dict | None] = {d.id: None for d in sc.ducks}
    cmd_hist: dict[str, list] = {d.id: [] for d in sc.ducks}
    act_hist: dict[str, list] = {d.id: [] for d in sc.ducks}
    half_h = math.radians(next(iter(w.ducks.values())).detector.spec.fov_h_deg) / 2
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]
            intent = b.step(s)
            ticks += 1
            head = intent.head or (0.0, 0.0, 0.0, 0.0)
            cmd_hist[d.id].append(float(head[2]))
            # The camera's true yaw off the body, now.
            R = w.data.site_xmat[d.detector.site_id].reshape(3, 3)
            Rb = w.data.xmat[d.detector.own_root].reshape(3, 3)
            act_hist[d.id].append(_wrap(math.atan2(R[1, 0], R[0, 0]) - math.atan2(Rb[1, 0], Rb[0, 0])))
            track = b.tracker.best("ball", w.t, min_hits=1)
            track_age = None if track is None else w.t - track.last_t
            fr = s.fresh_tof(Chase.TOF_MAX_AGE)
            ahead = tof_clearance_bearings(fr)[0] if fr is not None else math.inf
            gated = p0.yaw_clear > 0.0 and ahead < p0.yaw_clear
            if det is not None and det.t > last_frame_t[d.id]:
                last_frame_t[d.id] = det.t
                frames += 1
                truth = getattr(det, "truth", None)
                if truth is None:                                     # the frame carries a ball
                    if open_loss[d.id] is not None:
                        ev = open_loss[d.id]
                        ev["dur"] = round(w.t - ev["t0"], 2)
                        losses.append(ev)
                        open_loss[d.id] = None
                    last_ball_t[d.id] = w.t
                    last_truth[d.id] = None
                else:
                    blind_frames += 1
                    cause_frames[truth["cause"]] += 1
                    cause_state[truth["cause"]][b.state] += 1
                    last_truth[d.id] = truth
                    if open_loss[d.id] is not None:
                        open_loss[d.id]["causes"][truth["cause"]] += 1
                    if track_age is not None and p0.predict_s < track_age <= b.tracker.p.coast_s:
                        coast_gap += 1
                        if truth["cause"] == "h_out" and abs(truth["bearingBody"]) <= half_h + p0.head_yaw_max:
                            coast_gap_reach += 1
            if w.t - last_ball_t[d.id] <= Chase.DET_MAX_AGE:
                seen_ticks += 1
            elif open_loss[d.id] is None:
                tr = last_truth[d.id] or {}
                open_loss[d.id] = {
                    "duck": d.id, "t0": round(last_ball_t[d.id] + Chase.DET_MAX_AGE, 2),
                    "cause": tr.get("cause", "?"), "causes": Counter(),
                    "state": b.state, "cmdYaw": round(float(head[2]), 3), "cmdPitch": round(float(head[1]), 3),
                    "camYaw": tr.get("camYaw"), "camPitch": tr.get("camPitch"),
                    "bearingBody": tr.get("bearingBody"), "rng": tr.get("rng"),
                    "elevCam": tr.get("elevCam"),
                    "trackAge": None if track_age is None else round(track_age, 2),
                    "predicted": b.predicted is not None, "gated": bool(gated),
                    "standing": bool(intent.twist[0] == 0.0 and intent.twist[2] == 0.0),
                    "ballSpeed": round(float(np.hypot(*w.data.qvel[ball_dof:ball_dof + 2])), 3),
                    # Could a yawed head have kept it in the frame at all?
                    "reachable": (tr.get("bearingBody") is not None
                                  and abs(tr["bearingBody"]) <= half_h + p0.head_yaw_max),
                }
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        metrics.tick()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
    for ev in losses:
        ev["causes"] = dict(ev["causes"])
    servo = {k: _servo(np.array(cmd_hist[k]), np.array(act_hist[k])) for k in cmd_hist}
    row = {
        "seed": seed, "perSide": per_side, "seconds": seconds, "simSeconds": round(w.t, 1),
        "live": {k: (v if not isinstance(v, float) else round(v, 4)) for k, v in live.items()},
        "ticks": ticks, "viewFrac": round(seen_ticks / max(ticks, 1), 4),
        "frames": frames, "blindFrames": blind_frames,
        "causeFrames": dict(cause_frames),
        "causeState": {c: dict(v) for c, v in cause_state.items()},
        "coastGap": coast_gap, "coastGapReach": coast_gap_reach,
        "losses": losses,
        "servo": servo,
        "kicks": {k: b.kicks for k, b in brains.items()},
        "falls": {k: d.falls for k, d in w.ducks.items()},
        **metrics.row(),
    }
    sc_ = w.soccer_score()
    row.update(left=sc_["left"], right=sc_["right"])
    return row


def _mean_team(rows: list[dict], field: str) -> float:
    vals = [v for r in rows if isinstance(r.get(field), dict) for v in r[field].values() if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    view = np.array([r["viewFrac"] for r in rows])
    losses = [ev for r in rows for ev in r["losses"]]
    dur = np.array([ev["dur"] for ev in losses]) if losses else np.array([])
    blind = sum(r["blindFrames"] for r in rows)
    frames = sum(r["frames"] for r in rows)
    cf: Counter = Counter()
    for r in rows:
        cf.update(r["causeFrames"])
    by_cause: dict = {}
    for c in CAUSES:
        evs = [ev for ev in losses if ev["cause"] == c]
        d = np.array([ev["dur"] for ev in evs]) if evs else np.array([])
        by_cause[c] = {
            "frames": cf.get(c, 0), "events": len(evs),
            "median": float(np.median(d)) if len(d) else float("nan"),
            "over2s": float((d > 2.0).mean()) if len(d) else float("nan"),
            "reachable": float(np.mean([ev["reachable"] for ev in evs])) if evs else float("nan"),
            "gated": float(np.mean([ev["gated"] for ev in evs])) if evs else float("nan"),
            "tracked": float(np.mean([ev["trackAge"] is not None for ev in evs])) if evs else float("nan"),
            "standing": float(np.mean([ev["standing"] for ev in evs])) if evs else float("nan"),
            "rolling": float(np.mean([ev["ballSpeed"] > 0.1 for ev in evs])) if evs else float("nan"),
            "cmdYawMed": float(np.median([abs(ev["cmdYaw"]) for ev in evs])) if evs else float("nan"),
            "bearingMed": float(np.median([abs(ev["bearingBody"]) for ev in evs if ev["bearingBody"] is not None]))
            if any(ev["bearingBody"] is not None for ev in evs) else float("nan"),
            "rngMed": float(np.median([ev["rng"] for ev in evs if ev["rng"] is not None]))
            if any(ev["rng"] is not None for ev in evs) else float("nan"),
        }
    states: Counter = Counter(ev["state"] for ev in losses)
    lag = [r["servo"][k]["lagTicks"] for r in rows for k in r["servo"] if r["servo"][k]["lagTicks"] is not None]
    gain = [r["servo"][k]["gain"] for r in rows for k in r["servo"] if r["servo"][k]["gain"] is not None]
    rms = [r["servo"][k]["rms"] for r in rows for k in r["servo"] if r["servo"][k]["rms"] is not None]
    kicks = np.array([sum(r["kicks"].values()) for r in rows])
    falls = np.array([sum(r["falls"].values()) for r in rows])
    goals = np.array([r["left"] + r["right"] for r in rows])
    return {
        "label": label, "seeds": n,
        "viewFrac": float(view.mean()), "viewSeeds": view,
        "losses": len(losses), "lossPerRun": len(losses) / max(n, 1),
        "lossMedian": float(np.median(dur)) if len(dur) else float("nan"),
        "lossP90": float(np.percentile(dur, 90)) if len(dur) else float("nan"),
        "lossOver2s": float((dur > 2.0).mean()) if len(dur) else float("nan"),
        "lossSeedMed": np.array([np.median([ev["dur"] for ev in r["losses"]]) if r["losses"] else float("nan") for r in rows]),
        "blindFrames": blind, "frames": frames,
        "causeFrames": {c: cf.get(c, 0) for c in CAUSES},
        "byCause": by_cause, "states": states,
        "coastGap": sum(r["coastGap"] for r in rows), "coastGapReach": sum(r["coastGapReach"] for r in rows),
        "lagMed": float(np.median(lag)) if lag else float("nan"),
        "gainMed": float(np.median(gain)) if gain else float("nan"),
        "rmsMed": float(np.median(rms)) if rms else float("nan"),
        "kicks": kicks, "falls": falls, "goals": goals,
        "possession": np.array([np.mean([v for v in r["possession"].values() if v is not None]) for r in rows]),
        "ballProgress": _mean_team(rows, "ballProgress"),
    }


def report(s: dict) -> None:
    print(f"\n=== {s['label']}  ({s['seeds']} seeds) ===")
    print(f"  ball in view   {100 * s['viewFrac']:.1f}% of ticks")
    print(f"  losses         {s['losses']} events ({s['lossPerRun']:.1f} a run); median {s['lossMedian']:.2f} s, "
          f"p90 {s['lossP90']:.2f}, {100 * s['lossOver2s']:.0f}% over 2 s")
    print(f"  blind frames   {s['blindFrames']} of {s['frames']} ({100 * s['blindFrames'] / max(s['frames'], 1):.1f}%)")
    print(f"  coast gap      {s['coastGap']} blind frames with a track older than predict_s and younger than coast_s "
          f"({100 * s['coastGap'] / max(s['blindFrames'], 1):.1f}% of blind frames); "
          f"{s['coastGapReach']} of them h_out inside the head's reach")
    print(f"  head servo     lag {s['lagMed']:.0f} ticks ({20 * s['lagMed']:.0f} ms), gain {s['gainMed']:.2f}, "
          f"rms {s['rmsMed']:.3f} rad (medians over ducks)")
    print(f"  ledger         kicks {s['kicks'].mean():.2f}/run  falls {s['falls'].mean():.2f}  goals {s['goals'].mean():.2f}"
          f"  possession {s['possession'].mean():.2f} s/min")
    print(f"  {'cause':<10}{'blind frames':>14}{'events':>8}{'median s':>10}{'>2 s':>7}{'reach':>7}{'gated':>7}"
          f"{'stand':>7}{'roll':>6}{'|cmd|':>7}{'|bear|':>8}{'range':>7}")
    for c in CAUSES:
        b = s["byCause"][c]
        if not b["frames"] and not b["events"]:
            continue
        print(f"  {c:<10}{b['frames']:>7} {100 * b['frames'] / max(s['blindFrames'], 1):>5.1f}%{b['events']:>8}"
              f"{b['median']:>10.2f}{100 * b['over2s']:>6.0f}%{100 * b['reachable']:>6.0f}%{100 * b['gated']:>6.0f}%"
              f"{100 * b['standing']:>6.0f}%{100 * b['rolling']:>5.0f}%{b['cmdYawMed']:>7.2f}"
              f"{math.degrees(b['bearingMed']) if not math.isnan(b['bearingMed']) else float('nan'):>7.0f}°"
              f"{b['rngMed']:>7.2f}")
    top = ", ".join(f"{k} {v}" for k, v in s["states"].most_common(8))
    print(f"  loss starts by state: {top}")


def compare(base: dict, arm: dict, rows_a: list[dict], rows_b: list[dict]) -> None:
    print(f"\n--- {arm['label']} vs {base['label']} (paired on {min(base['seeds'], arm['seeds'])} seeds) ---")
    # (name, baseline, arm, lower is better): "better on" counts the seeds the
    # arm IMPROVED, whichever way the metric runs.
    rows = [
        ("ball in view (pp)", 100 * base["viewSeeds"], 100 * arm["viewSeeds"], False),
        ("median loss (s, per seed)", base["lossSeedMed"], arm["lossSeedMed"], True),
        ("kicks / run", base["kicks"].astype(float), arm["kicks"].astype(float), False),
        ("falls / run", base["falls"].astype(float), arm["falls"].astype(float), True),
        ("goals / run", base["goals"].astype(float), arm["goals"].astype(float), False),
        ("possession s/min", base["possession"], arm["possession"], False),
    ]
    for name, a, b, lower in rows:
        m = ~(np.isnan(a) | np.isnan(b))
        d, half, p, _ = paired(a[m], b[m])
        won = int(((b[m] < a[m]) if lower else (b[m] > a[m])).sum())
        verdict = "effect" if half < abs(d) else ("null" if half <= 0.15 * abs(np.nanmean(a)) + 1e-9 else "NO RESULT")
        print(f"  {name:<28} {np.nanmean(a):>8.2f} -> {np.nanmean(b):>8.2f}   diff {d:+.3f} ± {half:.3f}  "
              f"p={p:.3f}  better on {won}/{int(m.sum())}  [{verdict}]")
    la = [ev["dur"] for r in rows_a for ev in r["losses"]]
    lb = [ev["dur"] for r in rows_b for ev in r["losses"]]
    print(f"  loss events {len(la)} -> {len(lb)}; over 2 s {100 * np.mean(np.array(la) > 2):.0f}% -> "
          f"{100 * np.mean(np.array(lb) > 2):.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ball-out-s", type=float, default=5.0, help="the lab's referee (0 = off)")
    ap.add_argument("--arm", action="append", default=None, metavar="LABEL=KNOBS",
                    help="a brain to run, MICRODUCK_CHASE syntax; the first is the baseline. Default: the shipped brain")
    ap.add_argument("--out", default=None, help="a directory: each arm's rows go to <out>/<label>.jsonl")
    args = ap.parse_args()
    arms = [a.split("=", 1) for a in (args.arm or ["shipped="])]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    results: list[tuple[dict, list[dict]]] = []
    for label, knobs in arms:
        todo = [(s, args.seconds, args.per_side, knobs, args.ball_out_s) for s in seeds]
        if args.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(min(args.jobs, len(todo))) as pool:
                rows = list(pool.starmap(run, todo))
        else:
            rows = [run(*a) for a in todo]
        rows.sort(key=lambda r: r["seed"])
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with open(Path(args.out) / f"{label}.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        print(f"\n[{label}] MICRODUCK_CHASE={knobs!r}  live: "
              + json.dumps({k: v for k, v in rows[0]["live"].items()
                            if v != getattr(ChaseParams(), k)}))
        s = summarize(rows, label)
        report(s)
        results.append((s, rows))
    for s, rows in results[1:]:
        compare(results[0][0], s, results[0][1], rows)


if __name__ == "__main__":
    main()
