"""During a long loss, what state is the brain in, and when it searches, does it turn TOWARD the true ball?"""
import math, os, sys, json
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from probe_ball_loss import brain_kwargs
from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.world import World, make_pitch

def _wrap(a): return math.atan2(math.sin(a), math.cos(a))

def run(seed, seconds, knobs):
    if knobs: os.environ["MICRODUCK_CHASE"] = knobs
    else: os.environ.pop("MICRODUCK_CHASE", None)
    sc = make_pitch(per_side=2)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=5.0)
    teams = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q+2] = rng.uniform(-0.2, 0.2, 2)
    last_ball_t = {d.id: 0.0 for d in sc.ducks}; last_frame_t = {d.id: -1.0 for d in sc.ducks}
    rec = {d.id: [] for d in sc.ducks}
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t, det=det,
                       det_age=None if det is None else w.t - det.t, speed=d.heading_speed(w.data),
                       odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]; intent = b.step(s)
            if det is not None and det.t > last_frame_t[d.id]:
                last_frame_t[d.id] = det.t
                if any(x.cls == "ball" for x in det.detections): last_ball_t[d.id] = w.t
            Rb = w.data.xmat[d.detector.own_root].reshape(3, 3); yaw = math.atan2(Rb[1, 0], Rb[0, 0])
            px, py = w.data.xpos[d.detector.own_root][:2]; bx, by = w.data.qpos[q:q+2]
            tb = _wrap(math.atan2(by - py, bx - px) - yaw)
            tr = b.tracker.best("ball", w.t, min_hits=1)
            rec[d.id].append((w.t, b.state, float(intent.twist[2]), tb, math.hypot(bx-px, by-py),
                              None if tr is None else _wrap(tr.bearing), None if tr is None else w.t - tr.last_t,
                              w.t - last_ball_t[d.id] > Chase.DET_MAX_AGE, b.last_bearing))
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
    return rec

def analyse(recs):
    dt = 0.02
    state_s = Counter(); search_n = 0; toward = 0; alive = 0; stale_ok = 0; err_live = []; err_last = []
    long_states = Counter()
    for rec in recs:
        for duck, rows in rec.items():
            # loss stretches
            i = 0; n = len(rows)
            while i < n:
                if rows[i][7]:
                    j = i
                    while j < n and rows[j][7]: j += 1
                    dur = (j - i) * dt
                    if dur > 2.0:
                        for r in rows[i:j]: long_states[r[1]] += dt
                    i = j
                else: i += 1
            for t, st, wz, tb, rng, trb, tage, blind, lb in rows:
                state_s[st] += dt
                if st == "search" and abs(tb) > 0.3 and abs(wz) > 0.05:
                    search_n += 1
                    if wz * tb > 0: toward += 1
                    if lb * tb > 0: stale_ok += 1
                    if trb is not None:
                        alive += 1; err_live.append(abs(_wrap(trb - tb)))
                    err_last.append(abs(_wrap(lb - tb)))
    tot = sum(state_s.values())
    print("  time by state (% of duck-seconds): " + ", ".join(f"{s} {100*v/tot:.0f}%" for s, v in state_s.most_common(9)))
    lt = sum(long_states.values())
    print(f"  losses >2 s: {lt:.0f} duck-s; by state: " + ", ".join(f"{s} {100*v/lt:.0f}%" for s, v in long_states.most_common(9)))
    if search_n:
        print(f"  search ticks (ball >17° off nose, turning): {search_n}; body turning TOWARD the true ball {100*toward/search_n:.0f}%; "
              f"frozen last_bearing on the right side {100*stale_ok/search_n:.0f}%; a track alive {100*alive/search_n:.0f}%; "
              f"|track bearing - truth| median {np.degrees(np.median(err_live)) if err_live else float('nan'):.0f}°; "
              f"|last_bearing - truth| median {np.degrees(np.median(err_last)):.0f}°")

if __name__ == "__main__":
    import multiprocessing as mp
    # `--arm LABEL=KNOBS` (MICRODUCK_CHASE syntax), the first is the baseline; default: the 12af arms.
    argv = sys.argv[1:]
    arms = [tuple(a.split("=", 1)) for a in argv[argv.index("--arm") + 1::2]] if "--arm" in argv else [
        ("shipped", ""), ("pair", "rest_coast_s=30,search_sided=1"),
        ("live", "rest_coast_s=30,rest_predict_s=30,predict_steer=1,search_sided=1")]
    seeds = [0, 1, 2, 3]
    ctx = mp.get_context("forkserver")
    with ctx.Pool(6) as pool:
        out = pool.starmap(run, [(s, 180.0, k) for _, k in arms for s in seeds])
    for i, (label, knobs) in enumerate(arms):
        print(f"\n[{label}] {knobs!r}")
        analyse(out[i*len(seeds):(i+1)*len(seeds)])
