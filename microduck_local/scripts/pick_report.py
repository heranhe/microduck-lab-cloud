"""THE STANDARD REPORT for a MOSS pick policy — every number this project has
learned to care about, in one place.

    uv run python scripts/pick_report.py runs/<run>/policy.onnx [seeds]

Built by consolidating a day of throwaway probes (2026-09-25). Each line here
exists because a claim was once made without it and turned out to be wrong:

  * picks / can displacement  — reward curves lied about both, repeatedly.
  * wrist-vs-axis SLOPE       — "the wrist does not aim" was argued from
                                TRAVEL, which cannot tell moving a lot from
                                arriving anywhere. The slope, with its p, is
                                the claim's actual subject.
  * base command + chassis turn — the spinning a human watches.
  * floor drag                — the gripper scraped along the ground under
                                power on 3.2% of ticks with nothing charging
                                for it.
  * torque saturation         — shoulder_lift sat at its clamp 38% of ticks;
                                MOSS has no BAM, so nothing else reports this.

It reads every flag off the RUN (`eval_env_kwargs`) — what the policy SAW
and what it was ALLOWED TO DO — because a
policy evaluated in the wrong observation reports a collapse that is not real:
that mistake measured 3 grips where the policy makes 63.
"""
import math
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from microduck_local.robots import moss
from microduck_local.robots.moss_env import MossPickEnv, eval_env_kwargs


def report(path: str, seeds: int = 40, rung: int = 2) -> dict:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    kw = eval_env_kwargs(path)
    picks = ticks = floor = drag = low = sat = 0
    base, yaw, disp, wrist, axis = [], [], [], [], []
    for seed in range(seeds):
        env = MossPickEnv(seed=seed, pick_rung=rung, **kw)
        obs, _ = env.reset()
        ids = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g): g
               for g in range(env.model.ngeom)}
        acts = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                for i in range(env.model.nu)}
        ag = getattr(env, "_arm_geoms", set())
        c, sn, up = env._true_attitude()
        lying = up <= 0.5
        theta = math.degrees(0.5 * math.atan2(sn, c))
        spawn = np.array(env.data.xpos[env.can_body][:2], float)
        y0 = env.driver.pose(env.data)[2]
        ymin = ymax = y0
        worst = bs = 0.0
        grip_w = None
        n = 0
        info = {}
        while n < int(env.max_steps):
            a = sess.run(None, {nm: obs.reshape(1, -1).astype(np.float32)})[0][0]
            held = env._held()
            obs, _r, t, tr, info = env.step(a)
            n += 1
            ticks += 1
            bs += (abs(float(a[moss.ACT_BASE.start]))
                   + abs(float(a[moss.ACT_BASE.start + 1])))
            y = env.driver.pose(env.data)[2]
            ymin, ymax = min(ymin, y), max(ymax, y)
            if not held:
                worst = max(worst, float(np.linalg.norm(
                    np.array(env.data.xpos[env.can_body][:2], float) - spawn)))
            if grip_w is None and env._closed_on_can():
                grip_w = math.degrees(float(env.data.qpos[
                    env.model.joint("wrist_roll").qposadr[0]]))
            scrape = False
            for k in range(env.data.ncon):
                con = env.data.contact[k]
                n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                       con.geom1) or ""
                n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                       con.geom2) or ""
                pr = {n1, n2}
                if (pr & {"floor", "ground"}) and (pr & ag):
                    scrape = True
                    break
            if scrape:
                floor += 1
                if abs(float(a[moss.ACT_BASE.start])) > 0.1:
                    drag += 1
            ai = acts.get("shoulder_lift")
            if ai is not None:
                lim = float(env.model.actuator_forcerange[ai][1])
                if abs(float(env.data.actuator_force[ai])) >= 0.995 * lim:
                    sat += 1
            tcp = np.asarray(env.data.site_xpos[env.tcp_site], float)
            can = np.asarray(env.data.xpos[env.can_body], float)
            u = abs(env._true_attitude()[2])
            top = float(can[2]) + (env.prop.half_height * u
                                   + env.prop.radius * (1.0 - u))
            if (not held
                    and float(np.hypot(tcp[0] - can[0], tcp[1] - can[1])) > 0.08
                    and float(tcp[2]) < top):
                low += 1
            if t or tr:
                break
        picks += int(bool(info.get("picked")))
        base.append(bs / max(n, 1))
        yaw.append(math.degrees(ymax - ymin))
        disp.append(worst)
        if lying and grip_w is not None:
            wrist.append(grip_w)
            axis.append(theta)

    out = {"picked": picks, "seeds": seeds,
           "base_cmd": float(np.mean(base)),
           "chassis_turn_deg": float(np.median(yaw)),
           "can_moved_cm": float(np.median(disp) * 100),
           "floor_pct": floor / max(ticks, 1),
           "drag_pct": drag / max(ticks, 1),
           "low_pct": low / max(ticks, 1),
           "sat_pct": sat / max(ticks, 1),
           "obs_flags": kw}
    if len(axis) > 3:
        A, W = np.array(axis), np.array(wrist)
        r = float(np.corrcoef(A, W)[0, 1])
        out["aim_slope"] = float(np.polyfit(A, W, 1)[0])
        out["aim_r"] = r
        out["aim_n"] = len(A)
        t = abs(r) * math.sqrt(max(len(A) - 2, 1)) / math.sqrt(max(1 - r * r, 1e-9))
        # two-sided normal approximation; n is 40-70 here so it is close enough
        out["aim_p"] = float(2.0 * (1.0 - 0.5 * (
            1.0 + math.erf(t / math.sqrt(2.0)))))
        out["wrist_sd_deg"] = float(W.std())
    return out


def main() -> None:
    path = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    d = report(path, seeds)
    name = path.split("/")[-2]
    print(f"== {name}   ({seeds} seeds, rung 2)")
    print(f"   obs flags        {d['obs_flags']}")
    print(f"   picked           {d['picked']}/{d['seeds']}")
    print(f"   can moved        {d['can_moved_cm']:.1f} cm")
    print(f"   base command     {d['base_cmd']:.3f}     chassis turn "
          f"{d['chassis_turn_deg']:.1f} deg")
    print(f"   floor contact    {d['floor_pct']:.1%}     DRAGGING "
          f"{d['drag_pct']:.1%}")
    print(f"   low approach     {d['low_pct']:.1%}     shoulder_lift at clamp "
          f"{d['sat_pct']:.1%}")
    if "aim_slope" in d:
        verdict = "AIMS" if d["aim_p"] < 0.05 else "no aim (inside noise)"
        print(f"   wrist vs axis    slope {d['aim_slope']:+.3f}  r {d['aim_r']:+.3f}"
              f"  p {d['aim_p']:.3f}  n {d['aim_n']}   -> {verdict}")
        print(f"   wrist motion     sd {d['wrist_sd_deg']:.1f} deg")
    else:
        print("   wrist vs axis    too few lying cans gripped to say")


if __name__ == "__main__":
    main()
