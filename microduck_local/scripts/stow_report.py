"""THE STANDARD REPORT for a MOSS stow policy — delivery AND retraction.

    uv run python scripts/stow_report.py runs/<run>/policy.onnx [seeds] [rung]

`pick_report.py` evaluates in the PICK env, so pointing it at a stow policy
reports "0/40 picked" — which is what the training loop did on 2026-09-25 and
is not a result, it is the wrong environment. A stow run needs its own report.

It prints TWO evaluations, because they answer different questions:

  * THE TASK, from the rung the run TRAINED on (read off its `run.json`).
    Until 2026-09-26 this report never passed `stow_rung`, so every stow
    number it printed — "delivered 14/24" in the 09-25 handoff included —
    was measured at rung 0, with the arm STARTING already swung over the
    bin, for policies trained from the lift pose at rung 2.
  * THE FOLD ALONE, from a post-delivery pose (`retract_drill`): object in
    the bin, arm over it. Printed for every stow policy, so a drill-trained
    run has a baseline to beat and a task-trained run shows whether its
    fold failure is the fold or the delivery in front of it.

What each line measures:

  * delivered          — the can at rest in the bin.
  * arm HOME           — folded to `moss.tuck_pose()` within RETRACT_TOL_RAD
                         on every fold joint. Without it the arm finishes
                         extended over the bin, the pose the robot would then
                         try to DRIVE in.
  * bin contact        — the arm scraping the bin, which is how a gripper
                         drags its own load back out with it.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from microduck_local.robots import moss
from microduck_local.robots.moss_env import (MossStowEnv, RETRACT_JOINTS,
                                             RETRACT_TOL_RAD, eval_env_kwargs)

#: reward knobs a run predating `run.json`'s env_kwargs was scored with; they
#: change what the env PAYS, and the retract ones also keep the episode
#: running after delivery so the fold can be seen at all.
_DEFAULTS = dict(retract=6.0, bin_scrape=0.25, home_bonus=25.0,
                 prop_variety=True)
_STOW_KNOBS = ("retract", "bin_scrape", "home_bonus", "retract_staged",
               "retract_wp_bonus", "prop_variety", "stow_rung", "bin_clutter",
               "drop_target", "drop_accuracy", "overspeed", "cmd_leash",
               "valid_start")


def _episodes(sess, nm, kw, seeds, tuck, blind_drop=False):
    """`blind_drop`: score landings against the chosen drop point but feed
    the policy ZEROS in `moss.OBS_DROP` — for a policy trained without the
    drop slots, whose normaliser would clip any value there."""
    delivered = home = scrape = ticks = 0
    resid, steps, land, peak = [], [], [], []
    for seed in range(seeds):
        env = MossStowEnv(seed=seed, **kw)
        # the env lets MICRODUCK_MOSS_STOW_RUNG override its kwarg; say so
        # rather than print a rung that was not the one run
        assert env.stow_rung == kw["stow_rung"], (env.stow_rung, kw)
        assert env.retract_drill == kw["retract_drill"]
        obs, _ = env.reset()
        n, info = 0, {}
        vmax = 0.0
        while n < 600:
            if blind_drop:
                obs = obs.copy()
                obs[moss.OBS_DROP] = 0.0
            a = sess.run(None, {nm: obs.reshape(1, -1).astype(np.float32)})[0][0]
            obs, _r, t, tr, info = env.step(a)
            n += 1
            ticks += 1
            vmax = max(vmax, max(abs(float(env.data.qvel[
                env.model.joint(j).dofadr[0]])) for j in moss.ARM_JOINTS[:5]))
            for c in range(env.data.ncon):
                con = env.data.contact[c]
                n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                       con.geom1) or ""
                n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                       con.geom2) or ""
                if ({n1, n2} & env._arm_geoms) and any(
                        "bin" in x for x in (n1, n2)):
                    scrape += 1
                    break
            if t or tr:
                break
        ok = bool(getattr(env, "_stowed_ok", False) or info.get("stowed"))
        delivered += int(ok)
        if ok and not kw.get("retract_drill"):
            land.append((*env.landing(), getattr(env, "_release_err", None)))
        cur = np.asarray([env.data.qpos[env.model.joint(j).qposadr[0]]
                          for j in RETRACT_JOINTS], float)
        resid.append(float(np.abs(cur - tuck).sum()))
        home += int(float(np.abs(cur - tuck).max()) < RETRACT_TOL_RAD)
        steps.append(n)
        peak.append(vmax)
    _episodes.landings = land
    _episodes.peak = peak
    return delivered, home, resid, scrape / max(ticks, 1), steps


def main() -> None:
    path = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    kw = eval_env_kwargs(path)
    kw.pop("base_lock", None)          # the stow leg keeps its base
    trained = {}
    rj = Path(path).parent / "run.json"
    if rj.is_file():
        trained = json.loads(rj.read_text()).get("env_kwargs") or {}
    kw.update(_DEFAULTS)
    kw.update({k: trained[k] for k in _STOW_KNOBS if k in trained})
    kw.setdefault("stow_rung", 0)
    # EVERY stow kwarg the run recorded must reach the eval — a missing one
    # does not look like a bug, it looks like a finding (drop_target once
    # evaluated a policy with its drop slots zeroed: 8/24 in 18 steps).
    missing = sorted(set(trained) - set(kw) - {"domain_rand", "obs_noise",
                                              "retract_drill",
                                              "retract_drill_wp",
                                              "retract_drill_path",
                                              "retract_drill_bank"})
    if missing:
        raise SystemExit(f"run.json records {missing}, which this report "
                         "does not pass to the env — add them to _STOW_KNOBS")
    if len(sys.argv) > 3:
        kw["stow_rung"] = int(sys.argv[3])
    # the joints the fold is ABOUT — wrist_roll is excluded because it does
    # not move the arm's silhouette at all (measured, 0.0 mm over its range)
    _all = np.asarray(moss.tuck_pose(), float)
    tuck = _all[[moss.ARM_JOINTS.index(j) for j in RETRACT_JOINTS]]
    print(f"== {path.split('/')[-2]}   ({seeds} seeds; trained "
          f"{'on the FOLD DRILL' if trained.get('retract_drill') else 'on the task'}"
          f", rung {trained.get('stow_rung', '?')})")
    d, h, r, s, n = _episodes(sess, nm, {**kw, "retract_drill": False},
                              seeds, tuck)
    print(f" THE TASK from rung {kw['stow_rung']}")
    print(f"   delivered        {d}/{seeds}")
    print(f"   arm HOME         {h}/{seeds}   (within "
          f"{RETRACT_TOL_RAD} rad of every tuck joint)")
    print(f"   distance to rest {np.median(r):.2f} rad median")
    print(f"   bin contact      {s:.1%} of ticks")
    print(f"   episode length   {np.median(n):.0f} steps")
    if kw.get("drop_target") and _episodes.landings:
        e = np.array([x[0] for x in _episodes.landings])
        p = sum(x[1] for x in _episodes.landings)
        print(f"   landing error    {np.median(e)*1000:.0f} mm median from the "
              f"chosen spot (<3 cm: {int((e < 0.03).sum())}/{len(e)})")
        print(f"   perched          {p}/{len(e)} delivered resting on clutter")
        r = np.array([x[2] for x in _episodes.landings if x[2] is not None])
        if r.size:
            print(f"   release error    {np.median(r)*1000:.0f} mm median from the "
                  f"chosen spot at the moment it was let go")
    _d, h, r, s, n = _episodes(sess, nm, {**kw, "retract_drill": True},
                               seeds, tuck)
    print(" THE FOLD ALONE, from a post-delivery pose")
    print(f"   arm HOME         {h}/{seeds}")
    print(f"   peak joint speed {np.median(_episodes.peak):.2f} rad/s median, "
          f"{np.max(_episodes.peak):.2f} worst (commands cap at 0.75)")
    print(f"   distance to rest {np.median(r):.2f} rad median")
    print(f"   bin contact      {s:.1%} of ticks")
    print(f"   episode length   {np.median(n):.0f} steps")


if __name__ == "__main__":
    main()
