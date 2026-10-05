"""DAgger for the retract: label the STUDENT's own states with the teacher.

Plain behaviour cloning put the fold in the weights only partially — forward
reach came in from 178 mm to 128 mm (tuck is 114.3) but the arm folded home
0/28. That is the textbook failure: the student only ever sees the TEACHER's
trajectory, so the moment its own small errors carry it off that path it is in
states it was never shown, and the errors compound. This repo already recorded
the same shape of failure on the duck clone — "the teacher can turn; a student
that has never seen a turn cannot".

DAgger fixes exactly this: roll out the STUDENT, ask the TEACHER what it would
have done at each of the student's states, and add those labels. The dataset
then covers where the student actually goes, not where the teacher went.

The teacher here is cheap to query at any state — it is "drive toward the
universal waypoint, then toward the tuck pose" — so labelling is exact rather
than approximate.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from microduck_local.distill import fit
from microduck_local.robots import moss
import microduck_local.robots.moss_env as me
from microduck_local.robots.moss_env import (ARM_DELTA_RAD, MossStowEnv,
                                             eval_env_kwargs)

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("plan_retract", HERE / "plan_retract.py")
pr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pr)
TUCK = np.asarray(moss.tuck_pose(), float)


def teacher_action(e, adrs, reached_wp):
    """What the teacher would do HERE: waypoint first, then tuck."""
    cur = np.asarray([e.data.qpos[a] for a in adrs], float)
    tgt = TUCK if reached_wp else pr.UNIVERSAL_WAYPOINT
    if not reached_wp and float(np.abs(cur - tgt).max()) < 0.08:
        reached_wp = True
        tgt = TUCK
    a = np.zeros(e.action_space.shape[0], np.float32)
    a[moss.ACT_ARM] = np.clip((tgt - cur) / ARM_DELTA_RAD, -1, 1)
    return a, reached_wp


def main() -> None:
    stow, student, out = sys.argv[1], sys.argv[2], sys.argv[3]
    eps = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    me.RETRACT_MAX_STEPS = 3000
    pol = ort.InferenceSession(f"runs/{stow}/policy.onnx",
                               providers=["CPUExecutionProvider"])
    pn = pol.get_inputs()[0].name
    stu = ort.InferenceSession(f"runs/{student}/policy.onnx",
                               providers=["CPUExecutionProvider"])
    sn = stu.get_inputs()[0].name
    kw = eval_env_kwargs(f"runs/{stow}/policy.onnx")
    kw.pop("base_lock", None)
    O, A = [], []
    for ep in range(eps):
        e = MossStowEnv(seed=5000 + ep, retract=6.0, bin_scrape=0.25,
                        prop_variety=True, **kw)
        obs, _ = e.reset()
        n = 0
        while n < 200 and not getattr(e, "_retracting", False):
            a = pol.run(None, {pn: obs.reshape(1, -1).astype(np.float32)})[0][0]
            O.append(obs.copy())
            A.append(np.asarray(a, np.float32).copy())
            obs, _r, t, tr, _i = e.step(a)
            n += 1
            if t or tr:
                break
        if not getattr(e, "_retracting", False):
            continue
        adrs = [e.model.joint(j).qposadr[0] for j in moss.ARM_JOINTS[:5]]
        # AGGREGATE, do not replace. DAgger is D <- D union D_new; refitting on
        # the new labels ALONE collapsed delivery 17/32 -> 6/32, because a
        # student rollout is ~420 retract steps against ~10 delivery ones, so
        # the delivery behaviour was simply outvoted. The teacher's own
        # demonstration of this episode goes in first.
        snap_q, snap_v = e.data.qpos.copy(), e.data.qvel.copy()
        snap_cmd = dict(e.arm_cmd)
        got = False
        tobs = obs.copy()
        for _ in range(420):
            lab, got = teacher_action(e, adrs, got)
            O.append(tobs.copy())
            A.append(lab.copy())
            tobs, _r, t, tr, _i = e.step(lab)
            if t or tr:
                break
        e.data.qpos[:], e.data.qvel[:] = snap_q, snap_v
        e.arm_cmd = snap_cmd
        import mujoco as _mj
        _mj.mj_forward(e.model, e.data)
        got_wp = False
        for _ in range(420):
            # the STUDENT drives...
            act = stu.run(None, {sn: obs.reshape(1, -1).astype(np.float32)})[0][0]
            # ...and the TEACHER labels the state it put itself in
            lab, got_wp = teacher_action(e, adrs, got_wp)
            O.append(obs.copy())
            A.append(lab.copy())
            obs, _r, t, tr, _i = e.step(act)
            if t or tr:
                break
    print(f"  {len(O)} labelled steps from {eps} student rollouts")
    mse = fit(np.asarray(O, np.float32), np.asarray(A, np.float32),
              Path("runs") / out, epochs=120, robot="moss", task="stow")
    print(f"  DAgger round into runs/{out}   action MSE {mse:.5f}")


if __name__ == "__main__":
    main()
