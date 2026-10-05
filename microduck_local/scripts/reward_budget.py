"""WHAT IS THIS POLICY ACTUALLY PAID? Term by term, per episode.

    uv run python scripts/reward_budget.py runs/<run>/policy.onnx

**Why this exists.** MOSS's pick reward is a sum of nine hand-tuned terms and
nothing reported their relative sizes, so on 2026-09-25 the progress term was
retargeted from the chassis to the gripper WITHOUT rescaling, fell from about
+18 per episode to +1.47, and two full training runs went by before anyone
noticed the task had lost its main shaping signal. The penalties meanwhile
summed to -16.01 against a +17.50 success bonus: completing the whole task
netted +4.38, down from roughly +26, which reads from the outside as a robot
wandering aimlessly -- and it was, because almost nothing was left to aim at.

Run this after ANY change to what a term measures. A term whose contribution
moved by an order of magnitude is a redesign, not a tweak.

The question behind "it just drives around": if driving still earns more than
everything else costs, driving is correct behaviour and no amount of extra
penalty that it can absorb will change it.

Also checks each new knob's REACHABLE SET — a penalty on something that never
happens is a no-op, which this repo has shipped before.
"""
import sys, math
import numpy as np, onnxruntime as ort
from microduck_local.robots import moss
from microduck_local.robots import moss_env as me
from microduck_local.robots.moss_env import MossPickEnv

path = sys.argv[1]
# READ THE FLAGS OFF THE RUN, never hardcode them: a hardcoded set measured a
# base-locked policy turning its chassis 26.7 degrees, which it physically
# could not do. Same mistake as `pick_report.py` had, found the same day.
from microduck_local.robots.moss_env import eval_env_kwargs
KW = dict(pick_rung=2, wrist_start_rand=math.pi/2, jaw_align=6.0,
          wrist_free=True, can_topple=15.0, **eval_env_kwargs(path))
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
nm = sess.get_inputs()[0].name

tot = dict(progress=0.0, base=0.0, jerk=0.0, topple=0.0, align=0.0,
           disturb=0.0, bodyhit=0.0)
picks = 0
eps = 40
upright_lost, wrist_travel, base_cmd, chassis_turn = [], [], [], []
for seed in range(eps):
    env = MossPickEnv(seed=seed, **KW)
    obs, _ = env.reset()
    prev_gap = env._reward_gap()
    prev_al, _l = env._jaw_alignment()
    prev_up = abs(env._true_attitude()[2])
    prev_a = np.zeros(env.action_space.shape[0], np.float32)
    w0 = float(env.data.qpos[env.model.joint("wrist_roll").qposadr[0]])
    wmin = wmax = w0
    y0 = env.driver.pose(env.data)[2]; ymin = ymax = y0
    lost = 0.0; bsum = 0.0; n = 0
    info = {}
    while n < int(env.max_steps):
        a = sess.run(None, {nm: obs.reshape(1, -1).astype(np.float32)})[0][0]
        held = env._held()
        obs, _r, t, tr, info = env.step(a)
        g = env._reward_gap()
        tot["progress"] += me.W_PROGRESS * (prev_gap - g); prev_gap = g
        # ...but NOT when the base is locked: the env stops charging for a
        # command it discards, and a script that keeps charging reports a
        # -11.95 penalty that is not in the reward at all.
        if not env.base_lock:
            tot["base"] += me.W_BASE_EFFORT * (
                abs(float(a[moss.ACT_BASE.start]))
                + abs(float(a[moss.ACT_BASE.start+1])))
            tot["base"] += me.W_BASE_YAW * abs(float(a[moss.ACT_BASE.start+1]))
        j = np.abs(a - prev_a); j[me.ACT_WRIST_ROLL] = 0.0
        tot["jerk"] += me.W_ACTION_RATE * float(j.sum()); prev_a = a
        al, ly = env._jaw_alignment()
        tot["align"] += 6.0 * ly * (al - prev_al); prev_al = al
        up = abs(env._true_attitude()[2])
        if not held:
            d = max(0.0, prev_up - up); lost += d
            tot["topple"] -= 15.0 * d
        prev_up = up
        bsum += abs(float(a[moss.ACT_BASE.start])) + abs(float(a[moss.ACT_BASE.start+1]))
        w = float(env.data.qpos[env.model.joint("wrist_roll").qposadr[0]])
        wmin, wmax = min(wmin, w), max(wmax, w)
        y = env.driver.pose(env.data)[2]; ymin, ymax = min(ymin, y), max(ymax, y)
        n += 1
        if t or tr:
            break
    picks += int(bool(info.get("picked")))
    upright_lost.append(lost); wrist_travel.append(math.degrees(wmax-wmin))
    base_cmd.append(bsum/max(n,1)); chassis_turn.append(math.degrees(ymax-ymin))

print(f"{path.split('/')[-2]}  ({eps} episodes, rung 2)   picked {picks}/{eps}")
print()
print("  REWARD, per episode, by term:")
for k, v in sorted(tot.items(), key=lambda kv: -abs(kv[1])):
    print(f"     {k:9s} {v/eps:+8.2f}")
print(f"     {'SUCCESS':9s} {20.0*picks/eps:+8.2f}   (the +20 bonus, averaged)")
print()
print("  REACHABLE SETS of the new knobs:")
print(f"     uprightness LOST per episode : median {np.median(upright_lost):.3f}  "
      f"mean {np.mean(upright_lost):.3f}   -> topple charge is "
      f"{'LIVE' if np.mean(upright_lost) > 0.02 else 'A NO-OP'}")
print(f"     wrist travel                 : median {np.median(wrist_travel):5.1f} deg")
print(f"     chassis turned               : median {np.median(chassis_turn):5.1f} deg")
print(f"     base command per step        : {np.mean(base_cmd):.3f}")
