"""WHO touches the ball during the settle? (roadmap 12ak) kick_gym's episode, with the real
chase brain and walker; while the brain is in `settle` (and the kick window
after it), every contact between the ball and a duck geom is counted by the
body it belongs to, and the ball's displacement over the settle is recorded.
    cd microduck_local
    uv run python scripts/probe_settle_contact.py --arm "base=" --arm "sg=settle_gaze_neck=1.0,settle_head_down=1.0"
"""
import argparse, math, os, sys, statistics as st
from collections import Counter
import mujoco, numpy as np
sys.path.insert(0, "scripts")
from kick_gym import EPISODE_S, _drive, _place, gym_scenario  # noqa: E402
from microduck_local.brain import REGISTRY  # noqa: E402
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.world.arena import World  # noqa: E402

def run(seed, episodes, knobs):
    if knobs: os.environ["MICRODUCK_CHASE"] = knobs
    else: os.environ.pop("MICRODUCK_CHASE", None)
    sc = gym_scenario(); infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed)
    bk = __import__("microduck_local.brain.team", fromlist=["brain_kwargs"]).brain_kwargs
    brains = {x.id: REGISTRY.make("chase", **bk(x, w, {})) for x in sc.ducks}
    brain = brains["d0"]; d = w.ducks["d0"]; m = w.model
    ball_bid = m.body("ball0").id
    ball_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == ball_bid}
    rng = np.random.default_rng(seed); out = []
    for ep in range(episodes):
        q, v = _place(w, rng, 0.8)
        for b in brains.values(): b.reset()
        t0 = w.t; touched = Counter(); settle_t = None; ball_at_settle = None; disp = None; swung = False; settle_steps = 0
        prev = None
        while w.t - t0 < EPISODE_S:
            _drive(w, brains); w.step()
            if brain.state == "settle":
                if settle_t is None:
                    settle_t = w.t; ball_at_settle = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
                settle_steps += 1
                for c in w.data.contact[:w.data.ncon]:
                    for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                        if g in ball_geoms and other not in ball_geoms:
                            touched[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])] += 1
            if d.skill is not None and prev is None and str(d.skill).startswith("kick"):
                swung = True
                if ball_at_settle is not None:
                    disp = math.dist(ball_at_settle, (float(w.data.qpos[q]), float(w.data.qpos[q + 1])))
                break
            prev = d.skill
        out.append({"ep": ep, "swung": swung, "settle_s": None if settle_t is None else settle_steps * 0.02, "disp": disp, "touched": dict(touched)})
    return out

ap = argparse.ArgumentParser(); ap.add_argument("--arm", action="append", default=None); ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()
for spec in (a.arm or ["base="]):
    label, knobs = spec.split("=", 1)
    rows = [r for s in range(a.seeds) for r in run(s, a.episodes, knobs)]
    sw = [r for r in rows if r["swung"] and r["disp"] is not None]
    tot = Counter(); n_touch = 0
    for r in sw:
        tot.update(r["touched"]); n_touch += bool(r["touched"])
    print(f"\n== {label} ({knobs or 'shipped'}): {len(rows)} episodes, {len(sw)} settled and swung; settle length med {st.median(r['settle_s'] for r in sw):.2f} s")
    print(f"   ball moved during the settle: median {st.median(r['disp'] for r in sw)*100:.1f} cm, > 2 cm on {sum(1 for r in sw if r['disp']>0.02)/len(sw):.0%} of swings; episodes with any ball contact in the settle: {n_touch}/{len(sw)}")
    print("   contact steps by body:", dict(tot.most_common(8)))
