"""THE PIN: a prop that lodges against the rover and rides there for the rest
of the run.

    uv run python scripts/probe_moss_pin.py 300 27,31,38,60 out.json

Measured in the `bystander_avoid_m` battery: in 4 of 64 seeds — in BOTH arms,
so unrelated to that change — one prop is in contact with the driving surfaces
for 130-207 s of a 300 s run, against a normal run's 5 s. This finds which
prop, WHICH GEOMS it is wedged between, where it sits in the base frame, how
hard it is pressed, and what the brain knows about it while it happens.

Reports per pin: the geom pair, the cap's base-frame pose and whether it is
RIDING (fixed relative to the base) or being shoved along, the force, the
brain state timeline, and whether the detector ever reports it, the memory
ever holds it, or the machine ever targets it.
"""
import collections
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

def _seen(br, name):
    """Does the CURRENT detector frame carry this prop? `det.name` is truth —
    used to ask the question, never given to the brain."""
    fr = getattr(br, "_last_frame", None) or getattr(br, "_det_last", None)
    fr = fr if fr is not None else _last_det(br)
    if fr is None:
        return False
    return any(dt.cls == "toy" and dt.name == name for dt in fr.detections)


def _last_det(br):
    return getattr(br, "_det_frame", None)


def _in_mem(br, q):
    mem = getattr(br, "_mem", None)
    if mem is None:
        return None
    e = mem.near(float(q[0]), float(q[1]), 0.15)
    return None if e is None else bool(mem.confirmed(e))


def _report(seed, sc0, pairs, force, best, start, log):
    worst = max(best, key=lambda k: best[k]) if best else None
    print(f"\n=== seed {seed} ===")
    if worst is None or best[worst] * 0.002 < 20.0:
        print("   no pin (longest unbroken rover-prop contact under 20 s)")
        return {"seed": seed, "pin": None}
    t0 = start.get(worst + "@best", 0.0)
    dur = best[worst] * 0.002
    print(f"   PINNED: {worst}  {dur:.1f}s unbroken, starting at t={t0:.1f}s")
    print("   geoms it is wedged against (substeps, peak N):")
    for (pg, g), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if pg != worst:
            continue
        print(f"      {g:24s} {n * 0.002:7.1f}s  peak {force[(pg, g)]:5.2f} N")
    rows = [r for r in log[worst] if r[0] >= t0]
    if rows:
        xs = [r[1] for r in rows]; ys = [r[2] for r in rows]; zs = [r[3] for r in rows]
        import statistics as stt
        print(f"   in the BASE frame while pinned: x {stt.mean(xs):+.3f} "
              f"+-{stt.pstdev(xs):.3f}  y {stt.mean(ys):+.3f} +-{stt.pstdev(ys):.3f}  "
              f"z {stt.mean(zs):+.3f} +-{stt.pstdev(zs):.3f}")
        print("   -> RIDING with the rover" if stt.pstdev(xs) < 0.02
              and stt.pstdev(ys) < 0.02 else "   -> sliding/being shoved")
        sts = collections.Counter(r[4] for r in rows)
        print("   brain state while pinned: "
              + "  ".join(f"{k or '-'}:{v * 0.02:.0f}s" for k, v in sts.most_common(5)))
        print(f"   detector reports it on {sum(1 for r in rows if r[5])}/{len(rows)} "
              f"ticks;  in the memory on "
              f"{sum(1 for r in rows if r[6] is not None)}/{len(rows)}")
    return {"seed": seed, "pin": {"prop": worst, "t0": t0, "seconds": dur,
                                  "geoms": {g: n for (pg, g), n in pairs.items()
                                            if pg == worst}}}


SECONDS = float(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
out = []
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    br, w = st.brains["m0"], st.world
    m, d = w.model, w.data
    pre = "m0/"

    def bname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""

    def gname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    base_g = {g for g in range(m.ngeom)
              if bname(g).startswith(pre)
              and not any(k in bname(g) for k in
                          ("upper_arm", "lower_arm", "wrist", "gripper", "finger"))
              and not gname(g).startswith(pre + "bin")}
    prop_g, pid = {}, {}
    for p_ in sc0.props:
        pid[p_.id] = m.body(p_.id).id
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == pid[p_.id]:
                prop_g[g] = p_.id

    pairs = collections.Counter()      # (prop, rover geom) -> substeps
    force = collections.defaultdict(float)
    run = collections.Counter()        # current unbroken run per prop
    best = collections.Counter()       # longest unbroken run per prop
    start = {}                         # first substep of the longest run
    live = set()
    six = np.zeros(6)
    tick = [0]
    real = mujoco.mj_step

    def watched(mm, dd, nstep=1):
        real(mm, dd, nstep)
        if mm is not m:
            return
        now = set()
        for c in range(dd.ncon):
            con = dd.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            pg = prop_g.get(g1) or prop_g.get(g2)
            rg = g1 if g1 in base_g else (g2 if g2 in base_g else None)
            if pg is None or rg is None:
                continue
            if dd.xpos[pid[pg]][2] >= moss.BIN_FLOOR_Z:
                continue                       # delivered, riding in the basket
            mujoco.mj_contactForce(mm, dd, c, six)
            pairs[(pg, gname(rg))] += 1
            force[(pg, gname(rg))] = max(force[(pg, gname(rg))],
                                         float(np.linalg.norm(six[:3])))
            now.add(pg)
        for pg in now:
            if run[pg] == 0:
                start[pg] = tick[0] * 0.02
            run[pg] += 1
            if run[pg] > best[pg]:
                best[pg] = run[pg]
                start[pg + "@best"] = start[pg]
        for pg in live - now:
            run[pg] = 0
        live.clear()
        live.update(now)

    # per-control-tick record of the worst prop's situation
    log = collections.defaultdict(list)
    mujoco.mj_step = watched
    try:
        for k in range(int(SECONDS / 0.02)):
            tick[0] = k
            st.drive(np.zeros(3), "auto")
            w.step()
            bx, by, byaw = w.ducks["m0"].driver.pose(d)
            c_, s_ = math.cos(-byaw), math.sin(-byaw)
            for p_ in sc0.props:
                q = d.xpos[pid[p_.id]]
                dx, dy = q[0] - bx, q[1] - by
                log[p_.id].append((k * 0.02, dx * c_ - dy * s_, dx * s_ + dy * c_,
                                   float(q[2]), str(getattr(br, "state", "")),
                                   _seen(br, p_.id), _in_mem(br, q)))
    finally:
        mujoco.mj_step = real
    out.append(_report(seed, sc0, pairs, force, best, start, log))
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
