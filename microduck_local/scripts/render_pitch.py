"""Look at a ROSTER before believing its numbers (AGENTS.md, verification #2).

    cd microduck_local
    uv run python scripts/render_pitch.py --left "chase+defender,chase+striker" --per-side 2
    uv run python scripts/render_pitch.py --left chase --out /tmp/rp-plain      # the control

`render_striker` renders a `StrikerEnv`, which is one duck's training world.
This renders the thing a battery actually runs: `eval_striker`'s pitch, with
whatever roster each side is given, from a top-down camera — and burns the
numbers a roster is judged on into every tile, so the sheet answers the
question the table cannot: *is the defender where a defender should be, or is
it stuck against the boards?*

Per tile: the ball, each duck's distance to it, the widest gap between
teammates (`spread`), how many of a side are inside the pile-up radius
(`crowd`), and the deepest duck's distance from the goal it defends
(`depth`). A colorway is painted in the model (`world/compose.paint_team`),
so the two teams are the colours the /sim page shows.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from microduck_local.brain import Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import kickoff_brains
from microduck_local.eval_striker import apply_roster, home_away, make_brain, pitch_scenario
from microduck_local.render_rollout import build_sheet, sheet_indices
from microduck_local.render_striker import top_camera
from microduck_local.world import World
from microduck_local.world.metrics import CROWD_R, PitchMetrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="chase+defender,chase+striker", help="the home roster (eval-striker's spec)")
    ap.add_argument("--right", default="chase")
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="/tmp/rp")
    ap.add_argument("--sheet-frames", type=int, default=12)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=10, help="render every Nth control step (50 Hz)")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=320)
    args = ap.parse_args()

    sc = apply_roster(pitch_scenario(args.per_side, solo=False), args.left, args.right)
    home, away = home_away(sc)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=args.seed)
    teams: dict = {}
    brains = {d.id: make_brain(d.brain or "chase", d, w, teams) for d in sc.ducks}
    metrics = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    mine = [d.id for d in sc.ducks if d.team == home]
    goal_seq = 0

    renderer = mujoco.Renderer(w.model, height=args.height, width=args.width)
    cam = top_camera(distance=max(sc.floor) * 1.25)
    frames: list[np.ndarray] = []
    caps: list[list[str]] = []
    while w.t < args.seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            intent = brains[d.id].step(s)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        metrics.tick()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams)
        if w.tick % args.stride:
            continue
        renderer.update_scene(w.data, cam)
        frames.append(renderer.render().copy())
        ball = w.ball_xy()
        pos = metrics.positions()
        near = [(did, math.dist(pos[did], ball)) for did in mine]
        pairs = [math.dist(pos[a], pos[b]) for i, a in enumerate(mine) for b in mine[i + 1:]]
        line = -metrics.sign[home] * metrics.half_x
        deep = min(abs(pos[did][0] - line) for did in mine)
        jobs = " ".join(f"{did}:{(brains[did].job or '-')[:3]}/{brains[did].state[:6]}" for did in mine)
        caps.append([
            f"t {w.t:5.1f}s  ball {ball[0]:+.2f},{ball[1]:+.2f}  goals {w.goals['left']}-{w.goals['right']}",
            f"{home}: " + "  ".join(f"{k} {v:.2f}m" for k, v in near),
            f"spread {max(pairs) if pairs else 0:.2f}  crowd {sum(1 for _, v in near if v <= CROWD_R)}"
            f"  depth {deep:.2f}",
            jobs,
        ])

    idx = sheet_indices(len(frames), args.sheet_frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    row = metrics.row()
    header = [f"{home} [{args.left}] v {away} [{args.right}] — seed {args.seed}, {args.seconds:g} s",
              f"{home}: crowd {row['crowd'][home]}  spread {row['spread'][home]}m  depth {row['depth'][home]}m"
              f"  possession {row['possession'][home]}s/min",
              f"{away}: crowd {row['crowd'][away]}  spread {row['spread'][away]}m  depth {row['depth'][away]}m"
              f"  possession {row['possession'][away]}s/min"]
    footer = [f"crowd = how many of {home} are within {CROWD_R} m of the ball THIS FRAME (the pile-up).",
              "spread = the widest gap between them. depth = the deepest one's distance from its own goal line.",
              "the last line is each duck's role / state: a defender should sit deep and say 'suppor'."]
    build_sheet([frames[i] for i in idx], [caps[i] for i in idx], header, footer,
                [caps[i][2].startswith("spread") and " crowd 2" in caps[i][2] for i in idx],
                out / "sheet.png")
    imageio.mimsave(out / "pitch.mp4", frames, fps=args.fps, macro_block_size=1)
    print(f"wrote {out / 'sheet.png'} and {out / 'pitch.mp4'}")
    print(f"{home}: crowd {row['crowd'][home]} spread {row['spread'][home]} depth {row['depth'][home]}")
    print(f"{away}: crowd {row['crowd'][away]} spread {row['spread'][away]} depth {row['depth'][away]}")
    print("READ the sheet: the tiles marked with a highlight are frames where BOTH of "
          f"{home} are on the ball at once — a roster that works should have very few.")


if __name__ == "__main__":
    main()
