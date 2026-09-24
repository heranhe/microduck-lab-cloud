---
name: record-world
description: >-
  Record a /sim WORLD scenario — the living room, the playroom tidy loop, a
  soccer pitch, any scenario JSON — to an mp4, a captioned contact sheet and
  an events log, headless and under a seed, then READ the sheet and the log
  to see what the ducks actually did. This is the debugging eye for the lab's
  world mode: the same brains, policies, team boards, tether and kickoff rule
  as duck-lab, minus the socket. Use when a duck falls, stalls, crowds,
  scores on itself or "does something weird" on the /sim page, when
  eval-tidy / eval-pitch numbers move and you want to see why, after touching
  brain/, world/, sensors/ or a scenario, and to capture a clip for a PR.
  Trigger on: "record the sim", "video of the soccer", "what is the duck doing
  in the living room", "watch the tidy loop", "show me the pitch", "why did it
  fall on the pitch", "record a scenario".
---

# record-world — video, sheet and log of a world scenario

The lab streams a world to a browser; nobody is watching most of the time and
a bug report is "it looked wrong". This runs the SAME world headlessly
(`world_server.WorldState` builds it, the loop is `world_loop` without the
socket), so a scenario is reproduced under a seed and what you record is what
the lab would do.

From `microduck_local/`:

```bash
uv run record-world pitch-2v2 --seconds 30 --out /tmp/rw-pitch                          # a 2v2 match, whole-floor camera
uv run record-world living-room --seconds 20 --camera follow:d0 --out /tmp/rw-room     # follow one duck (wander brain)
uv run record-world playroom --seconds 120 --skip 60 --brain d0=tidy --out /tmp/rw-tidy # tidy loop, minute 2 only
uv run record-world scenarios/my-room.json --seed 3 --tether-ms 250 --out /tmp/rw-mine  # a user scenario, over a tether
```

Scenarios: any name `GET /scenarios` lists (built-ins: `empty-floor`,
`wall-test`, `living-room`, `follow-me`, `playroom`, `pitch`, `pitch-2v2`,
`pitch-3v3`) or a `.json` path. `--brain DUCK=KIND` (repeatable) swaps a
brain the way the inspector does; kinds are `REGISTRY.available()` (`wander`,
`tidy`, `chase`, `follow`, `learned:<run>`, ...). Runs ~2x real time on a Mac
for a 4-duck pitch, faster for a room.

## Then READ it — in this order

1. **`events.txt`** — every brain state transition, fall (with the state,
   skill, what was held, and where it respawned), pick-up / release, goal and
   kickoff, stamped with sim time. This is the whole run in a few hundred
   lines and it tells you WHEN to look. A run that is all `carry`/`deliver`
   turning is not seeing the basket; a chase duck cycling
   `blocked`/`chase`/`lineup` every 0.1 s is fighting a teammate.
2. **`sheet.png`** — Read it (it is an image). Header: scenario, seed,
   per-duck brain and policy, total falls, final score. Per tile: time,
   score line (`goals a-b  ball x,y` on a pitch, `tidy k/n`
   in the playroom), then one line per duck:
   `id role-or-brain/state v<speed> f<falls> x,y`, plus a second line
   for `hold=`, `skill=` and the brain's note when there is one.
   A **highlighted** tile is one in which some duck fell since the previous
   tile — go to that second in `events.txt`.
3. **`world.mp4`** — for the human. Attach it to the PR or SendUserFile it.
   Default `--stride 5 --fps 10` is real time; `--stride 2 --fps 25` for a
   slow-motion look at a fall or a grasp.

## Cameras

- `top` (default) — the whole floor from near-plan, the pitch's view. Ducks
  are small; the captions carry the positions.
- `follow:<duck>` — three-quarter camera tracking that duck at ~1.1 m. Use
  it for a fall, a grasp, a kick, a wall bump.
- `side` / `front` / `three-quarter` — fixed at the room centre.

## What it is not

- Not a replay of the live lab session: the lab is unseeded and a browser
  may have driven it. Reproduce the scenario with `--seed` and `--brain`
  instead; if the bug does not appear, try a few seeds before saying so.
- Not `render-rollout` (one policy in its training env, with reward
  diagnostics) and not `scripts/render_pitch.py` (a roster from the
  battery's side, with the pitch metrics burned in). Use those for those
  questions; use this one for "what happens in the room".
- Nothing here restarts or touches a running lab; safe at any time.
- Not the browser's own 🎥 record button (the /sim top bar): that films
  what the USER is looking at, their camera, no captions, and lands in
  `microduck_local/captures/` as mp4 + gif via `POST /captures`. Point a
  human at that; use this one when you need the numbers.

MuJoCo renders offscreen on the Mac's CGL backend with **no `MUJOCO_GL`
set**; set `MUJOCO_GL=egl` only on Linux.
