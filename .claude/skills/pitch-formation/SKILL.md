---
name: pitch-formation
description: >-
  Verify soccer team formations on the pitch: roster-size roles (defender /
  midfielder / striker), the 2v2 halfway zone split, cover-without-job-swap,
  and the post-kick exit line. Use after touching brain/team.py, Chase,
  make_pitch, world_server builtins, eval-pitch / eval-striker, or when asking
  whether the ducks are playing positions. Trigger on: "soccer formation",
  "pitch roles", "is the defender holding", "eval-striker roster", "render the
  pitch", "2v2 / 3v3 roles".
---

# Pitch formation — look before you believe a soccer number

Roles are a POST plus a zone, not a new brain. The lab builtins stamp them;
`eval-pitch` must not. Confirm with pytest on the shipped constructors, then
a short `eval-striker` and a contact sheet.

## 1. Pytest (the bar)

From `microduck_local/`:

```
uv run --with pytest pytest tests/test_team.py tests/test_world.py tests/test_striker.py tests/test_world_server.py -q
```

What those files lock on the real code:

| claim | where |
|---|---|
| `pitch-2v2` / `pitch-3v3` builtins stamp defender+striker / defender+mid+striker; 1v1 and `make_pitch` without `formation=` have no jobs | `test_world.py` `test_formation_roles_stamp_jobs_by_roster_size_and_eval_pitch_stays_role_free` |
| `eval-pitch` and chase-vs-chase `eval-striker` stay role-free; `test_the_chase_arm_reproduces_eval_pitch_exactly` still byte-for-byte | `test_striker.py` |
| defender+striker board: a ball at midfield is **not** "everybody may" | `test_team.py` `test_a_defender_striker_roster_owns_midfield_instead_of_falling_back_to_everybody` |
| cover: zone owner far/slow → quicker teammate is a candidate; `job` unchanged | `test_team.py` `test_cover_lets_a_quicker_teammate_attack_without_changing_jobs` |
| kick with `hunt_exit`: hunt heading includes the foot exit angle; board `ball_vel` matches that line above kick-like speed and is ignored below it | `test_team.py` `test_a_kick_publishes_the_exit_line_only_at_kick_like_speed` |
| `GET /world` after loading `pitch-2v2` reports the jobs | `test_world_server.py` `test_pitch_2v2_builtin_reports_formation_roles` |

Do not mock the board or Chase. Do not hard-code metric tables.

## 2. Short eval and a sheet (evidence)

```
uv run python -m microduck_local.eval_striker --per-side 2 --seeds 1 --seconds 8 \
    --left "chase+defender,chase+striker" --right chase --out /tmp/eval-2v2.jsonl --tag form
uv run python scripts/render_pitch.py --left "chase+defender,chase+striker" --right chase \
    --per-side 2 --seed 0 --seconds 12 --out /tmp/rp-form
```

Then **Read** the sheet PNG the script prints (typically `/tmp/rp-form_sheet.png`). Two colorways; in the late tiles the defender's depth stays back while a teammate is on the ball. If MuJoCo offscreen cannot run, write that failure down — pytest is still the bar.

Roster syntax is `eval-striker`'s: `chase+defender,chase+striker` one entry a duck; a single `chase` covers the side.

## 3. What to trust, and the baseline trap

Trust `crowd`, `spread`, `depth`, `possession`, signed `ballProgress`. Do not judge on `goals` or `ownGoals` (Track 4.1.5: 136 / 347 seeds for a 25% shift).

**Never turn roles on inside `eval-pitch`.** That function is the role-free chase-vs-chase control. Lab builtins take `make_pitch(..., formation=True)`. A battery that stamps jobs onto `eval-pitch` silently moves every future A/B.

## 4. /sim (optional)

```
bash .claude/skills/sim-smoke/bringup.sh pitch-2v2 --restart
curl -s :8788/world | python -c "import json,sys; d=json.load(sys.stdin); print([(x['id'], x.get('role')) for x in d.get('ducks',[])])"
```

Expect `d0` defender, `d1` striker (and the same on the other colorway). Follow `sim-smoke` for the screenshot path.
