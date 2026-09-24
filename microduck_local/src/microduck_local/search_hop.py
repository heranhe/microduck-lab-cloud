"""Finite CEM trajectory experiments, NOT a learned/deployable ONNX policy.

Reuse BehaviorEnv's real BAM stepping and complete reset. Only the RL commander's
timeout is bypassed: this open-loop controller supplies its own timing. Physical
falls, rear-foot touchdown and strict geometric requirements remain audited.
"""
import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import time

import mujoco
import numpy as np

from . import contract as C
from .behaviors import (BehaviorEnv, HOP_LOAD_POSE, HOP_THRUST_POSE,
                        HOP_READY_POSE, _slh_advance, _slh_clean, _slh_head, _wc_measure, _wc_sole_z)

ENV = None
SEEDS = (0, 1, 2)


def make_env():
    return BehaviorEnv('single_leg_hop', obs_noise=False, domain_rand=False,
                       random_yaw=False, action_delay=True, actuator_force='bam',
                       spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS': '1'}, seed=0)


def rollout(env, x, seed, capture=None, null=False):
    env.reset(seed=seed)  # Includes mjData, caches, BAM history, delays and RNG.
    certificate = dict(env._slh)  # Independent of the RL commander's timeout.
    certified_landing = False
    start = env._joint_qpos().copy()
    assert env._foot_contacts() == {'left': True, 'right': False}
    assert np.max(np.abs(env.data.qvel)) == 0 and env.bam is not None
    poses = np.asarray(x[:42]).reshape(3, 14)
    load_n, thrust_n = np.maximum(np.rint(np.asarray(x[42:])/C.CTRL_DT).astype(int), 1)
    forward = env._trunk_xmat.reshape(3, 3)[:2, 0].copy()
    forward /= np.linalg.norm(forward)
    initial_length = float(env._trunk_xpos[2]) - _wc_sole_z(env, 'left')
    shortest = initial_length
    compression = extension = vz = peak = max_air = fwd = landing_s = 0.
    air_start = None
    air_xy = None
    air_peak = air_vz = 0.
    takeoff_vz, descending_gap, descending_steps = 0., 0., 0
    catch_reach, left_landing_s = 0., 0.
    landed = valid_flight = False
    constraints_ok = True
    reasons = set()
    min_head, min_rear, min_clear = 1., 1., 1.
    max_tilt = 0.
    trace = []
    if capture:
        capture(0)
    for k in range(load_n+thrust_n+40):
        if null:
            target = env._joint_qpos().copy()
        elif k < load_n:
            target = start + ((k+1)/load_n)*(poses[0]-start)
        elif k < load_n+thrust_n:
            target = poses[1]
        else:
            target = poses[2]
        # Targets are bounded by the real joint ranges in the search, never
        # added impulses, altered gains or a stronger actuator.
        env.step((target-C.DEFAULT_POSE).astype(np.float32))
        c, s = env._foot_contacts(), _wc_measure(env)
        mujoco.mj_subtreeVel(env.model, env.data)
        com_vz = float(env.data.subtree_linvel[env._wc_robot_root, 2])
        com_xy = env.data.subtree_com[env._wc_robot_root, :2]
        head = _slh_head(env)
        tilt = float(np.arccos(np.clip(-env._projected_gravity()[2], -1, 1)))
        min_head = min(min_head, head)
        min_rear = min(min_rear, -float(s['foot'][0]))
        min_clear = min(min_clear, s['clearance'])
        max_tilt = max(max_tilt, tilt)
        for bad, reason in ((head < .35, 'head'), (s['foot'][0] > -.06, 'rear_direction'),
                            (s['clearance'] < .03, 'rear_height'), (tilt > np.deg2rad(40), 'tilt')):
            if bad:
                constraints_ok = False
                reasons.add(reason)
        length = float(env._trunk_xpos[2])-_wc_sole_z(env, 'left')
        _slh_advance(certificate, k, left=c['left'], right=c['right'],
                     lifted=s['clearance'] >= .03,
                     good=(_slh_clean(env) and tilt <= np.deg2rad(40) and head > .35
                           and s['clearance'] >= .03 and s['foot'][0] <= -.06),
                     knee=float(env._joint_qpos()[3]), knee_speed=float(env._joint_vel()[3]),
                     leg_height=length, vz=com_vz, clearance=_wc_sole_z(env, 'left'),
                     xy=com_xy, forward=forward)
        certified_landing |= certificate['landed']
        if c['left']:
            shortest = min(shortest, length)
            compression = max(compression, initial_length-shortest)
        extension = max(extension, length-shortest)
        vz = max(vz, com_vz)
        if not c['left'] and not c['right']:
            if air_start is None:
                air_start, air_xy = k, com_xy.copy()
                air_peak, air_vz = 0., com_vz
                takeoff_vz = max(takeoff_vz, air_vz)
            if com_vz < 0:
                descending_steps += 1
                descending_gap += float(np.clip((s['clearance']-_wc_sole_z(env, 'left'))/.05, 0, 1))
                catch_reach += float(np.clip((length-.07)/.07, 0, 1))
            air_peak = max(air_peak, _wc_sole_z(env, 'left'))
            peak = max(peak, air_peak)
            max_air = max(max_air, (k-air_start+1)*C.CTRL_DT)
        elif air_start is not None:
            flight_s = (k-air_start)*C.CTRL_DT
            dx = float((com_xy-air_xy) @ forward)
            fwd = max(fwd, dx)
            landed = c['left'] and not c['right']
            valid_flight |= (landed and flight_s >= .06 and air_peak >= .008
                             and dx >= .01 and air_vz > .03
                             and compression >= .004 and extension >= .004)
            air_start = None
        if valid_flight and c['left'] and not c['right']:
            landing_s += C.CTRL_DT
        elif not c['left']:
            landing_s = 0.
        if max_air >= .06 and takeoff_vz > .03 and c['left'] and not c['right']:
            left_landing_s += C.CTRL_DT
        elif not c['left']:
            left_landing_s = 0.
        if capture:
            trace.append(dict(t=round((k+1)*C.CTRL_DT, 3), contacts=c,
                              vz=float(env.data.qvel[2]), com_vz=com_vz, sole=_wc_sole_z(env, 'left'),
                              com_z=float(env.data.subtree_com[env._wc_robot_root, 2]),
                              trunk_z=float(env._trunk_xpos[2]),
                              knee=float(env._joint_qpos()[3]), knee_speed=float(env._joint_vel()[3]),
                              joint_qpos=env._joint_qpos().tolist(), target=target.tolist(),
                              head=head, rear=float(s['foot'][0]), tilt=float(np.degrees(tilt))))
            capture(k+1)
        if c['right'] or not _slh_clean(env) or tilt > np.deg2rad(65) or env._trunk_xpos[2] < .065:
            constraints_ok = False
            reasons.add('right_contact' if c['right'] else
                        'nonfoot_collision' if not _slh_clean(env) else
                        'excessive_tilt' if tilt > np.deg2rad(65) else 'body_too_low')
            break
    success = bool(valid_flight and landing_s >= .26 and constraints_ok and certified_landing)
    if not success and not reasons:
        reasons.add('no_valid_flight' if not valid_flight else
                    'landing_unstable' if landing_s < .26 else 'strict_certification_failed')
    # Bounded progress scores for search only, not new PPO reward terms.
    clip = lambda a: float(np.clip(a, 0, 1))
    ascending = clip(takeoff_vz/.15)
    score = (2*clip(vz/.45) + ascending*(3*clip(peak/.015) + 2*clip(max_air/.10))
             + clip(fwd/.025) + 2*clip(landing_s/.3) + .5*clip(compression/.015)
             + .5*clip(extension/.02)
             + 2.5*descending_gap/max(descending_steps, 1)
             - 2.*('right_contact' in reasons)
             - .7*clip((.35-min_head)/.35) - .7*clip((.06-min_rear)/.06)
             - .7*clip((.03-min_clear)/.03) - .5*clip((max_tilt-.7)/.4))
    # Landing outranks airborne spectacle. A folded support leg can earn
    # flight credit yet hit the floor body-first; reach and left support
    # provide a gradient before the strict whole-hop success gate fires.
    score += (4*catch_reach/max(descending_steps, 1) + 12*clip(left_landing_s/.26)
              - 8*bool(reasons & {'nonfoot_collision', 'excessive_tilt', 'body_too_low'}))
    return dict(success=success, score=round(score, 6), seconds=round((k+1)*.02, 3),
                certified_landing=bool(certified_landing),
                air_s=round(max_air, 4), peak_m=round(peak, 5), forward_m=round(fwd, 5),
                vz=round(vz, 4), landing_s=round(landing_s, 3), valid_flight=bool(valid_flight),
                takeoff_com_vz=round(takeoff_vz, 4),
                left_landing_s=round(left_landing_s, 3),
                compression_m=round(compression, 5), extension_m=round(extension, 5),
                min_head=round(min_head, 4), min_rear_m=round(min_rear, 4),
                min_clear_m=round(min_clear, 4), max_tilt_deg=round(float(np.degrees(max_tilt)), 2),
                reasons=sorted(reasons), trace=trace)


def evaluate(x):
    reports = [rollout(ENV, x, seed) for seed in SEEDS]
    scores = np.array([r['score'] for r in reports])
    return (sum(r['success'] for r in reports), float(scores.mean()+.5*scores.min()), reports)


def render(env, x, out, seed, null=False):
    import imageio.v2 as imageio
    import mujoco
    from .render_rollout import (Probe, make_camera, build_sheet, sheet_indices,
                                format_caption, sheet_footer)
    renderer = mujoco.Renderer(env.model, height=280, width=360)
    cam, probe = make_camera('side', .65), Probe(env)
    frames, diags = [], []
    def capture(k):
        cam.lookat[:] = [*env._trunk_xpos[:2], .16]
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render().copy())
        diags.append(probe.sample(k, k, 'null' if null else 'OPEN-LOOP SEARCH', False))
    report = rollout(env, x, seed, capture, null)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{"null" if null else "candidate"}-{seed}'
    with imageio.get_writer(out/f'{tag}.mp4', fps=50, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    ids = sheet_indices(len(frames), 8)
    build_sheet([frames[i] for i in ids],
                [format_caption(diags[i], probe.stand_z, probe.head_ref_z) for i in ids],
                ['OPEN-LOOP FEASIBILITY TEST, not learned ONNX',
                 f'seed={seed} success={report["success"]} air={report["air_s"]}s peak={report["peak_m"]}m',
                 f'reasons={report["reasons"]}; honest BAM, no injected velocity'],
                sheet_footer(probe), [False]*len(ids), out/f'{tag}_sheet.png')
    (out/f'{tag}.json').write_text(json.dumps(report, indent=2))
    renderer.close()


def main():
    global ENV
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--generations', type=int, default=60)
    p.add_argument('--population', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path)
    p.add_argument('--render', action='store_true')
    args = p.parse_args()
    if min(args.generations, args.population, args.workers) < 1:
        p.error('budgets must be positive')
    ENV = make_env()
    ranges = np.array([ENV.model.joint(n).range for n in C.JOINT_NAMES])
    mean = np.r_[HOP_LOAD_POSE, HOP_THRUST_POSE, HOP_READY_POSE, .24, .16]
    if args.init:
        mean = np.array(json.loads(args.init.read_text())['params'])
    if mean.shape != (44,) or not np.isfinite(mean).all():
        p.error('expected 44 finite trajectory parameters')
    lo, hi = np.r_[np.tile(ranges[:, 0], 3), .08, .06], np.r_[np.tile(ranges[:, 1], 3), .5, .4]
    mean = np.clip(mean, lo, hi)
    if args.render:
        for seed in (100000, 100001):
            render(ENV, mean, args.out, seed)
        render(ENV, mean, args.out, 100000, null=True)
        ENV.close()
        return
    ids = ([2, 3, 4] if args.round == 1 else
           [1, 2, 3, 4, 5, 6, 10, 11, 12, 13] if args.round == 2 else list(range(14)))
    active = [j+14*k for k in range(3) for j in ids]+[42, 43]
    std = np.r_[np.full(14, .30), np.full(14, .35), np.full(14, .18), .08, .06]
    floor = np.r_[np.full(42, .025), .008, .008]
    rng = np.random.default_rng(900+args.round)
    args.out.mkdir(parents=True, exist_ok=True)
    t0, best, stale = time.monotonic(), None, 0
    # One precompiled model inherited COW; every worker owns its mjData/BAM.
    with mp.get_context('fork').Pool(args.workers) as pool:
        for generation in range(args.generations):
            pop = np.repeat(mean[None], args.population, axis=0)
            pop[:, active] += rng.normal(size=(args.population, len(active)))*std[active]
            pop = np.clip(pop, lo, hi)
            pop[0] = mean if best is None else best[2]
            results = pool.map(evaluate, pop, chunksize=4)
            order = sorted(range(len(pop)), key=lambda i: results[i][:2])
            i = order[-1]
            if best is None or results[i][:2] > best[:2]:
                best = (*results[i][:2], pop[i].copy(), results[i][2]); stale = 0
            else:
                stale += 1
            elite = pop[order[-max(4, args.population//8):]]
            mean[active] = .3*mean[active]+.7*elite[:, active].mean(axis=0)
            std[active] = np.maximum(.3*std[active]+.7*elite[:, active].std(axis=0), floor[active])
            if stale and stale % 15 == 0:
                std[active] = np.minimum(std[active]*1.8, .35)
            record = dict(round=args.round, generation=generation+1,
                          candidates=(generation+1)*args.population,
                          elapsed_s=round(time.monotonic()-t0, 2), successes=best[0],
                          score=best[1], trials=best[3])
            with (args.out/'progress.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            (args.out/'best.json').write_text(json.dumps({**record, 'params': best[2].tolist()}, indent=2))
            print(json.dumps(record), flush=True)
    heldout = [rollout(ENV, best[2], seed) for seed in range(100000, 100010)]
    null = [rollout(ENV, best[2], seed, null=True) for seed in range(100000, 100010)]
    result = dict(round=args.round, candidates=args.generations*args.population,
                  rollout_count=args.generations*args.population*len(SEEDS),
                  elapsed_s=round(time.monotonic()-t0, 2), params=best[2].tolist(),
                  successes=sum(r['success'] for r in heldout), trials=heldout, null=null,
                  failures=dict(Counter(reason for r in heldout for reason in r['reasons'])),
                  controller='open_loop', actuator='bam', current_scale=1.,
                  no_initial_velocity=True, rl_commander_timeout_bypassed=True)
    (args.out/'evaluation.json').write_text(json.dumps(result, indent=2))
    print('FINAL '+json.dumps(result), flush=True)
    ENV.close()


if __name__ == '__main__':
    main()
