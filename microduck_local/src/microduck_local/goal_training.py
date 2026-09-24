"""Independent ONNX certification for poses and ground-launched hopping drills."""

import hashlib
import math
import os
from pathlib import Path

EVAL_INTERVAL = 500_000
EVAL_TRIALS = 10
REQUIRED_SUCCESSES = 9
STALL_EVALUATIONS = 12
MAX_GOAL_STEPS = 40_000_000
TERMINAL_GOAL_STATES = {"passed", "stalled", "limit", "error"}
GOAL_HOLD_SECONDS = {"one_leg_5s": 5.0, "white_crane": 6.0,
                     "single_leg_hop": 3.0, "jump_turn_180": 1.0}


def new_goal(behavior_id="one_leg_5s") -> dict:
    hops = int(os.environ.get("MICRODUCK_HOP_GOAL_HOPS", "0")) if behavior_id == "single_leg_hop" else 0
    if hops not in (0, 1, 2):
        raise ValueError("MICRODUCK_HOP_GOAL_HOPS must be 0, 1 or 2")
    recovery = behavior_id == 'single_leg_hop' and os.environ.get('MICRODUCK_HOP_RECOVERY', '0') == '1'
    if recovery and hops:
        raise ValueError('recovery rehearsal cannot also certify a hop')
    trials = 20 if behavior_id == 'single_leg_hop' else EVAL_TRIALS
    return {"status": "training", "round": 0, "steps": 0,
            "required_hops": hops, "recovery": recovery,
            "trials": trials, "required": 18 if trials == 20 else REQUIRED_SUCCESSES,
            "behavior": behavior_id,
            "hold_seconds": .5 if recovery else GOAL_HOLD_SECONDS[behavior_id], "eval_interval": EVAL_INTERVAL,
            "max_steps": MAX_GOAL_STEPS, "stall_evaluations": STALL_EVALUATIONS,
            "stale": 0, "best_successes": -1, "best_mean_hold": -1.0}


def evaluate_policy(path: Path, round_index: int, weights: dict | None = None,
                    behavior_id="one_leg_5s") -> dict:
    """No sampled PPO actions, no training normalizer, no pose/spotter assist.

    A fresh cohort of seeds is reserved for each round. The normalizer is
    already inside this ONNX. Full goals start on both feet under BAM.
    Hop drills start grounded on the support foot; a valid hop only counts
    after 0.5 seconds of stable same-foot contact. They never certify the
    ordinary-start three-second goal.
    """
    import numpy as np
    import onnxruntime as ort

    from .behaviors import BehaviorEnv

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy_bytes = path.read_bytes()
    session = ort.InferenceSession(policy_bytes, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    goal = new_goal(behavior_id)
    trials, recovery = goal['trials'], goal['recovery']
    seeds = list(range(100_000 + round_index * trials,
                       100_000 + (round_index + 1) * trials))
    holds = []
    drill_hops = goal["required_hops"]
    hop_counts, diagnostics = [], []
    env = BehaviorEnv(behavior_id, weight_overrides=weights,
                      obs_noise=False, domain_rand=False, random_yaw=False,
                      action_delay=True, standing_spawns=not (drill_hops or recovery),
                      spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "1.0",
                                       "MICRODUCK_HOP_RECOVERY": '1' if recovery else '0',
                                       "MICRODUCK_EPISODE_S": "10"} if drill_hops or recovery else
                                      {"MICRODUCK_HOP_RECOVERY": '0'},
                      actuator_force="bam", seed=seeds[0])
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            best = 0.0
            landed_step, stable_hops, flight, right_contacts = None, 0, 0, 0
            peak, rear_min, rear_max = 0., 1., -1.
            start_step = env.step_count
            for _ in range(env.max_steps):
                action = session.run(None, {input_name: obs[None]})[0][0]
                if not np.isfinite(action).all():
                    raise ValueError("ONNX produced non-finite actions")
                obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
                best = max(best, float(info["best_hold_s"]))
                if behavior_id == "single_leg_hop":
                    from .behaviors import _slh_clean, _wc_sole_z
                    state, contacts = env._slh, env.foot_contact_state
                    if recovery:
                        best = state['recovery_best']*.02
                    flight += not contacts['left'] and not contacts['right']
                    right_contacts += contacts['right']
                    peak = max(peak, _wc_sole_z(env, 'left'))
                    rear_min = min(rear_min, float(env._wc_state['foot'][0]))
                    rear_max = max(rear_max, float(env._wc_state['foot'][0]))
                    if state['landed']:
                        landed_step = env.step_count
                    if (landed_step is not None and contacts['left'] and not contacts['right']
                            and not terminated and _slh_clean(env)
                            and env.step_count-landed_step >= 25 and state['stable'] >= 25):
                        stable_hops = max(stable_hops, state['hops'])
                    if not contacts['left'] or contacts['right'] or state['failed']:
                        landed_step = None
                if info["is_success"] or (recovery and best >= .5) or (drill_hops and stable_hops >= drill_hops) or terminated or truncated:
                    break
            holds.append(round(best, 4))
            hop_counts.append(stable_hops)
            if behavior_id == "single_leg_hop":
                diagnostics.append(dict(seconds=round((env.step_count-start_step)*.02, 3),
                    flight_s=round(flight*.02, 3), peak_m=round(peak, 4),
                    right_contacts=right_contacts, rear_x=[round(rear_min, 3), round(rear_max, 3)],
                    stage=env._slh['stage'], failed=env._slh['failed']))
    finally:
        env.close()
    return {"holds": holds, "seeds": seeds, "env_seed": seeds[0], "stable_hops": hop_counts,
            "diagnostics": diagnostics,
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest()}


def record_evaluation(goal: dict, report: dict, steps: int) -> dict:
    """Only certified holds count; reward totals and episode lengths do not."""
    holds = report["holds"]
    trials = goal['trials']
    if (len(holds) != trials
            or not all(math.isfinite(h) and 0 <= h <= 20 for h in holds)):
        raise ValueError(f"evaluation needs {trials} finite hold durations in [0, 20]")
    target = goal.get("required_hops") or goal["hold_seconds"]
    scores = report.get("stable_hops") if goal.get("required_hops") else holds
    if scores is None or len(scores) != trials or not all(math.isfinite(h) and h >= 0 for h in scores):
        raise ValueError(f"evaluation needs {trials} finite nonnegative scores")
    successes = sum(h >= target for h in scores)
    mean_hold = sum(min(h, goal['hold_seconds']) for h in holds) / trials
    mean_score = sum(min(h, target) for h in scores) / trials
    improved = (successes > goal["best_successes"]
                or (successes == goal["best_successes"]
                    and mean_score >= goal.get("best_mean_score", goal["best_mean_hold"]) + 0.1))
    out = {**goal, **report, "round": goal["round"] + 1, "steps": steps,
           "successes": successes, "best_hold_s": max(holds),
           "mean_hold_s": mean_hold, "stale": 0 if improved else goal["stale"] + 1}
    if improved:
        out.update(best_successes=successes, best_mean_hold=mean_hold, best_mean_score=mean_score)
    out["status"] = (
        "passed" if successes >= goal['required'] else
        "limit" if steps >= MAX_GOAL_STEPS else
        "stalled" if out["stale"] >= STALL_EVALUATIONS else "training"
    )
    return out


def goal_job_status(goal: dict) -> str:
    """A clean process exit alone is NOT proof that the skill was learned."""
    return {"passed": "done", "error": "failed"}.get(goal.get("status"), "stopped")


if __name__ == '__main__':
    import argparse
    import json
    import zipfile

    parser = argparse.ArgumentParser(description='Independent deterministic ONNX certification')
    parser.add_argument('run', type=Path)
    parser.add_argument('--hops', type=int, choices=(0, 1, 2), default=0,
                        help='0: full three-second goal; 1/2: grounded hop-and-land drill')
    parser.add_argument('--recovery', action='store_true', help='grounded recovery only, not a hop')
    args = parser.parse_args()
    os.environ['MICRODUCK_HOP_GOAL_HOPS'] = str(args.hops)
    os.environ['MICRODUCK_HOP_RECOVERY'] = '1' if args.recovery else '0'
    metadata = json.loads((args.run / 'behavior.json').read_text())
    behavior = metadata['behavior']
    if (args.hops or args.recovery) and behavior != 'single_leg_hop':
        parser.error('--hops/--recovery are only valid for single_leg_hop')
    if args.hops and args.recovery:
        parser.error('--hops and --recovery are mutually exclusive')
    report = evaluate_policy(args.run / 'policy.onnx', 0, metadata.get('weights'), behavior)
    with zipfile.ZipFile(args.run / 'model.zip') as checkpoint:
        steps = json.loads(checkpoint.read('data'))['num_timesteps']
    result = record_evaluation(new_goal(behavior), report, steps)
    # This CLI verifies a completed bounded experiment; it launches no trainer.
    if result['status'] != 'passed':
        result['status'] = 'not_passed'
    result['starting_state'] = ('grounded_recovery' if args.recovery else
                               'grounded_single_leg_drill' if args.hops else 'ordinary_standing')
    result['source_sha256'] = hashlib.sha256(
        (Path(__file__).parent / 'behaviors/single_leg_hop.py').read_bytes()).hexdigest()
    out = args.run / ('verification-recovery.json' if args.recovery else f'verification-hops-{args.hops}.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
