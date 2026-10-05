"""Teachable tasks for MOSS — the 🎓 panel's entry to the fourth body.

Same shape as `g1_tasks.py` and `mars_tasks.py`, for the same reason:
everything else in this package is a MICRODUCK reward recipe evaluated
inside `BehaviorEnv`, and another body's task is an env subclass run by a
different trainer. Registering it here anyway is what gives MOSS the teach
panel, the job card, the live-snapshot preview and the trainee slot on stage
without any of those learning what a MOSS is.

The `terms` are what the panel SHOWS — display rows, not callables the env
evaluates. `robots/moss_env.MossPickEnv` owns the reward, and `_env_owned`
says so at a glance.
"""

import dataclasses

from .core import *  # noqa: F401,F403 — the package's namespace cascade
from .core import Behavior, CurriculumStage, RewardTerm, _register


def _env_owned(env):  # noqa: ARG001 — see the module docstring
    """Placeholder: MOSS's rewards live in the env, not in this library."""
    return 0.0


MOSS_PICK = Behavior(
    id="moss_pick",
    emoji="🥫",
    title="Scoop a can (MOSS)",
    description="From the deployed pose, drive the can between the open jaws, "
                "close on it and lift it clear — the 30 cm the scripted "
                "litter loop keeps losing.",
    how_it_learns=(
        "It is paid for GROUND MADE UP, not for being near the can: every "
        "control step it earns for however much closer the can got to the "
        "pocket between its pads, and loses exactly the same for shoving it "
        "away — so an episode's total is the distance it closed, and a robot "
        "that bulldozes the can around earns nothing at all. Closing the jaws "
        "on it adds a small per-step trickle, deliberately small: at ten "
        "times the size a hand-written script that simply shut its claw on "
        "the can and sat there out-earned one that actually picked it up, "
        "which is a policy that learns to park. What pays properly is LIFT, "
        "and again as progress — per centimetre gained while holding, so "
        "standing still with a can in the jaws earns nothing and the only way "
        "to keep earning is to keep raising it. The ladder is in the SPAWN "
        "BOX rather than the weights: rung 0 puts the can already between the "
        "pads so the first thing it can learn is what closing does, rung 1 "
        "just outside them, and rung 2 at the full creep distance the room's "
        "brain hands over at. A policy whose rollouts never once contain a "
        "can between its pads never learns to close on one, and no reward "
        "weight fixes that."
    ),
    keywords=("pick up the can", "scoop the can", "grab that can",
              "collect the litter", "pick it up", "get the can"),
    #: The panel's chip. A recipe with no `suggest` is trainable but INVISIBLE.
    suggest="pick up the can",
    robot="moss",
    # The 🎓 preview's episode length. Left at the 8 s default the preview
    # runs a different clock from the trainer, and for the stow task that
    # meant resetting at 8 s just as the ~7 s carry arrived over the bin —
    # so a watcher never once saw it let go.
    episode_s=8.0,
    # `train-walk --robot moss --task pick`; TrainingJob appends --run-name,
    # --envs, --steps, --snap-steps and --init-from.
    trainer=("-m", "microduck_local.train", "--robot", "moss",
             "--task", "pick"),
    # THE LADDER, as stages rather than as three commands somebody remembers
    # to run in order. Each one warm-starts from the last, and the rung
    # reaches the trainer as an environment variable — which is what
    # `CurriculumStage.env` is for, and it means the whole curriculum runs
    # from the 🎓 panel and streams to the viewer instead of living in a
    # shell history.
    curriculum=(
        CurriculumStage(
            label="jaws around it", steps=900_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "0"},
            detail="The can spawns BETWEEN the pads, 0.25-0.27 m ahead and "
                   "within 15 mm of centre. Nothing to drive to: the only "
                   "thing to learn is what closing on something does, and "
                   "what lifting it does. A policy whose rollouts never once "
                   "contain a can between its pads never learns to close on "
                   "one, and no reward weight fixes that."),
        CurriculumStage(
            label="just out of reach", steps=700_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "1"},
            detail="0.30-0.36 m ahead, up to 40 mm off centre — a few "
                   "centimetres of driving before the grip it already knows. "
                   "Short enough that the approach is a correction rather "
                   "than a journey."),
        CurriculumStage(
            label="the brain's own handover", steps=900_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "2"},
            detail="0.36-0.55 m ahead and up to 120 mm off centre, which is "
                   "where `brain/tidy_moss.py` stops driving and hands over. "
                   "Finish here and the scripted loop can call this policy "
                   "for the last 30 cm."),
    ),
    terms=(
        RewardTerm("gap_progress",
                   "the ground made up between the can and the pocket "
                   "between its pads — and the same taken back for pushing "
                   "it away",
                   60.0, _env_owned),
        RewardTerm("held",
                   "a small trickle per step while the jaws are shut on the "
                   "can; small on purpose, because a big one teaches it to "
                   "park",
                   0.3, _env_owned),
        RewardTerm("lift_progress",
                   "per metre GAINED in height while holding it, so the only "
                   "way to keep earning is to keep lifting",
                   120.0, _env_owned),
        RewardTerm("picked",
                   "once, for a can lifted clear of the floor",
                   20.0, _env_owned),
        RewardTerm("knocked_away",
                   "once, for shoving the can out of the band it could be "
                   "picked from",
                   -3.0, _env_owned),
        RewardTerm("action_rate",
                   "a flat nudge against twitchy commands — flat, because a "
                   "ramped one ate another body's whole task",
                   -0.01, _env_owned),
    ),
)


_register(MOSS_PICK)


#: THE PICK THAT CAN SEE ITS OWN GRIP (2026-09-27). Trained from scratch, not
#: fine-tuned: its inputs mean something new. Four things differ from
#: MOSS_PICK, each measured first (the curriculum adds one difficulty per
#: stage; the first version turned everything on at once and never closed):
#:   * `grip_xyz` — the wrist depth camera's fix of the object in the TOOL
#:     frame (is it centred between the jaws, and how deep). A distance alone
#:     (8956a1) improved picks in the env and not the bin.
#:   * yard handover states in the last two stages — the yard's fingertip
#:     grips come from the states it hands the pick (9/24 settled grips v 3/43
#:     from the env's spawns).
#:   * `deep_grip_m` 0.035 — only a grip the carry survives counts.
#:   * 15 s episodes — a missed grasp does not end the episode, so it learns
#:     to re-grasp with the arm out instead of the brain tucking and
#:     redeploying (12 s a time, half of every yard run).
MOSS_PICK_GRIP = dataclasses.replace(
    MOSS_PICK,
    id="moss_pick_grip",
    emoji="🫳",
    title="Pick with the depth camera (MOSS)",
    description="The pick, trained from scratch with the wrist depth camera's "
                "view of where the object sits in its jaws, from the yard's "
                "real handover states, long enough to re-grasp.",
    keywords=("pick with the depth camera", "grip-aware pick",
              "pick it up by feel"),
    suggest="pick with the depth camera",
    episode_s=15.0,
    curriculum=(
        CurriculumStage(
            label="cans between the jaws", steps=1_500_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "0",
                 "MICRODUCK_MOSS_PROP_VARIETY": "0"},
            detail="One shape, already between the pads, base free, wrist "
                   "where it deploys, nothing charged: learn what closing and "
                   "lifting do. MEASURED at 1.5M (20 deterministic probes): "
                   "19/20 like this, 14/20 with the wrist randomised, 8/20 "
                   "with the base locked, 4/20 with all of 478dad's settings; "
                   "every penalty on from step one: 0/20 at 900k."),
        CurriculumStage(
            label="every shape between the jaws", steps=2_000_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "0",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1"},
            detail="All the yard's shapes, still between the pads."),
        CurriculumStage(
            label="just out of reach", steps=1_500_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "1",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1"},
            detail="A few centimetres of reaching before the grip."),
        CurriculumStage(
            label="the handover distance", steps=2_000_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "2",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1"},
            detail="Where the brain hands over."),
        CurriculumStage(
            label="the shipped inputs, half from the yard", steps=2_500_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "2",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1",
                 "MICRODUCK_MOSS_ATTITUDE": "1",
                 "MICRODUCK_MOSS_SIZE_OBS": "1",
                 "MICRODUCK_MOSS_PROXIMITY": "1",
                 "MICRODUCK_MOSS_WRIST_FREE": "1",
                 "MICRODUCK_MOSS_WRIST_START": "1.5708",
                 "MICRODUCK_MOSS_JAW_ALIGN": "6.0",
                 "MICRODUCK_MOSS_ALIGN_HOLD": "0.3",
                 "MICRODUCK_MOSS_HANDOVER_STATES": "1",
                 "MICRODUCK_MOSS_HANDOVER_FRAC": "0.5"},
            detail="The object's axis, size and range, a wrist that starts "
                   "anywhere in its arc, and half the episodes starting where "
                   "moss-yard hands over."),
        CurriculumStage(
            label="deep and careful, from the yard", steps=2_500_000,
            env={"MICRODUCK_MOSS_PICK_RUNG": "2",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1",
                 "MICRODUCK_MOSS_ATTITUDE": "1",
                 "MICRODUCK_MOSS_SIZE_OBS": "1",
                 "MICRODUCK_MOSS_PROXIMITY": "1",
                 "MICRODUCK_MOSS_WRIST_FREE": "1",
                 "MICRODUCK_MOSS_WRIST_START": "1.5708",
                 "MICRODUCK_MOSS_JAW_ALIGN": "6.0",
                 "MICRODUCK_MOSS_ALIGN_HOLD": "0.3",
                 "MICRODUCK_MOSS_HANDOVER_STATES": "1",
                 "MICRODUCK_MOSS_HANDOVER_FRAC": "1.0",
                 "MICRODUCK_MOSS_DEEP_GRIP": "0.035",
                 "MICRODUCK_MOSS_TOPPLE": "15.0",
                 "MICRODUCK_MOSS_TORQUE_SAT": "0.05",
                 "MICRODUCK_MOSS_OVERSPEED": "0.3",
                 "MICRODUCK_MOSS_ARM_FLOOR": "0.2",
                 "MICRODUCK_MOSS_LOW_APPROACH": "0.1"},
            detail="Only a deep grip counts, the shipped pick's penalties come "
                   "on, and every episode starts from a yard handover state."),
    ),
)


_register(MOSS_PICK_GRIP)


MOSS_APPROACH = Behavior(
    id="moss_approach",
    emoji="🛞",
    title="Drive up to a can (MOSS)",
    description="Cross the room to a can and stop at the handover pose "
                "WITHOUT bulldozing it — the leg that decides whether the "
                "pickup policy is handed a standing can or a rolling one.",
    how_it_learns=(
        "Progress-pay on the distance to the handover pose, and a large "
        "per-second penalty for TOUCHING the can before it gets there. The "
        "penalty is per second rather than once because knocking a can is "
        "not an event the robot can undo: a can nudged at 2 m/s keeps "
        "rolling, and paying once for the first contact let a policy barge "
        "through and eat a single fine. MEASURED from the scripted loop: the "
        "approach topples the can more often than not, which is why the "
        "pickup env has to spend half its episodes on a can lying on its "
        "side. Fix the approach and that stops being the common case. It is "
        "also paid a little to keep the arm folded inside the shell while it "
        "drives, because the arm does NOT reach its tuck — it jams on the "
        "hull 0.74 rad short and rides 75 mm proud of the chassis, which is "
        "the wall-catching Laurent described. Since the policy holds all "
        "eight actions it can fold the arm itself, so the shell is a thing "
        "it learns rather than a pose somebody asserted."
    ),
    # The SUGGEST string must be one of these: the panel's chip posts its own
    # text back to /teach, so a suggestion that is not also a keyword is a
    # button that answers "I don't know that moss task yet".
    keywords=("drive up to the can", "drive to the can", "go to the can",
              "approach the can", "drive up to it", "head for the litter"),
    suggest="drive up to the can",
    robot="moss",
    episode_s=13.6,          # `moss_env.approach_seconds(1)`
    trainer=("-m", "microduck_local.train", "--robot", "moss",
             "--task", "approach"),
    curriculum=(
        CurriculumStage(
            label="the last half metre", steps=600_000,
            env={"MICRODUCK_MOSS_APPROACH_RUNG": "0"},
            detail="The can starts 0.8-1.5 m out and nearly straight ahead. "
                   "Short enough that arriving is a matter of stopping in "
                   "the right place rather than of finding anything."),
        CurriculumStage(
            label="across the room", steps=800_000,
            env={"MICRODUCK_MOSS_APPROACH_RUNG": "1"},
            detail="Further out and well off the bearing, so the turn is "
                   "part of the approach and the can leaves the camera's "
                   "87-degree frame during it."),
    ),
    terms=(
        RewardTerm("reach_progress",
                   "the ground made up toward the pose the pickup policy "
                   "takes over from",
                   25.0, _env_owned),
        RewardTerm("disturbed",
                   "per SECOND the can is being touched before arrival — the "
                   "failure that hands the next policy a rolling can",
                   -60.0, _env_owned),
        RewardTerm("arrived",
                   "once, for stopping at the handover pose facing the can",
                   25.0, _env_owned),
        RewardTerm("out_of_shell",
                   "per metre any arm link sticks out past the chassis while "
                   "driving",
                   -0.5, _env_owned),
        RewardTerm("action_rate",
                   "a flat nudge against twitchy commands",
                   -0.01, _env_owned),
    ),
)


MOSS_STOW = Behavior(
    id="moss_stow",
    emoji="🗑️",
    title="Put it in the bin (MOSS)",
    description="Carry a gripped can to the rover's own bin and let go of it "
                "INSIDE — the leg the scripted loop loses the can on.",
    how_it_learns=(
        "One task for carry AND release, not two, because the seam between "
        "them is the bug: a carry policy that ended when the arm arrived "
        "would be scored a success at exactly the moment the scripted loop "
        "fails, and a handover is one more place for the can to fall. So the "
        "only thing that counts as done is the can at rest inside the bin. "
        "Progress-pay is on the CAN's distance to the bin's mouth rather "
        "than the gripper's, so an arm that swings over the bin having left "
        "the can behind has made no progress at all, and letting go anywhere "
        "outside that mouth is paid -15. The ladder is in the GRIP, not the "
        "weights: rung 0 starts already lifted with the can seated square "
        "between the pads so the first thing learned is the swing, and the "
        "later rungs seat it off-centre and tilted. That is deliberate — the "
        "pickup policy does not hand over a tidy centred can, it hands over "
        "whatever it managed to close on, and a stow trained on a perfect "
        "grip learns a swing that only works for one."
    ),
    # Same rule as the approach recipe: the chip's own text is a keyword.
    keywords=("put the can in the bin", "put it in the bin", "stow the can",
              "drop it in the bin", "bin the can", "put the litter away"),
    suggest="put the can in the bin",
    robot="moss",
    episode_s=20.0,          # `moss_env.STOW_EPISODE_S`
    trainer=("-m", "microduck_local.train", "--robot", "moss",
             "--task", "stow"),
    curriculum=(
        CurriculumStage(
            label="let go over the bin", steps=400_000,
            env={"MICRODUCK_MOSS_STOW_RUNG": "0"},
            detail="The episode STARTS with the arm already swung round over "
                   "the bin, can still held. The only thing to learn is the "
                   "descent into the mouth and when to open. It reaches 12/12 "
                   "into the bin inside 40k steps here — the same task "
                   "trained from the lift pose scored 0/12 across four "
                   "chains of 2M."),
        CurriculumStage(
            label="turn round with it", steps=700_000,
            env={"MICRODUCK_MOSS_STOW_RUNG": "1"},
            detail="Now it starts folded up and clear of the bin, facing "
                   "forward, and has to swing the shoulder round about 3.1 "
                   "rad before the descent it already knows. This is the "
                   "hard rung: turning the right way makes the can's "
                   "distance to the bin WORSE for a third of the move, so a "
                   "greedy progress term punishes exactly the manoeuvre that "
                   "works. The one-shot bonus for getting the HAND over the "
                   "bin is what pays for it."),
        CurriculumStage(
            label="the whole move", steps=700_000,
            env={"MICRODUCK_MOSS_STOW_RUNG": "2"},
            detail="From the lift pose, where the pickup policy really hands "
                   "over: fold the arm up clear of the bin, rotate, descend, "
                   "release — with the can sitting up to 12 mm off centre "
                   "and 0.4 rad tilted in the jaws, which is the spread the "
                   "pickup actually produces."),
    ),
    terms=(
        RewardTerm("stow_progress",
                   "the ground the CAN makes up toward the bin's mouth — "
                   "measured on the can, so an empty gripper earns nothing",
                   80.0, _env_owned),
        RewardTerm("carrying",
                   "a small trickle per step while the pads are shut on it; "
                   "small on purpose, or it learns to stand and hold",
                   0.3, _env_owned),
        RewardTerm("stowed",
                   "once, for a can at rest inside the bin",
                   40.0, _env_owned),
        RewardTerm("dropped",
                   "once, for letting go of it anywhere else — the failure "
                   "this whole task exists to train out",
                   -15.0, _env_owned),
        RewardTerm("action_rate",
                   "a flat nudge against twitchy commands",
                   -0.01, _env_owned),
    ),
)


MOSS_FOLD = Behavior(
    id="moss_fold",
    emoji="🦾",
    title="Fold the arm home (MOSS)",
    description="After a delivery, get the arm out of the bin and folded "
                "back to its tuck pose — its own leg, from scratch.",
    how_it_learns=(
        "Every episode starts where a delivery ENDS: object in the bin, arm "
        "over it. There is no straight path home — the bin is in the way — so "
        "the pay is in two legs, first to a waypoint a planner found clear of "
        "the bin, then to the tuck. A third of episodes start AT the waypoint "
        "so the second leg is practised from its own start. It trains from "
        "SCRATCH on purpose: warm-started from the stow, the fold's joint "
        "angles sat 13-25 standard deviations outside the frozen observation "
        "normaliser, and two runs learned the first leg and never the second "
        "(teach-moss_stow-bd60ff, -4be5a4). From scratch it still never "
        "finished once in 2M steps (92301e), so now most episodes START "
        "along the clear path, many close to home — a reverse curriculum. "
        "Touching the bin costs 1 a tick, and any joint turning faster than "
        "its rated 1.5 rad/s (commands cap at 0.75) is charged."
    ),
    keywords=("fold the arm", "fold the arm home", "tuck the arm",
              "arm home", "put the arm away", "retract the arm"),
    suggest="fold the arm home",
    robot="moss",
    episode_s=8.0,           # `moss_env.RETRACT_DRILL_S`
    trainer=("-m", "microduck_local.train", "--robot", "moss",
             "--task", "stow"),
    curriculum=(
        CurriculumStage(
            label="fold home from the bin", steps=2_000_000,
            env={"MICRODUCK_MOSS_RETRACT_DRILL": "1",
                 "MICRODUCK_MOSS_RETRACT_DRILL_WP": "0",
                 "MICRODUCK_MOSS_RETRACT_DRILL_PATH": "0.5",
                 "MICRODUCK_MOSS_RETRACT_DRILL_BANK": "moss_tuck_entry_poses_v2.npy",
                 "MICRODUCK_MOSS_BIN_CLUTTER": "6",
                 "MICRODUCK_MOSS_CMD_LEASH": "0.08",
                 "MICRODUCK_MOSS_OVERSPEED": "2.0",
                 "MICRODUCK_MOSS_RETRACT_STAGED": "1",
                 "MICRODUCK_MOSS_RETRACT": "6.0",
                 "MICRODUCK_MOSS_RETRACT_WP_BONUS": "15.0",
                 "MICRODUCK_MOSS_HOME_BONUS": "25.0",
                 "MICRODUCK_MOSS_BIN_SCRAPE": "1.0",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1",
                 "MICRODUCK_MOSS_STOW_RUNG": "2"},
            detail="70% of episodes start somewhere ALONG the known clear "
                   "path, weighted toward home, so the finish is sampled "
                   "from the first update. Scripted straight from the waypoint, the fold takes 97 "
                   "steps and reaches home 16/16; the pay for it is +36.5 "
                   "an episode against -105 for sitting still. So the leg "
                   "is reachable and paid — what it lacked was inputs the "
                   "policy could resolve."),
    ),
    terms=(
        RewardTerm("retract",
                   "per radian folded toward the current leg's goal — the "
                   "waypoint first, then the tuck",
                   6.0, _env_owned),
        RewardTerm("waypoint",
                   "once, for reaching the waypoint clear of the bin",
                   15.0, _env_owned),
        RewardTerm("home",
                   "once, for the arm folded to the tuck pose",
                   25.0, _env_owned),
        RewardTerm("bin_scrape",
                   "per tick the arm is touching the bin",
                   -0.25, _env_owned),
        RewardTerm("action_rate",
                   "a flat nudge against twitchy commands",
                   -0.01, _env_owned),
    ),
)


_register(MOSS_APPROACH)
_register(MOSS_STOW)
_register(MOSS_FOLD)


MOSS_STOW_AIM = Behavior(
    id="moss_stow_aim",
    emoji="🎯",
    title="Put it in the bin, where there's room (MOSS)",
    description="Carry the object to the bin and let go over the spot the "
                "arm camera picked as clearest, in a bin already holding up "
                "to six pieces of litter.",
    how_it_learns=(
        "The arm camera looks into the bin and a rule picks the spot with the "
        "most room (`moss_bin`); the policy sees that spot in slots 28-29 and "
        "is paid for letting go OVER it — the release point, because objects "
        "move a median 4 cm after they are dropped and the resting place is "
        "half luck. Two earlier runs that paid only on progress, or on the "
        "resting place, ignored the spot entirely (slope 0). So the ladder "
        "starts with the arm already over the bin, where aiming is the ONLY "
        "thing left to learn, before the whole move."
    ),
    keywords=("put it in the bin where there's room", "aim into the bin",
              "place it in the bin", "stow with the camera"),
    suggest="put it in the bin where there's room",
    robot="moss",
    episode_s=20.0,
    trainer=("-m", "microduck_local.train", "--robot", "moss",
             "--task", "stow"),
    curriculum=(
        CurriculumStage(
            label="aim from over the bin", steps=600_000,
            env={"MICRODUCK_MOSS_STOW_RUNG": "0",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1", "MICRODUCK_MOSS_RETRACT": "6.0",
                 "MICRODUCK_MOSS_BIN_SCRAPE": "0.25", "MICRODUCK_MOSS_HOME_BONUS": "25.0",
                 "MICRODUCK_MOSS_BIN_CLUTTER": "6", "MICRODUCK_MOSS_DROP_TARGET": "1",
                 "MICRODUCK_MOSS_DROP_ACCURACY": "0.6"},
            detail="Starts holding the object over the bin; shift over the "
                   "chosen spot and let go."),
        CurriculumStage(
            label="the whole move, aimed", steps=1_400_000,
            env={"MICRODUCK_MOSS_STOW_RUNG": "2",
                 "MICRODUCK_MOSS_PROP_VARIETY": "1", "MICRODUCK_MOSS_RETRACT": "6.0",
                 "MICRODUCK_MOSS_BIN_SCRAPE": "0.25", "MICRODUCK_MOSS_HOME_BONUS": "25.0",
                 "MICRODUCK_MOSS_BIN_CLUTTER": "6", "MICRODUCK_MOSS_DROP_TARGET": "1",
                 "MICRODUCK_MOSS_DROP_ACCURACY": "0.6"},
            detail="From the lift pose, as the pickup hands over."),
    ),
    terms=MOSS_STOW.terms,
)
_register(MOSS_STOW_AIM)
