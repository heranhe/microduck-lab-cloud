"""How SOFT an object's contact is — one definition for the training envs and
the world, so a cigarette butt squeezes the same in both.

`soft` runs 0..1. 0 is MuJoCo's default rigid contact and returns None, so
every existing geom compiles exactly as before. Towards 1 the contact gets a
slower time constant and a lower, wider impedance: squeezed, the object gives
by a few millimetres instead of pushing back at full stiffness. That is the
difference between a jaw PINCHING a filter and a jaw SQUIRTING a hard 8 mm
stick sideways out of its pads — MEASURED on a rigid butt (2026-09-27), 7 of
19 failed grasps had the jaws close all the way to 0 mm on nothing.

NOT USED BY ANY LITTER (2026-09-27), and why: softening the butt to 1.0 took
the shipped pick from 11/30 butts to 3/30. In MuJoCo a softer contact gives
LESS normal force at the same squeeze, so less friction — the opposite of a
real filter, which compresses AND grips. It stays as plumbing (Prop.soft,
GraspProp.soft) for a better-calibrated model; friction alone (1.2) was 12/30.
"""
from __future__ import annotations


def contact_softness(soft: float):
    """`(solref, solimp)` for a geom of this softness, or None when rigid."""
    s = float(soft)
    if s <= 0.0:
        return None
    s = min(s, 1.0)
    solref = [0.02 + 0.04 * s, 1.0]                       # time constant, damping
    solimp = [0.9 - 0.4 * s, 0.95, 0.001 + 0.004 * s, 0.5, 2.0]
    return solref, solimp
