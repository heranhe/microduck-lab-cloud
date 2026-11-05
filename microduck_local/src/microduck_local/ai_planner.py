"""Turn a person's description into a training plan, without running anything the
model wrote.

The model is never trusted with code. It may do exactly two things:

1. pick an existing recipe and set the weights of reward terms that already exist
   (the recipe's own terms, or the shared catalog), or
2. choreograph a motion as keyframes (joint angles) for the imitation recipe.

Both come back as a plan that goes through the same validator an imported plan
does, so a made-up reward key, an out-of-range weight or an impossible clip is
refused there. A refusal is handed back to the model once so it can repair its
own answer; if it still fails, the person sees the reason and nothing started.

This module is free of the web layer so the rules can be tested with a fake model.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

MAX_ATTEMPTS = 2
MAX_PROMPT_CHARS = 4000


class PlannerError(Exception):
    """The model never produced a plan the validator accepts. `reason` is safe to show."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _term_line(term: dict) -> str:
    kind = "penalty (weight is a positive size)" if term.get("isPenalty") else "reward"
    return f'  - {term["key"]}: {term["friendly"]} [{kind}, default {term["weight"]}]'


def build_system_prompt(*, robot: str, behaviors: list[dict], catalog: list[dict],
                        joints: dict | None) -> str:
    """Everything the model may use, and the rules it must follow."""
    lines = [
        "You turn a person's description of a skill into ONE JSON training plan for a robot, "
        "then stop. Reply with the JSON object only, no prose, no code fences.",
        "",
        "HARD RULES",
        "- Never write code. Never invent a reward key, a recipe id or a joint name.",
        "- local.behavior must be one of the recipe ids listed below. local.weights may only use "
        "term keys listed for that recipe or in the shared catalog; each weight is a number "
        "from 0 to 1000. Penalty terms take a positive size, not a negative number.",
        "- Do not set local.servo, cloud, id or revision. The person chooses those.",
        "- source must be \"ai\". title, description and success are short Chinese sentences "
        "the person will read; success says how they can tell it worked.",
        "- steps is the training length, 100000 to 40000000; use the recipe's default unless "
        "the description clearly needs more or less.",
        "",
        "TWO WAYS TO ANSWER",
        "A) An existing recipe fits: set local.behavior and, if the description asks for a "
        "different emphasis, adjust or add weights. Say in description what you changed and why.",
        "B) The person wants a NEW motion no recipe covers and a joint list is given below: set "
        "local.behavior to \"imitate\" and add a top-level \"reference\" with keyframes. The robot "
        "then learns to reproduce your motion while staying balanced. Keyframe rules: "
        "{\"duration\": seconds (at most 10), \"loop\": false, \"keys\": [{\"t\": seconds, "
        "\"joints\": [one angle in radians per joint, in the listed order], \"rootPitch\": 0}]}; "
        "the first key is t=0, times strictly ascend, 3 to 40 keys, and the first and last key "
        "return to the standing pose so the robot starts and ends balanced. Stay inside the "
        "listed limits and move gradually: neighbouring keys differ by less than 0.6 rad per joint.",
        "If nothing fits and no joint list is given, choose the closest recipe and say in "
        "description what is missing.",
        "",
        f"ROBOT: {robot}",
        "RECIPES",
    ]
    for b in behaviors:
        lines.append(f'- id "{b["id"]}": {b["title"]} — {b.get("description", "")}'.rstrip())
        lines.append(f'  default steps {b.get("defaultSteps")}; its terms:')
        lines.extend(_term_line(t) for t in b.get("terms", []))
    lines += ["", "SHARED CATALOG (any recipe may add these)"]
    lines.extend(_term_line(t) for t in catalog)
    if joints:
        lines += ["", "JOINTS (angles in radians, in this order)"]
        for name, lo, hi, stand in zip(joints["names"], joints["lower"], joints["upper"],
                                       joints["stand"], strict=True):
            lines.append(f"  {name}: {lo:.3f} to {hi:.3f}, standing {stand:.3f}")
    lines += ["", "ANSWER SHAPE",
              '{"version":1,"title":"…","description":"…","success":"…","robot":"' + robot + '",'
              '"source":"ai","local":{"behavior":"<id>","steps":<int>,"weights":{"<key>":<number>}}'
              ',"reference":<only for way B>}']
    return "\n".join(lines)


def extract_json(raw: str) -> dict:
    """The object in a model reply, tolerating the fences and chatter models add."""
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the reply contained no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("the reply was not a JSON object")
    return value


def draft_plan(call_model: Callable[[list[dict]], str], validate: Callable[[dict], dict], *,
               system: str, prompt: str, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Ask, validate, and give the model one chance to repair what the validator refused.

    Returns `{"draft": plan, "attempts": n}`. The plan is the VALIDATOR's output, never
    the model's raw JSON, so nothing the validator dropped or normalised comes back.
    """
    if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise PlannerError(f"请填写目标，最多 {MAX_PROMPT_CHARS} 字符")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    last = "AI 没有返回可用内容"
    for attempt in range(1, max_attempts + 1):
        reply = call_model(messages)
        try:
            plan = extract_json(reply)
            plan["source"] = "ai"
            plan.pop("id", None)
            plan.pop("revision", None)
            if isinstance(plan.get("local"), dict):
                plan["local"].pop("servo", None)  # the person's choice, not the model's
            plan["cloud"] = None
            return {"draft": validate(plan), "attempts": attempt}
        except PlannerError:
            raise
        except Exception as exc:  # noqa: BLE001 — every refusal is the model's to repair
            last = getattr(exc, "detail", None) or str(exc) or exc.__class__.__name__
            last = str(last)[:400]
            messages += [{"role": "assistant", "content": (reply or "")[:6000]},
                         {"role": "user", "content":
                          f"That plan was refused: {last}\nReturn a corrected JSON plan only."}]
    raise PlannerError(f"AI 生成的方案没有通过校验：{last}")
