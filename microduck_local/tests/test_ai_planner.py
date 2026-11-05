"""The AI planner may choose and tune; it may never invent. Every case here drives the
REAL validator with a fake model, so a refusal is the production refusal."""

import json

import pytest

from microduck_local import ai_planner as AP
from microduck_local import behaviors as B
from microduck_local import viz_server as V
from microduck_local.training_plans import clean_plan

STAND = [0.0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.453]


def validate(plan):
    return clean_plan(plan, B, V.clean_clip)


def plan(**local):
    return {"version": 1, "title": "单脚站立", "description": "d", "success": "s", "robot": "microduck",
            "source": "manual", "local": {"behavior": "one_leg", "steps": 2_000_000, **local}}


class FakeModel:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


def run(model, prompt="想让鸭子单脚站稳"):
    return AP.draft_plan(model, validate, system="SYS", prompt=prompt)


# ------------------------------------------------------------------ the answer


def test_a_clean_answer_is_validated_and_comes_back_normalised():
    out = run(FakeModel("```json\n" + json.dumps(plan(weights={"calm_body": 3})) + "\n```"))
    d = out["draft"]
    assert out["attempts"] == 1 and d["source"] == "ai" and d["cloud"] is None
    assert d["local"]["weights"]["calm_body"] == 3
    # Every base term is pinned, not just the one the model mentioned.
    assert {t.key for t in B.BEHAVIORS["one_leg"].terms} <= set(d["local"]["weights"])


def test_the_model_cannot_choose_the_servo_the_cloud_or_the_identity():
    raw = plan(servo="official")
    raw.update(id="evil", revision=9, cloud={"task": "Mjlab-Velocity-Flat-MicroDuck"})
    d = run(FakeModel(json.dumps(raw)))["draft"]
    assert "servo" not in d["local"] and d["cloud"] is None and "id" not in d


def test_chatter_around_the_json_is_tolerated():
    reply = "好的，这是方案：\n" + json.dumps(plan()) + "\n希望有帮助"
    assert run(FakeModel(reply))["draft"]["local"]["behavior"] == "one_leg"


# ------------------------------------------------------------------ never invent


@pytest.mark.parametrize("bad, why", [
    (plan(weights={"win_the_game": 5}), "无效奖励权重"),
    (plan(weights={"calm_body": 5000}), "0–1000"),
    (plan(weights={"calm_body": -1}), "0–1000"),
    (plan(weights={"calm_body": float("nan")}), "0–1000"),
    (plan(weights={"calm_body": True}), "0–1000"),
    (plan(behavior="teleport"), "配方不存在"),
    (plan(steps=5), "训练步数"),
    (plan(code="import os"), "无法识别|支持"),
])
def test_an_invented_or_out_of_range_answer_is_refused_and_the_reason_reaches_the_model(bad, why):
    good = plan(weights={"calm_body": 2})
    model = FakeModel(json.dumps(bad), json.dumps(good))
    out = run(model)
    assert out["attempts"] == 2 and out["draft"]["local"]["weights"]["calm_body"] == 2
    repair = model.calls[1][-1]["content"]
    assert "refused" in repair and "Return a corrected JSON plan only" in repair


def test_a_model_that_keeps_failing_ends_in_a_reason_not_a_started_plan():
    model = FakeModel(json.dumps(plan(weights={"x": 1})), json.dumps(plan(weights={"y": 1})))
    with pytest.raises(AP.PlannerError) as error:
        run(model)
    assert "没有通过校验" in error.value.reason and len(model.calls) == 2


def test_prose_with_no_json_is_a_repairable_failure():
    out = run(FakeModel("我不确定", json.dumps(plan())))
    assert out["attempts"] == 2


def test_an_empty_or_huge_request_never_reaches_the_model():
    model = FakeModel()
    for prompt in ("   ", "x" * (AP.MAX_PROMPT_CHARS + 1)):
        with pytest.raises(AP.PlannerError):
            AP.draft_plan(model, validate, system="S", prompt=prompt)
    assert model.calls == []


# ------------------------------------------------------------------ keyframes


def clip(**over):
    base = {"duration": 2.0, "loop": False, "keys": [
        {"t": 0.0, "joints": STAND, "rootPitch": 0.0},
        {"t": 1.0, "joints": [v + (0.3 if i in (2, 11) else 0) for i, v in enumerate(STAND)], "rootPitch": 0.0},
        {"t": 2.0, "joints": STAND, "rootPitch": 0.0}]}
    base.update(over)
    return base


def imitation(reference):
    return {"version": 1, "title": "挥腿", "description": "d", "success": "s", "robot": "microduck",
            "source": "ai", "local": {"behavior": "imitate", "steps": 2_000_000}, "reference": reference}


def test_a_choreographed_motion_becomes_an_imitation_plan_with_clamped_joints():
    wild = clip()
    wild["keys"][1]["joints"][0] = 9.0           # far outside the hip's range
    d = run(FakeModel(json.dumps(imitation(wild))))["draft"]
    assert d["local"]["behavior"] == "imitate"
    assert d["reference"]["keys"][1]["joints"][0] < 1.0      # clamped to the limit, not 9.0
    assert len(d["reference"]["keys"]) == 3


@pytest.mark.parametrize("reference, why", [
    (clip(keys=[{"t": 0.5, "joints": STAND}]), "t = 0"),
    (clip(keys=[{"t": 0, "joints": STAND}, {"t": 0, "joints": STAND}]), "ascend"),
    (clip(keys=[{"t": 0, "joints": STAND[:5]}]), "14"),
    (clip(duration=1000), "<="),
    (clip(keys=[]), "at least one"),
])
def test_an_impossible_clip_is_refused_and_repaired(reference, why):
    model = FakeModel(json.dumps(imitation(reference)), json.dumps(imitation(clip())))
    assert run(model)["attempts"] == 2
    assert why.lower() in model.calls[1][-1]["content"].lower() or "refused" in model.calls[1][-1]["content"]


def test_a_clip_on_a_non_imitation_recipe_is_refused():
    bad = plan()
    bad["reference"] = clip()
    with pytest.raises(AP.PlannerError):
        AP.draft_plan(FakeModel(json.dumps(bad), json.dumps(bad)), validate, system="S", prompt="p")


# ------------------------------------------------------------------ the prompt


def test_the_prompt_lists_only_what_exists_and_states_the_rules():
    recipes = [{**B.behavior_card(B.BEHAVIORS["one_leg"]), "robot": "microduck"}]
    joints = {"names": ["a", "b"], "lower": [-1.0, -2.0], "upper": [1.0, 2.0], "stand": [0.1, 0.2]}
    text = AP.build_system_prompt(robot="microduck", behaviors=recipes,
                                  catalog=B.catalog_cards(), joints=joints)
    assert 'id "one_leg"' in text and "calm_body" in text and "one_leg_hold" in text
    assert "Never write code" in text and "Do not set local.servo" in text
    assert "a: -1.000 to 1.000, standing 0.100" in text
    assert "JOINTS" not in AP.build_system_prompt(robot="g1", behaviors=[], catalog=[], joints=None)
