"""Fixtures every test in this directory gets.

Deliberately almost empty: anything autouse here is paid for by all ~1400
tests, so only things that are WRONG to leave undone belong in it.
"""
from __future__ import annotations

import sys

import pytest


def _clear_moss_scene_cache() -> None:
    """Drop `robots/moss_env._RECURRING_SCENES`, if that module is loaded.

    Asked of `sys.modules` rather than imported, so a session that never
    touches MOSS never pays for mujoco's import just to clear an empty dict.
    """
    mod = sys.modules.get("microduck_local.robots.moss_env")
    if mod is not None:
        mod.clear_scene_cache()


@pytest.fixture(autouse=True)
def moss_scene_cache():
    """Compiled MOSS scenes must not cross a test boundary.

    The cache keys on the inputs that vary at RUNTIME, not on every `moss`
    constant a scene reads — `_scene_inputs` says why that is the right
    trade and what it leaves open. What it leaves open is exactly a test:
    monkeypatch `moss.FINGER_FRICTION` or `moss.BIN_FLOOR_PRIORITY`, take
    the handover path, and `_bind_model` hands back the model compiled
    BEFORE the patch — so the test measures the unpatched physics and passes
    green. Clearing on both sides means neither the test that patches nor
    the one after it can be served a stale model.
    """
    _clear_moss_scene_cache()
    yield
    _clear_moss_scene_cache()
