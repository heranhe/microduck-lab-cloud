"""Background resets (`vec_env._SpareReset`): the swap must not touch physics.

A spare env resets on a thread while the active one steps, so a lockstep
fleet stops idling on one worker's expensive reset. What that may change is
which random draws an episode gets; what it must not change is any episode
itself. These tests pin both halves: the swapped sequence is a bit-for-bit
replay of plain envs taking turns, and config reaches every instance.
"""

from __future__ import annotations

import itertools
import warnings

import gymnasium as gym
import numpy as np
import pytest

from microduck_local import vec_env
from microduck_local.vec_env import _build_env, _SpareReset


class Toy(gym.Env):
    """One generator, reseeded by reset(seed=...), random episode lengths —
    the MOSS env's shape at no cost."""

    background_reset = True
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.knob = self.k = 0.0
        self.resets = 0

    def bump(self):
        self.knob += 1.0
        return self.knob

    def _obs(self):
        return np.array([self.x, self.k], np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.resets += 1
        self.k = self.knob          # a RESET-time knob, like a spawn range
        self.x = float(self.rng.normal())
        self.t = 0
        self.len = int(self.rng.integers(1, 6))
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        self.x += float(action[0]) + 0.1 * float(self.rng.normal())
        return self._obs(), self.x, self.t >= self.len, False, {}


def _episodes(pairs, n):
    """Concatenated observations of `n` episodes; `pairs` yields
    (reset_fn, env) per episode."""
    out = []
    for reset, env in itertools.islice(pairs, n):
        o = reset()
        out.append(o)
        while True:
            o, _r, term, trunc, _ = env.step(np.array([0.1], np.float32))
            out.append(o)
            if term or trunc:
                break
    return np.array(out)


def _plain_replay(make, n_spares, n):
    """What the wrapper should produce: the spares' first episodes on seeds
    drawn from the active env's generator, then every instance in turn."""
    active = make()
    seeds = [int(active.rng.integers(2**62)) for _ in range(n_spares)]
    spares = [make() for _ in range(n_spares)]

    def pairs():
        for env, s in zip(spares, seeds):
            yield (lambda e=env, s=s: e.reset(seed=s)[0]), env
        ring = [active, *spares]
        while True:
            for env in ring:
                yield (lambda e=env: e.reset()[0]), env
    return _episodes(pairs(), n)


@pytest.mark.parametrize("n_spares", [1, 2])
def test_swaps_replay_plain_envs_bit_for_bit(n_spares):
    w = _SpareReset(Toy(3), [Toy(3) for _ in range(n_spares)])
    got = _episodes((((lambda: w.reset()[0]), w) for _ in itertools.count()), 25)
    w.close()
    assert np.array_equal(got, _plain_replay(lambda: Toy(3), n_spares, 25))


def test_spares_draw_their_own_streams():
    # Two instances built by the same factory share a seed; without the
    # reseed every spare would replay the active env's episodes exactly.
    w = _SpareReset(Toy(3), [Toy(3), Toy(3)])
    firsts = [w.reset()[0][0] for _ in range(3)]
    w.close()
    assert len(set(firsts)) == 3


def test_set_attr_reaches_every_instance_and_the_next_episode():
    w = _SpareReset(Toy(0), [Toy(0), Toy(0)])
    w.reset()
    w.set_attr("knob", 5.0)
    # the spares' episodes were prepared before the knob moved: redone
    assert w.reset()[0][1] == 5.0
    assert w.reset()[0][1] == 5.0
    assert all(e.knob == 5.0 for e in (w.active, *(j[0] for j in w._queue)))
    w.close()


def test_env_method_reaches_every_instance():
    w = _SpareReset(Toy(0), [Toy(0), Toy(0)])
    w.reset()
    assert w.call("bump", (), {}) == 1.0     # the active env's answer
    assert all(e.knob == 1.0 for e in (w.active, *(j[0] for j in w._queue)))
    w.close()


def test_a_failed_background_reset_surfaces_on_the_swap():
    class Breaks(Toy):
        def reset(self, **kw):
            if self.resets >= 1:
                raise RuntimeError("reset blew up")
            return super().reset(**kw)
    w = _SpareReset(Toy(0), [Breaks(0)])
    w.reset()                       # the spare's first reset is fine
    w.step(np.array([0.1], np.float32))
    w.reset()                       # the Toy's first; the Breaks one relaunches
    with pytest.raises(RuntimeError, match="blew up"):
        w.reset()


def test_only_opted_in_envs_get_spares(monkeypatch):
    class Plain(Toy):
        background_reset = False
    assert isinstance(_build_env(lambda: Plain(0)), Plain)
    assert isinstance(_build_env(lambda: Toy(0)), _SpareReset)
    monkeypatch.setattr(vec_env, "BACKGROUND_RESET", False)
    assert isinstance(_build_env(lambda: Toy(0)), Toy)


def test_fork_vec_env_steps_through_spares():
    v = vec_env.ForkVecEnv([lambda i=i: Toy(i) for i in range(3)])
    try:
        v.reset()
        dones = 0
        for _ in range(40):
            _o, _r, d, infos = v.step(np.full((3, 1), 0.1, np.float32))
            for i in np.flatnonzero(d):
                assert "terminal_observation" in infos[i]
            dones += int(d.sum())
        assert dones > 10
    finally:
        v.close()


def test_moss_pick_swaps_replay_plain_envs_bit_for_bit():
    """The real thing: prop recompiles, the 2 s deploy, handover models
    shared between instances — all on a thread beside a stepping env."""
    from microduck_local.robots import moss
    if not moss.moss_ready():
        pytest.skip("MOSS's assets are not downloaded — `uv run fetch-robot moss`")
    from microduck_local.robots.moss_env import MossPickEnv
    kw = dict(prop_variety=True, handover_states=True, handover_frac=0.5,
              wrist_start_rand=1.5708)

    def make():
        return MossPickEnv(seed=5, **kw)

    def run(pairs):
        out, W = [], None
        for reset, env in itertools.islice(pairs, 6):
            o = reset()
            if W is None:
                W = np.random.default_rng(0).normal(
                    0, 0.3, (env.action_space.shape[0], o.shape[0]))
            out.append(o)
            for _ in range(30):             # a prefix is enough to diverge
                o, _r, term, trunc, _ = env.step(np.tanh(W @ o).astype(np.float32))
                out.append(o)
                if term or trunc:
                    break
        return np.array(out)

    w = _SpareReset(make(), [make()])
    got = run(((lambda: w.reset()[0]), w) for _ in itertools.count())
    w.close()
    active = make()
    seed = int(active.rng.integers(2**62))
    spare = make()

    def plain():
        yield (lambda: spare.reset(seed=seed)[0]), spare
        while True:
            for env in (active, spare):
                yield (lambda e=env: e.reset()[0]), env
    assert np.array_equal(got, run(plain()))


def test_an_env_without_a_seeded_rng_resets_in_line():
    # gymnasium's `np_random` is OS entropy until the first seeded reset;
    # spares seeded from it would make a fixed --seed irreproducible.
    class NoRng(Toy):
        def __init__(self, seed: int = 0):
            super().__init__(seed)
            del self.rng
    with pytest.warns(UserWarning, match="no seeded `rng`"):
        env = _build_env(lambda: NoRng(0))
    assert isinstance(env, NoRng)


def test_envs_sharing_one_compiled_model_reset_in_line():
    # `shared_model_scope` hands every instance ONE model, and the walk env
    # writes each step's domain randomization into it: a spare resetting on
    # a thread would race those writes.
    shared = object()
    built = []

    class Shared(Toy):
        def __init__(self, seed: int = 0):
            super().__init__(seed)
            self.model = shared
            self.closed = False
            built.append(self)

        def close(self):
            self.closed = True
    with pytest.warns(UserWarning, match="shares one compiled model"):
        env = _build_env(lambda: Shared(0))
    assert env is built[0] and all(s.closed for s in built[1:])


def test_the_scene_cache_key_sees_what_the_scene_build_reads(monkeypatch):
    """A cached handover scene is reused only while everything the MOSS scene
    build reads from outside its arguments still matches."""
    from microduck_local.robots import moss, moss_env
    base = moss_env._scene_inputs()
    monkeypatch.setattr(moss, "ARM_CAMERA_POS", (0.0, 0.0, 0.0))
    assert moss_env._scene_inputs() != base
    monkeypatch.undo()
    monkeypatch.setattr(moss, "CACHE_DIR", moss.CACHE_DIR / "elsewhere")
    assert moss_env._scene_inputs() != base
    monkeypatch.undo()
    monkeypatch.setenv("MICRODUCK_MOSS_COLLISION_V04", "0")
    assert moss_env._scene_inputs() != base


def test_an_env_may_declare_its_compiled_model_read_only():
    """The share check runs ONCE, at construction, so it cannot see a cache
    that hands the same model out at RESET time. An env that never writes
    into its model says so instead, and keeps its spares."""
    shared = object()
    built = []

    class SharedReadOnly(Toy):
        shares_compiled_model = True

        def __init__(self, seed: int = 0):
            super().__init__(seed)
            self.model = shared
            self.closed = False
            built.append(self)

        def close(self):
            self.closed = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning here fails the test
        env = _build_env(lambda: SharedReadOnly(0))
    assert isinstance(env, _SpareReset)
    assert not any(s.closed for s in built)
    env.close()


def test_moss_pick_really_does_share_the_model_it_declares():
    """The declaration is load-bearing, not decoration.

    `_RECURRING_SCENES` is process-wide, so two instances asking for the
    same handover scene get the SAME `MjModel` — which is exactly what
    `_build_env` refuses over unless the env promises never to write into
    one. If this stops being true the promise should go too.
    """
    from microduck_local.robots import moss
    if not moss.moss_ready():
        pytest.skip("MOSS's assets are not downloaded — `uv run fetch-robot moss`")
    from microduck_local.robots import moss_env
    from microduck_local.robots.moss_env import MossPickEnv

    assert MossPickEnv.shares_compiled_model is True
    a = MossPickEnv(seed=5, handover_states=True, handover_frac=1.0)
    b = MossPickEnv(seed=6, handover_states=True, handover_frac=1.0)
    # Same scene asked for twice, through the path that caches.
    b.prop = a.prop
    b.rung = a.rung
    a._bind_model(recurring=True)
    b._bind_model(recurring=True)
    assert a.model is b.model, "the cache stopped sharing; drop the promise"

    # ...and clearing it really does hand back a fresh compile.
    moss_env.clear_scene_cache()
    assert not moss_env._RECURRING_SCENES
    a._bind_model(recurring=True)
    assert a.model is not b.model


def test_a_patched_scene_constant_can_be_served_a_stale_model(monkeypatch):
    """Why `conftest.moss_scene_cache` exists, demonstrated rather than
    asserted.

    `_scene_inputs` reads the handful of scene inputs that vary at runtime,
    not every `moss` constant a scene build touches — it says which, and
    why that is the right trade. So a test that patches one of the others
    and rebuilds is handed the model compiled BEFORE the patch: green, and
    measuring the physics it thought it had just changed. Clearing is the
    only thing that picks it up, which is why the fixture does it for every
    test instead of leaving each one to remember.
    """
    from microduck_local.robots import moss
    if not moss.moss_ready():
        pytest.skip("MOSS's assets are not downloaded — `uv run fetch-robot moss`")
    from microduck_local.robots import moss_env
    from microduck_local.robots.moss_env import MossPickEnv
    env = MossPickEnv(seed=5, handover_states=True, handover_frac=1.0)
    env._bind_model(recurring=True)
    first = env.model

    before = moss_env._scene_inputs()
    monkeypatch.setattr(moss, "BIN_FLOOR_PRIORITY", moss.BIN_FLOOR_PRIORITY + 7)
    assert moss_env._scene_inputs() == before      # the key cannot see it...
    env._bind_model(recurring=True)
    assert env.model is first                      # ...so the stale model comes back

    moss_env.clear_scene_cache()
    env._bind_model(recurring=True)
    assert env.model is not first
