"""A LEARNED scorer for the kick candidates (roadmap E.2).

`brain/kickselect.py` ranks the candidate fan by SIMULATING each line 30
times under the measured kick model and reading where the ball stops
(A.3). That roll-out is a model of the BALL; it knows nothing about the
body that has to walk round to the spot, and it is calibrated to a kick
that connects — while half the swings in the gym do not. This module is
the other half of E.2's question: keep the same candidate fan and the same
own-goal veto, and learn the RANKING from the gym's own realised
outcomes.

The shape is a contextual regressor, not a policy gradient. One row per
line-up: the candidate the brain actually swung, its context, and the
metres the ball then ran toward the attacked mouth over the carry window.
Exploration comes from the data script picking a random safe candidate per
episode (`scripts/kick_choice_data.py`), so the labels cover candidates the
shipped selector would never take — a regressor fit only on its own picks
would be extrapolating everywhere it matters.

The roll-out's own outputs (`p_goal`, `p_own`, `value`, ...) are INPUT
features, so the learned scorer can at worst re-derive the shipped ranking
and the comparison is "does the context add anything to the simulation",
not "model against model".

Pure numpy, no torch: the whole thing is a 19-feature ridge (or a 16-unit
one-layer MLP) scored 40 times a tick.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .kickselect import PUSH, Pitch, Verdict, inside, potential

# The feature names, in the order `features()` emits them. Written into the
# weights file so a model fitted before a feature moved refuses to load
# rather than silently scoring the wrong column.
FEATURES: tuple[str, ...] = (
    # --- what the roll-out simulation already said about this candidate ---
    "p_goal", "p_own", "p_block", "p_pass", "value",
    # --- the action ---
    "is_push", "is_left",
    # --- the line, against the body and against the pitch ---
    "turn", "abs_turn", "cos_att", "sin_att", "goal_off",
    # --- where the ball is ---
    "ball_ax", "ball_ay", "ball_board", "pot",
    # --- a push is a different animal: let the fit give it its own slope ---
    "abs_turn_push", "value_push",
    # --- how far the body is from the ball right now ---
    "range",
)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def features(v: Verdict, ball, pitch: Pitch, los: float, goal, me) -> list[float]:
    """One candidate's row. `los` is the line of sight to the ball at this
    line-up and `me` the body's (x, y): both are context the roll-out does
    not see — `turn` is how far round the ball this line asks the duck to
    walk, which is the quantity a whiff is made of."""
    bx, by = inside(ball, pitch)
    turn = _wrap(v.heading - los)
    att = 0.0 if pitch.attack_sign >= 0 else math.pi
    d_att = _wrap(v.heading - att)
    u_goal = math.atan2(goal[1] - by, goal[0] - bx)
    push = 1.0 if v.foot == PUSH else 0.0
    return [
        float(v.p_goal), float(v.p_own), float(v.p_block), float(v.p_pass), float(v.value),
        push, 1.0 if v.foot == "kick_left" else 0.0,
        turn, abs(turn), math.cos(d_att), math.sin(d_att), abs(_wrap(v.heading - u_goal)),
        pitch.attack_sign * bx / max(pitch.half_x, 1e-6), by / max(pitch.half_y, 1e-6),
        min(pitch.half_x - abs(bx), pitch.half_y - abs(by)), potential(bx, by, pitch),
        abs(turn) * push, float(v.value) * push,
        math.hypot(bx - float(me[0]), by - float(me[1])),
    ]


class ChoiceModel:
    """Two heads over the same features: the metres the ball is predicted to
    run toward the attacked mouth (`adv`), and the probability the line it
    leaves on points backward (`back`). The score is `adv - lam_back * back`
    — `lam_back` is fitted-in, not a live knob, so a weights file is one
    decision and not two."""

    def __init__(self, kind: str, mean, scale, params: dict, lam_back: float = 0.0,
                 feat: tuple[str, ...] = FEATURES, meta: dict | None = None):
        if tuple(feat) != FEATURES:
            raise ValueError(f"kickchoice: weights were fitted on {tuple(feat)!r}, code has {FEATURES!r}")
        self.kind = kind
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.lam_back = float(lam_back)
        self.meta = dict(meta or {})
        self.P = {k: np.asarray(v, float) for k, v in params.items()}

    # --- the two heads -------------------------------------------------
    def heads(self, X: np.ndarray) -> np.ndarray:
        """(n, 2): predicted advance and predicted backward-line share."""
        Z = (np.asarray(X, float) - self.mean) / self.scale
        if self.kind == "ridge":
            return Z @ self.P["W"] + self.P["b"]
        if self.kind == "mlp":
            H = np.tanh(Z @ self.P["W1"] + self.P["b1"])
            return H @ self.P["W2"] + self.P["b2"]
        raise ValueError(f"kickchoice: unknown model kind {self.kind!r}")

    def score(self, X: np.ndarray) -> np.ndarray:
        h = self.heads(X)
        return h[:, 0] - self.lam_back * h[:, 1]

    # --- on disk --------------------------------------------------------
    def to_json(self) -> dict:
        return {"kind": self.kind, "feat": list(FEATURES), "lam_back": self.lam_back,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "params": {k: v.tolist() for k, v in self.P.items()}, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "ChoiceModel":
        return cls(d["kind"], d["mean"], d["scale"], d["params"], d.get("lam_back", 0.0),
                   tuple(d.get("feat", FEATURES)), d.get("meta"))

    @classmethod
    def load(cls, path) -> "ChoiceModel":
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))


class Chooser:
    """What `kickselect.select` calls instead of its own ranking. `bind`
    captures the line-up's context (the body and the line of sight), so
    `kickselect` stays a pure function of the ball and the pitch."""

    def __init__(self, model: ChoiceModel):
        self.model = model
        self.calls = 0

    @classmethod
    def load(cls, path) -> "Chooser":
        return cls(ChoiceModel.load(path))

    def bind(self, odom, los: float, goal):
        def choose(ball, pitch: Pitch, safe: list[Verdict]) -> Verdict:
            self.calls += 1
            X = np.array([features(v, ball, pitch, los, goal, odom) for v in safe], float)
            return safe[int(np.argmax(self.model.score(X)))]
        return choose


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Standardise, then a penalised least squares per head (the intercept is
    the column mean and is never penalised). Returns (mean, scale, params)."""
    X = np.asarray(X, float)
    mean = X.mean(0)
    scale = np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    Z = (X - mean) / scale
    b = np.asarray(Y, float).mean(0)
    Yc = np.asarray(Y, float) - b
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ Yc)
    return mean, scale, {"W": W, "b": b}


def fit_mlp(X: np.ndarray, Y: np.ndarray, hidden: int = 16, epochs: int = 400,
            lr: float = 0.02, l2: float = 1e-4, seed: int = 0):
    """One tanh layer, full-batch Adam, two linear outputs. Small enough that
    full batch is faster than minibatching and exactly reproducible."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    mean = X.mean(0)
    scale = np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    Z = (X - mean) / scale
    rng = np.random.default_rng(seed)
    n, d = Z.shape
    P = {"W1": rng.normal(0, 1.0 / math.sqrt(d), (d, hidden)), "b1": np.zeros(hidden),
         "W2": rng.normal(0, 1.0 / math.sqrt(hidden), (hidden, Y.shape[1])), "b2": Y.mean(0).copy()}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    vv = {k: np.zeros_like(v) for k, v in P.items()}
    for ep in range(1, epochs + 1):
        A = Z @ P["W1"] + P["b1"]
        H = np.tanh(A)
        out = H @ P["W2"] + P["b2"]
        E = (out - Y) / n
        g = {"W2": H.T @ (2 * E) + l2 * P["W2"], "b2": (2 * E).sum(0)}
        dH = (2 * E) @ P["W2"].T * (1 - H * H)
        g["W1"] = Z.T @ dH + l2 * P["W1"]
        g["b1"] = dH.sum(0)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            vv[k] = 0.999 * vv[k] + 0.001 * g[k] ** 2
            P[k] -= lr * (m[k] / (1 - 0.9 ** ep)) / (np.sqrt(vv[k] / (1 - 0.999 ** ep)) + 1e-8)
    return mean, scale, P
