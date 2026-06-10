"""
Two-Gate Lock Environment (1D corridor, Rule Swap)
====================================================
Layout:  S . A . . B D G
State:   0 1 2 3 4 5 6 7

Rule:
  A→B (default): visit A then B (in order, within Δ=6) → door opens
  B→A:           visit B then A (in order, within Δ=6) → door opens

Purpose:
  Rule-swap hysteresis test.
  Phase 1 (A→B): PAO crystallises "go right" skill.
  Phase 2 (B→A): Agent must reverse → "go right, then go left".
                 Cached skill fights this → inertia (PAO slower).
  Phase 3 (A→B): "Go right" is correct again → reuse (PAO faster).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal

N_STATES = 8
SWITCH_A = 2
SWITCH_B = 5
DOOR = 6
GOAL = 7
START = 0

SEQ_TIMEOUT = 8  # max steps between first and second switch
MAX_EPISODE_STEPS = 50
NUM_ACTIONS = 2  # 0=LEFT, 1=RIGHT

Rule = Literal["A→B", "B→A"]


@dataclass
class EnvState:
    pos: int = START
    first_pressed: bool = False    # the "first" switch per active rule
    second_pressed: bool = False   # the "second" switch per active rule
    sequence_ok: bool = False
    door_open: bool = False
    steps: int = 0
    steps_since_first: int = 0
    done: bool = False
    first_just_pressed: bool = False
    second_just_pressed: bool = False


class TwoGateLockEnv:
    """1D corridor env with rule-swap support."""

    def __init__(self, rule: Rule = "A→B", seed: int = 0):
        self.rule = rule
        self.rng = np.random.RandomState(seed)
        self.reset()

    def set_rule(self, rule: Rule):
        self.rule = rule

    def reset(self) -> np.ndarray:
        self.s = EnvState()
        return self._obs()

    def _obs(self) -> np.ndarray:
        s = self.s
        # 5-D: pos, first_pressed, second_pressed, door_open, steps_since_first/timeout
        return np.array([
            s.pos / (N_STATES - 1),
            float(s.first_pressed),
            float(s.second_pressed),
            float(s.door_open),
            s.steps_since_first / SEQ_TIMEOUT,
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < NUM_ACTIONS
        s = self.s
        dx = 1 if action else -1
        nx = int(np.clip(s.pos + dx, 0, N_STATES - 1))

        if nx == DOOR and not s.door_open:
            nx = s.pos

        s.pos = nx
        s.steps += 1
        s.first_just_pressed = False
        s.second_just_pressed = False

        if self.rule == "A→B":
            # first = A (pos 2), second = B (pos 5)
            if s.pos == SWITCH_A and not s.first_pressed:
                s.first_pressed = True
                s.steps_since_first = 0
                s.first_just_pressed = True
            if s.pos == SWITCH_B and s.first_pressed and not s.second_pressed \
                    and s.steps_since_first <= SEQ_TIMEOUT:
                s.second_pressed = True
                s.sequence_ok = True
                s.door_open = True
                s.second_just_pressed = True
        else:  # B→A
            # first = B (pos 5), second = A (pos 2)
            if s.pos == SWITCH_B and not s.first_pressed:
                s.first_pressed = True
                s.steps_since_first = 0
                s.first_just_pressed = True
            if s.pos == SWITCH_A and s.first_pressed and not s.second_pressed \
                    and s.steps_since_first <= SEQ_TIMEOUT:
                s.second_pressed = True
                s.sequence_ok = True
                s.door_open = True
                s.second_just_pressed = True

        if s.first_pressed and not s.second_pressed:
            s.steps_since_first += 1

        # ── Reward shaping ──
        reward = -0.02
        if s.second_just_pressed:
            reward = 0.5
        elif s.first_just_pressed:
            reward = 0.1
        if s.pos == GOAL and s.door_open:
            reward = 1.0
            s.done = True
        if s.steps >= MAX_EPISODE_STEPS:
            s.done = True

        return self._obs(), reward, s.done, {
            "at_goal": s.pos == GOAL and s.door_open,
            "door_open": s.door_open,
            "first_pressed": s.first_pressed,
            "second_pressed": s.second_pressed,
            "sequence_ok": s.sequence_ok,
        }

    def render(self) -> str:
        chars = [".", ".", "A", ".", ".", "B", "▣", "G"]
        if self.s.door_open:
            chars[DOOR] = " "
        chars[GOAL] = "G"
        chars[self.s.pos] = "@"
        f, s = "AB" if self.rule == "A→B" else "BA"
        return (
            "".join(f"{c:^3}" for c in chars)
            + f"  first={f}={int(self.s.first_pressed)} "
            f"second={s}={int(self.s.second_pressed)} "
            f"door={int(self.s.door_open)} t={self.s.steps}"
        )


if __name__ == "__main__":
    for rule in ["A→B", "B→A"]:
        print(f"\n--- Rule: {rule} ---")
        env = TwoGateLockEnv(rule=rule)
        print(env.render())
        if rule == "A→B":
            seq = [1] * 7  # all RIGHT
        else:
            seq = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  # R to B, L to A, R to G
        for a in seq:
            obs, r, done, info = env.step(a)
            print(f"  {'R' if a else 'L'} r={r:+.3f} | {env.render()}")
            if done:
                break
