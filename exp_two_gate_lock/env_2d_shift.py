"""
Two-Gate Lock — 2D with Location Shift
========================================
5×5 grid. Phase 1: A=(2,1), B=(3,3). 
Phase 2 (location shift): A=(4,0), B=(0,4).
Phase 3: restore A/B to Phase 1 positions.

Observation: 29-dim flattened grid (absolute coordinates only)
- No relative A/B features (prevents skill from generalising to shifted locations)
- Skill must encode absolute spatial memory → location shift invalidates content
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal

GRID_W, GRID_H = 5, 5
START = (0, 0)

# Phase 1 (default) positions
P1_A = (2, 1); P1_B = (3, 3)
# Phase 2 (shifted) positions — no overlap with P1
P2_A = (4, 0); P2_B = (0, 4)

DOOR = (2, 2); GOAL = (4, 4)

SEQ_TIMEOUT = 12
MAX_EPISODE_STEPS = 50
NUM_ACTIONS = 4
ACTION_DELTA = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTION_NAMES = ["N", "S", "E", "W"]
Rule = Literal["A→B", "B→A"]
OBS_DIM = 29  # 25 grid + 4 flags


@dataclass
class EnvState:
    x: int; y: int
    a_pressed: bool = False; b_pressed: bool = False
    sequence_ok: bool = False; door_open: bool = False
    steps: int = 0; steps_since_first: int = 0
    done: bool = False; a_just: bool = False; b_just: bool = False


class TwoGate2DShiftEnv:
    """5×5 grid with location shift support."""

    def __init__(self, rule: Rule = "A→B", seed: int = 0, shifted: bool = False):
        self.rule = rule
        self.shifted = shifted
        self.rng = np.random.RandomState(seed)
        self.reset()

    @property
    def _a(self): return P2_A if self.shifted else P1_A

    @property
    def _b(self): return P2_B if self.shifted else P1_B

    def set_shifted(self, shifted: bool):
        self.shifted = shifted

    def set_rule(self, rule: Rule):
        self.rule = rule

    def reset(self) -> np.ndarray:
        self.s = EnvState(x=START[0], y=START[1])
        return self._obs()

    def _obs(self) -> np.ndarray:
        s = self.s
        grid = np.zeros(GRID_H * GRID_W, dtype=np.float32)
        grid[s.y * GRID_W + s.x] = 1.0  # agent
        grid[self._a[1] * GRID_W + self._a[0]] = 1.0  # A
        grid[self._b[1] * GRID_W + self._b[0]] = 1.0  # B
        if not s.door_open:
            grid[DOOR[1] * GRID_W + DOOR[0]] = 1.0  # blocked door
        grid[GOAL[1] * GRID_W + GOAL[0]] = 1.0  # goal
        flags = np.array([
            float(s.a_pressed), float(s.b_pressed),
            float(s.door_open), s.steps_since_first / SEQ_TIMEOUT,
        ], dtype=np.float32)
        return np.concatenate([grid, flags])  # 29-dim

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        s = self.s
        dx, dy = ACTION_DELTA[action]
        nx = int(np.clip(s.x + dx, 0, GRID_W - 1))
        ny = int(np.clip(s.y + dy, 0, GRID_H - 1))
        if nx == DOOR[0] and ny == DOOR[1] and not s.door_open:
            nx, ny = s.x, s.y
        s.x, s.y = nx, ny
        s.steps += 1; s.a_just = s.b_just = False

        at_a, at_b = (s.x, s.y) == self._a, (s.x, s.y) == self._b
        if self.rule == "A→B":
            if at_a and not s.a_pressed:
                s.a_pressed = True; s.steps_since_first = 0; s.a_just = True
            if at_b and s.a_pressed and not s.b_pressed and s.steps_since_first <= SEQ_TIMEOUT:
                s.b_pressed = True; s.sequence_ok = True; s.door_open = True; s.b_just = True
        else:
            if at_b and not s.b_pressed:
                s.b_pressed = True; s.steps_since_first = 0; s.b_just = True
            if at_a and s.b_pressed and not s.a_pressed and s.steps_since_first <= SEQ_TIMEOUT:
                s.a_pressed = True; s.sequence_ok = True; s.door_open = True; s.a_just = True

        fd = (s.a_pressed if self.rule == "A→B" else s.b_pressed)
        sd = (s.b_pressed if self.rule == "A→B" else s.a_pressed)
        if fd and not sd: s.steps_since_first += 1

        reward = -0.02
        if s.b_just: reward = 0.5
        elif s.a_just: reward = 0.1
        if not s.sequence_ok:
            target = self._a if not s.a_pressed else self._b
            dist = np.sqrt((target[0] - s.x)**2 + (target[1] - s.y)**2)
            max_d = np.sqrt(GRID_W**2 + GRID_H**2)
            reward += 0.03 * (1 - dist / max_d)
        if (s.x, s.y) == GOAL and s.door_open:
            reward = 1.0; s.done = True
        if s.steps >= MAX_EPISODE_STEPS: s.done = True
        return self._obs(), reward, s.done, {
            "at_goal": (s.x, s.y) == GOAL and s.door_open,
            "door_open": s.door_open, "sequence_ok": s.sequence_ok,
        }

    def render(self) -> str:
        g = [["•" for _ in range(GRID_W)] for _ in range(GRID_H)]
        g[START[1]][START[0]] = "S"
        g[self._a[1]][self._a[0]] = "A"
        g[self._b[1]][self._b[0]] = "B"
        g[GOAL[1]][GOAL[0]] = "G"
        g[DOOR[1]][DOOR[0]] = " " if self.s.door_open else "▣"
        g[self.s.y][self.s.x] = "@"
        lines = [f"Rule {self.rule}  Shift={'YES' if self.shifted else 'NO'}"]
        for row in g: lines.append(" ".join(f"{c:^3}" for c in row))
        lines.append(f"A={int(self.s.a_pressed)} B={int(self.s.b_pressed)} t={self.s.steps}")
        return "\n".join(lines)
