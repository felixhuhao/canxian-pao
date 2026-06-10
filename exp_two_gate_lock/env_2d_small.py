"""
Two-Gate Lock — 2D Small Grid (6×4 / 5×5)
=============================================
Reduced from 7×5 → 5×5 for reliable PPO convergence.

Layout (5×5):
  0 1 2 3 4
0 S . . . .
1 . . A . .
2 . . D . .
3 . . . B .
4 . . . . G

S=(0,0) Start   A=(2,1) Switch A   D=(2,2) Door
B=(3,3) Switch B   G=(4,4) Goal

Non-convex path: agent must go up-right to A, circle around door,
reach B, then door opens, passage through door to goal.

Rule A→B: press A first, then B → door opens
Rule B→A: press B first, then A → door opens
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal

GRID_W, GRID_H = 5, 5
START = (0, 0)
SWITCH_A = (2, 1)
SWITCH_B = (3, 3)
DOOR = (2, 2)
GOAL = (4, 4)

SEQ_TIMEOUT = 12
MAX_EPISODE_STEPS = 50
NUM_ACTIONS = 4  # N,S,E,W
ACTION_DELTA = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTION_NAMES = ["N", "S", "E", "W"]
Rule = Literal["A→B", "B→A"]


@dataclass
class EnvState:
    x: int; y: int
    a_pressed: bool = False; b_pressed: bool = False
    sequence_ok: bool = False; door_open: bool = False
    steps: int = 0; steps_since_first: int = 0
    done: bool = False
    a_just: bool = False; b_just: bool = False


class TwoGate2DSmallEnv:
    """5×5 grid environment with configurable rule."""
    def __init__(self, rule: Rule = "A→B", seed: int = 0):
        self.rule = rule
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.s = EnvState(x=START[0], y=START[1])
        return self._obs()

    def _obs(self) -> np.ndarray:
        s = self.s
        # Part 1: flattened 5×5 grid with spatial encoding (25 dims)
        grid = np.zeros(GRID_H * GRID_W, dtype=np.float32)
        idx = s.y * GRID_W + s.x
        grid[idx] = 1.0  # agent position
        grid[SWITCH_A[1] * GRID_W + SWITCH_A[0]] = 1.0  # switch A
        grid[SWITCH_B[1] * GRID_W + SWITCH_B[0]] = 1.0  # switch B
        if not s.door_open:
            grid[DOOR[1] * GRID_W + DOOR[0]] = 1.0  # blocked door
        # Goal marker
        grid[GOAL[1] * GRID_W + GOAL[0]] = 1.0
        # Part 2: relative features + flags (10 dims, as before)
        rel = np.array([
            (SWITCH_A[0] - s.x) / GRID_W, (SWITCH_A[1] - s.y) / GRID_H,
            (SWITCH_B[0] - s.x) / GRID_W, (SWITCH_B[1] - s.y) / GRID_H,
            float(s.a_pressed), float(s.b_pressed),
            float(s.door_open), s.steps_since_first / SEQ_TIMEOUT,
        ], dtype=np.float32)
        return np.concatenate([grid, rel])  # 35-dim total

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        s = self.s
        dx, dy = ACTION_DELTA[action]
        nx = int(np.clip(s.x + dx, 0, GRID_W - 1))
        ny = int(np.clip(s.y + dy, 0, GRID_H - 1))
        if nx == DOOR[0] and ny == DOOR[1] and not s.door_open:
            nx, ny = s.x, s.y  # blocked
        s.x, s.y = nx, ny
        s.steps += 1
        s.a_just = s.b_just = False

        at_a, at_b = (s.x, s.y) == SWITCH_A, (s.x, s.y) == SWITCH_B
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

        first_done = (s.a_pressed if self.rule == "A→B" else s.b_pressed)
        second_done = (s.b_pressed if self.rule == "A→B" else s.a_pressed)
        if first_done and not second_done:
            s.steps_since_first += 1

        reward = -0.02
        if s.b_just: reward = 0.5
        elif s.a_just: reward = 0.1
        # Distance shaping: guide agent toward next subgoal
        if not s.sequence_ok:
            if not s.a_pressed:
                dist = np.sqrt((SWITCH_A[0]-s.x)**2 + (SWITCH_A[1]-s.y)**2)
                max_d = np.sqrt(GRID_W**2 + GRID_H**2)
                reward += 0.03 * (1 - dist / max_d)
            elif not s.b_pressed:
                dist = np.sqrt((SWITCH_B[0]-s.x)**2 + (SWITCH_B[1]-s.y)**2)
                max_d = np.sqrt(GRID_W**2 + GRID_H**2)
                reward += 0.04 * (1 - dist / max_d)
            else:
                dist = np.sqrt((GOAL[0]-s.x)**2 + (GOAL[1]-s.y)**2)
                max_d = np.sqrt(GRID_W**2 + GRID_H**2)
                reward += 0.05 * (1 - dist / max_d)
        if (s.x, s.y) == GOAL and s.door_open:
            reward = 1.0; s.done = True
        if s.steps >= MAX_EPISODE_STEPS:
            s.done = True
        return self._obs(), reward, s.done, {
            "at_goal": (s.x, s.y) == GOAL and s.door_open,
            "door_open": s.door_open, "sequence_ok": s.sequence_ok,
        }

    def render(self) -> str:
        g = [["•" for _ in range(GRID_W)] for _ in range(GRID_H)]
        g[START[1]][START[0]] = "S"
        g[SWITCH_A[1]][SWITCH_A[0]] = "A"
        g[SWITCH_B[1]][SWITCH_B[0]] = "B"
        g[GOAL[1]][GOAL[0]] = "G"
        g[DOOR[1]][DOOR[0]] = " " if self.s.door_open else "▣"
        g[self.s.y][self.s.x] = "@"
        lines = [f"Rule {self.rule}"]
        for row in g: lines.append(" ".join(f"{c:^3}" for c in row))
        lines.append(f"A={int(self.s.a_pressed)} B={int(self.s.b_pressed)} tk={int(self.s.sequence_ok)} t={self.s.steps}")
        return "\n".join(lines)

    def set_rule(self, rule: Rule):
        self.rule = rule
