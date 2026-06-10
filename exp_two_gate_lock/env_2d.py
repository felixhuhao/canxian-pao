"""
Two-Gate Lock — 2D Grid (with Rule Swap)
==========================================
7×5 grid:
  0 1 2 3 4 5 6
0 . . . . . . .
1 . . . . . . .
2 S . A . . D .
3 . . . . . . .
4 . . . . B . G

S(0,2) Start   A(2,2) Switch A   B(4,4) Switch B
D(5,2) Door    G(6,4) Goal

Rule:
  A→B (default): press A first, then B → door opens
  B→A:           press B first, then A → door opens

The non-trivial navigation (agent must navigate around the grid,
not just walk a straight line) makes PPO's gradient flow no longer
naturally align with the optimal solution.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal

GRID_W, GRID_H = 7, 5
START = (0, 2)
SWITCH_A = (2, 2)
SWITCH_B = (4, 4)
DOOR = (5, 2)
GOAL = (6, 4)

SEQ_TIMEOUT = 12  # steps between first and second switch
MAX_EPISODE_STEPS = 60
NUM_ACTIONS = 4  # N,S,E,W

ACTION_DELTA = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTION_NAMES = ["N", "S", "E", "W"]

Rule = Literal["A→B", "B→A"]


@dataclass
class EnvState:
    x: int
    y: int
    a_pressed: bool = False
    b_pressed: bool = False
    sequence_ok: bool = False
    door_open: bool = False
    steps: int = 0
    steps_since_first: int = 0
    done: bool = False
    a_just_pressed: bool = False
    b_just_pressed: bool = False


class TwoGate2DEnv:
    """2D grid environment with configurable rule."""

    def __init__(self, rule: Rule = "A→B", seed: int = 0):
        self.rule = rule
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.s = EnvState(x=START[0], y=START[1])
        return self._obs()

    def _obs(self) -> np.ndarray:
        s = self.s
        # 10-D: x, y, rel_A_x, rel_A_y, rel_B_x, rel_B_y, a, b, door, steps
        # Relative features provide translation invariance for skill matching.
        return np.array([
            s.x / (GRID_W - 1),
            s.y / (GRID_H - 1),
            (SWITCH_A[0] - s.x) / GRID_W,   # rel_to_A_x
            (SWITCH_A[1] - s.y) / GRID_H,   # rel_to_A_y
            (SWITCH_B[0] - s.x) / GRID_W,   # rel_to_B_x
            (SWITCH_B[1] - s.y) / GRID_H,   # rel_to_B_y
            float(s.a_pressed),
            float(s.b_pressed),
            float(s.door_open),
            s.steps_since_first / SEQ_TIMEOUT,
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < NUM_ACTIONS
        s = self.s
        dx, dy = ACTION_DELTA[action]
        nx = int(np.clip(s.x + dx, 0, GRID_W - 1))
        ny = int(np.clip(s.y + dy, 0, GRID_H - 1))

        # Door blocks passage to (5,2) unless open
        if nx == DOOR[0] and ny == DOOR[1] and not s.door_open:
            nx, ny = s.x, s.y

        s.x, s.y = nx, ny
        s.steps += 1
        s.a_just_pressed = False
        s.b_just_pressed = False

        # --- Switch activation with rule-awareness ---
        at_a = (s.x, s.y) == SWITCH_A
        at_b = (s.x, s.y) == SWITCH_B

        if self.rule == "A→B":
            if at_a and not s.a_pressed:
                s.a_pressed = True
                s.steps_since_first = 0
                s.a_just_pressed = True
            if at_b and s.a_pressed and not s.b_pressed and s.steps_since_first <= SEQ_TIMEOUT:
                s.b_pressed = True
                s.sequence_ok = True
                s.door_open = True
                s.b_just_pressed = True

        elif self.rule == "B→A":
            if at_b and not s.b_pressed:
                s.b_pressed = True
                s.steps_since_first = 0
                s.b_just_pressed = True
            if at_a and s.b_pressed and not s.a_pressed and s.steps_since_first <= SEQ_TIMEOUT:
                s.a_pressed = True
                s.sequence_ok = True
                s.door_open = True
                s.a_just_pressed = True

        # Track time since first switch press (only if first pressed, second not yet)
        first_pressed = (s.a_pressed if self.rule == "A→B" else s.b_pressed)
        second_pressed = (s.b_pressed if self.rule == "A→B" else s.a_pressed)
        if first_pressed and not second_pressed:
            s.steps_since_first += 1

        # ── Reward shaping ──
        reward = -0.02  # step penalty
        if s.b_just_pressed and self.rule == "A→B":
            reward = 0.5
        elif s.a_just_pressed and self.rule == "A→B":
            reward = 0.1
        elif s.a_just_pressed and self.rule == "B→A":
            reward = 0.5
        elif s.b_just_pressed and self.rule == "B→A":
            reward = 0.1

        if (s.x, s.y) == GOAL and s.door_open:
            reward = 1.0
            s.done = True

        if s.steps >= MAX_EPISODE_STEPS:
            s.done = True

        return self._obs(), reward, s.done, {
            "at_goal": (s.x, s.y) == GOAL and s.door_open,
            "door_open": s.door_open,
            "a_pressed": s.a_pressed,
            "b_pressed": s.b_pressed,
            "sequence_ok": s.sequence_ok,
            "a_just_pressed": s.a_just_pressed,
            "b_just_pressed": s.b_just_pressed,
        }

    def render(self) -> str:
        grid = [["•" for _ in range(GRID_W)] for _ in range(GRID_H)]
        grid[START[1]][START[0]] = "S"
        grid[SWITCH_A[1]][SWITCH_A[0]] = "A"
        grid[SWITCH_B[1]][SWITCH_B[0]] = "B"
        grid[GOAL[1]][GOAL[0]] = "G"
        if self.s.door_open:
            grid[DOOR[1]][DOOR[0]] = " "
        else:
            grid[DOOR[1]][DOOR[0]] = "▣"
        grid[self.s.y][self.s.x] = "@"
        lines = [f"Rule: {self.rule}"]
        for row in grid:
            lines.append(" ".join(f"{c:^3}" for c in row))
        lines.append(f"  A={int(self.s.a_pressed)} B={int(self.s.b_pressed)} "
                     f"Seq={int(self.s.sequence_ok)} Door={int(self.s.door_open)} t={self.s.steps}")
        return "\n".join(lines)

    def set_rule(self, rule: Rule):
        """Swap the rule (used for hysteresis Phase 2/3)."""
        self.rule = rule


if __name__ == "__main__":
    # Test both rules
    for rule in ["A→B", "B→A"]:
        print(f"\n{'='*40}\nRule: {rule}\n{'='*40}")
        env = TwoGate2DEnv(rule=rule)
        print(env.render())
        # Manual test: navigate the correct path
        # S→A→B→D→G for A→B, S→B→A→D→G for B→A
        if rule == "A→B":
            seq = [2, 2, 1, 1, 2, 1, 2, 2, 2, 0, 2, 2, 1]  # E,E,S,S,E,S,E,E,E,N,E,E,S
        else:
            seq = [2, 2, 1, 1, 2, 1, 2, 2, 0, 2, 2, 0, 2, 2, 1]  # E,E,S,S,E,S,E,E,N,E,E,N,E,E,S
        for a in seq:
            obs, rew, done, info = env.step(a)
            action_name = ACTION_NAMES[a]
            print(f"  {action_name} r={rew:+.3f} | {env.render().split(chr(10))[-1]}")
            if done:
                break
