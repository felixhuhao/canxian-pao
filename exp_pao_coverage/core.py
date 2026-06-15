"""
R4 core — multi-task corridor env + base PPO (stage 1).
=======================================================
Tests the prediction from the RDD ladder (coverage-gating's payoff <=> fragile combiner) in the
PAO/RL regime. Stage 1 only builds the env + base PPO and verifies the precondition: a single shared
policy LEARNS one task but FORGETS across conflicting tasks (the interference that makes a skill
library worthwhile). Library + admission gate are added in later stages.

Env: 1D corridor, K conflicting goal tasks. Start at centre; each task has a distinct goal cell, so
optimal actions conflict across tasks in shared states. Task id is signalled (one-hot) in the obs, so
a high-capacity goal-conditioned policy *could* solve all K — interference comes from limited capacity
+ continual (phase-by-phase) presentation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CELLS = 11
START = 5
GOALS = [0, 3, 7, 10]          # K=4 conflicting goals
K = len(GOALS)
MAX_STEPS = 30
STEP_COST = -0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultiTaskCorridor:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.task = 0
        self.reset()

    def set_task(self, k):
        self.task = k

    def reset(self):
        self.pos = START
        self.steps = 0
        return self._obs()

    def _obs(self):
        oh = np.zeros(K, dtype=np.float32); oh[self.task] = 1.0
        return np.concatenate([[self.pos / (N_CELLS - 1)], oh]).astype(np.float32)

    def step(self, a):                      # 0=left, 1=right
        self.pos = int(np.clip(self.pos + (1 if a else -1), 0, N_CELLS - 1))
        self.steps += 1
        g = GOALS[self.task]
        done = False; r = STEP_COST
        if self.pos == g:
            r = 1.0; done = True
        elif self.steps >= MAX_STEPS:
            done = True
        return self._obs(), r, done, {"success": self.pos == g}


OBS_DIM = 1 + K
ACT_DIM = 2


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=16):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h)


def discount_adv(rews, vals, gamma=0.98, lam=0.95):
    adv = np.zeros(len(rews), dtype=np.float32); last = 0.0
    for t in reversed(range(len(rews))):
        nextv = vals[t + 1] if t + 1 < len(vals) else 0.0
        delta = rews[t] + gamma * nextv - vals[t]
        adv[t] = last = delta + gamma * lam * last
    ret = adv + np.array(vals[:len(rews)], dtype=np.float32)
    return ret, adv


class PPO:
    def __init__(self, lr=3e-3, seed=0, extra_bias=None):
        torch.manual_seed(seed)
        self.net = ActorCritic().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buf = []
        self.extra_bias = extra_bias        # callable(obs_np)->logits tensor (skills); None for base

    def act(self, obs, train=True):
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, val = self.net(o)
        if self.extra_bias is not None:
            logits = logits + self.extra_bias(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        if train:
            self.buf.append([obs.copy(), int(a.item()), float(dist.log_prob(a).item()),
                             float(val.item()), 0.0, False])
        return int(a.item())

    def store(self, r, done):
        self.buf[-1][4] = r; self.buf[-1][5] = done

    def finish(self, epochs=4):
        if not self.buf:
            return
        obs = torch.as_tensor(np.array([b[0] for b in self.buf]), dtype=torch.float32, device=DEVICE)
        acts = torch.as_tensor([b[1] for b in self.buf], device=DEVICE)
        oldlp = torch.as_tensor([b[2] for b in self.buf], device=DEVICE)
        rews = [b[4] for b in self.buf]; vals = [b[3] for b in self.buf]
        ret, adv = discount_adv(rews, vals)
        ret = torch.as_tensor(ret, device=DEVICE); adv = torch.as_tensor(adv, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(epochs):
            logits, v = self.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(acts)
            ratio = torch.exp(lp - oldlp)
            clip = torch.clamp(ratio, 0.8, 1.2)
            ploss = -torch.min(ratio * adv, clip * adv).mean()
            vloss = F.mse_loss(v.squeeze(-1), ret)
            loss = ploss + 0.5 * vloss - 0.01 * dist.entropy().mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.buf = []


def run_episode(agent, env, train=True, eps=0.0, rng=None):
    o = env.reset(); done = False; succ = False
    while not done:
        a = agent.act(o, train)
        if eps and rng.random() < eps:
            a = rng.randint(0, ACT_DIM)
        o, r, done, info = env.step(a)
        if train:
            agent.store(r, done)
        succ = info["success"]
    if train:
        agent.finish()
    return succ


def eval_success(agent, seed=999, n=20):
    """Mean success per task with the current (frozen-eval) agent."""
    env = MultiTaskCorridor(seed)
    out = []
    for k in range(K):
        env.set_task(k); s = 0
        for _ in range(n):
            s += run_episode(agent, env, train=False)
        out.append(s / n)
    return np.array(out)


if __name__ == "__main__":
    # Stage-1 calibration: (a) single-task learnable? (b) continual -> forgetting?
    print(f"K={K} goals={GOALS} device={DEVICE}")
    rng = np.random.RandomState(0)
    # (a) single task
    ag = PPO(seed=0); env = MultiTaskCorridor(0); env.set_task(0)
    for _ in range(300):
        run_episode(ag, env, train=True, eps=0.1, rng=rng)
    print("single-task(0) success after 300 eps:", eval_success(ag)[0])
    # (b) continual: train tasks in sequence, watch forgetting of task 0
    ag = PPO(seed=0)
    for k in range(K):
        env.set_task(k)
        for _ in range(300):
            run_episode(ag, env, train=True, eps=0.1, rng=rng)
        print(f"after training up to task {k}: per-task success = {np.round(eval_success(ag),2)}")
    print("=> if later rows show task-0 success dropping, interference/forgetting is present.")
