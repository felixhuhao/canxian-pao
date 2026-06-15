"""
R4 grid env — multi-task 2D grid with 4 genuinely-distinct goal niches.
======================================================================
The 1D corridor of core.py is task-degenerate: a single monotone policy ("always-left")
solves two goals at once (it trips pos==goal on the way past), so there are only 2 real niches,
not 4. This 2D grid fixes that: 4 goals at the cardinal edge-midpoints, each reachable only by a
distinct dominant direction, so no single behaviour covers two niches. A fresh policy learns any
one task; continual training over conflicting tasks interferes — the precondition for a skill library.

obs = [x/(S-1), y/(S-1), onehot(task, K)]; actions 0=up 1=down 2=left 3=right; terminal at goal.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 5
START = (2, 2)
GOALS = [(2, 0), (2, 4), (0, 2), (4, 2)]   # up, down, left, right — 4 distinct niches
K = len(GOALS)
MAX_STEPS = 20
STEP_COST = -0.02
GAMMA = 0.98
SHAPE = 0.1                          # potential-based shaping weight
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBS_DIM = 2 + K
ACT_DIM = 4
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # up, down, left, right


class GridTasks:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.task = 0
        self.reset()

    def set_task(self, k):
        self.task = k

    def reset(self):
        self.x, self.y = START
        self.steps = 0
        return self._obs()

    def _obs(self):
        oh = np.zeros(K, dtype=np.float32); oh[self.task] = 1.0
        return np.concatenate([[self.x / (SIZE - 1), self.y / (SIZE - 1)], oh]).astype(np.float32)

    def step(self, a):
        g = GOALS[self.task]
        d0 = abs(self.x - g[0]) + abs(self.y - g[1])
        dx, dy = MOVES[a]
        self.x = int(np.clip(self.x + dx, 0, SIZE - 1))
        self.y = int(np.clip(self.y + dy, 0, SIZE - 1))
        self.steps += 1
        d1 = abs(self.x - g[0]) + abs(self.y - g[1])
        done = False; r = STEP_COST
        # potential-based shaping (preserves optimal policy): gamma*Phi(s')-Phi(s), Phi=-dist
        r += SHAPE * (GAMMA * (-d1) - (-d0))
        if (self.x, self.y) == g:
            r = 1.0; done = True
        elif self.steps >= MAX_STEPS:
            done = True
        return self._obs(), r, done, {"success": (self.x, self.y) == g}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
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
    def __init__(self, lr=3e-3, seed=0):
        torch.manual_seed(seed)
        self.net = ActorCritic().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buf = []

    def act(self, obs, train=True):
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, val = self.net(o)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        if train:
            self.buf.append([obs.copy(), int(a.item()), float(dist.log_prob(a).item()),
                             float(val.item()), 0.0, False])
        return int(a.item())

    def store(self, r, done):
        self.buf[-1][4] = r; self.buf[-1][5] = done

    def finish(self, epochs=4, ent_coef=0.01):
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
            loss = ploss + 0.5 * vloss - ent_coef * dist.entropy().mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.buf = []


def run_episode(agent, env, train=True, eps=0.0, rng=None, ent_coef=0.01):
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
        agent.finish(ent_coef=ent_coef)
    return succ


def greedy_action(net, obs):
    with torch.no_grad():
        logits, _ = net(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
    return int(torch.argmax(logits))


def policy_success(net, k, seed=999, n=20):
    """Greedy success of a frozen net on task k."""
    env = GridTasks(seed); env.set_task(k); s = 0
    for _ in range(n):
        o = env.reset(); done = False; succ = False
        while not done:
            o, r, done, info = env.step(greedy_action(net, o)); succ = info["success"]
        s += succ
    return s / n


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    print(f"K={K} goals={GOALS} device={DEVICE}")
    # (a) single-task learnability + (b) non-degeneracy: a policy trained on task k should
    #     solve ONLY task k (not subsume another niche).
    print("\nfresh specialist cross-task success matrix (row=trained task, col=eval task):")
    for k in range(K):
        ag = PPO(seed=0); env = GridTasks(0); env.set_task(k)
        for _ in range(300):
            run_episode(ag, env, train=True, eps=0.15, rng=rng)
        row = [policy_success(ag.net, j) for j in range(K)]
        print(f"  train {k} (goal {GOALS[k]}): {np.round(row,2)}")
    print("=> diagonal should be ~1.0, off-diagonal ~0 (each niche distinct).")
