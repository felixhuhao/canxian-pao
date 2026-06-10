"""
Option-Critic baseline for Two-Gate Lock (1D Rule Swap).
Tests: does temporal abstraction alone (without discrete crystallisation)
produce hysteresis under rule reversal? Prediction: NO.

Reference: Bacon, Harb, Precup (2017). The Option-Critic Architecture. AAAI.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis/exp_two_gate_lock'))
from env import TwoGateLockEnv
import torch
import torch.nn as nn
import torch.optim as optim


class OptionCriticNet(nn.Module):
    """Shared torso + per-option policy + term heads + Q head."""
    def __init__(self, obs_dim: int, act_dim: int, n_options: int = 2, hidden: int = 64):
        super().__init__()
        self.n_options = n_options
        self.torso = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Intra-option policies: [n_options, hidden -> act_dim]
        self.policy_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, act_dim))
            for _ in range(n_options)
        ])
        # Termination functions: [n_options, hidden -> 1]
        self.term_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 1))
            for _ in range(n_options)
        ])
        # Option-Q function: hidden -> n_options
        self.Q_head = nn.Linear(hidden, n_options)

    def forward(self, obs):
        h = self.torso(obs)
        # Policy logits per option: [B, n_opt, act_dim]
        pi_logits = torch.stack([head(h) for head in self.policy_heads], dim=1)
        # Termination logits per option: [B, n_opt]
        beta_logits = torch.stack([head(h).squeeze(-1) for head in self.term_heads], dim=1)
        # Option-Q values: [B, n_opt]
        Q = self.Q_head(h)
        return pi_logits, beta_logits, Q


class OCAgent:
    """Option-Critic agent with proper gradients for intra-option + termination."""

    def __init__(self, obs_dim: int, act_dim: int, n_options: int = 2,
                 lr: float = 3e-4, gamma: float = 0.99, entropy_coef: float = 0.01):
        self.net = OptionCriticNet(obs_dim, act_dim, n_options)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.n_options = n_options
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.option = 0  # current option index

        # Episode buffers
        self.buf_obs = []
        self.buf_actions = []
        self.buf_options = []
        self.buf_rewards = []
        self.buf_dones = []
        self.buf_log_probs = []  # log pi(a|s,o) for intra-option PG
        self.buf_beta_log_probs = []  # log beta termination events

        self.episode_returns = []
        self.episode_entropies = []

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pi_logits, beta_logits, Q = self.net(obs_t)
            # Intra-option policy for current option
            opt_logits = pi_logits[0, self.option]
            dist = torch.distributions.Categorical(logits=opt_logits)
            a = dist.sample()
            log_prob = dist.log_prob(a)

            # Termination: Bernoulli from sigmoid(beta)
            beta = torch.sigmoid(beta_logits[0, self.option])
            term_sample = torch.bernoulli(beta)

        action = int(a.item())

        if training:
            self.buf_obs.append(obs.copy())
            self.buf_actions.append(action)
            self.buf_options.append(self.option)
            self.buf_log_probs.append(float(log_prob.item()))
            self.buf_beta_log_probs.append(float(torch.log(beta + 1e-10).item()))

            # Termination: if terminate, sample new option from Q
            if term_sample > 0.5:
                Q_probs = torch.softmax(Q[0], dim=-1)
                new_opt = int(torch.multinomial(Q_probs, 1).item())
                self.option = new_opt

        return action

    def step_end(self, reward: float, done: bool):
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)
        if done:
            self._finish_episode()

    def _finish_episode(self):
        """Compute Option-Critic gradients and update."""
        T = len(self.buf_rewards)
        if T == 0:
            return

        obs_t = torch.as_tensor(np.array(self.buf_obs), dtype=torch.float32)
        actions_t = torch.as_tensor(self.buf_actions, dtype=torch.long)
        options_t = torch.as_tensor(self.buf_options, dtype=torch.long)
        rewards_t = torch.as_tensor(self.buf_rewards, dtype=torch.float32)
        dones_t = torch.as_tensor(self.buf_dones, dtype=torch.float32)

        pi_logits, beta_logits, Q = self.net(obs_t)

        # Compute TD targets for Q
        with torch.no_grad():
            next_Q = torch.zeros(T, self.n_options)
            if T > 1:
                _, _, Q_next = self.net(obs_t[1:])
                next_Q[:-1] = Q_next.detach()
            # For terminal states, target = reward
            targets = rewards_t.unsqueeze(-1) + self.gamma * (1 - dones_t.unsqueeze(-1)) * next_Q
            # target value for current option: Q(s_t, o_t)
            targets_o = targets.gather(1, options_t.unsqueeze(-1)).squeeze(-1)

        # ---- Losses ----

        # 1. Q-learning loss (option-value)
        Q_o = Q.gather(1, options_t.unsqueeze(-1)).squeeze(-1)
        q_loss = nn.MSELoss()(Q_o, targets_o.detach())

        # 2. Intra-option policy gradient
        # ∇ log π(a_t | s_t, o_t) * (Q(s_t, o_t) - V(s_t))
        with torch.no_grad():
            V = Q.mean(dim=-1)  # baseline V(s) = average Q
            advantages = Q_o - V

        log_probs = torch.zeros(T)
        for t in range(T):
            d = torch.distributions.Categorical(logits=pi_logits[t, options_t[t]])
            log_probs[t] = d.log_prob(actions_t[t])
        pg_loss = -(log_probs * advantages.detach()).mean()

        # 3. Termination gradient
        # -∇β(s_t, o_t) * (Q(s_t, o_t) - V(s_t)) * (1 - β)
        # Simplified: -log(β) * advantage for termination
        beta = torch.sigmoid(beta_logits.gather(1, options_t.unsqueeze(-1)).squeeze(-1))
        term_loss = (beta * advantages.detach()).mean()  # minimize advantage-weighted termination

        # 4. Entropy bonus (intra-option)
        entropies = torch.zeros(T)
        for t in range(T):
            d = torch.distributions.Categorical(logits=pi_logits[t, options_t[t]])
            entropies[t] = d.entropy()
        entropy_bonus = -self.entropy_coef * entropies.mean()

        total_loss = q_loss + pg_loss + term_loss + entropy_bonus

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

        ep_return = sum(self.buf_rewards)
        self.episode_returns.append(ep_return)
        self.episode_entropies.append(float(entropies.mean().detach()))

        # Clear buffers
        self.buf_obs.clear()
        self.buf_actions.clear()
        self.buf_options.clear()
        self.buf_rewards.clear()
        self.buf_dones.clear()
        self.buf_log_probs.clear()
        self.buf_beta_log_probs.clear()

    def get_log(self):
        return {'returns': self.episode_returns, 'entropies': self.episode_entropies,
                'n_options': self.n_options}


# ═══════════════════════════════════════════════════════════════════════════════
# Run 1D Rule Swap
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Option-Critic 1D Rule Swap (N=5 seeds)")
print("=" * 60)
print(f"{'Seed':>5s}  {'P1':>7s}  {'P2':>7s}  {'P3':>7s}  {'Lock?':>6s}  {'Reuse?':>7s}")
print("-" * 50)

for seed in range(5):
    agent = OCAgent(obs_dim=5, act_dim=2, n_options=2, lr=3e-4, entropy_coef=0.02)

    # Phase 1: A→B
    env = TwoGateLockEnv(rule='A→B', seed=seed)
    for ep in range(80):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, training=True)
            obs, r, done, _ = env.step(a)
            agent.step_end(r, done)
    p1 = np.mean(agent.episode_returns[-20:]) if agent.episode_returns else 0.0

    # Phase 2: B→A + 10% ε-greedy
    env.set_rule('B→A')
    rng = np.random.RandomState(seed + 500)
    for ep in range(80):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, training=True)
            if rng.random() < 0.1:
                a = rng.randint(0, 2)
            obs, r, done, _ = env.step(a)
            agent.step_end(r, done)
    p2 = np.mean(agent.episode_returns[-20:]) if agent.episode_returns else 0.0

    # Phase 3: A→B restored
    env.set_rule('A→B')
    for ep in range(60):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, training=True)
            obs, r, done, _ = env.step(a)
            agent.step_end(r, done)
    p3 = np.mean(agent.episode_returns[-20:]) if agent.episode_returns else 0.0

    lock_str = 'LOCK' if p2 < p1 - 0.3 else 'no'
    reuse_str = 'REUSE' if p3 > p2 + 0.3 else 'no'
    print(f"{seed:5d}  {p1:7.3f}  {p2:7.3f}  {p3:7.3f}  {lock_str:>6s}  {reuse_str:>7s}")

print("-" * 50)
print("Prediction: Option-Critic does NOT produce hysteresis.")
print("  Phase 2: options should adapt to B→A smoothly (no lock-in).")
print("  Phase 3: no discrete skill to reactivate (no reuse acceleration).")
