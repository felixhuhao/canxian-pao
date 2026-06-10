import sys, numpy as np
sys.path.insert(0, 'analysis/exp_two_gate_lock')
from env import TwoGateLockEnv
import torch, torch.nn as nn, torch.optim as optim
from agents import PPOData, compute_gae, ActorCritic

class EWCAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, ewc_lambda=50):
        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma; self.ewc_lambda = ewc_lambda
        self.fisher = {}; self.opt_params = {}
        self.buf = PPOData()
        self.ep_returns = []

    def estimate_fisher(self, env, n_samples=100):
        self.fisher = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        for _ in range(n_samples):
            obs = env.reset(); done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = self.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                self.optimiser.zero_grad(); log_prob.backward()
                for n, p in self.policy.named_parameters():
                    if p.grad is not None: self.fisher[n] += p.grad.detach() ** 2
                obs, _, done, _ = env.step(int(a.item()))
        for n in self.fisher: self.fisher[n] /= n_samples

    def ewc_loss(self):
        loss = 0.0
        for n, p in self.policy.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return self.ewc_lambda * loss

    def act(self, obs, training=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, val = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample(); logp = dist.log_prob(a)
        if training:
            self.buf.states.append(obs.copy()); self.buf.actions.append(int(a.item()))
            self.buf.log_probs.append(float(logp.item())); self.buf.vals.append(float(val.item()))
        return int(a.item())

    def step_end(self, reward, done):
        self.buf.rewards.append(reward); self.buf.dones.append(done)

    def finish_episode(self):
        if not self.buf.states: return
        with torch.no_grad():
            last_obs = torch.as_tensor(self.buf.states[-1], dtype=torch.float32).unsqueeze(0)
            _, last_val = self.policy(last_obs); last_val = float(last_val.item()) if not self.buf.dones[-1] else 0.0
        returns_np, adv_np = compute_gae(self.buf.rewards, self.buf.vals, self.gamma, last_val=last_val)
        states_t = torch.as_tensor(np.array(self.buf.states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(self.buf.actions), dtype=torch.long)
        old_lp_t = torch.as_tensor(np.array(self.buf.log_probs), dtype=torch.float32)
        returns_t = torch.as_tensor(returns_np, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        logits, vals = self.policy(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        new_lp = dist.log_prob(actions_t)
        ratio = torch.exp(new_lp - old_lp_t)
        clip = torch.clamp(ratio, 0.8, 1.2)
        pg_loss = -torch.min(ratio * adv_t, clip * adv_t).mean()
        v_loss = nn.MSELoss()(vals, returns_t)
        total_loss = pg_loss + 0.5 * v_loss - 0.01 * dist.entropy().mean()
        if self.fisher: total_loss += self.ewc_loss()
        self.optimiser.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimiser.step()
        self.ep_returns.append(sum(self.buf.rewards))
        self.buf.clear()

    def get_log(self): return {'returns': self.ep_returns}

# Run
print("=== EWC 1D Rule Swap ===")
print("Seed  P1     P2     P3     hyst")
for seed in [0, 1, 2, 3, 4]:
    agent = EWCAgent(obs_dim=5, act_dim=2, ewc_lambda=50)
    env = TwoGateLockEnv(rule='A-B', seed=seed)
    for ep in range(80):
        obs=env.reset();done=False
        while not done: a=agent.act(obs,training=True);obs,r,done,_=env.step(a);agent.step_end(r,done)
        agent.finish_episode()
    p1=np.mean(agent.ep_returns[-20:])
    agent.estimate_fisher(env, n_samples=50)
    agent.opt_params = {n: p.detach().clone() for n, p in agent.policy.named_parameters()}
    
    env.set_rule('B-A'); rng=np.random.RandomState(seed+500)
    for ep in range(80):
        obs=env.reset();done=False
        while not done:
            a=agent.act(obs,training=True)
            if rng.random()<0.1: a=rng.randint(0,2)
            obs,r,done,_=env.step(a);agent.step_end(r,done)
        agent.finish_episode()
    p2=np.mean(agent.ep_returns[-20:])
    
    env.set_rule('A-B')
    for ep in range(60):
        obs=env.reset();done=False
        while not done: a=agent.act(obs,training=True);obs,r,done,_=env.step(a);agent.step_end(r,done)
        agent.finish_episode()
    p3=np.mean(agent.ep_returns[-20:])
    
    lock = "LOCK" if p2 < p1-0.3 else "~"
    reuse = "reuse" if p3 > p2+0.3 else "no_reuse"
    print(f"{seed:5d} {p1:7.3f} {p2:7.3f} {p3:7.3f} {lock}/{reuse}")
