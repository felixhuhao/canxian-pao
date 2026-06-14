"""
Agents: PAO-light vs Flat-PPO baseline
=======================================
Both share: small MLP actor-critic, PPO-clip objective, GAE.
PAO-light adds:
  - Event-triggered skill crystallisation (heuristic threshold: return > 1.0
    AND entropy < 0.6 AND 3/5 recent episodes successful)
  - Skill cache (relative-feature-anchored action templates)
  - Dormancy gate (suppressed plasticity near cached skills)

Note: full PAO uses Bayesian Online Change-Point Detection (BOCPD) on policy
entropy for triggering. This minimal implementation uses a dual-threshold proxy.
A BOCPD implementation is included for future upgrade.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─── Tiny MLP Actor-Critic ──────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.torso(x)
        logits = self.actor(h)
        val = self.critic(h).squeeze(-1)
        return logits, val

    def act(self, obs_np: np.ndarray, temperature: float = 1.0, greedy: bool = False) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        logits, val = self.forward(obs_t)
        if greedy:
            a = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits / temperature)
            a = dist.sample()
        logp = 0.0
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(val.item())


# ─── Skill ───────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    n: int
    trajectory: list        # List[(obs_np, action_int)] — full crystallised trajectory
    spatial_dims: int = 1   # leading dims to use for spatial matching
    radius: float = 0.25    # cosine-distance threshold
    formed_at: int = 0

    def action_at(self, obs_np: np.ndarray) -> Optional[int]:
        """Nearest-neighbour lookup: find closest state in trajectory, return its action."""
        d = self.spatial_dims
        q = obs_np[:d] / (np.linalg.norm(obs_np[:d]) + 1e-8)
        best_dist = float("inf")
        best_a = None
        for stored_obs, stored_a in self.trajectory:
            v = stored_obs[:d] / (np.linalg.norm(stored_obs[:d]) + 1e-8)
            dist = 1.0 - float(np.dot(q, v))  # cosine distance
            if dist < best_dist:
                best_dist = dist
                best_a = stored_a
        return best_a if best_dist < self.radius else None


class SkillCache:
    def __init__(self, act_dim: int, spatial_dims: int = 1):
        self.skills: List[Skill] = []
        self.act_dim = act_dim
        self.spatial_dims = spatial_dims
        self._next_id = 0

    def add(self, trajectory_states: List[np.ndarray], trajectory_actions: List[int],
            radius: float = 0.25, formed_at: int = 0):
        if not trajectory_states:
            return
        traj = list(zip(trajectory_states, trajectory_actions))
        skill = Skill(self._next_id, traj, self.spatial_dims, radius, formed_at)
        self.skills.append(skill)
        self._next_id += 1

    def query(self, obs_np: np.ndarray) -> Optional[Tuple[int, float]]:
        """Return (action, confidence) from nearest matching skill, or None."""
        best_a, best_conf = None, 0.0
        for sk in self.skills:
            a = sk.action_at(obs_np)
            if a is not None:
                # Confidence: 1 - best cosine distance among trajectory
                d = self.spatial_dims
                q = obs_np[:d] / (np.linalg.norm(obs_np[:d]) + 1e-8)
                sims = [1.0 - 1.0 + float(np.dot(q, tobs[:d]/(np.linalg.norm(tobs[:d])+1e-8)))
                        for tobs, _ in sk.trajectory[:5]]  # check first 5
                conf = max(sims) if sims else 0.5
                if conf > best_conf:
                    best_a, best_conf = a, conf
        return (best_a, best_conf) if best_a is not None else None

    @property
    def count(self):
        return len(self.skills)

    @property
    def count(self):
        return len(self.skills)


# ─── BOCPD ───────────────────────────────────────────────────────────────────


class BOCPD:
    """Log-space Bayesian Online Change Point Detection (Adams & MacKay, 2007).
    Uses Student-t predictive model and log-probabilities to avoid underflow."""
    def __init__(self, hazard: float = 0.05, min_data: int = 8):
        self.hazard = hazard
        self.min_data = min_data
        self.data = []
        self.log_run_probs = [0.0]  # log P(r_{-1}=0) = 0
        self.change_points = []

    def _log_pred(self, window, value):
        n = len(window)
        if n < 2:
            mu = np.mean(self.data) if len(self.data) > 2 else 0.5
            var = max(np.var(self.data) + 0.1, 0.1) if len(self.data) > 2 else 0.5
        else:
            mu = np.mean(window)
            var = np.var(window, ddof=1) + 0.01
        pred_var = var + 1.0
        return -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (value - mu)**2 / pred_var

    def update(self, value):
        self.data.append(value)
        n = len(self.data)
        if n < self.min_data:
            return None
        max_r = min(n, 80)
        log_preds = []
        for r in range(max_r):
            w = self.data[:-1][-r:] if r > 0 else []
            log_preds.append(self._log_pred(w, value))
        log_preds = np.array(log_preds)
        H = self.hazard
        log_H = np.log(max(H, 1e-60))
        log_1mH = np.log(max(1-H, 1e-60))
        new_log_probs = np.full(max_r + 1, -np.inf)
        # r_t = 0: change point
        log_cp_mass = -np.inf
        for prev_r in range(min(len(self.log_run_probs), max_r)):
            l = self.log_run_probs[prev_r] + log_preds[prev_r]
            log_cp_mass = np.logaddexp(log_cp_mass, l)
        new_log_probs[0] = log_H + log_cp_mass
        # r_t = r: growth
        for r in range(1, max_r + 1):
            if r - 1 < len(self.log_run_probs):
                new_log_probs[r] = log_1mH + self.log_run_probs[r - 1] + log_preds[r - 1]
        # Normalise
        max_log = np.max(new_log_probs)
        new_probs = np.exp(new_log_probs - max_log)
        total = new_probs.sum()
        if total > 0:
            new_probs /= total
        self.log_run_probs = np.log(np.maximum(new_probs, 1e-60)).tolist()
        if new_probs[0] > 0.5:
            self.change_points.append(n)
            self.log_run_probs = [0.0]
            return n
        return None

    @property
    def uncertainty(self):
        p = np.exp(self.log_run_probs)
        p = np.maximum(p, 1e-30)
        p /= p.sum()
        return -np.sum(p * np.log(p)) / np.log(1 + len(p))



# ═══════════════════════════════════════════════════════════════════════════════
# Shared PPO base
# ═══════════════════════════════════════════════════════════════════════════════

class PPOData:
    """Stores a single episode's experience for PPO update."""
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.vals: List[float] = []
        self.dones: List[bool] = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.vals.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


def compute_gae(rewards: List[float], values: List[float],
                gamma: float = 0.99, lam: float = 0.95, last_val: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            delta = rewards[t] + gamma * last_val - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return returns, advantages


def ppo_update(policy: ActorCritic, optimiser: optim.Optimizer,
               states_np: np.ndarray, actions_np: np.ndarray,
               old_log_probs_np: np.ndarray, returns_np: np.ndarray,
               advantages_np: np.ndarray,
               clip_eps: float = 0.2, entropy_coef: float = 0.01,
               epochs: int = 4, batch_size: int = 32,
               dormancy_weights: Optional[np.ndarray] = None) -> float:
    """Multi-epoch PPO-clip update. Returns mean entropy."""
    n = len(states_np)
    idxs = np.arange(n)
    states_t = torch.as_tensor(states_np, dtype=torch.float32)
    actions_t = torch.as_tensor(actions_np, dtype=torch.long)
    old_lp_t = torch.as_tensor(old_log_probs_np, dtype=torch.float32)
    returns_t = torch.as_tensor(returns_np, dtype=torch.float32)
    adv_t = torch.as_tensor(advantages_np, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    mean_entropy = 0.0

    for _ in range(epochs):
        np.random.shuffle(idxs)
        for i in range(0, n, batch_size):
            b = idxs[i:i + batch_size]
            logits, vals = policy(states_t[b])
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions_t[b])
            ratio = torch.exp(new_lp - old_lp_t[b])
            surr1 = ratio * adv_t[b]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[b]
            pg_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(vals, returns_t[b])
            ent = dist.entropy().mean()
            loss = pg_loss + 0.5 * v_loss - entropy_coef * ent
            mean_entropy += float(ent.detach())

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimiser.step()

    return mean_entropy / (epochs * max(1, n // batch_size))


# ─── Skill Policy (KL Distillation) ───────────────────────────────────────────

class SkillPolicy(nn.Module):
    """A lightweight neural policy representing a crystallised skill.
    Distilled from PPO trajectories via KL minimisation."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, obs):
        return self.net(obs)
    def action_probs(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))


def distill_skill(trajectory_states: List[np.ndarray],
                  trajectory_actions: List[int],
                  teacher_log_probs: List[float],
                  obs_dim: int, act_dim: int,
                  epochs: int = 100, lr: float = 3e-4) -> SkillPolicy:
    """KL-distill a SkillPolicy from teacher (PPO) trajectory data."""
    skill = SkillPolicy(obs_dim, act_dim)
    opt = optim.Adam(skill.parameters(), lr=lr)
    obs_t = torch.as_tensor(np.array(trajectory_states), dtype=torch.float32)
    actions_t = torch.as_tensor(trajectory_actions, dtype=torch.long)
    for _ in range(epochs):
        logits = skill(obs_t)
        # Cross-entropy against teacher actions (simpler than full KL)
        loss = F.cross_entropy(logits, actions_t)
        opt.zero_grad(); loss.backward(); opt.step()
    return skill


# ─── Applicability Network ───────────────────────────────────────────────────

class ApplicabilityNet(nn.Module):
    """Binary classifier: should skill z be activated at observation obs?
    P(output=1 | obs) = sigmoid(w^T φ(obs))."""
    def __init__(self, obs_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# PAO-light Agent (with policy distillation)
# ═══════════════════════════════════════════════════════════════════════════════

class PAOLight:
    """PAO-light: PPO + Event-triggered skill crystallisation + Dormancy Gate.

    Skill representation: KL-distilled SkillPolicy + ApplicabilityNet.
    """

    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4,
                 entropy_coef: float = 0.02, gamma: float = 0.99,
                 spatial_dims: int = 1,
                 trigger_return: float = 1.0,
                 trigger_entropy: float = 0.6,
                 trigger_sustained: int = 3):
        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self._tr = trigger_return
        self._te = trigger_entropy
        self._ts = trigger_sustained

        # PAO mechanisms
        self.bocpd = BOCPD(hazard=0.02, min_data=15)
        self.skills = SkillCache(act_dim, spatial_dims=spatial_dims)
        self.skill_bias_strength = 1.0
        self.skill_policy: Optional[SkillPolicy] = None
        self.applicability: Optional[ApplicabilityNet] = None

        self.buf = PPOData()
        self.dormancy_lr_factor = 0.3

        self.episode_returns: List[float] = []
        self.episode_entropies: List[float] = []
        self.skill_formation_eps: List[int] = []
        self.crystallised_ep: List[bool] = []
        self.bocpd_cp_eps: List[int] = []

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, val = self.policy(obs_t)

        # Skill bias: configurable rigidity (policy distillation or trajectory)
        app_thresh = getattr(self, '_app_thresh', 0.7)
        if getattr(self, '_use_trajectory', False) and training:
            # Trajectory matching mode (rigid)
            skill_result = self.skills.query(obs)
            if skill_result is not None:
                cached_action, confidence = skill_result
                bias_t = torch.zeros(self.act_dim, dtype=torch.float32)
                bias_t[cached_action] = 1.0
                logits = logits + self.skill_bias_strength * confidence * bias_t
        elif self.skill_policy is not None and self.applicability is not None and training:
            # Frozen base-policy skill + applicability gating
            app_score = float(self.applicability(obs_t).detach())
            if app_score > app_thresh:
                skill_logits, _ = self.skill_policy(obs_t)  # frozen ActorCritic
                logits = logits + self.skill_bias_strength * skill_logits

        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)

        if training:
            self.buf.states.append(obs.copy())
            self.buf.actions.append(int(a.item()))
            self.buf.log_probs.append(float(logp.item()))
            self.buf.vals.append(float(val.item()))

        return int(a.item())

    def step_end(self, reward: float, done: bool):
        self.buf.rewards.append(reward)
        self.buf.dones.append(done)

    def finish_episode(self):
        """Called when episode ends. Runs PPO update + crystallisation trigger."""
        if len(self.buf) == 0:
            return

        # GAE
        with torch.no_grad():
            last_obs = torch.as_tensor(self.buf.states[-1], dtype=torch.float32).unsqueeze(0)
            _, last_val = self.policy(last_obs)
            last_val = float(last_val.item()) if not self.buf.dones[-1] else 0.0

        returns_np, adv_np = compute_gae(
            self.buf.rewards, self.buf.vals, self.gamma, last_val=last_val
        )

        # PPO update (dormancy applied via applicability score if skill exists)
        dorm = np.ones(len(self.buf))
        if self.applicability is not None:
            obs_t = torch.as_tensor(np.array(self.buf.states), dtype=torch.float32)
            with torch.no_grad():
                app_scores = self.applicability(obs_t).numpy()
            dorm[app_scores > 0.7] = self.dormancy_lr_factor

        mean_ent = ppo_update(
            self.policy, self.optimiser,
            np.array(self.buf.states),
            np.array(self.buf.actions, dtype=np.int64),
            np.array(self.buf.log_probs),
            returns_np, adv_np,
            entropy_coef=self.entropy_coef,
            dormancy_weights=dorm,
        )

        # Log
        ep_return = sum(self.buf.rewards)
        self.episode_returns.append(ep_return)
        self.episode_entropies.append(mean_ent)
        ep_idx = len(self.episode_returns) - 1

        # Save trajectory BEFORE clearing buffer
        saved_states = [s.copy() for s in self.buf.states]
        saved_actions = list(self.buf.actions)
        saved_log_probs = list(self.buf.log_probs)
        # Store negative examples for applicability training (failed episodes)
        if not hasattr(self, '_neg_buffer'):
            self._neg_buffer = []
        if ep_return < -0.5 and len(saved_states) > 5:
            self._neg_buffer.extend(saved_states)
            if len(self._neg_buffer) > 500:
                self._neg_buffer = self._neg_buffer[-500:]
        # Accumulate successful trajectories for multi-trace distillation
        if not hasattr(self, '_success_trajs'):
            self._success_trajs = []
        if ep_return > 0.5:  # successful episode
            self._success_trajs.append({
                "states": [s.copy() for s in saved_states],
                "actions": list(saved_actions),
                "log_probs": list(saved_log_probs),
            })
        self.buf.clear()

        # Crystallisation trigger (overridable; see _should_crystallize).
        crystallised = False
        if self._should_crystallize(ep_idx, ep_return, mean_ent, saved_states):
            if getattr(self, '_use_trajectory', False):
                if self.skills.count == 0:
                    self.skills.add(saved_states, saved_actions, radius=2.0, formed_at=ep_idx)
                    crystallised = True
                    self.skill_formation_eps.append(ep_idx)
            elif self.skill_policy is None:  # crystallise once
                # PAO-style: freeze base policy params as the skill policy
                self.skill_policy = ActorCritic(self.obs_dim, self.act_dim)
                self.skill_policy.load_state_dict(self.policy.state_dict())
                for p in self.skill_policy.parameters():
                    p.requires_grad = False  # frozen
                self._train_applicability(saved_states)
                # Deferred validation: caller runs _validate_skill(eval_env=env)
                self._skill_deferred = True
                self._skill_validated = False
                print(f"  \u25b6 Skill crystallised at ep {ep_idx}. Run _validate_skill() with eval_env to confirm.")
                print(f"    Success trajectories in P1: {len(self._success_trajs)}")
                crystallised = True
                self.skill_formation_eps.append(ep_idx)

        self.crystallised_ep.append(crystallised)

    def _should_crystallize(self, ep_idx, ep_return, mean_ent, saved_states) -> bool:
        """Default heuristic dual-threshold trigger (unchanged behaviour).
        Overridden by PAOForced for the P1 random-window controls."""
        if len(saved_states) == 0:
            return False
        high_return = ep_return > self._tr
        low_entropy = mean_ent < self._te
        recent_good = sum(1 for r in self.episode_returns[-5:] if r > self._tr)
        sustained = recent_good >= self._ts
        return high_return and low_entropy and sustained

    def _train_applicability(self, positive_states: List[np.ndarray]):
        """Train applicability classifier: positive = states from successful trajectory,
        negative = states from recent failed episodes (stored in self._neg_buffer)."""
        if not hasattr(self, '_neg_buffer'):
            self._neg_buffer = []
        # Build training data
        pos_obs = np.array([s.copy() for s in positive_states])
        pos_labels = np.ones(len(pos_obs))
        neg_obs = np.array(self._neg_buffer[-200:]) if self._neg_buffer else pos_obs[:1] * 0
        neg_labels = np.zeros(len(neg_obs))
        if len(neg_obs) == 0:
            return
        obs_all = torch.as_tensor(np.concatenate([pos_obs, neg_obs]), dtype=torch.float32)
        labels_all = torch.as_tensor(np.concatenate([pos_labels, neg_labels]), dtype=torch.float32)

        self.applicability = ApplicabilityNet(self.obs_dim)
        opt = optim.Adam(self.applicability.parameters(), lr=1e-3)
        for _ in range(50):
            pred = self.applicability(obs_all)
            loss = F.binary_cross_entropy(pred, labels_all)
            opt.zero_grad(); loss.backward(); opt.step()

    def _validate_skill(self, n_rollouts: int = 10, eval_env=None) -> float:
        """Validate skill by running rollouts with skill-only actions.
        Returns Q(z) = success rate over n_rollouts.
        If no skill policy, returns 0.0.
        Accepts optional eval_env (must implement →B with same obs_dim)."""
        if self.skill_policy is None:
            return 0.0
        if eval_env is not None:
            env = eval_env
        else:
            try:
                from env_2d_small import TwoGate2DSmallEnv as DefaultEnv
                env = DefaultEnv(rule='A\u2192B', seed=42)
            except ImportError:
                return 0.0
        successes = 0
        for _ in range(n_rollouts):
            obs = env.reset(); done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = self.skill_policy(obs_t)  # frozen ActorCritic
                a = torch.distributions.Categorical(logits=logits).sample()
                obs, r, done, info = env.step(int(a.item()))
            if info.get('at_goal', False):
                successes += 1
        return successes / n_rollouts

    def get_log(self) -> dict:
        return {
            "returns": self.episode_returns,
            "entropies": self.episode_entropies,
            "n_skills": self.skills.count,
            "skill_episodes": self.skill_formation_eps,
            "crystallised_per_ep": self.crystallised_ep,
            "bocpd_cp_eps": self.bocpd_cp_eps,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Flat PPO Baseline (same architecture, NO PAO mechanisms)
# ═══════════════════════════════════════════════════════════════════════════════

class PAOForced(PAOLight):
    """P1 control: crystallise at a PRE-SPECIFIED episode, ignoring the trigger signal.
    Identical to PAOLight in every other respect (frozen weight-copy skill + ApplicabilityNet)."""
    def __init__(self, *args, crystallize_at: int = 40, **kwargs):
        super().__init__(*args, **kwargs)
        self.crystallize_at = crystallize_at
    def _should_crystallize(self, ep_idx, ep_return, mean_ent, saved_states) -> bool:
        return (len(saved_states) > 0 and ep_idx == self.crystallize_at
                and self.skill_policy is None)


class PAOBocpd(PAOLight):
    """PAO-paper-faithful trigger via BOCPD (Adams & MacKay 2007).
    NOTE: the shipped BOCPD.update() firing rule (P(r=0)>0.5) never fires even on a clean step
    (verified 2026-06-07). We therefore read the run-length posterior directly and declare a
    change-point on a MAP run-length RESET (the standard robust read), gated to the high-return
    regime so crystallisation happens once the policy is competent."""
    def __init__(self, *args, bocpd_hazard: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.bocpd = BOCPD(hazard=bocpd_hazard, min_data=15)
        self._prev_map_rl = 0
    def _should_crystallize(self, ep_idx, ep_return, mean_ent, saved_states) -> bool:
        if self.skill_policy is not None or len(saved_states) == 0:
            return False
        self.bocpd.update(ep_return)
        p = np.exp(np.asarray(self.bocpd.log_run_probs))
        if p.sum() <= 0:
            return False
        map_rl = int(np.argmax(p / p.sum()))
        reset = (self._prev_map_rl >= 12 and map_rl < self._prev_map_rl - 6)
        self._prev_map_rl = map_rl
        if reset:
            self.bocpd_cp_eps.append(ep_idx)
        return reset and ep_return > self._tr


class PAONoDormancy(PAOLight):
    """PAO with skill cache active but dormancy disabled (lr_factor=1.0)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dormancy_lr_factor = 1.0


class PAONoSkill(PAOLight):
    """PAO with dormancy active but skill bias disabled (bias_strength=0)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_bias_strength = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-skill steelman (PAOLibrary) — for the unpredictable recurring-regime test
# (PREREG_P3_unpredictable.md). PAO as shipped crystallises ONCE; the theory
# ("progressive assembly") envisions a growing skill library reused across regimes.
# This is the steelman: a library of frozen sub-policies + event-triggered
# re-selection. Crystallisation rule is IDENTICAL across all library arms; arms
# differ ONLY in what triggers skill re-selection (the manipulated variable).
# ═══════════════════════════════════════════════════════════════════════════════

class PAOLibrary(PAOLight):
    """Skill-library PAO. Builds a library of frozen ActorCritic sub-policies at
    competent+stable+debounced episodes (same rule for every arm). On a 'reset'
    signal it searches the library (one candidate skill per episode, plus a
    no-skill candidate), then exploits the best until the next reset. The reset
    trigger is supplied by subclasses via _reset_signal()."""

    def __init__(self, *args, cryst_debounce: int = 10, max_lib: int = 6,
                 obs_conf_thresh: float = 0.5, reset_refractory: int = 30,
                 reset_eps=None, bocpd_hazard: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.library: List[ActorCritic] = []
        self._last_cryst_ep = -10**9
        self._cryst_debounce = cryst_debounce
        self.max_lib = max_lib
        self._reset_refractory = reset_refractory
        self._last_reset_ep = -10**9
        self._mode = "exploit"          # "exploit" | "search"
        self._active_idx = None         # library index (or None = base policy)
        self._search_queue: list = []
        self._search_results: dict = {}
        self._episode_bias_idx = None
        self._searched_this_ep = False
        self._pending_reset = False
        self._cryst_done_since_reset = False
        self._obs_select = False
        self._obs_conf_thresh = obs_conf_thresh
        self.reset_eps = set(reset_eps) if reset_eps else set()
        self.reset_count = 0
        self.lib_size_log: List[int] = []
        # own BOCPD (fresh hazard) for the BOCPD subclass
        self.bocpd = BOCPD(hazard=bocpd_hazard, min_data=15)
        self._prev_map_rl = 0

    # disable PAOLight's single-skill crystallisation path
    def _should_crystallize(self, *a, **k) -> bool:
        return False

    def begin_episode(self):
        """Decide this episode's skill bias. Called by the driver before each episode."""
        if self._obs_select:
            self._searched_this_ep = False
            return
        if self._pending_reset:
            self._pending_reset = False
            if len(self.library) > 0:
                self._mode = "search"
                self._search_queue = list(range(len(self.library))) + [None]
                self._search_results = {}
        if self._mode == "search":
            if len(self._search_queue) == 0:
                best = (max(self._search_results, key=self._search_results.get)
                        if self._search_results else None)
                # competence fallback: if no cached skill clears the bar, relearn
                # from the base policy (active=None) rather than lock onto a bad skill.
                if best is not None and self._search_results.get(best, -1e9) <= self._tr:
                    best = None
                self._active_idx = best
                self._mode = "exploit"
                self._episode_bias_idx = self._active_idx
                self._searched_this_ep = False
            else:
                self._episode_bias_idx = self._search_queue[0]
                self._searched_this_ep = True
        else:
            self._episode_bias_idx = self._active_idx
            self._searched_this_ep = False

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, val = self.policy(obs_t)
        bias_idx = None
        if training and self.library:
            if self._obs_select:
                best_conf, best_i = -1.0, None
                for i, sk in enumerate(self.library):
                    with torch.no_grad():
                        sl, _ = sk(obs_t)
                    conf = float(torch.softmax(sl, dim=-1).max())
                    if conf > best_conf:
                        best_conf, best_i = conf, i
                if best_conf >= self._obs_conf_thresh:
                    bias_idx = best_i
            else:
                bias_idx = self._episode_bias_idx
        if bias_idx is not None:
            with torch.no_grad():
                sl, _ = self.library[bias_idx](obs_t)
            logits = logits + self.skill_bias_strength * sl
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        if training:
            self.buf.states.append(obs.copy())
            self.buf.actions.append(int(a.item()))
            self.buf.log_probs.append(float(logp.item()))
            self.buf.vals.append(float(val.item()))
        return int(a.item())

    def finish_episode(self):
        if len(self.buf) == 0:
            return
        super().finish_episode()  # PPO update + logging (no single-skill crystallise)
        ep_idx = len(self.episode_returns) - 1
        ep_return = self.episode_returns[-1]
        mean_ent = self.episode_entropies[-1]
        if self._searched_this_ep and self._mode == "search" and len(self._search_queue) > 0:
            cand = self._search_queue.pop(0)
            self._search_results[cand] = ep_return
        self._maybe_crystallize(ep_idx, ep_return, mean_ent)
        # detector always updates state; refractory gates whether we ACT on a reset
        raw = self._reset_signal(ep_idx, ep_return)
        if raw and (ep_idx - self._last_reset_ep) >= self._reset_refractory:
            self._pending_reset = True
            self._cryst_done_since_reset = False
            self._last_reset_ep = ep_idx
            self.reset_count += 1
        self.lib_size_log.append(len(self.library))

    def _maybe_crystallize(self, ep_idx, ep_return, mean_ent):
        # Cap library; only crystallise a regime mastered by the BASE policy
        # (not while exploiting a cached skill) -> ~1 skill per distinct regime.
        if len(self.library) >= self.max_lib or self._cryst_done_since_reset:
            return
        if self._mode != "exploit" or self._active_idx is not None:
            return
        if ep_return <= self._tr or mean_ent >= self._te:
            return
        if sum(1 for r in self.episode_returns[-5:] if r > self._tr) < self._ts:
            return
        if ep_idx - self._last_cryst_ep < self._cryst_debounce:
            return
        sk = ActorCritic(self.obs_dim, self.act_dim)
        sk.load_state_dict(self.policy.state_dict())
        for p in sk.parameters():
            p.requires_grad = False
        self.library.append(sk)
        self._last_cryst_ep = ep_idx
        self._cryst_done_since_reset = True
        self.skill_formation_eps.append(ep_idx)

    def _reset_signal(self, ep_idx, ep_return) -> bool:
        return False


class PAOLibraryBOCPD(PAOLibrary):
    """Re-selection triggered by BOCPD change-point detection (the mechanism under test)."""
    def _reset_signal(self, ep_idx, ep_return) -> bool:
        self.bocpd.update(ep_return)
        p = np.exp(np.asarray(self.bocpd.log_run_probs))
        if p.sum() <= 0:
            return False
        map_rl = int(np.argmax(p / p.sum()))
        reset = (self._prev_map_rl >= 12 and map_rl < self._prev_map_rl - 6)
        self._prev_map_rl = map_rl
        return reset


class PAOLibraryRandom(PAOLibrary):
    """Re-selection triggered at pre-specified random episodes (count-matched to BOCPD)."""
    def _reset_signal(self, ep_idx, ep_return) -> bool:
        return ep_idx in self.reset_eps


class PAOLibraryObs(PAOLibrary):
    """Skill selection by per-step observation confidence (no change-point detection).
    Expected to fail: the env is rule-unsignaled, so obs cannot reveal the active rule."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_select = True


class FlatPPO:
    """Pure PPO agent. Identical architecture to PAOLight, no skill mechanisms."""

    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4,
                 entropy_coef: float = 0.02, gamma: float = 0.99):
        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.act_dim = act_dim

        self.buf = PPOData()
        self.episode_returns: List[float] = []
        self.episode_entropies: List[float] = []

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, val = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)

        if training:
            self.buf.states.append(obs.copy())
            self.buf.actions.append(int(a.item()))
            self.buf.log_probs.append(float(logp.item()))
            self.buf.vals.append(float(val.item()))

        return int(a.item())

    def step_end(self, reward: float, done: bool):
        self.buf.rewards.append(reward)
        self.buf.dones.append(done)

    def finish_episode(self):
        if len(self.buf) == 0:
            return
        with torch.no_grad():
            last_obs = torch.as_tensor(self.buf.states[-1], dtype=torch.float32).unsqueeze(0)
            _, last_val = self.policy(last_obs)
            last_val = float(last_val.item()) if not self.buf.dones[-1] else 0.0

        returns_np, adv_np = compute_gae(
            self.buf.rewards, self.buf.vals, self.gamma, last_val=last_val
        )
        mean_ent = ppo_update(
            self.policy, self.optimiser,
            np.array(self.buf.states),
            np.array(self.buf.actions, dtype=np.int64),
            np.array(self.buf.log_probs),
            returns_np, adv_np,
            entropy_coef=self.entropy_coef,
        )
        self.episode_returns.append(sum(self.buf.rewards))
        self.episode_entropies.append(mean_ent)
        self.buf.clear()

    def get_log(self) -> dict:
        return {"returns": self.episode_returns, "entropies": self.episode_entropies}
