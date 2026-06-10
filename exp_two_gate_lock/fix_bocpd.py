"""Replace BOCPD class with log-space version."""
with open('analysis/exp_two_gate_lock/agents.py', 'r') as f:
    content = f.read()

# Find old BOCPD class
import re
old_start = content.find('class BOCPD:')
old_end = content.find('\n\n\n', old_start)
if old_end == -1:
    old_end = content.find('\n# ', old_start)
if old_end == -1:
    old_end = content.find('class PPOData', old_start)
old_class = content[old_start:old_end]

new_class = '''
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
'''

content = content.replace(old_class, new_class)
with open('analysis/exp_two_gate_lock/agents.py', 'w') as f:
    f.write(content)
print('BOCPD replaced')
