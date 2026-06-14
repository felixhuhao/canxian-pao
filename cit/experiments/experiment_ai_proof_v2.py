"""
AI Proof-of-Concept V2: Forced Collapse Scenario
=================================================
Demonstrates that lateral inhibition (LI) is critical when the
gradient signal alone would collapse channels.

Design: 
- Input is a mixture where one frequency dominates (amplitude 3x others)
- Model initialized with all channels at SAME τ (forcing symmetric start)
- Without LI: symmetry broken slowly, channels converge to similar τ
- With LI: channels rapidly differentiate

Expected:
  No LI → τ diversity < 3 (near-collapse)
  With LI → τ diversity > 10 (robust separation)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class MultiChannelSSM(nn.Module):
    """Minimal multi-channel linear SSM with learnable τ."""
    
    def __init__(self, M=3, input_dim=1, hidden_dim=8, output_dim=1,
                 tau_init=None):
        super().__init__()
        self.M = M
        self.hidden_dim = hidden_dim
        
        if tau_init is None:
            tau_init = [15.0] * M  # All same — symmetric start
        
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init, dtype=torch.float32)))
        self.input_proj = nn.Linear(input_dim, hidden_dim * M)
        self.output_proj = nn.Linear(hidden_dim * M, output_dim)
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        M, H = self.M, self.hidden_dim
        
        tau = torch.exp(self.log_tau)
        alpha = torch.clamp(1.0 / tau, 0.01, 0.5)
        
        h = torch.zeros(batch, M, H, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            u = x[:, t, :]
            drive = self.input_proj(u).view(batch, M, H)
            
            h_new = torch.zeros_like(h)
            for m in range(M):
                h_new[:, m, :] = (1 - alpha[m]) * h[:, m, :] + alpha[m] * drive[:, m, :]
            h = h_new
            
            y_t = self.output_proj(h.reshape(batch, M * H))
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)
    
    def get_tau(self):
        return torch.exp(self.log_tau).detach().cpu().numpy()
    
    def lateral_inhibition_loss(self, scale=5.0, strength=10.0):
        """Gaussian repulsion — requires STRONG penalty to overcome gradient."""
        if self.M <= 1:
            return torch.tensor(0.0, device=self.log_tau.device)
        tau = torch.exp(self.log_tau)
        loss = 0.0
        for i in range(self.M):
            for j in range(i+1, self.M):
                d = torch.abs(tau[i] - tau[j])
                loss += strength * torch.exp(-d**2 / (2 * scale**2))
        return loss


def generate_data(n_samples=500, seq_len=100, noise=0.01):
    """Multi-frequency data with ONE dominant frequency (amplitude 3x)."""
    X = np.zeros((n_samples, seq_len, 1))
    Y = np.zeros((n_samples, seq_len, 1))
    
    for s in range(n_samples):
        t = np.arange(seq_len)
        # Dominant frequency (period ≈ 20)
        signal = 3.0 * np.sin(2 * np.pi * 0.05 * t + np.random.uniform(0, 2*np.pi))
        # Secondary frequencies
        for f in [0.02, 0.08, 0.12, 0.18]:
            signal += 0.5 * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2*np.pi))
        signal += np.random.normal(0, noise, seq_len)
        
        X[s, :, 0] = signal
        Y[s, :-1, 0] = signal[1:]
        Y[s, -1, 0] = signal[-1]
    
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


def train_model(model, X_train, Y_train, X_val, Y_val,
                n_epochs=200, lr=0.01, use_li=False):
    optimizer = optim.Adam([
        {'params': [model.input_proj.weight, model.input_proj.bias,
                     model.output_proj.weight, model.output_proj.bias], 'lr': lr},
        {'params': [model.log_tau], 'lr': lr * 0.3},
    ])
    mse = nn.MSELoss()
    history = {'val_loss': [], 'tau': []}
    
    for epoch in range(n_epochs):
        model.train()
        Y_pred = model(X_train)
        loss = mse(Y_pred, Y_train)
        if use_li:
            loss += model.lateral_inhibition_loss()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            history['val_loss'].append(mse(model(X_val), Y_val).item())
            history['tau'].append(model.get_tau().copy())
    
    return history


if __name__ == '__main__':
    print("=" * 60)
    print("AI Proof-of-Concept V2: Forced Collapse Test")
    print("=" * 60)
    print("All channels initialized at τ=15 (identical).")
    print("Data has ONE dominant frequency (3x amplitude).")
    print("Without LI, gradient pulls all channels to similar τ.")
    print()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_train, Y_train = generate_data(500, 100)
    X_val, Y_val = generate_data(100, 100)
    
    results = {}
    
    for use_li, label in [(False, 'No LI'), (True, 'With LI')]:
        print(f"\n--- {label} ---")
        model = MultiChannelSSM(M=3, tau_init=[15.0, 15.0, 15.0])
        hist = train_model(model, X_train, Y_train, X_val, Y_val,
                           n_epochs=200, use_li=use_li)
        tau_f = model.get_tau()
        diversity = float(np.mean([abs(tau_f[i] - tau_f[j])
                                    for i in range(3) for j in range(i+1, 3)]))
        results[label] = {
            'tau_final': tau_f,
            'diversity': diversity,
            'val_loss': hist['val_loss'][-1],
            'tau_traj': np.array(hist['tau']),
        }
        
        print(f"  Final τ: {np.round(tau_f, 2)}")
        print(f"  Diversity: {diversity:.2f}")
        print(f"  Val MSE: {hist['val_loss'][-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    no_li = results['No LI']
    with_li = results['With LI']
    
    print(f"No LI: τ diversity = {no_li['diversity']:.2f}," +
          f" val = {no_li['val_loss']:.6f}")
    print(f"With LI: τ diversity = {with_li['diversity']:.2f}," +
          f" val = {with_li['val_loss']:.6f}")
    
    collapse = no_li['diversity'] < 5.0
    li_helps = with_li['diversity'] > no_li['diversity'] * 1.5
    
    print(f"\nCollapse without LI: {'YES ✓' if collapse else 'NO'}")
    print(f"LI restores diversity: {'YES ✓' if li_helps else 'NO'}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (label, r) in zip(axes, results.items()):
        traj = r['tau_traj']
        for m in range(traj.shape[1]):
            ax.plot(traj[:, m], label=f'τ_{m}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('τ')
        ax.set_title(label)
        ax.legend()
        ax.set_ylim(0, 60)
    
    plt.tight_layout()
    outpath = '/Users/caihengjin/.openclaw/workspace/analysis/code/ai_ssm_collapse_test.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure: {outpath}")
