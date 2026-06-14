"""
AI Proof-of-Concept: Multi-Channel SSM with Lateral Inhibition
==============================================================
Demonstrates that channel competition regularization prevents
timescale collapse in learned multi-scale sequence models.

Task: Online prediction of multi-frequency sine mixtures.
Metrics: τ diversity, prediction MSE.

Three conditions:
  M=1: Single channel (baseline)
  M=3: Multi-channel, NO lateral inhibition
  M=3: Multi-channel WITH lateral inhibition

Prediction:
  Without LI → all channels learn same τ (collapse)
  With LI → channels separate into distinct timescales
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ============================================================
# Multi-channel SSM with learnable time constants
# ============================================================

class MultiChannelSSM(nn.Module):
    """
    Minimal multi-channel linear SSM.
    Each channel is a 1st-order low-pass filter with learnable τ.
    """
    
    def __init__(self, M=3, input_dim=1, hidden_dim=8, output_dim=1,
                 tau_init=None):
        super().__init__()
        self.M = M
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learnable time constants (one per channel)
        if tau_init is None:
            tau_init = [10.0, 25.0, 40.0][:M]
            if M == 1:
                tau_init = [20.0]
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init, dtype=torch.float32)))
        
        # Input projection to each channel
        self.input_proj = nn.Linear(input_dim, hidden_dim * M)
        
        # Output projection from all channels
        self.output_proj = nn.Linear(hidden_dim * M, output_dim)
        
        # Channel output weights (per timestep mixing)
        self.channel_mix = nn.Parameter(torch.ones(M, 1) / M)
    
    def forward(self, x, return_hidden=False):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len, output_dim)
        """
        batch, seq_len, _ = x.shape
        M = self.M
        hidden_dim = self.hidden_dim
        
        tau = torch.exp(self.log_tau)  # (M,)
        
        # Initialize hidden states
        h = torch.zeros(batch, M, hidden_dim, device=x.device)  # (B, M, H)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            u = x[:, t, :]  # (B, 1)
            
            # Input driving per channel
            drive = self.input_proj(u)  # (B, M*H)
            drive = drive.view(batch, M, hidden_dim)
            
            # Channel update: h_m = (1 - α_m) * h_m + α_m * drive_m
            # where α_m = 1/τ_m (normalized by max τ)
            alpha = 1.0 / tau  # (M,) — larger α = faster channel
            alpha = torch.clamp(alpha, 0.01, 0.5)  # stability
            
            h_new = torch.zeros_like(h)
            for m in range(M):
                h_new[:, m, :] = (1 - alpha[m]) * h[:, m, :] + alpha[m] * drive[:, m, :]
            h = h_new
            
            hidden_states.append(h)
            
            # Output: weighted sum across channels
            h_flat = h.reshape(batch, M * hidden_dim)
            y_t = self.output_proj(h_flat)
            outputs.append(y_t)
        
        out = torch.stack(outputs, dim=1)  # (B, seq_len, 1)
        if return_hidden:
            hidden_states = torch.stack(hidden_states, dim=1)  # (B, seq_len, M, H)
            return out, hidden_states, tau
        return out
    
    def get_tau(self):
        return torch.exp(self.log_tau).detach().cpu().numpy()
    
    def lateral_inhibition_loss(self, scale=8.0, strength=0.5):
        """Gaussian repulsion between channels."""
        if self.M <= 1:
            return torch.tensor(0.0, device=self.log_tau.device)
        tau = torch.exp(self.log_tau)
        loss = 0.0
        count = 0
        for i in range(self.M):
            for j in range(i+1, self.M):
                d = torch.abs(tau[i] - tau[j])
                loss += strength * torch.exp(-d**2 / (2 * scale**2))
                count += 1
        return loss / max(count, 1)


# ============================================================
# Dataset: Multi-frequency sine mixtures
# ============================================================

def generate_multifreq_data(n_samples=500, seq_len=100,
                             freq_range=(0.02, 0.25), n_freqs=3,
                             noise=0.02):
    """
    Generate sequences that are sums of sine waves at random frequencies.
    Each sample: mixture of n_freqs sine components with random phases.
    """
    X = np.zeros((n_samples, seq_len, 1))
    Y = np.zeros((n_samples, seq_len, 1))
    
    for s in range(n_samples):
        # Pick random frequencies for this sample
        freqs = np.random.uniform(freq_range[0], freq_range[1], n_freqs)
        amps = np.random.uniform(0.3, 1.0, n_freqs)
        amps = amps / np.sum(amps)  # Normalize
        phases = np.random.uniform(0, 2*np.pi, n_freqs)
        
        t = np.arange(seq_len)
        
        signal = np.zeros(seq_len)
        for f, a, p in zip(freqs, amps, phases):
            signal += a * np.sin(2 * np.pi * f * t + p)
        
        signal += np.random.normal(0, noise, seq_len)
        
        X[s, :, 0] = signal
        Y[s, :-1, 0] = signal[1:]  # Predict next step
        Y[s, -1, 0] = signal[-1]  # Last step predicts itself
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def train_model(model, X_train, Y_train, X_val, Y_val,
                n_epochs=200, lr=0.01, use_li=False,
                li_scale=8.0, li_strength=0.5):
    """Train multi-channel SSM."""
    optimizer = optim.Adam([
        {'params': model.input_proj.parameters(), 'lr': lr},
        {'params': model.output_proj.parameters(), 'lr': lr},
        {'params': [model.log_tau], 'lr': lr * 0.5},  # Slower τ update
        {'params': [model.channel_mix], 'lr': lr},
    ])
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'tau': []}
    
    for epoch in range(n_epochs):
        model.train()
        
        # Forward
        Y_pred = model(X_train)
        loss = mse_loss(Y_pred, Y_train)
        
        if use_li:
            loss += model.lateral_inhibition_loss(scale=li_scale, strength=li_strength)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            Y_val_pred = model(X_val)
            val_loss = mse_loss(Y_val_pred, Y_val).item()
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['tau'].append(model.get_tau().copy())
    
    return history


# ============================================================
# Main experiment
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("AI Proof-of-Concept: Multi-Channel SSM with LI")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Generate data
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_train, Y_train = generate_multifreq_data(500, 80, noise=0.02)
    X_val, Y_val = generate_multifreq_data(100, 80, noise=0.02)
    
    print(f"\nData: train {X_train.shape}, val {X_val.shape}")
    print(f"Frequencies range: [0.02, 0.25] (periods 4-50 timesteps)")
    
    # ---- Condition 1: M=1 (single channel baseline) ----
    print("\n" + "-" * 40)
    print("Condition 1: Single Channel (M=1)")
    
    model_1 = MultiChannelSSM(M=1, input_dim=1, hidden_dim=16, output_dim=1)
    hist_1 = train_model(model_1, X_train, Y_train, X_val, Y_val,
                          n_epochs=150, lr=0.01, use_li=False)
    tau_final_1 = model_1.get_tau()[0]
    final_val_1 = hist_1['val_loss'][-1]
    print(f"  Final τ: {tau_final_1:.2f}")
    print(f"  Val loss: {final_val_1:.6f}")
    
    # ---- Condition 2: M=3 without LI ----
    print("\n" + "-" * 40)
    print("Condition 2: Multi-Channel (M=3), NO lateral inhibition")
    
    model_2 = MultiChannelSSM(M=3, input_dim=1, hidden_dim=8, output_dim=1)
    hist_2 = train_model(model_2, X_train, Y_train, X_val, Y_val,
                          n_epochs=150, lr=0.01, use_li=False)
    tau_final_2 = model_2.get_tau()
    tau_diversity_2 = float(np.mean([abs(tau_final_2[i] - tau_final_2[j])
                                      for i in range(3) for j in range(i+1, 3)]))
    final_val_2 = hist_2['val_loss'][-1]
    print(f"  Final τ: {np.round(tau_final_2, 2)}")
    print(f"  τ diversity: {tau_diversity_2:.2f}")
    print(f"  Val loss: {final_val_2:.6f}")
    
    # ---- Condition 3: M=3 WITH lateral inhibition ----
    print("\n" + "-" * 40)
    print("Condition 3: Multi-Channel (M=3), WITH lateral inhibition")
    
    model_3 = MultiChannelSSM(M=3, input_dim=1, hidden_dim=8, output_dim=1)
    hist_3 = train_model(model_3, X_train, Y_train, X_val, Y_val,
                          n_epochs=150, lr=0.01, use_li=True,
                          li_scale=8.0, li_strength=0.5)
    tau_final_3 = model_3.get_tau()
    tau_diversity_3 = float(np.mean([abs(tau_final_3[i] - tau_final_3[j])
                                      for i in range(3) for j in range(i+1, 3)]))
    final_val_3 = hist_3['val_loss'][-1]
    print(f"  Final τ: {np.round(tau_final_3, 2)}")
    print(f"  τ diversity: {tau_diversity_3:.2f}")
    print(f"  Val loss: {final_val_3:.6f}")
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':>30} | {'τ diversity':>12} | {'Val MSE':>10}")
    print("-" * 60)
    
    collapse_no_li = tau_diversity_2 < 3.0
    
    print(f"{'Single channel (M=1)':>30} | {'N/A':>12} | {final_val_1:.6f}")
    print(f"{'Multi w/o LI (M=3)':>30} | {tau_diversity_2:>12.2f} | {final_val_2:.6f}")
    print(f"{'Multi WITH LI (M=3)':>30} | {tau_diversity_3:>12.2f} | {final_val_3:.6f}")
    
    print(f"\nCollapse without LI: {'YES ✓' if collapse_no_li else 'NO'}")
    print(f"Diversity with LI restored: {'YES ✓' if tau_diversity_3 > tau_diversity_2 else 'NO'}")
    
    if final_val_3 < final_val_2:
        print(f"LI improves prediction: YES ✓ ({final_val_3:.6f} < {final_val_2:.6f})")
    else:
        print(f"LI improves prediction: marginal ({final_val_3:.6f} vs {final_val_2:.6f})")
    
    # Save τ trajectory plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, hist, title in zip(axes,
                                [hist_1, hist_2, hist_3],
                                ['M=1 (single)', 'M=3 no LI', 'M=3 + LI']):
        tau_hist = np.array(hist['tau'])
        for m in range(tau_hist.shape[1]):
            ax.plot(tau_hist[:, m], label=f'τ_{m}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('τ')
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 60)
    
    plt.tight_layout()
    plt.savefig('/Users/caihengjin/.openclaw/workspace/analysis/code/ai_ssm_tau_trajectories.png',
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to ai_ssm_tau_trajectories.png")
