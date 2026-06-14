"""
Cognitive Inertia Theorem — Model Definition
=============================================
Defines the minimal delayed adaptive system (Eqs. 1-4 of the main text).

Default parameters correspond to Table 1.

Usage:
    import model
    result = model.run(seed=42)
    print(result['k_sil'])  # 0.862
"""

import numpy as np

# Default parameters (Table 1)
DEFAULT_PARAMS = {
    'alpha': 1.8,       # stimulus coupling gain
    'beta': 2.2,         # self-coupling gain
    'gamma': 0.006,      # kappa relaxation rate
    'eta': 0.05,         # autocatalytic consolidation strength
    'eta_C': 0.01,       # delayed self-correlation accumulation rate
    'lambda': 1.5,       # correlation-to-target scaling
    'theta_k': 0.35,     # autocatalytic activation threshold
    'sigma': 0.015,      # noise amplitude
    'T': 24,             # stimulus period (timesteps)
    'delta': 24,         # recurrence delay (timesteps)
    't_sync': 400,       # synchronization phase duration
    't_sil': 600,        # silence onset
    'total': 3000,       # total simulation steps
}


def run(seed=42, params=None, return_trajectory=False):
    """
    Run a single simulation of the cognitive inertia model.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    params : dict or None
        Parameter overrides (keys from DEFAULT_PARAMS).
    return_trajectory : bool
        If True, return full time series.

    Returns
    -------
    dict with keys:
        regime : str ('S' = sustained / 'L' = locked-only / 'R' = reactive)
        k_sil : float — mean kappa during silence window
        a_sil : float — half peak-to-peak amplitude during silence
        period : float — detected autonomous period (0 if none)
        kappa_t : ndarray (if return_trajectory)
        x_t : ndarray (if return_trajectory)
        C_t : ndarray (if return_trajectory)
    """
    if params is None:
        params = {}
    p = {**DEFAULT_PARAMS, **params}

    np.random.seed(seed)

    # Initialise history buffer
    x_buf = list(np.random.uniform(-0.1, 0.1, p['delta']))
    x_hist = list(x_buf)
    kappa, C = 0.0, 0.0
    kappa_hist = [kappa]

    for t in range(p['total']):
        # Stimulus
        if t < p['t_sync']:
            S = np.sin(2 * np.pi * t / p['T'])
        elif t < p['t_sil']:
            S = np.sin(2 * np.pi * t / (2 * p['T']))
        else:
            S = 0.0

        # State update (Eq. 1)
        x_delayed = x_hist[-p['delta']]
        x_t = ((1 - kappa) * np.tanh(p['alpha'] * S)
               + kappa * np.tanh(p['beta'] * x_delayed)
               + np.random.normal(0, p['sigma']))
        x_hist.append(x_t)

        # Delayed self-correlation accumulation (Eq. 2)
        C += p['eta_C'] * (np.tanh(x_t) * np.tanh(x_delayed) - C)

        # Kappa update: base (Eq. 3)
        kappa_target = np.tanh(p['lambda'] * C)
        kappa_base = kappa + p['gamma'] * (kappa_target - kappa)

        # Kappa update: autocatalytic lift (Eq. 4)
        if kappa > p['theta_k']:
            kappa_raw = kappa_base + p['eta'] * kappa * (1 - kappa)
        else:
            kappa_raw = kappa_base
        kappa = np.clip(kappa_raw, 0.0, 1.0)
        kappa_hist.append(kappa)

    # Analysis
    x_arr = np.array(x_hist)
    k_arr = np.array(kappa_hist)
    x_sil = x_arr[p['t_sil']:]
    k_sil = float(np.mean(k_arr[p['t_sil']:]))
    a_sil = float(np.max(np.abs(x_sil)))

    # Period detection
    signs = np.sign(x_sil)
    zero_crossings = np.where(np.diff(signs))[0]
    if len(zero_crossings) > 4:
        half_periods = np.diff(zero_crossings)
        period = float(2.0 * np.median(half_periods))
    else:
        period = 0.0

    # Regime classification
    k_sync = k_arr[p['t_sync'] - 100:p['t_sync']]
    sync_locked = float(np.mean(k_sync > p['theta_k']))
    sustained = (k_sil > 0.5 and a_sil > 0.1 and period > 0)
    if sustained:
        regime = 'S'
    elif sync_locked > 0.8:
        regime = 'L'
    else:
        regime = 'R'

    result = {
        'regime': regime,
        'k_sil': k_sil,
        'a_sil': a_sil,
        'period': period,
    }
    if return_trajectory:
        result['x_t'] = x_arr
        result['kappa_t'] = k_arr
    return result
