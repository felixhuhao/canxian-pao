"""
High-Dimensional Manifold Topology
====================================
Extends τ from a 1D scalar to a D-dimensional vector, defining resonance
as a projection onto skill-specific direction vectors. Computes deadlock
volume fraction in D-space and visualizes the topology of deadlock regions.

Key concepts:
  - Skill i defines a direction v_i ∈ ℝ^D and period P_i
  - Resonance condition: |⟨τ, v_i⟩ - n·P_i| < ε for some integer n
  - Deadlock: no skill satisfies resonance for any integer n
  - In D dimensions, the deadlock region is the complement of intersecting
    parallel hyperplane families — a topologically complex void space.

Predictions:
  1. Deadlock volume fraction increases with D (curse of dimensionality)
  2. BUT each deadlock region is smaller — more numerous, more fragmented
  3. At D ≥ 3, deadlock becomes a "Swiss cheese" topology
    
Usage:
    python experiment_highdim.py --deadlock-volume D=1,2,3
    python experiment_highdim.py --visualize-2d
    python experiment_highdim.py --visualize-3d
"""

import numpy as np
import time
import os
import sys
import json
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# For 3D plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle, Circle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# Core: High-Dimensional Resonance & Deadlock
# ============================================================

def make_random_directions(N: int, D: int, seed: int = 42) -> np.ndarray:
    """Generate N random unit vectors in ℝ^D."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(N, D)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def resonance_condition(tau: np.ndarray, v: np.ndarray, P: float,
                        eps: float = 2.0) -> bool:
    """
    Check if τ resonates with skill (v, P).
    Condition: |⟨τ, v⟩ - n·P| < ε for some integer n.
    """
    proj = np.dot(tau, v)
    # Find the nearest integer multiple of P
    n = np.round(proj / P)
    return abs(proj - n * P) < eps


def in_deadlock(tau: np.ndarray, skills: List[Tuple[np.ndarray, float]],
                eps: float = 2.0) -> bool:
    """Check if τ is in deadlock (no skill resonates)."""
    for v, P in skills:
        if resonance_condition(tau, v, P, eps):
            return False
    return True


def compute_deadlock_volume(D: int, N_skills: int = 5,
                            periods: Optional[List[float]] = None,
                            n_samples: int = 50000,
                            seed: int = 42,
                            eps: float = 2.0,
                            verbose: bool = True) -> Dict:
    """
    Monte Carlo estimate of deadlock volume fraction in [0, L]^D.
    """
    if periods is None:
        periods = [12.0, 24.0, 24.0, 48.0, 30.0]  # C, A, B, D, E
    
    L = max(periods) * 2.0  # Box size covers up to 2× the longest period
    
    # Generate random skill directions
    directions = make_random_directions(N_skills, D, seed=seed)
    skills = list(zip(directions, periods))
    
    # Monte Carlo sampling
    rng = np.random.RandomState(seed + 1)
    samples = rng.uniform(0, L, (n_samples, D))
    
    deadlock_count = 0
    deadlock_samples = []
    
    for i, sample in enumerate(samples):
        if in_deadlock(sample, skills, eps):
            deadlock_count += 1
            if len(deadlock_samples) < min(100, n_samples):
                deadlock_samples.append(sample.tolist())
    
    vol_fraction = deadlock_count / n_samples
    
    # Theory prediction for comparison
    # In 1D with N skills: will the theoretical fraction be different?
    # Each skill creates a family of "resonance strips" of width 2ε/P_i
    # For N independent families in D dimensions, the fraction of deadlock
    # is approximately product over skills of (1 - 2ε/P_i × factor_from_D)
    
    results = {
        'D': D,
        'N_skills': N_skills,
        'periods': periods,
        'n_samples': n_samples,
        'deadlock_count': deadlock_count,
        'deadlock_volume_fraction': vol_fraction,
        'eps': eps,
        'length_scale': L,
        'deadlock_samples': deadlock_samples[:20],  # Keep some for inspection
        'directions': directions.tolist(),
    }
    
    if verbose:
        print(f"  D={D}: deadlock fraction = {vol_fraction:.4f} "
              f"({deadlock_count}/{n_samples})")
    
    return results


# ============================================================
# Experiment 1: Deadlock Volume vs Dimensionality
# ============================================================

def scan_dimensionality(D_values: List[int] = [1, 2, 3, 4, 5, 6],
                        n_samples: int = 100000,
                        seed: int = 42,
                        verbose: bool = True) -> Dict:
    """Scan deadlock volume fraction across dimensions."""
    periods = [12.0, 24.0, 24.0, 48.0, 30.0]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Deadlock Volume vs Dimensionality — {n_samples} samples per D")
    print(f"{'='*60}")
    print(f"Skills: P ∈ {periods}")
    print()
    
    for D in D_values:
        r = compute_deadlock_volume(D, N_skills=len(periods),
                                    periods=periods,
                                    n_samples=n_samples,
                                    seed=seed, verbose=True)
        results[D] = r
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary: Deadlock Volume Fraction vs D")
    print(f"{'='*60}")
    print(f"{'D':<5} {'Fraction':<12} {'Count':<10} {'N_samples':<12}")
    print('-' * 40)
    for D in D_values:
        r = results[D]
        print(f"{D:<5} {r['deadlock_volume_fraction']:.6f}  "
              f"{r['deadlock_count']:<10} {r['n_samples']:<12}")
    
    return results


# ============================================================
# Experiment 2: Nearest Deadlock Distance
# ============================================================

def nearest_deadlock_distance(D: int = 2, N_skills: int = 5,
                               n_grid: int = 200, seed: int = 42,
                               verbose: bool = True) -> Dict:
    """
    For a grid of τ points in [0, L]^D, compute the L2 distance
    to the nearest resonant point. Visualize the "deadlock topology."
    """
    periods = [12.0, 24.0, 24.0, 48.0, 30.0]
    L = max(periods) * 2.0
    directions = make_random_directions(N_skills, D, seed=seed)
    skills = list(zip(directions, periods))
    
    if D > 2:
        if verbose:
            print(f"Nearest-deadlock-distance grid: D={D} requires "
                  f"{n_grid**D} points → sampling instead")
        # Monte Carlo for D > 2
        rng = np.random.RandomState(seed + 10)
        n_pts = min(10000, n_grid ** 2)
        samples = rng.uniform(0, L, (n_pts, D))
        
        distances = []
        for sample in samples:
            # Compute min distance to any resonance hyperplane
            min_dist = float('inf')
            for v, P in skills:
                proj = np.dot(sample, v)
                n = np.round(proj / P)
                dist = abs(proj - n * P)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        distances = np.array(distances)
        deadlock_mask = distances > 2.0
        
        return {
            'D': D,
            'distances_mean': float(np.mean(distances)),
            'distances_std': float(np.std(distances)),
            'deadlock_fraction': float(np.mean(deadlock_mask)),
            'mean_deadlock_dist': float(np.mean(distances[deadlock_mask])) if np.any(deadlock_mask) else 0,
            'mean_resonant_dist': float(np.mean(distances[~deadlock_mask])) if np.any(~deadlock_mask) else 0,
        }
    
    # For D ≤ 2, make a grid
    xs = np.linspace(0, L, n_grid)
    if D == 1:
        grid = xs.reshape(-1, 1)
    else:
        ys = np.linspace(0, L, n_grid)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    n_total = grid.shape[0]
    distances = np.zeros(n_total)
    
    for i, pt in enumerate(grid):
        min_dist = float('inf')
        for v, P in skills:
            proj = np.dot(pt, v)
            n = np.round(proj / P)
            dist = abs(proj - n * P)
            min_dist = min(min_dist, dist)
        distances[i] = min_dist
    
    distances = distances.reshape((n_grid, n_grid)) if D == 2 else distances
    deadlock_mask = distances > 2.0
    deadlock_fraction = float(np.mean(deadlock_mask))
    
    if verbose:
        print(f"  D={D} grid={n_grid}×{n_grid}: "
              f"deadlock fraction = {deadlock_fraction:.4f}, "
              f"mean distance = {float(np.mean(distances)):.3f}")
    
    return {
        'D': D,
        'distances': distances.tolist() if D == 1 else None,
        'distances_grid': distances.tolist() if D == 2 else None,
        'deadlock_fraction': deadlock_fraction,
        'deadlock_mask': deadlock_mask.tolist() if D == 2 else None,
        'n_grid': n_grid,
        'L': L,
    }


# ============================================================
# Visualization (2D / 3D)
# ============================================================

def visualize_2d(n_grid: int = 400, seed: int = 42, 
                  save_dir: str = '.') -> str:
    """
    Generate a 2D heatmap of resonance distance.
    Shows the "Swiss cheese" topology of deadlock regions.
    """
    if not HAS_MPL:
        print("matplotlib not available — skipping visualization")
        return ""
    
    periods = [12.0, 24.0, 24.0, 48.0, 30.0]
    N_skills = len(periods)
    L = max(periods) * 2.0
    directions = make_random_directions(N_skills, 2, seed=seed)
    skills = list(zip(directions, periods))
    
    xs = np.linspace(0, L, n_grid)
    ys = np.linspace(0, L, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    distances = np.zeros(len(grid))
    for i, pt in enumerate(grid):
        min_dist = float('inf')
        for v, P in skills:
            proj = np.dot(pt, v)
            n = np.round(proj / P)
            dist = abs(proj - n * P)
            min_dist = min(min_dist, dist)
        distances[i] = min_dist
    
    distances = distances.reshape((n_grid, n_grid))
    deadlock_mask = distances > 2.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: resonance distance heatmap
    im1 = ax1.imshow(distances, extent=[0, L, 0, L], origin='lower',
                     cmap='RdYlBu_r', vmin=0, vmax=8)
    plt.colorbar(im1, ax=ax1, label='Distance to nearest resonance')
    ax1.set_title(f'Resonance Distance Map (D=2, {N_skills} skills)')
    ax1.set_xlabel('τ₁')
    ax1.set_ylabel('τ₂')
    
    # Overlay skill direction vectors
    for i, (v, P) in enumerate(skills):
        center = np.array([L/3 + i*L/10, L/3 + i*L/10])
        ax1.arrow(center[0], center[1], v[0]*L/6, v[1]*L/6,
                  head_width=2, head_length=2, fc=f'C{i}', ec=f'C{i}',
                  alpha=0.7)
        ax1.text(center[0] + v[0]*L/6 + 1, center[1] + v[1]*L/6 + 1,
                 f'P={P:.0f}', fontsize=9, color=f'C{i}')
    
    # Right: deadlock mask (black = deadlock)
    ax2.imshow(deadlock_mask, extent=[0, L, 0, L], origin='lower',
               cmap='gray_r')
    ax2.set_title(f'Deadlock Regions (fraction = {np.mean(deadlock_mask):.3f})')
    ax2.set_xlabel('τ₁')
    ax2.set_ylabel('τ₂')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'highdim_deadlock_2d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  2D visualization saved to {save_path}")
    return save_path


def visualize_3d(n_grid: int = 100, seed: int = 42,
                  save_dir: str = '.') -> str:
    """
    Generate a 3D scatter plot of deadlock vs resonant points.
    """
    if not HAS_MPL:
        print("matplotlib not available — skipping visualization")
        return ""
    
    periods = [12.0, 24.0, 24.0, 48.0, 30.0]
    N_skills = len(periods)
    L = max(periods) * 2.0
    directions = make_random_directions(N_skills, 3, seed=seed)
    skills = list(zip(directions, periods))
    
    # Sub-sampled Monte Carlo for 3D
    n_pts = n_grid ** 3
    actual_pts = min(20000, n_pts)
    rng = np.random.RandomState(seed)
    samples = rng.uniform(0, L, (actual_pts, 3))
    
    deadlock_pts = []
    resonant_pts = []
    
    for sample in samples:
        dl = in_deadlock(sample, skills, eps=2.0)
        if dl:
            deadlock_pts.append(sample)
        else:
            resonant_pts.append(sample)
    
    deadlock_pts = np.array(deadlock_pts)
    resonant_pts = np.array(resonant_pts)
    
    fig = plt.figure(figsize=(14, 7))
    
    # Left: resonant points
    ax1 = fig.add_subplot(121, projection='3d')
    if len(resonant_pts) > 0:
        ax1.scatter(resonant_pts[:, 0], resonant_pts[:, 1], resonant_pts[:, 2],
                    c='blue', s=2, alpha=0.3, label='Resonant')
    ax1.set_title(f'Resonant Points ({len(resonant_pts)})')
    ax1.set_xlabel('τ₁')
    ax1.set_ylabel('τ₂')
    ax1.set_zlabel('τ₃')
    
    # Right: deadlock points
    ax2 = fig.add_subplot(122, projection='3d')
    if len(deadlock_pts) > 0:
        ax2.scatter(deadlock_pts[:, 0], deadlock_pts[:, 1], deadlock_pts[:, 2],
                    c='red', s=2, alpha=0.3, label='Deadlock')
    ax2.set_title(f'Deadlock Points ({len(deadlock_pts)})')
    ax2.set_xlabel('τ₁')
    ax2.set_ylabel('τ₂')
    ax2.set_zlabel('τ₃')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'highdim_deadlock_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D visualization saved to {save_path}")
    return save_path


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadlock-volume', action='store_true')
    parser.add_argument('--visualize-2d', action='store_true')
    parser.add_argument('--visualize-3d', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=100000)
    args = parser.parse_args()
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not any([args.deadlock_volume, args.visualize_2d, args.visualize_3d]):
        print("Running all high-dimensional experiments...")
        args.deadlock_volume = args.visualize_2d = args.visualize_3d = True
    
    if args.deadlock_volume:
        scan_dimensionality(D_values=[1, 2, 3, 4, 5], 
                            n_samples=args.samples, seed=args.seed)
    
    if args.visualize_2d:
        print(f"\n2D Deadlock Topology Visualization")
        visualize_2d(seed=args.seed, save_dir=save_dir)
    
    if args.visualize_3d:
        print(f"\n3D Deadlock Topology Visualization")
        visualize_3d(seed=args.seed, save_dir=save_dir)
