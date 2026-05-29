"""
exp_synthetic.py
─────────────────────────────────────────────────────────────────────────────
Synthetic experiments for the CL ↔ CC project.

Four experiments, each on the same synthetic CC instance generator from
cc_core. Each writes a .npy file (raw numbers) and a .png figure to
results/.

  Experiment 1 (density × size)
    Vary E++ density p in [0.1, 1.0] and graph size n in {80, 160, 320, 500}.
    Tests: does CL track Pivot/MFP as density grows? (Claim C3)

  Experiment 2 (TC violation)            <-- the main result
    Vary epsilon, the fraction of within-cluster pairs flipped from
    '++' to '--' (i.e., triadic-closure violations).
    Tests: does CL collapse uniquely when TC is violated? (Claim C4)

  Experiment 3 (positive-edge purity)
    Vary cross-cluster noise eta and measure resulting purity rho.
    Tests: does rho predict CL's relative advantage? (Claim C5)
    NOTE: a cleaner controlled-purity version is in exp_purity.py.

  Experiment 4 (scaling)
    Fix k, vary n in {50, 100, 200, 400, 600, 800}.
    Tests: does CL hold up as graphs get larger at fixed cluster count?

Usage:
    python exp_synthetic.py              # run all four
    python exp_synthetic.py --exp 2      # run just exp 2
    python exp_synthetic.py --exp 1 3    # run exp 1 and exp 3
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, time
from collections import defaultdict

from cc_core import (
    generate_sparse_cc_instance, pivot, train_cl, cluster_emb,
    cc_cost, norm_cost,
)
from cc_baselines import local_search, mfp

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defaults shared by all experiments. Override per-experiment as needed.
N_TRIALS = 5
N_EPOCHS = 80    # CL training epochs (empirically sufficient for these sizes)


# ═════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════════════

def run_one(n, k, edges, gt, seed):
    """
    Run all four methods on a single CC instance, return a dict of metrics.

    Pivot is run first; its output is reused as the initialization for
    Local Search (this is the standard setup, not a tuning choice).
    """
    pa = pivot(n, edges, seed=seed)
    la = local_search(n, edges, pa)
    ma = mfp(n, edges, seed=seed)
    emb = train_cl(n, edges, k=k, n_epochs=N_EPOCHS, seed=seed)
    ca = cluster_emb(emb, k, seed=seed)
    return {
        'pivot': {'cost': norm_cost(edges, pa), 'ari': adjusted_rand_score(gt, pa)},
        'ls':    {'cost': norm_cost(edges, la), 'ari': adjusted_rand_score(gt, la)},
        'mfp':   {'cost': norm_cost(edges, ma), 'ari': adjusted_rand_score(gt, ma)},
        'cl':    {'cost': norm_cost(edges, ca), 'ari': adjusted_rand_score(gt, ca)},
    }


def empty_arrays(n_params, n_trials):
    """Allocate the (n_params, n_trials) zero matrices we accumulate into."""
    methods = ['pivot', 'ls', 'mfp', 'cl']
    return {m: {'cost': np.zeros((n_params, n_trials)),
                'ari':  np.zeros((n_params, n_trials))}
            for m in methods}


def print_row(label, arrays, pi):
    """Print one progress row across all four methods."""
    print(f"  {label:20s} | " + " | ".join(
        f"{m:5s} ARI={arrays[m]['ari'][pi].mean():.3f} "
        f"cost={arrays[m]['cost'][pi].mean():.3f}"
        for m in ['pivot', 'ls', 'mfp', 'cl']
    ))


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

# Consistent color/marker scheme across all four experiments
METHOD_STYLES = {
    'pivot': dict(color='#185FA5', marker='o', ls='-',  label='Pivot'),
    'ls':    dict(color='#2E8B57', marker='D', ls='-',  label='Local Search'),
    'mfp':   dict(color='#8B4513', marker='^', ls='-',  label='Veldt MFP'),
    'cl':    dict(color='#D85A30', marker='s', ls='--', label='CL (2-hop)'),
}


def plot_results(xs, arrays, xlabel, title, save_path,
                 ari_ylim=(-0.05, 1.05), cost_ylim=None):
    """Single-row figure: ARI on the left, normalized CC cost on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12)

    for ax, metric, ylabel, ylim in [
        (axes[0], 'ari',  'Adjusted Rand Index  (higher is better)', ari_ylim),
        (axes[1], 'cost', 'Normalized CC cost  (lower is better)',   cost_ylim),
    ]:
        for m, sty in METHOD_STYLES.items():
            mean = arrays[m][metric].mean(axis=1)
            std  = arrays[m][metric].std(axis=1)
            ax.plot(xs, mean, marker=sty['marker'], ls=sty['ls'],
                    color=sty['color'], label=sty['label'], lw=2, ms=5)
            # ±1 std shaded band
            ax.fill_between(xs, mean - std, mean + std,
                            alpha=0.12, color=sty['color'])
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


def plot_grid_by_size(xs, all_arrays, sizes, xlabel, title, save_path):
    """
    Experiment 1 plot: one curve per (method, size) combination.
    Color intensity encodes size; line style encodes method.
    """
    pivot_colors = ['#B5D4F4', '#378ADD', '#185FA5', '#042C53']
    cl_colors    = ['#F0997B', '#D85A30', '#993C1D', '#4A1B0C']
    ls_colors    = ['#B5EAC9', '#52B788', '#2E8B57', '#1B4332']
    mfp_colors   = ['#F5C6A0', '#E08A40', '#A05010', '#5A2500']
    color_maps = {'pivot': pivot_colors, 'cl': cl_colors,
                  'ls': ls_colors, 'mfp': mfp_colors}
    markers = {'pivot': 'o', 'cl': 's', 'ls': 'D', 'mfp': '^'}
    lstyles = {'pivot': '-', 'cl': '--', 'ls': '-', 'mfp': '-.'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    for ax, metric, ylabel, ylim in [
        (axes[0], 'ari',  'ARI (higher is better)',              (-0.05, 1.05)),
        (axes[1], 'cost', 'Normalized CC cost (lower is better)', None),
    ]:
        for ni, n in enumerate(sizes):
            arrays = all_arrays[n]
            for m in ['pivot', 'cl', 'ls', 'mfp']:
                mean = arrays[m][metric].mean(axis=1)
                std  = arrays[m][metric].std(axis=1)
                color = color_maps[m][ni]
                ax.plot(xs, mean,
                        marker=markers[m], ls=lstyles[m],
                        color=color, lw=1.8, ms=4,
                        label=f'{m.upper()} n={n}')
                ax.fill_between(xs, mean - std, mean + std,
                                alpha=0.08, color=color)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 1: density × size
# ═════════════════════════════════════════════════════════════════════════════

def exp1(densities=None, sizes=None, noise=0.02):
    """Sweep E++ density × graph size. CL should track the baselines as
    density grows."""
    if densities is None:
        densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if sizes is None:
        sizes = [80, 160, 320, 500]

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: E++ density × graph size  (epsilon=0)")
    print("=" * 70)

    all_arrays = {}
    for n in sizes:
        k = max(2, n // 25)
        print(f"\n  n={n}  k={k}  (feat_dim={k*4} d_h={k*8} d_out={k*4})")
        arrays = empty_arrays(len(densities), N_TRIALS)

        for di, d in enumerate(densities):
            for trial in range(N_TRIALS):
                seed = 100 + di * 1000 + trial * 7 + n
                edges, gt = generate_sparse_cc_instance(
                    n, k, density=d, epsilon=0.0, noise=noise, seed=seed)
                r = run_one(n, k, edges, gt, seed)
                for m in ['pivot', 'ls', 'mfp', 'cl']:
                    arrays[m]['cost'][di, trial] = r[m]['cost']
                    arrays[m]['ari'][di, trial]  = r[m]['ari']

            print_row(f"p={d:.1f}", arrays, di)

        all_arrays[n] = arrays

    np.save(f'{OUTPUT_DIR}/exp1_results.npy', all_arrays)
    plot_grid_by_size(
        densities, all_arrays, sizes,
        xlabel='E++ density  p',
        title='Exp 1: CL vs Baselines across E++ density and graph size\n'
              r'($\varepsilon$=0, noise=0.02, k=n/25)',
        save_path=f'{OUTPUT_DIR}/exp1_all_baselines.png',
    )
    return all_arrays


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2: TC violation rate  (THE MAIN RESULT)
# ═════════════════════════════════════════════════════════════════════════════

def exp2(epsilons=None, sizes=None, density=0.7, noise=0.02):
    """Sweep epsilon, the fraction of within-cluster pairs flipped to '--'
    (TC violations). CL should be uniquely sensitive to this knob."""
    if epsilons is None:
        epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if sizes is None:
        sizes = [160, 500]   # two representative sizes for the plot

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: TC violation rate  (density={density})")
    print("=" * 70)

    all_arrays = {}
    for n in sizes:
        k = max(2, n // 25)
        print(f"\n  n={n}  k={k}")
        arrays = empty_arrays(len(epsilons), N_TRIALS)

        for ei, eps in enumerate(epsilons):
            for trial in range(N_TRIALS):
                seed = 200 + ei * 1000 + trial * 7 + n
                edges, gt = generate_sparse_cc_instance(
                    n, k, density=density, epsilon=eps, noise=noise, seed=seed)
                r = run_one(n, k, edges, gt, seed)
                for m in ['pivot', 'ls', 'mfp', 'cl']:
                    arrays[m]['cost'][ei, trial] = r[m]['cost']
                    arrays[m]['ari'][ei, trial]  = r[m]['ari']

            print_row(f"eps={eps:.1f}", arrays, ei)

        all_arrays[n] = arrays

    np.save(f'{OUTPUT_DIR}/exp2_results.npy', all_arrays)
    for n in sizes:
        plot_results(
            epsilons, all_arrays[n],
            xlabel='TC violation rate  ε',
            title=f'Exp 2: Effect of TC violation  (n={n}, p={density})',
            save_path=f'{OUTPUT_DIR}/exp2_n{n}.png',
        )
    return all_arrays


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 3: positive-edge purity ρ via noise sweep
# ═════════════════════════════════════════════════════════════════════════════
# This was the "first attempt" at the purity sweep; the cleaner version with
# exact control over rho lives in exp_purity.py. Kept here for completeness
# / reproducibility of the original Exp 3 figure.

def _measure_purity(edges, n_nodes):
    """
    Approximate positive-edge purity: treat E++-connected components as
    approximate clusters, then compute the fraction of E++ edges that lie
    *within* a component.

    Caveat: at high density the E++ subgraph becomes one giant component,
    so this measure saturates to 1.0 regardless of true noise. exp_purity.py
    sidesteps this by sampling positive edges directly with a target ρ.
    """
    from collections import deque
    pos_nbrs = defaultdict(list)
    for (i, j), label in edges.items():
        if label == '++':
            pos_nbrs[i].append(j); pos_nbrs[j].append(i)

    visited = np.full(n_nodes, -1, dtype=int)
    cid = 0
    for start in range(n_nodes):
        if visited[start] >= 0:
            continue
        queue = deque([start])
        visited[start] = cid
        while queue:
            v = queue.popleft()
            for u in pos_nbrs[v]:
                if visited[u] < 0:
                    visited[u] = cid
                    queue.append(u)
        cid += 1

    total_pos = sum(1 for l in edges.values() if l == '++')
    if total_pos == 0:
        return 0.0
    within = sum(
        1 for (i, j), l in edges.items()
        if l == '++' and visited[i] == visited[j]
    )
    return within / total_pos


def exp3(noise_levels=None, n=320, density=0.7, epsilon=0.0):
    """Vary cross-cluster noise eta, measure resulting purity rho, plot
    method performance vs rho."""
    if noise_levels is None:
        noise_levels = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05,
                        0.10, 0.15, 0.20, 0.30, 0.40]

    k = max(2, n // 25)
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: positive-edge purity via noise sweep  (n={n}, density={density})")
    print("=" * 70)

    arrays = empty_arrays(len(noise_levels), N_TRIALS)
    rho_vals = np.zeros((len(noise_levels), N_TRIALS))

    for ni, eta in enumerate(noise_levels):
        for trial in range(N_TRIALS):
            seed = 300 + ni * 1000 + trial * 7 + n
            edges, gt = generate_sparse_cc_instance(
                n, k, density=density, epsilon=epsilon, noise=eta, seed=seed)
            rho = _measure_purity(edges, n)
            rho_vals[ni, trial] = rho
            r = run_one(n, k, edges, gt, seed)
            for m in ['pivot', 'ls', 'mfp', 'cl']:
                arrays[m]['cost'][ni, trial] = r[m]['cost']
                arrays[m]['ari'][ni, trial]  = r[m]['ari']

        rho_mean = rho_vals[ni].mean()
        print_row(f"eta={eta:.3f} rho={rho_mean:.3f}", arrays, ni)

    np.save(f'{OUTPUT_DIR}/exp3_results.npy',
            {'arrays': arrays, 'rho': rho_vals, 'noise': noise_levels})

    # Scatter: each trial is a point at (rho_actual, method_score)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Exp 3: Method performance vs positive-edge purity ρ\n'
        f'(n={n}, density={density}, ε={epsilon})', fontsize=12)
    rho_flat = rho_vals.ravel()

    for ax, metric, ylabel, ylim in [
        (axes[0], 'ari',  'ARI (higher is better)',              (-0.05, 1.05)),
        (axes[1], 'cost', 'Normalized CC cost (lower is better)', None),
    ]:
        for m, sty in METHOD_STYLES.items():
            vals = arrays[m][metric].ravel()
            ax.scatter(rho_flat, vals, color=sty['color'],
                       label=sty['label'], alpha=0.6, s=25)
            order = np.argsort(rho_vals.mean(1))
            xs_trend = rho_vals.mean(1)[order]
            ys_trend = arrays[m][metric].mean(1)[order]
            ax.plot(xs_trend, ys_trend, color=sty['color'], lw=2)

        ax.set_xlabel('Positive-edge purity  ρ', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/exp3_purity.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    return arrays, rho_vals


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 4: scaling at fixed k
# ═════════════════════════════════════════════════════════════════════════════

def exp4(ns=None, k=4, density=0.7, noise=0.02):
    """Fix the number of clusters and vary n. Tests scalability with
    increasing graph size at constant cluster count."""
    if ns is None:
        ns = [50, 100, 200, 400, 600, 800]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4: fixed k={k}, varying n  (density={density})")
    print("=" * 70)

    arrays = empty_arrays(len(ns), N_TRIALS)

    for ni, n in enumerate(ns):
        for trial in range(N_TRIALS):
            seed = 400 + ni * 1000 + trial * 7
            edges, gt = generate_sparse_cc_instance(
                n, k, density=density, epsilon=0.0, noise=noise, seed=seed)
            r = run_one(n, k, edges, gt, seed)
            for m in ['pivot', 'ls', 'mfp', 'cl']:
                arrays[m]['cost'][ni, trial] = r[m]['cost']
                arrays[m]['ari'][ni, trial]  = r[m]['ari']

        print_row(f"n={n}", arrays, ni)

    np.save(f'{OUTPUT_DIR}/exp4_results.npy', arrays)
    plot_results(
        ns, arrays,
        xlabel='Graph size  n',
        title=f'Exp 4: Scalability with fixed k={k}  (density={density}, ε=0)',
        save_path=f'{OUTPUT_DIR}/exp4_scaling.png',
    )
    return arrays


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # --exp N M ... runs only those experiment numbers; no flag runs all.
    args = sys.argv[1:]
    if '--exp' in args:
        idx = args.index('--exp')
        to_run = set(int(x) for x in args[idx + 1:] if x.isdigit())
    else:
        to_run = {1, 2, 3, 4}

    t_total = time.time()

    if 1 in to_run:
        t0 = time.time(); exp1(); print(f"\n  Exp 1 done in {time.time()-t0:.1f}s")
    if 2 in to_run:
        t0 = time.time(); exp2(); print(f"\n  Exp 2 done in {time.time()-t0:.1f}s")
    if 3 in to_run:
        t0 = time.time(); exp3(); print(f"\n  Exp 3 done in {time.time()-t0:.1f}s")
    if 4 in to_run:
        t0 = time.time(); exp4(); print(f"\n  Exp 4 done in {time.time()-t0:.1f}s")

    print(f"\nAll done in {time.time()-t_total:.1f}s")
    print(f"Results saved to {OUTPUT_DIR}/")
