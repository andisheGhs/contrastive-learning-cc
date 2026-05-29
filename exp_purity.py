"""
exp_purity.py
─────────────────────────────────────────────────────────────────────────────
Experiment 3, controlled-purity variant.

The original Experiment 3 in exp_synthetic.py tried to control positive-edge
purity ρ indirectly by varying the cross-cluster noise eta and then measuring
ρ from E++ connected components. That approach has a known failure mode: at
high density the E++ subgraph collapses to one giant component, so the
component-based ρ measure saturates at 1.0 regardless of the true noise.

This file fixes that by *constructing* instances with a target ρ directly:

  - n nodes, k balanced clusters, total positive edges m_pos = pos_density · C(n, 2)
  - Within-cluster positive edges:  W = round(ρ · m_pos)
  - Cross-cluster positive edges:   X = m_pos - W
  - Sample W within-cluster pairs uniformly without replacement
  - Sample X across-cluster pairs uniformly without replacement
  - All other pairs are E--

This gives exact control over ρ, making the scatter plot of "method
performance vs ρ" meaningful — which is the headline figure for Claim C5.

Usage:
    python exp_purity.py
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, os

from cc_core import pivot, train_cl, cluster_emb, norm_cost
from cc_baselines import local_search, mfp

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TRIALS = 5
N_EPOCHS = 80


# ═════════════════════════════════════════════════════════════════════════════
# Controlled-purity graph generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_controlled_purity(n_nodes, n_clusters, rho, pos_density=0.15,
                                seed=None):
    """
    Generate a CC instance with positive-edge purity exactly = rho.

    Parameters
    ----------
    n_nodes     : int
    n_clusters  : int
    rho         : float in [0, 1]   target fraction of E++ edges that are
                                    within-cluster (purity)
    pos_density : float in [0, 1]   total E++ edges as fraction of all pairs
    seed        : int or None

    Returns
    -------
    edges      : dict (i, j) -> '++' | '--'
    labels     : np.ndarray of GT cluster assignments
    rho_actual : float — achieved ρ (may differ slightly due to rounding
                 or pair-availability limits)
    """
    rng = np.random.default_rng(seed)

    # Balanced clusters: [0,0,..,1,1,..,k-1,k-1,..]
    labels = np.zeros(n_nodes, dtype=int)
    sizes  = [n_nodes // n_clusters] * n_clusters
    for i in range(n_nodes % n_clusters):
        sizes[i] += 1
    idx = 0
    for c, s in enumerate(sizes):
        labels[idx:idx+s] = c
        idx += s

    # Partition all pairs by within / across-cluster
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
    within = [(i, j) for (i, j) in all_pairs if labels[i] == labels[j]]
    across = [(i, j) for (i, j) in all_pairs if labels[i] != labels[j]]

    # Compute target counts. May need clamping if the desired counts exceed
    # the number of available within / across pairs.
    m_total  = int(pos_density * len(all_pairs))
    m_within = int(round(rho * m_total))
    m_across = m_total - m_within
    m_within = min(m_within, len(within))
    m_across = min(m_across, len(across))

    # Sample without replacement
    rng_py = random.Random(int(rng.integers(0, 2**31)))
    pos_within = set(map(tuple, [within[i] for i in
                                  rng_py.sample(range(len(within)), m_within)]))
    pos_across = set(map(tuple, [across[i] for i in
                                  rng_py.sample(range(len(across)), m_across)]))
    pos_edges  = pos_within | pos_across

    edges = {}
    for (i, j) in all_pairs:
        edges[(i, j)] = '++' if (i, j) in pos_edges else '--'

    rho_actual = len(pos_within) / max(len(pos_edges), 1)
    return edges, labels, rho_actual


# ═════════════════════════════════════════════════════════════════════════════
# Single-instance runner
# ═════════════════════════════════════════════════════════════════════════════

def run_one(n, k, edges, gt, seed):
    """Run all four methods on a single instance, return metrics dict."""
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


# ═════════════════════════════════════════════════════════════════════════════
# Experiment driver
# ═════════════════════════════════════════════════════════════════════════════

def exp_purity(n=200, k=4, rho_values=None, pos_density=0.15,
                n_trials=N_TRIALS):
    """
    Sweep ρ from near 0 to 1 and run all four methods at each step.

    Default settings (n=200, k=4) chosen so:
      - Clusters are large enough for CL to learn (~50 nodes each)
      - The problem is hard enough that methods differ meaningfully
      - Total runtime is manageable on a laptop (a few minutes)
    """
    if rho_values is None:
        rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      0.92, 0.95, 0.97, 0.99, 1.0]

    methods = ['pivot', 'ls', 'mfp', 'cl']
    results = {m: {'cost': np.zeros((len(rho_values), n_trials)),
                   'ari':  np.zeros((len(rho_values), n_trials))}
               for m in methods}
    rho_actual = np.zeros((len(rho_values), n_trials))

    print(f"\nExp purity: n={n}, k={k}, pos_density={pos_density}")
    print(f"{'rho_target':>12} {'rho_actual':>12} | "
          + " | ".join(f"{'  '+m:>16}" for m in methods))
    print("-" * 80)

    for ri, rho in enumerate(rho_values):
        for trial in range(n_trials):
            seed = 300 + ri * 1000 + trial * 7
            edges, gt, rho_act = generate_controlled_purity(
                n, k, rho=rho, pos_density=pos_density, seed=seed)
            rho_actual[ri, trial] = rho_act

            r = run_one(n, k, edges, gt, seed)
            for m in methods:
                results[m]['cost'][ri, trial] = r[m]['cost']
                results[m]['ari'][ri, trial]  = r[m]['ari']

        ra = rho_actual[ri].mean()
        print(f"  rho={rho:.2f}  actual={ra:.3f} | " +
              " | ".join(
                  f"ARI={results[m]['ari'][ri].mean():.3f}"
                  for m in methods))

    np.save(f'{OUTPUT_DIR}/exp_purity_results.npy',
            {'results': results, 'rho_actual': rho_actual,
             'rho_targets': rho_values})

    _plot_purity_results(results, rho_actual, n, k, pos_density)
    return results, rho_actual


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

METHOD_STYLES = {
    'pivot': dict(color='#185FA5', marker='o', ls='-',  label='Pivot'),
    'ls':    dict(color='#2E8B57', marker='D', ls='-',  label='Local Search'),
    'mfp':   dict(color='#8B4513', marker='^', ls='-',  label='Veldt MFP'),
    'cl':    dict(color='#D85A30', marker='s', ls='--', label='CL (2-hop)'),
}


def _plot_purity_results(results, rho_actual, n, k, pos_density):
    """Two figures:
       (1) ARI and cost vs ρ for all four methods (the main panel)
       (2) CL advantage (CL ARI − baseline ARI) vs ρ (the single-panel summary)
    """
    rho_means = rho_actual.mean(axis=1)

    # Figure 1: ARI + cost
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Performance vs positive-edge purity ρ\n'
        f'(n={n}, k={k}, pos_density={pos_density})', fontsize=12)

    for ax, metric, ylabel, ylim in [
        (axes[0], 'ari',  'ARI (higher is better)',              (-0.05, 1.05)),
        (axes[1], 'cost', 'Normalized CC cost (lower is better)', None),
    ]:
        for m, sty in METHOD_STYLES.items():
            mean = results[m][metric].mean(axis=1)
            std  = results[m][metric].std(axis=1)
            ax.plot(rho_means, mean,
                    marker=sty['marker'], ls=sty['ls'],
                    color=sty['color'], label=sty['label'], lw=2, ms=5)
            ax.fill_between(rho_means, mean - std, mean + std,
                            alpha=0.12, color=sty['color'])

        ax.set_xlabel('Positive-edge purity  ρ', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.05, 1.05)

    # Vertical line at the ρ=0.5 random-baseline level
    for ax in axes:
        ax.axvline(0.5, color='gray', ls=':', lw=1, alpha=0.6,
                    label='ρ=0.5 (random)')

    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/exp_purity_main.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")

    # Figure 2: CL advantage gap
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.set_title(
        f'CL advantage over baselines vs ρ\n(n={n}, k={k})', fontsize=12)

    cl_ari = results['cl']['ari'].mean(axis=1)
    for m, sty in METHOD_STYLES.items():
        if m == 'cl':
            continue
        gap = cl_ari - results[m]['ari'].mean(axis=1)
        # Standard-deviation envelope, treating CL and baseline as independent
        gstd = np.sqrt(results['cl']['ari'].std(axis=1) ** 2 +
                       results[m]['ari'].std(axis=1) ** 2)
        ax2.plot(rho_means, gap,
                  marker=sty['marker'], ls=sty['ls'],
                  color=sty['color'], label=f'CL − {sty["label"]}', lw=2)
        ax2.fill_between(rho_means, gap - gstd, gap + gstd,
                          alpha=0.10, color=sty['color'])

    ax2.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax2.axvline(0.5, color='gray', ls=':', lw=1, alpha=0.6)
    ax2.set_xlabel('Positive-edge purity  ρ', fontsize=11)
    ax2.set_ylabel('ARI gap  (CL minus baseline)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.05, 1.05)

    save_path2 = f'{OUTPUT_DIR}/exp_purity_advantage.png'
    fig2.tight_layout()
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Advantage figure saved: {save_path2}")


if __name__ == '__main__':
    exp_purity()
