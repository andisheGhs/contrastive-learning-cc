"""
diagnose_signed_subgraph.py
─────────────────────────────────────────────────────────────────────────────
Diagnostic script for understanding CL behavior on individual ego-networks
of a SNAP signed graph (default: Epinions).

For each of a handful of high-degree centers, this script:

  1. Extracts and cleans the ego-network around that center.
  2. Prints structural statistics:
       - n, density, |E++|, |E--|, TC violation rate
       - positive-edge connected-component distribution
       - k_est = #pos-components clamped to n/5
  3. Runs Pivot/LS/MFP once for reference.
  4. Sweeps CL across k ∈ {2, 5, 10, 20, k_est} and prints cost at each k.

Intended use:
  When CL surprises you on real signed data, run this first to understand
  whether the issue is structural (e.g., sparsity, singleton dominance) or
  whether it's a k-selection problem (CL prefers small k, k_est too large).

Usage:
  python diagnose_signed_subgraph.py
"""

import numpy as np
from collections import defaultdict

from exp_real_data import (
    load_snap_signed, symmetrize, extract_ego_network,
    compute_tc_violation_rate, SNAP_DEFAULT_FILES, DATA_DIR,
)
from cc_core import pivot, train_cl, cluster_emb, norm_cost
from cc_baselines import local_search, mfp

import os


def diagnostics(sub_edges, n):
    """
    Compute structural diagnostics for a single signed subgraph:
      density, positive-edge fraction, |E++|, |E--|,
      number of positive-edge components, top-5 component sizes,
      number of singleton positive components.
    """
    pos = sum(1 for l in sub_edges.values() if l == '++')
    neg = sum(1 for l in sub_edges.values() if l == '--')
    total_pairs = n * (n - 1) // 2
    density = (pos + neg) / total_pairs if total_pairs else 0.0
    pos_frac = pos / (pos + neg) if (pos + neg) else 0.0

    # Connected components of the positive-edge subgraph
    pos_nbrs = defaultdict(set)
    for (i, j), l in sub_edges.items():
        if l == '++':
            pos_nbrs[i].add(j); pos_nbrs[j].add(i)
    visited = set()
    comps = []
    for s in range(n):
        if s in visited:
            continue
        queue = [s]; visited.add(s); comp = [s]
        while queue:
            v = queue.pop()
            for u in pos_nbrs[v]:
                if u not in visited:
                    visited.add(u); queue.append(u); comp.append(u)
        comps.append(len(comp))
    comps = sorted(comps, reverse=True)
    return {
        'density':              density,
        'pos_frac':             pos_frac,
        'n_pos':                pos,
        'n_neg':                neg,
        'n_components':         len(comps),
        'top5_component_sizes': comps[:5],
        'n_singletons':         sum(1 for c in comps if c == 1),
    }


def main(snap_dataset='epinions', filepath=None):
    """
    Run the diagnostic on 3 centers × 2 target sizes (raw 200 and 500).

    Output is plain text printed to stdout — meant to be read directly
    or grepped for `cl(k=`.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, SNAP_DEFAULT_FILES[snap_dataset])

    centers = [25, 1652, 12168]   # high-degree centers in Epinions
    sizes   = [200, 500]
    k_sweep = [2, 5, 10, 20]

    print(f"Loading {snap_dataset} from {filepath}...")
    directed, nodes = load_snap_signed(filepath)
    edges, _, n_total = symmetrize(directed, nodes)

    for sz in sizes:
        print(f"\n{'='*72}\nSubgraph target size n≈{sz}\n{'='*72}")
        for center in centers:
            seed = 42 + center
            sub_edges, sub_n, _ = extract_ego_network(
                center, edges, n_total, hop=2, max_size=sz, seed=seed)
            if sub_n < 20:
                print(f"center={center}: too small (n={sub_n}), skip")
                continue

            d = diagnostics(sub_edges, sub_n)
            tc = compute_tc_violation_rate(sub_edges, sub_n)
            k_est_raw = d['n_components']
            k_est = max(2, min(k_est_raw, sub_n // 5))

            print(f"\ncenter={center}  n={sub_n}  TC_viol={tc:.4f}")
            print(f"  density={d['density']:.4f}  pos_frac={d['pos_frac']:.3f}"
                  f"  |E++|={d['n_pos']}  |E--|={d['n_neg']}")
            print(f"  pos-components: {d['n_components']}"
                  f"  (singletons={d['n_singletons']},"
                  f"  top5={d['top5_component_sizes']})")
            print(f"  k_est_raw={k_est_raw}  k_est_clamped={k_est}")

            # Reference baseline costs (one run each)
            pa = pivot(sub_n, sub_edges, seed=seed)
            la = local_search(sub_n, sub_edges, pa)
            ma = mfp(sub_n, sub_edges, seed=seed)
            print(f"  pivot={norm_cost(sub_edges, pa):.3f}"
                  f"  ls={norm_cost(sub_edges, la):.3f}"
                  f"  mfp={norm_cost(sub_edges, ma):.3f}")

            # CL k-sweep
            ks = sorted(set(k_sweep + [k_est]))
            for k in ks:
                emb = train_cl(sub_n, sub_edges, k=k, n_epochs=80, seed=seed)
                ca = cluster_emb(emb, k, seed=seed)
                tag = " <- k_est" if k == k_est else ""
                print(f"  cl(k={k:3d}) cost={norm_cost(sub_edges, ca):.3f}{tag}")


if __name__ == '__main__':
    main()
