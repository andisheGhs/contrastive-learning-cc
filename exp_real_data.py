"""
exp_real_data.py
─────────────────────────────────────────────────────────────────────────────
Real-data experiments for the CL ↔ CC project.

Two dataset types:

  1. SNAP signed social networks  (Epinions, Slashdot)
     Already signed: each directed edge u→v carries +1 (trust) or -1 (distrust).
     We clean the data (drop self-loops, drop reciprocal-disagreement pairs),
     symmetrize, then extract ego-network subgraphs for tractable CC instances.
     No ground-truth clustering is available, so we report CC cost only.

  2. 20 Newsgroups text                (text classification benchmark)
     Embed each document with sentence-transformers (all-MiniLM-L6-v2), then
     threshold pairwise cosine similarities: top-θ fraction of pairs become
     E++, the rest E--. Ground truth = document topic label, so we can report
     CC cost AND ARI vs ground truth.

Usage:
    # SNAP (data files in data/)
    python exp_real_data.py --snap epinions
    python exp_real_data.py --snap slashdot

    # 20 Newsgroups (auto-downloads via sklearn on first run)
    python exp_real_data.py --uci 20news

Cleaning policies applied to SNAP data:
    * Drop self-loops (u == v lines).
    * Drop reciprocal-disagreement pairs (u→v=+1 and v→u=-1).
    * In ego-net extraction: always include the center node; drop nodes
      with degree 0 within the kept subgraph (random-subsampling artifacts).

Cleaning policies applied to 20 Newsgroups:
    * Single-topic docs only (multi-label would contaminate ARI).
    * Top-K most frequent classes only (drops degenerate tail classes).
    * Filter texts shorter than 50 chars.
    * Random subsample to max_docs (seeded — not first-N, to avoid order bias).
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, os, sys, argparse
from collections import defaultdict, Counter

from cc_core import pivot, train_cl, cluster_emb, norm_cost
from cc_baselines import local_search, mfp

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.join(ROOT, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default filenames for SNAP datasets (looked up inside DATA_DIR)
SNAP_DEFAULT_FILES = {
    'epinions': 'soc-sign-epinions.txt',
    'slashdot': 'soc-sign-Slashdot090221.txt',
}

N_EPOCHS = 80


# ═════════════════════════════════════════════════════════════════════════════
# SNAP data loading & cleaning
# ═════════════════════════════════════════════════════════════════════════════

def load_snap_signed(filepath, verbose=True):
    """
    Load a SNAP signed network from a tab-separated text file.

    File format: each non-comment line is `fromNode<TAB>toNode<TAB>sign`,
    where sign is +1 (trust) or -1 (distrust). Lines starting with # are
    treated as comments.

    Cleaning: self-loops (u == v) are dropped.

    Returns
    -------
    directed_edges : dict (u, v) -> +1 | -1
    node_list      : sorted list of original node ids
    """
    directed_edges = {}
    self_loops = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v, s = int(parts[0]), int(parts[1]), int(parts[2])
            if u == v:
                self_loops += 1
                continue
            directed_edges[(u, v)] = s

    node_set = set()
    for (u, v) in directed_edges:
        node_set.add(u); node_set.add(v)
    node_list = sorted(node_set)
    if verbose:
        print(f"  Cleaning: dropped {self_loops} self-loops")
    return directed_edges, node_list


def symmetrize(directed_edges, node_list, verbose=True):
    """
    Collapse a directed signed graph into an undirected signed graph.

    Policy for each unordered pair (i, j):
      * All observed directed edges agree in sign  -> keep with that sign
      * Reciprocal directions DISAGREE (+1 and -1) -> drop the pair entirely
        (treated as no observation; becomes E+ zero-weight in the 3-label
         CC formulation — i.e., the pair has no opinion and is ignored in
         the CC cost)

    Returns
    -------
    edges      : dict (i, j) -> '++' | '--'   with re-indexed 0..n-1 nodes
    node_index : dict original_id -> 0-indexed integer
    n          : number of nodes (= len(node_list))
    """
    node_index = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)

    # Group all observed signs by unordered pair
    pair_signs = defaultdict(list)
    for (u, v), s in directed_edges.items():
        i, j = node_index[u], node_index[v]
        key = (min(i, j), max(i, j))
        pair_signs[key].append(s)

    edges = {}
    dropped_disagreement = 0
    for (i, j), signs in pair_signs.items():
        pos = sum(1 for x in signs if x > 0)
        neg = sum(1 for x in signs if x < 0)
        if pos > 0 and neg > 0:
            dropped_disagreement += 1
            continue
        edges[(i, j)] = '++' if pos > 0 else '--'

    if verbose:
        print(f"  Cleaning: dropped {dropped_disagreement} pairs with "
              f"reciprocal sign disagreement")
    return edges, node_index, n


def extract_ego_network(center_node, edges, n_nodes, hop=2, max_size=500,
                        seed=42):
    """
    Extract a subgraph centered on `center_node` by BFS up to `hop` hops
    over ALL edges (positive and negative).

    Cleaning:
      * The center node is ALWAYS retained, even when subsampling.
      * After subsampling, any node with degree 0 in the kept subgraph is
        dropped (it's a random-subsampling artifact — its real edges all
        point to nodes we didn't keep).

    Returns
    -------
    sub_edges : dict (i, j) -> label, with nodes re-indexed to 0..m-1
    sub_n     : number of nodes in the cleaned subgraph
    node_map  : dict old_index -> new_index
    """
    rng = random.Random(seed)

    # Full undirected adjacency (over all signed edges)
    adj = defaultdict(set)
    for (i, j) in edges:
        adj[i].add(j); adj[j].add(i)

    # BFS to collect the `hop`-hop neighborhood of the center
    visited = {center_node}
    frontier = {center_node}
    for _ in range(hop):
        next_frontier = set()
        for v in frontier:
            for u in adj[v]:
                if u not in visited:
                    visited.add(u)
                    next_frontier.add(u)
        frontier = next_frontier

    # Subsample if the visited set is too large — but ALWAYS keep the center
    nodes = list(visited)
    if len(nodes) > max_size:
        others = [v for v in nodes if v != center_node]
        sampled = rng.sample(others, max_size - 1)
        nodes = [center_node] + sampled

    # Collect edges between kept nodes; count degrees within the kept set
    node_set = set(nodes)
    kept_edges = {}
    sub_deg = defaultdict(int)
    for (i, j), label in edges.items():
        if i in node_set and j in node_set:
            kept_edges[(i, j)] = label
            sub_deg[i] += 1; sub_deg[j] += 1

    # Drop degree-0 nodes (sampling artifacts)
    nodes = sorted(v for v in nodes if sub_deg[v] > 0)
    node_map = {v: i for i, v in enumerate(nodes)}
    node_set = set(nodes)

    # Re-index surviving edges to compact 0..m-1 ids
    sub_edges = {}
    for (i, j), label in kept_edges.items():
        if i in node_set and j in node_set:
            ni, nj = node_map[i], node_map[j]
            sub_edges[(min(ni, nj), max(ni, nj))] = label

    return sub_edges, len(nodes), node_map


def compute_tc_violation_rate(edges, n_nodes):
    """
    Fraction of E++ wedges that are TC-violating.

    A wedge centered at v is a pair (u, w) such that both u-v and v-w are
    in E++. It is TC-violating exactly when u-w is in E-- (the closing
    edge contradicts the triadic-closure assumption).

    Returns 0 when no wedges exist; values close to 0 mean TC holds.
    """
    pos_nbrs = defaultdict(set)
    neg_set = set()
    for (i, j), label in edges.items():
        if label == '++':
            pos_nbrs[i].add(j); pos_nbrs[j].add(i)
        else:
            neg_set.add((i, j)); neg_set.add((j, i))

    total_wedges = 0
    violated = 0
    for v in range(n_nodes):
        nbrs = list(pos_nbrs[v])
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                u, w = nbrs[ai], nbrs[bi]
                total_wedges += 1
                key = (min(u, w), max(u, w))
                if key in neg_set or (u, w) in neg_set:
                    violated += 1

    if total_wedges == 0:
        return 0.0
    return violated / total_wedges


# ═════════════════════════════════════════════════════════════════════════════
# Single-instance runner (used by both SNAP and UCI experiments)
# ═════════════════════════════════════════════════════════════════════════════

def run_one(n, k, edges, gt, seed, has_gt=True, cl_k_grid=None):
    """
    Run all four methods on one CC instance.

    cl_k_grid:
      * If None, CL is run once at the given k.
      * If a list of integers, CL is trained at each k in the list and the
        assignment with the lowest CC cost is reported. This is the fair
        analog of letting Pivot/LS/MFP pick their own k from the graph —
        every method gets some freedom in cluster count.

    `gt` may be None (SNAP has no ground truth); in that case ARI is omitted.
    The chosen CL k is recorded under results['cl']['k_chosen'].
    """
    pa = pivot(n, edges, seed=seed)
    la = local_search(n, edges, pa)
    ma = mfp(n, edges, seed=seed)

    if cl_k_grid is None:
        cl_k_grid = [k]
    # Sanitize: integer, deduped, sorted, within valid range
    cl_k_grid = sorted(set(int(x) for x in cl_k_grid if 2 <= int(x) <= n))

    best_cl_cost = float('inf')
    best_cl_assignment = None
    best_cl_k = cl_k_grid[0]
    for k_try in cl_k_grid:
        emb = train_cl(n, edges, k=k_try, n_epochs=N_EPOCHS, seed=seed)
        ca = cluster_emb(emb, k_try, seed=seed)
        c = norm_cost(edges, ca)
        if c < best_cl_cost:
            best_cl_cost = c
            best_cl_assignment = ca
            best_cl_k = k_try

    results = {}
    for name, assignment in [('pivot', pa), ('ls', la), ('mfp', ma),
                              ('cl', best_cl_assignment)]:
        results[name] = {'cost': norm_cost(edges, assignment)}
        if has_gt and gt is not None:
            results[name]['ari'] = adjusted_rand_score(gt, assignment)
    results['cl']['k_chosen'] = best_cl_k
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SNAP experiment driver
# ═════════════════════════════════════════════════════════════════════════════

def run_snap_experiment(filepath, dataset_name, sizes=(1000, 2500, 5000),
                        n_ego_nets=5, n_trials=2):
    """
    Run the signed-network experiment on one SNAP dataset.

    For each target subgraph size:
      1. Pick the top-degree nodes in the full graph as ego-network centers.
      2. Extract a 2-hop ego-network around each center (with cleaning).
      3. Run Pivot, LS, MFP, and CL (with k-sweep) on each subgraph.
      4. Save bar-chart (cost by size) + scatter (CL gap vs TC violation).

    Parameters
    ----------
    filepath     : path to SNAP .txt file
    dataset_name : 'epinions' or 'slashdot'  (used for filenames + log labels)
    sizes        : tuple of target subgraph sizes (raw, before cleaning)
    n_ego_nets   : how many distinct ego-network centers to use per size
    n_trials     : how many trials per (size, center)
    """
    print(f"\nLoading {dataset_name} from {filepath}...")
    directed_edges, node_list = load_snap_signed(filepath)
    edges, node_index, n_total = symmetrize(directed_edges, node_list)

    print(f"  {n_total} nodes, {len(edges)} edges after symmetrization")
    print(f"  Positive: {sum(1 for l in edges.values() if l == '++')}, "
          f"Negative: {sum(1 for l in edges.values() if l == '--')}")

    # Use the highest-degree nodes as ego-network centers
    degree = defaultdict(int)
    for (i, j) in edges:
        degree[i] += 1; degree[j] += 1
    top_nodes = sorted(degree, key=lambda v: -degree[v])[:n_ego_nets * 2]

    all_results  = {sz: [] for sz in sizes}
    all_tc_rates = {sz: [] for sz in sizes}

    for sz in sizes:
        print(f"\n  Subgraph size ~ {sz}:")
        centers_used = 0

        for center in top_nodes:
            if centers_used >= n_ego_nets:
                break

            for trial in range(n_trials):
                seed = 42 + centers_used * 100 + trial
                sub_edges, sub_n, _ = extract_ego_network(
                    center, edges, n_total,
                    hop=2, max_size=sz, seed=seed)

                # Skip subgraphs that came back too small to learn from
                if sub_n < 20:
                    continue

                # Estimate k from positive-edge connected components
                # (clamped to sub_n // 5 so it stays reasonable)
                pos_nbrs = defaultdict(set)
                for (i, j), label in sub_edges.items():
                    if label == '++':
                        pos_nbrs[i].add(j); pos_nbrs[j].add(i)
                visited = set()
                k_est = 0
                for start in range(sub_n):
                    if start in visited:
                        continue
                    queue = [start]; visited.add(start); k_est += 1
                    while queue:
                        v = queue.pop()
                        for u in pos_nbrs[v]:
                            if u not in visited:
                                visited.add(u); queue.append(u)
                k_est = max(2, min(k_est, sub_n // 5))

                tc_rate = compute_tc_violation_rate(sub_edges, sub_n)
                cl_grid = sorted({2, 5, 10, k_est})
                r = run_one(sub_n, k_est, sub_edges, None, seed,
                            has_gt=False, cl_k_grid=cl_grid)

                all_results[sz].append(r)
                all_tc_rates[sz].append(tc_rate)

                print(f"    center={center} n={sub_n} k_est={k_est} "
                      f"cl_k*={r['cl']['k_chosen']} "
                      f"TC_violation={tc_rate:.3f} | "
                      + " | ".join(f"{m}={r[m]['cost']:.3f}"
                                   for m in ['pivot', 'ls', 'mfp', 'cl']))
            centers_used += 1

    _plot_snap_results(all_results, all_tc_rates, sizes, dataset_name)
    return all_results, all_tc_rates


def _plot_snap_results(all_results, all_tc_rates, sizes, dataset_name):
    """
    Two figures saved per SNAP run:
      1. Bar chart: mean normalized CC cost per method, grouped by subgraph size.
      2. Scatter: per-ego-net CL cost minus baseline cost, vs TC violation rate.
    """
    methods = ['pivot', 'ls', 'mfp', 'cl']
    colors = {'pivot': '#185FA5', 'ls': '#2E8B57',
              'mfp': '#8B4513', 'cl': '#D85A30'}

    # Figure 1: cost by size (bar chart)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sizes))
    width = 0.2
    for mi, m in enumerate(methods):
        means, stds = [], []
        for sz in sizes:
            vals = [r[m]['cost'] for r in all_results[sz] if m in r]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.bar(x + mi * width, means, width, label=m.upper(),
               color=colors[m], alpha=0.8, yerr=stds, capsize=3)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'n≈{sz}' for sz in sizes])
    ax.set_ylabel('Normalized CC cost (lower is better)')
    ax.set_title(f'{dataset_name}: CC cost by subgraph size')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path1 = f'{OUTPUT_DIR}/{dataset_name}_cost_by_size.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")

    # Figure 2: CL cost gap vs TC violation rate (scatter)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.set_title(
        f'{dataset_name}: CL vs baseline cost gap vs TC violation rate\n'
        f'(positive = CL worse than baseline)', fontsize=11)

    all_tc, all_gap_ls, all_gap_pivot = [], [], []
    for sz in sizes:
        for r, tc in zip(all_results[sz], all_tc_rates[sz]):
            if 'cl' in r and 'ls' in r and 'pivot' in r:
                all_tc.append(tc)
                all_gap_ls.append(r['cl']['cost'] - r['ls']['cost'])
                all_gap_pivot.append(r['cl']['cost'] - r['pivot']['cost'])

    ax2.scatter(all_tc, all_gap_ls, color='#2E8B57', alpha=0.6,
                label='CL − Local Search', s=30)
    ax2.scatter(all_tc, all_gap_pivot, color='#185FA5', alpha=0.6,
                label='CL − Pivot', s=30, marker='^')
    ax2.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax2.set_xlabel('TC violation rate', fontsize=11)
    ax2.set_ylabel('CC cost gap (CL minus baseline)', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    path2 = f'{OUTPUT_DIR}/{dataset_name}_cl_vs_tc.png'
    fig2.tight_layout()
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close('all')


# ═════════════════════════════════════════════════════════════════════════════
# UCI text experiment driver
# ═════════════════════════════════════════════════════════════════════════════

def run_uci_experiment(dataset_name, thresholds=(0.25, 0.50, 0.75),
                       max_docs=400, n_trials=3, top_k_classes=10,
                       min_text_chars=50, sample_seed=42):
    """
    Run the text-similarity-threshold experiment on a UCI text dataset.

    Steps:
      1. Load the dataset (currently supports '20news' via sklearn,
         'victorian' via HuggingFace datasets).
      2. Apply cleaning: single-topic docs, top-K classes, min length,
         random subsample to max_docs.
      3. Embed docs with sentence-transformers all-MiniLM-L6-v2.
      4. For each θ in `thresholds`, label the top-θ fraction of pairs
         (by cosine similarity) as E++, the rest as E--.
      5. Run all four methods at the ground-truth k; report CC cost + ARI.
      6. Save a 2-panel figure (ARI and cost vs θ).

    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install sentence-transformers")
        return None

    print(f"\nLoading {dataset_name} dataset...")

    # ── Step 1: load + initial filter ────────────────────────────────────────
    if dataset_name == '20news':
        from sklearn.datasets import fetch_20newsgroups
        # Strip headers/footers/quotes so topic isn't trivially leaked
        ds = fetch_20newsgroups(subset='train',
                                remove=('headers', 'footers', 'quotes'),
                                random_state=sample_seed)
        candidates = []
        for txt, lbl_idx in zip(ds.data, ds.target):
            if not txt or len(txt) < min_text_chars:
                continue
            # Truncate to 500 chars — leading text in news posts is dense
            candidates.append((txt[:500], ds.target_names[lbl_idx]))
    elif dataset_name == 'victorian':
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: pip install datasets")
            return None
        ds = load_dataset('rceborg/victorian-authors-dataset',
                          trust_remote_code=True)
        candidates = [(x['text'][:500], x['author']) for x in ds['train']
                      if x['text'] and len(x['text']) >= min_text_chars]
    else:
        print(f"Unknown dataset: {dataset_name}. Use '20news' or 'victorian'.")
        return None

    print(f"  After single-topic + text-length filter: {len(candidates)} docs")

    # ── Step 2: cleaning ─────────────────────────────────────────────────────
    label_counts = Counter(lbl for _, lbl in candidates)
    keep_labels = {l for l, _ in label_counts.most_common(top_k_classes)}
    candidates = [(t, l) for (t, l) in candidates if l in keep_labels]
    print(f"  After top-{top_k_classes} class filter: {len(candidates)} docs"
          f"  ({len(keep_labels)} classes)")

    rng = random.Random(sample_seed)
    if len(candidates) > max_docs:
        candidates = rng.sample(candidates, max_docs)
    print(f"  After random subsample to max_docs: {len(candidates)} docs")

    texts = [t for t, _ in candidates]
    labels_raw = [l for _, l in candidates]

    # Map string class labels to integers; the ground-truth k is just |classes|
    unique_labels = sorted(set(labels_raw))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    gt = np.array([label_map[l] for l in labels_raw])
    k  = len(unique_labels)
    n  = len(texts)
    class_dist = Counter(labels_raw).most_common()
    print(f"  Final: {n} documents, {k} classes")
    print(f"  Class sizes: " + ", ".join(f"{l}:{c}" for l, c in class_dist))

    # ── Step 3: embed + pairwise similarity ──────────────────────────────────
    print("  Embedding with sentence-transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True,
                              normalize_embeddings=True)
    # Embeddings are L2-normalized, so dot product == cosine similarity
    print("  Computing pairwise similarities...")
    sims = embeddings @ embeddings.T   # (n, n) cosine-sim matrix

    # ── Step 4 & 5: threshold sweep ──────────────────────────────────────────
    all_results = {theta: [] for theta in thresholds}
    for theta in thresholds:
        print(f"\n  Threshold theta={theta}:")
        for trial in range(n_trials):
            seed = 42 + trial

            # Sort pairs by similarity; top θ fraction become E++
            upper = [(sims[i, j], i, j)
                     for i in range(n) for j in range(i+1, n)]
            upper.sort(reverse=True)
            n_pos = int(theta * len(upper))
            pos_set = set((i, j) for _, i, j in upper[:n_pos])

            edges = {}
            for _, i, j in upper:
                edges[(i, j)] = '++' if (i, j) in pos_set else '--'

            tc_rate = compute_tc_violation_rate(edges, n)
            # UCI has ground-truth k; use it (no k-sweep here — would distort
            # ARI which compares against a fixed reference clustering)
            r = run_one(n, k, edges, gt, seed, has_gt=True)
            r['tc_rate'] = tc_rate
            all_results[theta].append(r)

            print(f"    trial={trial} TC_violation={tc_rate:.3f} | "
                  + " | ".join(
                      f"{m}: ARI={r[m].get('ari',0):.3f} cost={r[m]['cost']:.3f}"
                      for m in ['pivot', 'ls', 'mfp', 'cl']))

    _plot_uci_results(all_results, thresholds, dataset_name, k)
    return all_results


def _plot_uci_results(all_results, thresholds, dataset_name, k):
    """Save a 2-panel figure: ARI (left) and CC cost (right) vs θ."""
    methods = ['pivot', 'ls', 'mfp', 'cl']
    colors = {'pivot': '#185FA5', 'ls': '#2E8B57',
              'mfp': '#8B4513', 'cl': '#D85A30'}
    markers = {'pivot': 'o', 'ls': 'D', 'mfp': '^', 'cl': 's'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset_name} (k={k}): Performance vs similarity threshold θ',
                 fontsize=12)

    for ax, metric, ylabel, ylim in [
        (axes[0], 'ari',  'ARI (higher is better)',              (-0.05, 1.05)),
        (axes[1], 'cost', 'Normalized CC cost (lower is better)', None),
    ]:
        for m in methods:
            means, stds = [], []
            for theta in thresholds:
                vals = [r[m][metric] for r in all_results[theta]
                        if metric in r[m]]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            ax.plot(thresholds, means, marker=markers[m], color=colors[m],
                    label=m.upper(), lw=2, ms=7)
            ax.fill_between(thresholds,
                             np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds),
                             alpha=0.12, color=colors[m])
        ax.set_xlabel('Similarity threshold  θ', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(thresholds)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/{dataset_name}_threshold_sweep.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close('all')


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_snap_filepath(dataset_name, explicit_file):
    """
    Resolve the SNAP data file path. If --file was passed, use it.
    Otherwise look up the default filename in data/.
    """
    if explicit_file:
        return explicit_file
    default = SNAP_DEFAULT_FILES.get(dataset_name)
    if default is None:
        return None
    return os.path.join(DATA_DIR, default)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap', choices=['epinions', 'slashdot'],
                        help='Run SNAP signed-network experiment')
    parser.add_argument('--file', type=str,
                        help='Path to SNAP .txt file '
                             '(default: data/soc-sign-{dataset}.txt)')
    parser.add_argument('--uci', choices=['20news', 'victorian'],
                        help='Run UCI text experiment')
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[1000, 2500, 5000],
                        help='Subgraph sizes (raw, before cleaning) for SNAP')
    parser.add_argument('--n-ego-nets', type=int, default=5,
                        help='Ego networks per size (SNAP)')
    parser.add_argument('--n-trials', type=int, default=2,
                        help='Trials per (size, ego-net) (SNAP)')
    parser.add_argument('--max-docs', type=int, default=400,
                        help='Maximum documents for UCI experiment')
    args = parser.parse_args()

    if args.snap:
        filepath = _resolve_snap_filepath(args.snap, args.file)
        if not filepath or not os.path.exists(filepath):
            print(f"ERROR: SNAP file not found: {filepath}")
            print(f"  Place soc-sign-{args.snap}.txt in {DATA_DIR}/,")
            print(f"  or pass --file <path>. "
                  f"Download from https://snap.stanford.edu/data/")
            sys.exit(1)
        run_snap_experiment(filepath, args.snap,
                            sizes=tuple(args.sizes),
                            n_ego_nets=args.n_ego_nets,
                            n_trials=args.n_trials)

    elif args.uci:
        run_uci_experiment(args.uci, max_docs=args.max_docs)

    else:
        print("Usage:")
        print("  SNAP: python exp_real_data.py --snap epinions")
        print("  UCI:  python exp_real_data.py --uci 20news")
