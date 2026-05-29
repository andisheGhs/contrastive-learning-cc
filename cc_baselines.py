"""
cc_baselines.py
─────────────────────────────────────────────────────────────────────────────
Two combinatorial baselines for correlation clustering.

  - local_search : node-move local refinement seeded from Pivot's output.
                   Greedy: each node moves to whichever cluster (or a new
                   singleton) minimizes its local CC-cost contribution.

  - mfp          : Veldt's Match-Flip-Pivot (ICML 2022). Explicitly uses
                   triadic closure: finds a maximal matching of open
                   E++ wedges, flips closing edges to '++' to satisfy
                   strong triadic closure on the matched wedges, then
                   runs Pivot. Practical approximation ratio ~2.

Both functions share the edge-dict format from cc_core:
    edges      : dict (i, j) -> '++' | '--'   (i < j)
    assignment : np.ndarray shape (n_nodes,) of int cluster ids
"""

import numpy as np
import random
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# Local Search
# ═════════════════════════════════════════════════════════════════════════════

def _node_cost_in_cluster(v, cluster_id, assignment, adj_pos, adj_neg):
    """
    Local CC-cost contribution of placing node v into `cluster_id`.

      For each E++ neighbor of v already in this cluster:  contributes -1
      (we *want* same-cluster '++' edges; they reduce CC cost)

      For each E-- neighbor of v already in this cluster:  contributes +1
      (same-cluster '--' edges are violations; they increase CC cost)

    Lower is better. Used by local_search to decide single-node moves.
    """
    cost = 0
    for u in adj_pos[v]:
        if assignment[u] == cluster_id:
            cost -= 1
    for u in adj_neg[v]:
        if assignment[u] == cluster_id:
            cost += 1
    return cost


def local_search(n_nodes, edges, init_assignment, n_passes=15):
    """
    Iterative node-move local search seeded from `init_assignment` (usually
    Pivot's output).

    Each pass walks every node and considers three options:
      1. Stay in the current cluster
      2. Move to any cluster that touches v via at least one signed edge
      3. Form a new singleton cluster

    Apply the move with the lowest local cost. Repeat for up to `n_passes`
    full sweeps or until no node moves in a full pass.

    Because cost is a non-negative integer and each accepted move strictly
    lowers it, the procedure converges in finite time.

    Parameters
    ----------
    n_nodes         : int
    edges           : dict (i, j) -> '++' | '--'
    init_assignment : np.ndarray  starting cluster ids
    n_passes        : int  max number of full sweeps

    Returns
    -------
    assignment : np.ndarray with cluster ids remapped to 0..K-1 contiguous
    """
    assignment = init_assignment.copy()

    # Pre-build signed adjacency lists once (O(|E|))
    adj_pos = defaultdict(list); adj_neg = defaultdict(list)
    for (i, j), label in edges.items():
        if label == '++':
            adj_pos[i].append(j); adj_pos[j].append(i)
        else:
            adj_neg[i].append(j); adj_neg[j].append(i)

    for pass_num in range(n_passes):
        improved = False
        for v in range(n_nodes):
            current_cluster = assignment[v]

            # Candidate target clusters: only those that touch v (keeps the
            # inner loop O(|E|) total, not O(n·k))
            candidate_clusters = {current_cluster}
            for u in adj_pos[v]:
                candidate_clusters.add(assignment[u])
            for u in adj_neg[v]:
                candidate_clusters.add(assignment[u])

            # Also consider v becoming its own singleton cluster
            singleton_id = int(assignment.max()) + 1

            current_cost = _node_cost_in_cluster(
                v, current_cluster, assignment, adj_pos, adj_neg)

            # Search for a better cluster
            best_cost = current_cost
            best_cluster = current_cluster
            for c in candidate_clusters:
                if c == current_cluster:
                    continue
                # Temporarily move v, score, revert
                assignment[v] = c
                cost = _node_cost_in_cluster(v, c, assignment, adj_pos, adj_neg)
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = c
                assignment[v] = current_cluster

            # Check singleton option
            assignment[v] = singleton_id
            singleton_cost = _node_cost_in_cluster(
                v, singleton_id, assignment, adj_pos, adj_neg)
            if singleton_cost < best_cost:
                best_cost = singleton_cost
                best_cluster = singleton_id
            assignment[v] = current_cluster

            # Apply the best move (if any)
            if best_cluster != current_cluster:
                assignment[v] = best_cluster
                improved = True

        if not improved:
            break

    # Remap cluster ids to be 0-indexed contiguous integers
    unique = np.unique(assignment)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[a] for a in assignment])


# ═════════════════════════════════════════════════════════════════════════════
# Veldt Match-Flip-Pivot (MFP)
# ═════════════════════════════════════════════════════════════════════════════

def _build_open_wedges(n_nodes, edges):
    """
    Find every "open wedge" in the E++ subgraph.

    A wedge is a triple (i, k, j) where i-k and k-j are both E++.
    It is *open* when i-j is NOT E++ — i.e., either i-j is E-- or i-j
    is unobserved. Open wedges are exactly the spots where strong
    triadic closure (STC) is being violated.

    Returns a list of (i, k, j) tuples.
    """
    # E++ adjacency + fast (i, j)-pair lookup set
    pos_set = set()
    pos_nbrs = defaultdict(list)
    for (i, j), label in edges.items():
        if label == '++':
            pos_set.add((i, j)); pos_set.add((j, i))
            pos_nbrs[i].append(j); pos_nbrs[j].append(i)

    wedges = []
    for k in range(n_nodes):
        nbrs = pos_nbrs[k]
        # Every pair of pos-neighbors of k forms a wedge centered at k
        for idx_a in range(len(nbrs)):
            for idx_b in range(idx_a + 1, len(nbrs)):
                i, j = nbrs[idx_a], nbrs[idx_b]
                # Open iff (i, j) is NOT a positive edge
                if (min(i, j), max(i, j)) not in pos_set and (i, j) not in pos_set:
                    wedges.append((i, k, j))
    return wedges


def mfp(n_nodes, edges, seed=None):
    """
    Match-Flip-Pivot algorithm (Veldt, ICML 2022).

    Steps:
      1. Enumerate all open E++ wedges (TC-violation sites).
      2. Greedy maximal matching: process wedges in random order; accept a
         wedge only if none of its three node-pairs are already used by an
         earlier accepted wedge. This gives a lower-bound certificate for
         MINSTC+ — the minimum number of edge-flips needed to satisfy STC.
      3. For each matched wedge (i, k, j), set edge (i, j) to '++' (either
         adding it or flipping it from '--'). This closes the open wedge,
         resolving the STC violation on that wedge.
      4. Run Pivot on the modified edge set.

    The matching step yields a lower bound on MINSTC+; the flip+pivot step
    gives a 6-approximation for cluster editing in theory and ~2-approximation
    in practice (Veldt Theorem 4.1).
    """
    rng = random.Random(seed)

    # Step 1
    wedges = _build_open_wedges(n_nodes, edges)
    rng.shuffle(wedges)

    # Step 2: greedy node-pair-disjoint matching
    used_pairs = set()
    matched_wedges = []
    for (i, k, j) in wedges:
        p1 = (min(i, k), max(i, k))
        p2 = (min(k, j), max(k, j))
        p3 = (min(i, j), max(i, j))
        if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
            matched_wedges.append((i, k, j))
            used_pairs.add(p1); used_pairs.add(p2); used_pairs.add(p3)

    # Step 3: close matched wedges by flipping/adding the (i, j) edge to E++
    modified_edges = dict(edges)
    for (i, k, j) in matched_wedges:
        key = (min(i, j), max(i, j))
        modified_edges[key] = '++'

    # Step 4: run Pivot on the modified graph
    # (Import inside the function to avoid a hard module-load-time cycle if
    #  someone later tries to import cc_baselines from cc_core.)
    from cc_core import pivot
    return pivot(n_nodes, modified_edges, seed=seed)


# ═════════════════════════════════════════════════════════════════════════════
# Smoke test (run this file directly to sanity-check the baselines)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from cc_core import (
        generate_sparse_cc_instance, pivot, norm_cost, train_cl, cluster_emb
    )
    from sklearn.metrics import adjusted_rand_score
    import time

    print("Smoke test: 60 nodes, 3 clusters, density=0.7, cross-cluster noise=0.05")
    edges, gt = generate_sparse_cc_instance(
        60, 3, density=0.7, epsilon=0.0, noise=0.05, seed=7)

    pa = pivot(60, edges, seed=7)
    print(f"  Pivot  ARI={adjusted_rand_score(gt, pa):.3f}  "
          f"cost={norm_cost(edges, pa):.4f}")

    t0 = time.time()
    la = local_search(60, edges, pa)
    print(f"  LS     ARI={adjusted_rand_score(gt, la):.3f}  "
          f"cost={norm_cost(edges, la):.4f}  ({time.time()-t0:.2f}s)")

    t0 = time.time()
    ma = mfp(60, edges, seed=7)
    print(f"  MFP    ARI={adjusted_rand_score(gt, ma):.3f}  "
          f"cost={norm_cost(edges, ma):.4f}  ({time.time()-t0:.2f}s)")

    t0 = time.time()
    emb = train_cl(60, edges, k=3, n_epochs=80, seed=7)
    ca = cluster_emb(emb, 3, seed=7)
    print(f"  CL     ARI={adjusted_rand_score(gt, ca):.3f}  "
          f"cost={norm_cost(edges, ca):.4f}  ({time.time()-t0:.2f}s)")
