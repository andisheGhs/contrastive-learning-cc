"""
cc_core.py
─────────────────────────────────────────────────────────────────────────────
Core library for the Contrastive-Learning ↔ Correlation-Clustering project.

This file is a *library* — it contains no experiment entry point. The
modules below are imported by every experiment script.

CONTENTS
  1. Graph generation:    generate_sparse_cc_instance
  2. Cost metrics:        cc_cost, norm_cost
  3. Pivot baseline:      pivot
  4. Feature builder:     build_features  (2-hop positive neighborhood)
  5. MLP encoder:         MLP class (3-layer feed-forward, manual backprop)
  6. NT-Xent loss:        nt_xent_vectorized
  7. CL training driver:  train_cl
  8. k-means readout:     cluster_emb

The contrastive-learning model uses dimensions that *scale with k*:
  feat_dim = 4k    (random-projection feature dim)
  d_h      = 8k    (MLP hidden width)
  d_out    = 4k    (embedding output dim)

This is grounded in the spectral-clustering principle that separating
k clusters needs at least k embedding dimensions; we use 4k as a
safety multiplier.
"""

import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import random
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# 1. Graph generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_sparse_cc_instance(n_nodes, n_clusters, density=0.5,
                                 epsilon=0.0, noise=0.02, seed=None):
    """
    Sample a synthetic correlation-clustering instance with a known ground
    truth, suitable for controlled experiments.

    Construction:
      - Nodes are partitioned into n_clusters balanced groups (the GT).
      - For each within-cluster pair (i, j):
          with probability `density` the pair is labeled '++'
          (then with probability `epsilon` we flip it to '--', injecting
           a TC violation)
      - For each across-cluster pair (i, j):
          with probability `noise` it is labeled '++' (a cross-cluster
          positive edge — lowers positive-edge purity)
          otherwise it is labeled '--'

    Parameters
    ----------
    n_nodes    : int
    n_clusters : int
    density    : float in [0, 1]   probability that a within-cluster pair has E++
    epsilon    : float in [0, 1]   probability of flipping a within E++ to E--
                                    (this is the TC-violation knob in Exp 2)
    noise      : float in [0, 1]   probability that a cross-cluster pair is E++
                                    (the cross-cluster positive noise knob)
    seed       : int or None

    Returns
    -------
    edges  : dict (i, j) -> '++' | '--'   (i < j)
    labels : np.ndarray of GT cluster ids
    """
    rng = np.random.default_rng(seed)

    # Build balanced ground-truth labels [0, 0, ..., 1, 1, ..., k-1, ..., k-1]
    labels = np.zeros(n_nodes, dtype=int)
    sizes  = [n_nodes // n_clusters] * n_clusters
    for i in range(n_nodes % n_clusters):
        sizes[i] += 1
    idx = 0
    for c, s in enumerate(sizes):
        labels[idx:idx+s] = c
        idx += s

    # Walk every pair (O(n^2)) and assign a label
    edges = {}
    for i, j in combinations(range(n_nodes), 2):
        same = (labels[i] == labels[j])
        if same:
            # Within-cluster: observe a '++' with prob `density`,
            # then flip to '--' with prob `epsilon` to inject TC violation
            if rng.random() < density:
                edges[(i, j)] = '--' if rng.random() < epsilon else '++'
        else:
            # Across-cluster: rarely positive (cross-cluster noise),
            # otherwise negative
            if rng.random() < noise:
                edges[(i, j)] = '++'
            else:
                edges[(i, j)] = '--'
    return edges, labels


# ═════════════════════════════════════════════════════════════════════════════
# 2. CC cost metrics
# ═════════════════════════════════════════════════════════════════════════════

def cc_cost(edges, assignment):
    """
    Correlation-clustering cost = number of violated edges.

      A '++' edge is violated when its endpoints land in different clusters.
      A '--' edge is violated when its endpoints land in the same cluster.

    Missing edges (zero-weight, E+) contribute nothing.
    """
    cost = 0
    for (i, j), label in edges.items():
        same = (assignment[i] == assignment[j])
        if   label == '--' and     same: cost += 1
        elif label == '++' and not same: cost += 1
    return cost


def norm_cost(edges, assignment):
    """Normalized CC cost — fraction of labeled edges that are violated."""
    return cc_cost(edges, assignment) / max(len(edges), 1)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Pivot baseline (Ailon, Charikar, Newman 2008 — 3-approximation)
# ═════════════════════════════════════════════════════════════════════════════

def pivot(n_nodes, edges, seed=None):
    """
    Greedy randomized Pivot:
      1. Build E++ adjacency map.
      2. Walk nodes in a random order.
      3. When we hit an unclustered node v, create a new cluster
         containing v and all its still-unclustered E++ neighbors.
    Runs in O(|E|). Expected cost is at most 3× OPT.
    """
    rng = random.Random(seed)

    pos_nbrs = defaultdict(set)
    for (i, j), label in edges.items():
        if label == '++':
            pos_nbrs[i].add(j); pos_nbrs[j].add(i)

    remaining = set(range(n_nodes))
    order = list(range(n_nodes)); rng.shuffle(order)
    assignment = np.full(n_nodes, -1, dtype=int)
    cid = 0
    for v in order:
        if v not in remaining:
            continue
        # Pivot v: cluster contains v + all unclustered E++ neighbors
        members = {v} | (pos_nbrs[v] & remaining)
        for u in members:
            assignment[u] = cid
            remaining.discard(u)
        cid += 1
    return assignment


# ═════════════════════════════════════════════════════════════════════════════
# 4. Contrastive-learning feature builder (2-hop positive neighborhood)
# ═════════════════════════════════════════════════════════════════════════════

def build_features(n_nodes, edges, feat_dim, seed=42):
    """
    Build a per-node feature vector summarizing the local positive
    neighborhood structure.

    For each node v, the feature is a concatenation of two summaries:
      F1[v] = [pos_deg(v), neg_deg(v), avg(random_proj(u) for u in pos_nbrs(v))]
      F2[v] = avg(F1[w] for w in 2-hop pos neighbors of v)

    Final feature = concat([F1, F2]).

    Why 2-hop? Triadic closure is a 2-hop property: if u-v and v-w are
    both '++', TC says u-w should not be '--'. By aggregating over
    2-hop positive neighbors, the feature already encodes the TC
    assumption — which is exactly why CL collapses precisely when TC
    is violated (synthetic Experiment 2).

    Parameters
    ----------
    n_nodes  : int
    edges    : dict (i, j) -> '++' | '--'
    feat_dim : int — width of the per-node random projection
                     (scaled with k by the caller, see train_cl)
    seed     : int

    Returns
    -------
    X : np.ndarray of shape (n_nodes, 2*(2 + feat_dim))
    """
    rng = np.random.default_rng(seed)
    # Random per-node identity vector — used because CC has no node features
    proj = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)

    # Build signed adjacency lists
    pos_nbrs = defaultdict(list); neg_nbrs = defaultdict(list)
    for (i, j), label in edges.items():
        if label == '++': pos_nbrs[i].append(j); pos_nbrs[j].append(i)
        else:             neg_nbrs[i].append(j); neg_nbrs[j].append(i)

    # Normalization constants so degree features land in [0, 1]
    max_pos = max((len(v) for v in pos_nbrs.values()), default=1) + 1
    max_neg = max((len(v) for v in neg_nbrs.values()), default=1) + 1

    # F1[v] = 1-hop summary: [pos_deg, neg_deg, avg(proj over pos neighbors)]
    F1 = np.zeros((n_nodes, 2 + feat_dim), dtype=np.float32)
    for v in range(n_nodes):
        F1[v, 0] = len(pos_nbrs[v]) / max_pos
        F1[v, 1] = len(neg_nbrs[v]) / max_neg
        if pos_nbrs[v]:
            F1[v, 2:] = proj[pos_nbrs[v]].mean(axis=0)

    # F2[v] = 2-hop summary: average F1 over the union of pos-neighbors-of-pos-neighbors
    F2 = np.zeros_like(F1)
    for v in range(n_nodes):
        two_hop = set()
        for u in pos_nbrs[v]:
            two_hop.update(pos_nbrs[u])
        two_hop.discard(v)
        if two_hop:
            F2[v] = F1[list(two_hop)].mean(axis=0)

    return np.concatenate([F1, F2], axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 5. MLP encoder (3-layer, manual numpy backprop)
# ═════════════════════════════════════════════════════════════════════════════

def relu(x):
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)

def relu_g(x):
    """ReLU gradient: 1 where x>0, else 0."""
    return (x > 0).astype(np.float32)

def l2norm(x):
    """Row-wise L2 normalization. Places each row on the unit sphere
    so cosine similarity equals dot product."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class MLP:
    """
    Three-layer feed-forward network with ReLU activations and an
    L2-normalized output (places embeddings on the unit hypersphere).

    Hand-coded forward and backward passes so we don't need PyTorch.
    Use He initialization (scale by sqrt(2/d_in)) for stable training.
    """

    def __init__(self, d_in, d_h, d_out, seed=0):
        rng = np.random.default_rng(seed)
        # Layer 1: d_in -> d_h
        self.W1 = rng.standard_normal((d_in, d_h)).astype(np.float32) * np.sqrt(2/d_in)
        self.b1 = np.zeros(d_h, dtype=np.float32)
        # Layer 2: d_h -> d_h
        self.W2 = rng.standard_normal((d_h, d_h)).astype(np.float32) * np.sqrt(2/d_h)
        self.b2 = np.zeros(d_h, dtype=np.float32)
        # Layer 3: d_h -> d_out (no ReLU after this; we L2-normalize instead)
        self.W3 = rng.standard_normal((d_h, d_out)).astype(np.float32) * np.sqrt(2/d_h)
        self.b3 = np.zeros(d_out, dtype=np.float32)

    def fwd(self, X):
        """Forward pass. Caches intermediate activations for backward."""
        self.X  = X
        self.h1 = X @ self.W1 + self.b1;   self.a1 = relu(self.h1)
        self.h2 = self.a1 @ self.W2 + self.b2; self.a2 = relu(self.h2)
        self.h3 = self.a2 @ self.W3 + self.b3
        self.z  = l2norm(self.h3)
        return self.z

    def bwd(self, dz, lr):
        """
        Backward pass: takes the gradient dz of the loss w.r.t. the
        L2-normalized output z, propagates it back through the L2-norm
        and the three layers, then applies a vanilla SGD step.

        The L2-norm gradient term `(dz - (dz·z)*z) / ||h3||` is the
        standard formula for differentiating x / ||x||.
        """
        n = self.X.shape[0]
        nor = np.linalg.norm(self.h3, axis=1, keepdims=True) + 1e-8
        dh3 = (dz - (dz * self.z).sum(1, keepdims=True) * self.z) / nor

        # Layer 3 gradients
        dW3 = self.a2.T @ dh3 / n;   db3 = dh3.mean(0)
        da2 = dh3 @ self.W3.T;       dh2 = da2 * relu_g(self.h2)

        # Layer 2 gradients
        dW2 = self.a1.T @ dh2 / n;   db2 = dh2.mean(0)
        da1 = dh2 @ self.W2.T;       dh1 = da1 * relu_g(self.h1)

        # Layer 1 gradients
        dW1 = self.X.T @ dh1 / n;    db1 = dh1.mean(0)

        # SGD step on all parameters
        for name, grad in [('W1', dW1), ('b1', db1),
                           ('W2', dW2), ('b2', db2),
                           ('W3', dW3), ('b3', db3)]:
            setattr(self, name, getattr(self, name) - lr * grad)


# ═════════════════════════════════════════════════════════════════════════════
# 6. NT-Xent contrastive loss (vectorized)
# ═════════════════════════════════════════════════════════════════════════════

def nt_xent_vectorized(z, pos_idx, neg_idx, T=0.5, max_pos=300):
    """
    SimCLR-style normalized temperature-scaled cross-entropy loss.

    For each anchor i with a designated positive partner j (a '++' edge),
    and a shared pool of negative partners N:

         L_i = -log( exp(z_i·z_j / T)
                     / [ exp(z_i·z_j / T) + sum_{k in N} exp(z_i·z_k / T) ] )

    Intuition: the positive partner's similarity score should dominate
    the soft-max over all candidate partners (positive + negatives).

    Implementation notes:
      - pos_idx is a list of (i, j) anchor/partner pairs (one per '++' edge).
      - To control compute, we subsample at most `max_pos` of them per call.
      - neg_idx is a flat list of node indices used as the shared negatives.
      - We use the log-sum-exp trick (subtract row max) for numerical stability.
      - The gradient w.r.t. z is computed analytically; this is mathematically
        equivalent to autograd through the same loss.

    Returns
    -------
    loss : scalar
    dz   : gradient w.r.t. z, same shape as z
    """
    if len(pos_idx) > max_pos:
        sel = np.random.choice(len(pos_idx), max_pos, replace=False)
        pos_idx = [pos_idx[i] for i in sel]
    if not pos_idx or not neg_idx:
        return 0.0, np.zeros_like(z)

    anc = np.array([p[0] for p in pos_idx], dtype=int)
    prt = np.array([p[1] for p in pos_idx], dtype=int)
    neg_nodes = np.unique(np.array(neg_idx, dtype=int).ravel())

    # Gather embeddings for anchors, partners, and negatives
    Za = z[anc]; Zp = z[prt]; Zn = z[neg_nodes]
    T_inv = 1.0 / T

    # Pairwise similarities (cosine = dot product since embeddings are L2-normed)
    sp = (Za * Zp).sum(1) * T_inv          # (P,)   anchor·positive
    sn = Za @ Zn.T * T_inv                  # (P, N) anchor·negatives

    # Log-sum-exp normalizer (stable form)
    all_s = np.concatenate([sp[:, None], sn], axis=1)
    mx = all_s.max(1, keepdims=True)
    ld = mx.ravel() + np.log(np.exp(all_s - mx).sum(1))
    loss = -(sp - ld).mean()

    # Softmax probabilities, then subtract 1 from the positive column
    # to get the InfoNCE gradient
    sm = np.exp(all_s - mx - ld[:, None]); sm[:, 0] -= 1.0

    # Accumulate gradient w.r.t. z (anchors, partners, negatives)
    P = len(pos_idx)
    dz = np.zeros_like(z)
    dZa = (sm[:, 0:1] * Zp + sm[:, 1:] @ Zn) / (T * P)
    np.add.at(dz, anc, dZa)
    np.add.at(dz, prt, sm[:, 0:1] * Za / (T * P))
    np.add.at(dz, neg_nodes, sm[:, 1:].T @ Za / (T * P))
    return loss, dz


# ═════════════════════════════════════════════════════════════════════════════
# 7. CL training driver
# ═════════════════════════════════════════════════════════════════════════════

def train_cl(n_nodes, edges, k, n_epochs=150, lr=5e-3, T=0.5,
             neg_sample=400, seed=42):
    """
    Train a CL embedding for one correlation-clustering instance.

    Dimensions scale with k:
        feat_dim = 4k     (random-projection feature dim)
        d_h      = 8k     (MLP hidden width)
        d_out    = 4k     (embedding output dim)

    The spectral-clustering principle says we need ≥ k embedding dimensions
    to separate k clusters; we use 4k as a safety multiplier.

    Steps:
      1. Build 2-hop neighborhood features X (see build_features).
      2. Extract positive and negative edge sets; subsample negatives for speed.
      3. Initialize the MLP and run `n_epochs` of full-batch SGD on NT-Xent.
      4. Return the final embedding matrix.
    """
    feat_dim = k * 4
    d_h      = k * 8
    d_out    = k * 4

    X   = build_features(n_nodes, edges, feat_dim=feat_dim, seed=seed)
    pos = [(i, j) for (i, j), l in edges.items() if l == '++']
    neg = [(i, j) for (i, j), l in edges.items() if l == '--']
    rng = random.Random(seed)
    if len(neg) > neg_sample:
        neg = rng.sample(neg, neg_sample)

    # If there are no positives at all, return random embeddings (degenerate
    # case — the loss has no signal to push against)
    if not pos:
        return np.random.default_rng(seed).standard_normal(
            (n_nodes, d_out)).astype(np.float32)

    # Full-batch training loop
    model = MLP(X.shape[1], d_h, d_out, seed=seed)
    for _ in range(n_epochs):
        z = model.fwd(X)
        _, dz = nt_xent_vectorized(z, pos, neg, T=T)
        model.bwd(dz, lr)
    return model.fwd(X)


# ═════════════════════════════════════════════════════════════════════════════
# 8. k-means readout
# ═════════════════════════════════════════════════════════════════════════════

def cluster_emb(emb, k, seed=42):
    """
    Run k-means on the trained embeddings to produce a discrete partition.

    The CC objective requires a discrete clustering, but CL gives us
    continuous embeddings. k-means is the standard readout: nodes whose
    embeddings are close in Euclidean space land in the same cluster.

    `n_init=10` runs k-means 10 times with different random initializations
    and keeps the result with the lowest inertia (sklearn default behavior).
    """
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)
