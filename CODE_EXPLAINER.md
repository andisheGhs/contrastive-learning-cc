# Code Explainer

A technical guide to the codebase accompanying *Contrastive Learning as an
Implicit Solver for Correlation Clustering*. This document defines the
notation used throughout the repository, walks through each algorithm as
implemented in the source files, describes the two real-world datasets and
their cleaning policies, and reports the headline empirical finding.

---

## 1. Overview

The repository tests a single claim: that the SimCLR-style NT-Xent
contrastive loss is implicitly solving correlation clustering (CC) under a
triadic-closure assumption. The same input — a signed graph with positive
and negative edges — is run through four clustering procedures:

1. Pivot (Ailon, Charikar, Newman 2008)
2. Local Search seeded from Pivot
3. Veldt's Match-Flip-Pivot (MFP, ICML 2022)
4. Contrastive Learning (CL): a 3-layer MLP encoder trained with NT-Xent on
   2-hop positive-neighborhood features, followed by a k-means readout

Experiments are run on (a) synthetic graphs in which density, triadic-closure
violation rate, and positive-edge purity are independently controlled and
(b) two real-world data sources: the Epinions signed social network and the
20 Newsgroups text classification benchmark with a similarity-threshold
graph construction.

All algorithms are defined in [cc_core.py](cc_core.py) and
[cc_baselines.py](cc_baselines.py); the experiment drivers are in
[exp_synthetic.py](exp_synthetic.py), [exp_purity.py](exp_purity.py), and
[exp_real_data.py](exp_real_data.py).

---

## 2. Notation and definitions

### 2.1 Correlation clustering instances

A correlation clustering instance is a labeled undirected graph stored as a
Python dictionary:

```python
edges = {
    (0, 1): '++',   # nodes 0 and 1 should be in the same cluster
    (0, 2): '--',   # nodes 0 and 2 should be in different clusters
    (1, 5): '++',
    # pairs not in the dict are zero-weight (no observation)
}
```

Each labeled pair `(i, j)` with `i < j` carries one of two signs:

- `'++'` — positive edge: the endpoints should be co-clustered (E⁺⁺)
- `'--'` — negative edge: the endpoints should be separated (E⁻⁻)

Unlabeled pairs are zero-weight (E⁺ in the 3-label CC formulation) and
contribute nothing to the cost.

### 2.2 Clusterings and the CC cost

A clustering is an integer array of length `n`:

```python
assignment = np.array([0, 0, 1, 1, 2, 0])
# node 0 → cluster 0, node 1 → cluster 0, node 2 → cluster 1, ...
```

Cluster identifiers are arbitrary labels — `[0, 0, 1]` and `[5, 5, 9]`
represent the same clustering.

The correlation clustering cost of an assignment is the number of *violated*
labeled edges: positive edges whose endpoints fall in different clusters,
or negative edges whose endpoints fall in the same cluster. The
implementation in [cc_core.py:108-122](cc_core.py#L108-L122):

```python
def cc_cost(edges, assignment):
    cost = 0
    for (i, j), label in edges.items():
        same = (assignment[i] == assignment[j])
        if   label == '--' and     same: cost += 1
        elif label == '++' and not same: cost += 1
    return cost
```

The normalized cost `norm_cost` divides by `|edges|`, yielding a value in
`[0, 1]`.

### 2.3 Adjusted Rand Index (ARI)

The Adjusted Rand Index compares two clusterings of the same node set. It
is used here whenever a ground-truth clustering is available — i.e., on the
synthetic graphs and on 20 Newsgroups.

For two clusterings `A` and `B`:

1. Enumerate every pair of nodes `(i, j)`.
2. A pair is concordant if `A` and `B` agree on whether the two nodes are
   co-clustered.
3. The Rand Index is `(concordant pairs) / (total pairs)`.
4. The Adjusted Rand Index subtracts the expected Rand Index under random
   labeling. ARI = 1 indicates perfect agreement; ARI = 0 indicates chance
   performance; ARI may take small negative values for worse-than-chance
   clusterings.

The ARI is invariant to cluster relabeling, which is necessary because
clustering algorithms produce groupings without naming them. The
implementation used is `sklearn.metrics.adjusted_rand_score`.

### 2.4 Cosine similarity and the threshold-based graph construction

For two real-valued vectors `u, v`:

$$\operatorname{cos\_sim}(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|} \in [-1, 1].$$

When both vectors are L2-normalized (placed on the unit sphere), this
reduces to the dot product. All embeddings produced by the CL encoder are
L2-normalized for this reason.

20 Newsgroups is a corpus of text documents, not a graph. To produce a CC
instance from it, the pipeline constructs a graph from the pairwise
embedding similarities. For a chosen threshold θ ∈ (0, 1):

1. Embed each document with `sentence-transformers/all-MiniLM-L6-v2`,
   yielding L2-normalized 384-dimensional vectors.
2. Compute the full pairwise similarity matrix `S[i, j] = cos_sim(doc_i, doc_j)`.
3. Sort all upper-triangle pairs by descending similarity. The top θ
   fraction are labeled E⁺⁺; the remaining 1 − θ are labeled E⁻⁻.

The relevant construction is in
[exp_real_data.py:564-575](exp_real_data.py#L564-L575):

```python
upper = [(sims[i, j], i, j) for i in range(n) for j in range(i+1, n)]
upper.sort(reverse=True)
n_pos = int(theta * len(upper))
pos_set = set((i, j) for _, i, j in upper[:n_pos])

edges = {}
for _, i, j in upper:
    edges[(i, j)] = '++' if (i, j) in pos_set else '--'
```

The threshold θ controls how many pairs are positively labeled; varying θ
changes the density of positive edges and, less directly, the rate of
triadic-closure violations.

### 2.5 Positive 1- and 2-hop neighborhoods

For a node `v` in an edge dictionary, the **positive 1-hop neighborhood** is

```text
N⁺₁(v) = { u : (u, v) ∈ E⁺⁺ }.
```

The **positive 2-hop neighborhood** is the union of 1-hop neighborhoods of
the 1-hop neighbors, with `v` removed:

```text
N⁺₂(v) = ⋃_{u ∈ N⁺₁(v)} N⁺₁(u)  \  {v}.
```

The CL feature constructor (Section 3.4.1) summarizes both
neighborhoods. The 2-hop neighborhood is significant because triadic
closure is a 2-hop property: if `u-v ∈ E⁺⁺` and `v-w ∈ E⁺⁺`, then TC asserts
`u-w ∉ E⁻⁻`.

---

## 3. Algorithms

### 3.1 Pivot

A classical randomized 3-approximation for correlation clustering
(Ailon, Charikar, Newman 2008). Implementation:
[cc_core.py:134-164](cc_core.py#L134-L164).

```python
def pivot(n_nodes, edges, seed=None):
    pos_nbrs = defaultdict(set)
    for (i, j), label in edges.items():
        if label == '++': pos_nbrs[i].add(j); pos_nbrs[j].add(i)
    remaining = set(range(n_nodes))
    order = list(range(n_nodes)); rng.shuffle(order)
    assignment = np.full(n_nodes, -1, dtype=int)
    cid = 0
    for v in order:
        if v not in remaining: continue
        members = {v} | (pos_nbrs[v] & remaining)
        for u in members: assignment[u] = cid; remaining.discard(u)
        cid += 1
    return assignment
```

Operation:

1. Build an adjacency map of positive neighbors.
2. Traverse nodes in a uniformly random order.
3. When a still-unclustered node `v` is encountered, form a new cluster
   containing `v` and every positive neighbor of `v` that has not yet been
   assigned.

The procedure runs in `O(|E|)` time and yields a clustering whose expected
cost is at most three times the optimal correlation-clustering cost.

### 3.2 Local Search

A greedy refinement seeded from Pivot's output. Implementation:
[cc_baselines.py:52-143](cc_baselines.py#L52-L143).

```python
def local_search(n_nodes, edges, init_assignment, n_passes=15):
    assignment = init_assignment.copy()
    adj_pos = defaultdict(list); adj_neg = defaultdict(list)
    for (i, j), label in edges.items():
        if label == '++': adj_pos[i].append(j); adj_pos[j].append(i)
        else:             adj_neg[i].append(j); adj_neg[j].append(i)

    for pass_num in range(n_passes):
        improved = False
        for v in range(n_nodes):
            # Try moving v to each adjacent cluster or to its own singleton;
            # keep the move with the lowest local cost.
            ...
        if not improved: break
    return assignment
```

The local cost contribution of node `v` in cluster `c`
([cc_baselines.py:30-50](cc_baselines.py#L30-L50)) is

```text
local_cost(v, c) = |{u ∈ E⁻⁻ neighbors of v : assignment[u] = c}|
                 − |{u ∈ E⁺⁺ neighbors of v : assignment[u] = c}|.
```

At each pass, every node is offered the option of moving to any cluster
that touches it via a labeled edge, or of forming a fresh singleton. Moves
that strictly decrease the local cost are accepted. The procedure halts
either after `n_passes = 15` full sweeps or when no node moves in a full
pass. Convergence is guaranteed because the cost is a non-negative integer
and each accepted move strictly decreases it.

This is the strongest baseline in the experiments. It is not a published
approximation algorithm — it is the natural greedy local optimum and is
included as a reference for what aggressive combinatorial search can
achieve.

### 3.3 Match-Flip-Pivot (MFP)

Veldt's algorithm (ICML 2022) is the one baseline that *explicitly* uses
triadic closure as a preprocessing step. Implementation:
[cc_baselines.py:149-227](cc_baselines.py#L149-L227).

**Step 1.** Enumerate open wedges in the positive subgraph. A wedge `(i, k, j)`
is *open* when `i-k ∈ E⁺⁺`, `k-j ∈ E⁺⁺`, and `i-j ∉ E⁺⁺`. An open wedge is
precisely a site of strong-triadic-closure violation.

**Step 2.** Construct a greedy maximal matching over open wedges. Wedges are
visited in a uniformly random order, and a wedge is accepted only if none
of its three node-pairs is already used by a previously accepted wedge.

**Step 3.** Close the matched wedges by inserting (or flipping to) `'++'` on
the edge `(i, j)` of each matched wedge:

```python
for (i, k, j) in matched_wedges:
    modified_edges[(min(i, j), max(i, j))] = '++'
```

**Step 4.** Run Pivot on the modified edge set.

The matching step yields a lower-bound certificate for MINSTC+ (the
minimum number of edge flips required to satisfy strong triadic closure).
Veldt proves a theoretical 6-approximation for cluster editing; in practice
the algorithm achieves approximation ratios near 2.

### 3.4 Contrastive Learning

The CL pipeline has four components: a feature constructor, a 3-layer MLP
encoder, the NT-Xent contrastive loss, and a k-means readout.

#### 3.4.1 Two-hop feature construction

The CC problem provides no node features, only signed edges. The feature
constructor in [cc_core.py:170-232](cc_core.py#L170-L232) builds a per-node
vector by combining a random per-node "identity" projection with averages
over the positive 1- and 2-hop neighborhoods.

```python
def build_features(n_nodes, edges, feat_dim, seed=42):
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n_nodes, feat_dim))   # random per-node identity
    # ... build positive and negative adjacency lists ...

    F1 = np.zeros((n_nodes, 2 + feat_dim))
    for v in range(n_nodes):
        F1[v, 0] = len(pos_nbrs[v]) / max_pos        # normalized positive degree
        F1[v, 1] = len(neg_nbrs[v]) / max_neg        # normalized negative degree
        if pos_nbrs[v]:
            F1[v, 2:] = proj[pos_nbrs[v]].mean(axis=0)   # avg pos-neighbor projection

    F2 = np.zeros_like(F1)
    for v in range(n_nodes):
        two_hop = set()
        for u in pos_nbrs[v]: two_hop.update(pos_nbrs[u])
        two_hop.discard(v)
        if two_hop: F2[v] = F1[list(two_hop)].mean(axis=0)
```

`F1[v]` summarizes the 1-hop positive neighborhood; `F2[v]` averages `F1`
over the 2-hop positive neighbors. The final feature is the concatenation
`[F1, F2]`, a `2·(2 + feat_dim)`-dimensional vector per node.

Because the construction averages over 2-hop positive neighbors, the
feature already encodes the triadic-closure assumption. This is the
mechanism by which CL is sensitive to TC violation: when TC is violated,
the 2-hop average contains labels that contradict the supervised signal
provided by the loss.

#### 3.4.2 MLP encoder

A 3-layer feed-forward network with ReLU activations and L2-normalized
output. Implemented in numpy with manual backpropagation at
[cc_core.py:252-310](cc_core.py#L252-L310).

```python
class MLP:
    def __init__(self, d_in, d_h, d_out, seed=0):
        # Two hidden layers of size d_h, output of size d_out
        self.W1 = ...; self.W2 = ...; self.W3 = ...
    def fwd(self, X):
        h1 = relu(X @ W1 + b1)
        h2 = relu(h1 @ W2 + b2)
        h3 = h2 @ W3 + b3
        z = l2_normalize(h3)
        return z
    def bwd(self, dz, lr):
        # Manual backprop; SGD update
```

The output `z` is L2-normalized so that every embedding lies on the unit
hypersphere. With unit-norm embeddings the cosine similarity equals the
dot product, which the NT-Xent loss exploits.

#### 3.4.3 NT-Xent loss

Normalized temperature-scaled cross-entropy. For an anchor node `i` with a
positive partner `j` (a node such that `(i, j) ∈ E⁺⁺`) and a set of
negative partners `N_i`:

$$\mathcal{L}_i = -\log \frac{\exp(z_i \cdot z_j / \tau)}{\exp(z_i \cdot z_j / \tau) + \sum_{k \in N_i} \exp(z_i \cdot z_k / \tau)}.$$

Vectorized implementation at
[cc_core.py:317-378](cc_core.py#L317-L378):

```python
def nt_xent_vectorized(z, pos_idx, neg_idx, T=0.5):
    Za = z[anchors]; Zp = z[partners]; Zn = z[negatives]
    sp = (Za * Zp).sum(1) / T       # anchor·positive similarities
    sn = Za @ Zn.T / T              # anchor·negative similarities
    all_s = concat([sp, sn], axis=1)
    ld = logsumexp(all_s, axis=1)
    loss = -(sp - ld).mean()
    # ... gradient w.r.t. z computed analytically ...
```

The log-sum-exp trick is used for numerical stability, and the gradient is
computed analytically rather than via autograd. The proposition in
`docs/proposition.pdf` bounds the gap between positive-pair similarity and
mean negative similarity in terms of this loss, providing the formal link
between NT-Xent minimization and the continuous CC relaxation.

#### 3.4.4 Training driver and k-means readout

The training driver at [cc_core.py:385-428](cc_core.py#L385-L428) ties the
pieces together:

```python
def train_cl(n_nodes, edges, k, n_epochs=150, ...):
    feat_dim = k * 4
    d_h      = k * 8
    d_out    = k * 4

    X = build_features(n_nodes, edges, feat_dim=feat_dim, ...)
    pos = [(i, j) for ... if label == '++']
    neg = [(i, j) for ... if label == '--']
    if len(neg) > 400: neg = rng.sample(neg, 400)   # subsample for speed

    model = MLP(X.shape[1], d_h, d_out, ...)
    for _ in range(n_epochs):
        z = model.fwd(X)
        _, dz = nt_xent_vectorized(z, pos, neg, T=0.5)
        model.bwd(dz, lr=5e-3)
    return model.fwd(X)
```

The output is an `(n_nodes, 4k)` embedding matrix. The discrete clustering
is then obtained by k-means at [cc_core.py:434](cc_core.py#L434):

```python
def cluster_emb(emb, k, seed=42):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)
```

`n_init=10` runs k-means ten times with different random initializations and
returns the solution with the lowest inertia.

---

## 4. Datasets

### 4.1 Epinions signed network

Epinions.com was a consumer-review website (2000–2014). The version of the
dataset distributed by SNAP is a directed signed trust network:

- 131,828 users (nodes)
- 841,372 directed signed relationships. Each edge `u → v` is labeled
  `+1` (user `u` marked `v` as trusted) or `-1` (user `u` marked `v` as
  distrusted).

The dataset has no ground-truth clustering, so on Epinions only CC cost is
reported; ARI is not defined here.

Cleaning steps applied (see Section 6 for details):

- 573 self-loops dropped.
- 2,703 pairs with reciprocal sign disagreement (`u → v = +1`,
  `v → u = -1`) dropped.

The remaining directed edges are symmetrized: pairs with at least one
positive direction (and no disagreement) become `'++'`; all others
`'--'`. Symmetrization yields 711,210 undirected signed edges.

Because the full graph is too large for full-graph CL training, the
experiment extracts 2-hop ego-networks around high-degree nodes. Each
ego-network is treated as one independent CC instance.

### 4.2 20 Newsgroups

A canonical text-classification benchmark, distributed with scikit-learn:
roughly 11,000 short USENET posts across 20 topic categories
(`sci.med`, `comp.graphics`, `rec.sport.hockey`, etc.).

For this study:

- Nodes are documents (400 are sampled).
- Edges are induced by the cosine-similarity threshold construction
  described in Section 2.4. The graph is not an intrinsic property of the
  dataset; it is a function of the embedding model and the threshold θ.
- Ground truth is the document's topic label, so ARI is meaningful here.

Cleaning steps applied (see Section 6):

- Keep only single-topic documents.
- Retain only the top-10 most frequent classes (drops degenerate tail
  classes with very few documents).
- Drop documents shorter than 50 characters.
- Random subsample (seeded) to `max_docs`, instead of taking the first N
  documents in the dataset order.

The role of 20 Newsgroups in the study is as a real-data analog to the
synthetic Experiment 2 (TC-violation sweep): varying θ changes the
graph construction and induces a range of TC-violation rates and positive-
edge densities, allowing a sensitivity analysis on a non-synthetic dataset.

---

## 5. Implementation choices

### 5.1 Dimensions scale with k

The CL encoder uses dimensions that scale with the number of clusters `k`:

```text
feat_dim = 4·k     (random-projection feature dimension)
d_h      = 8·k     (MLP hidden width)
d_out    = 4·k     (embedding output dimension)
```

The rationale is the spectral-clustering principle that separating `k`
clusters with a linear partitioning (such as k-means) requires at least
`k − 1` independent directions in embedding space. Spectral clustering
makes this exact: it uses the `k` smallest non-trivial eigenvectors of the
graph Laplacian. The CL pipeline uses a looser version of the same idea:
scale the output dimension with `k` so the encoder has at least linear
representational capacity in the number of clusters. The multiplier 4 is
empirical and provides a comfortable margin without inflating compute.

### 5.2 k-means as readout

The CL encoder produces continuous embeddings, whereas the CC objective
requires a discrete partition. A readout step is therefore required.
k-means is the natural choice here:

1. The NT-Xent loss encourages within-cluster compactness and
   between-cluster separation in embedding space, which is exactly what
   k-means optimizes (Euclidean within-cluster variance under a partition).
2. It is fast and deterministic up to initialization.
3. It requires an explicit `k`.

The dependency on `k` is the main limitation of this readout. Pivot, Local
Search, and MFP infer their own cluster count from the graph structure; CL
must be told. Section 6.2 describes the k-sweep policy used to make the
comparison fair on signed-network data. Alternative readouts (hierarchical
clustering, DBSCAN, spectral clustering on the embedding similarity matrix)
remain plausible but are not used in the experiments reported here.

---

## 6. Real-data cleaning

The first attempt at running the SNAP signed-network experiments produced
misleading results that traced to data-cleaning issues rather than
algorithmic ones. This section documents the cleaning policy applied to
fix those issues, with code references and quantitative effect.

### 6.1 Self-loops in SNAP signed networks

The original loader at
[exp_real_data.py:70](exp_real_data.py#L70) accepts every line in the SNAP
file. The corrected loader skips self-loops:

```python
if u == v:
    self_loops += 1
    continue
directed_edges[(u, v)] = s
```

Effect on Epinions: 573 self-loops dropped.

### 6.2 Reciprocal sign disagreement in symmetrization

The directed SNAP data contains pairs `(u, v)` with `u → v = +1` and
`v → u = -1`. Symmetrization must choose a policy for these cases. The
original code at [exp_real_data.py:110](exp_real_data.py#L110) used a sign-
sum heuristic that arbitrarily assigned the pair to `'--'`:

```python
label = '++' if sum(signs) > 0 else '--'
```

The corrected version drops disagreeing pairs entirely, treating them as
unobserved (E⁺ zero-weight in the 3-label formulation):

```python
pos = sum(1 for x in signs if x > 0)
neg = sum(1 for x in signs if x < 0)
if pos > 0 and neg > 0:
    dropped_disagreement += 1
    continue
edges[(i, j)] = '++' if pos > 0 else '--'
```

Effect on Epinions: 2,703 pairs dropped.

### 6.3 Ego-network subsampling artifacts

The most consequential issue concerned ego-network extraction. The original
code performed a 2-hop BFS from a chosen center, subsampled uniformly to
the target size, and indexed the kept nodes. Two problems followed:

1. **The center could be dropped.** `rng.sample(visited, max_size)` is
   uniform over `visited`, with no guarantee of retaining the center node.
2. **Subsampled fringe nodes became forced singletons.** A 2-hop BFS from a
   high-degree center fans out to tens of thousands of nodes. Uniformly
   subsampling to 500 retains predominantly 2-hop-fringe nodes whose
   incident edges lead to *other* fringe nodes not kept by the sample, so
   the kept subgraph contains many degree-zero nodes.

The corrected ego-net extractor always retains the center, then drops any
node with degree zero in the kept subgraph:

```python
# Always keep the center
nodes = list(visited)
if len(nodes) > max_size:
    others = [v for v in nodes if v != center_node]
    sampled = rng.sample(others, max_size - 1)
    nodes = [center_node] + sampled

# Drop nodes with degree zero in the kept subgraph
node_set = set(nodes)
kept_edges = {}; sub_deg = defaultdict(int)
for (i, j), label in edges.items():
    if i in node_set and j in node_set:
        kept_edges[(i, j)] = label
        sub_deg[i] += 1; sub_deg[j] += 1
nodes = sorted(v for v in nodes if sub_deg[v] > 0)
```

Effect on Epinions: on six representative high-degree centers at raw target
size n = 200, the cleaned subgraphs contained ≈ 20–25 nodes (roughly 80% of
the original "nodes" were sampling artifacts), and the resulting graph
density increased by approximately two orders of magnitude.

### 6.4 CL k-selection on real signed networks

The original code passed `k_est = min(#positive-components, n // 5)` to the
k-means readout. On Epinions ego-networks this produced `k` values between
40 and 500, but the CC-cost-optimal cluster count on these graphs is `k = 2`
(one trust circle plus singletons).

Pivot, Local Search, and MFP infer their own cluster count from the graph
structure. To make the comparison fair, the corrected `run_one` in
[exp_real_data.py](exp_real_data.py) sweeps CL across a small grid
`k ∈ {2, 5, 10, k_est}` and returns the assignment with the lowest CC cost.
The chosen `k` is logged on every run. On cleaned Epinions ego-networks
the selected `k` is consistently 2.

### 6.5 20 Newsgroups cleaning

Four policies are applied:

- Single-topic documents only (multi-label documents would contaminate ARI).
- Retain only the top-10 most frequent classes.
- Drop documents shorter than 50 characters.
- Random subsample (seeded) to `max_docs = 400`.

These steps produce a balanced experimental setup with 10 well-populated
classes (typical class sizes 31–47 documents at `max_docs = 400`).

### 6.6 Summary of cleaning steps

| Step | Location | Effect on Epinions | Action |
|---|---|---|---|
| Self-loops kept | `load_snap_signed` | 573 spurious edges | Skip `u == v` lines |
| Sign disagreement collapsed | `symmetrize` | 2,703 arbitrary labels | Drop disagreeing pairs |
| Center could be dropped | `extract_ego_network` | Center not guaranteed in sample | Always include center |
| Degree-zero nodes kept | `extract_ego_network` | ~80% sampling artifacts | Drop degree-zero nodes |
| `k_est` too large for CL | `run_one` | CL forced into degenerate partitions | Sweep `k ∈ {2, 5, 10, k_est}`, report best |

After these corrections, the cleaned Epinions results align with the
synthetic Experiment 1 prediction: CL is the second-best method on average,
beats Pivot and MFP across all subgraph sizes, and approaches Local Search.

---

## 7. Repository layout

| File | Purpose |
|---|---|
| [cc_core.py](cc_core.py) | Library: graph generation, Pivot, CL model (features + MLP + NT-Xent + training), k-means readout |
| [cc_baselines.py](cc_baselines.py) | Library: Local Search and Veldt MFP |
| [exp_synthetic.py](exp_synthetic.py) | Unified runner for synthetic Experiments 1–4 |
| [exp_purity.py](exp_purity.py) | Controlled-purity variant of Experiment 3 |
| [exp_real_data.py](exp_real_data.py) | SNAP (Epinions, Slashdot) and 20 Newsgroups experiments, with cleaning |
| [diagnose_signed_subgraph.py](diagnose_signed_subgraph.py) | Per-ego-net structural diagnostic: density, component-size, k-sweep |
| [docs/proposition.pdf](docs/proposition.pdf) | Formal statement and proof of the NT-Xent ↔ CC bound |
| [docs/session_report.tex](docs/session_report.tex) / [docs/session_report.pdf](docs/session_report.pdf) | Writeup of the cleaning iteration and real-data findings |
| [results/](results/) | Output figures and `.npy` result arrays |

---

## 8. Headline finding

The synthetic experiments (proposition + Experiments 1, 2, 3) establish
that CL is implicitly solving correlation clustering under a triadic-
closure assumption. The real-data experiments confirm that, on cleaned
Epinions ego-networks where TC essentially holds, the synthetic prediction
is borne out: CL is consistently the second-best method, ahead of Pivot and
MFP and within a small margin of Local Search. The 20 Newsgroups threshold
sweep provides a complementary sensitivity analysis on text data,
demonstrating that CL degrades smoothly as the graph construction departs
from the regime that respects the underlying class structure.

Local Search remains the strongest method overall on every dataset. The
contribution of this work is therefore a *characterization* of when and why
CL succeeds, not a performance claim that CL outperforms classical CC
algorithms.
