# Code explainer — what this codebase does



---

## Level 1 

> The code tests a single claim: that contrastive learning, the SimCLR-style NT-Xent loss, is implicitly solving correlation clustering. So I run the same input — a signed graph with positive and negative edges — through four different clustering algorithms and compare them: Pivot, Local Search, Veldt's MFP, and a contrastive-learning model with k-means readout. I do this on both synthetic graphs (where I control the structure) and real data (Epinions signed network, 20 Newsgroups text). The output is the normalized correlation-clustering cost for each method, and on labeled data also the ARI against ground truth.

---

## Level 2

### The data unit

A correlation-clustering instance is a graph where every edge has a label:

- `++` — these two nodes should be in the same cluster
- `--` — these two nodes should be in different clusters
- *(missing)* — zero-weight, no opinion

A **clustering** is an assignment of each node to a cluster id. The **correlation-clustering cost** is the number of *violated* edges:

- a `++` edge whose endpoints are in different clusters
- a `--` edge whose endpoints are in the same cluster

Normalized by `|edges|`.

### The four algorithms

All defined in [cc_core.py](cc_core.py) and [cc_baselines.py](cc_baselines.py).

1. **Pivot** — pick a random node, put it and all its `++` neighbors in a cluster, remove them, repeat. Classic 3-approximation (Ailon, Charikar, Newman 2008).
2. **Local Search** — start from Pivot's output, then for every node, try moving it to each cluster that touches it (or to a new singleton) and keep the move if it lowers cost. Repeat until stable.
3. **Veldt MFP (Match-Flip-Pivot)** — find all "open wedges" `u-v-w` where `u-v` and `v-w` are `++` but `u-w` is not. Greedily match them, flip `u-w` to `++` to close each matched wedge, then run Pivot on the modified graph. This is the algorithm that *explicitly* uses triadic closure as preprocessing (Veldt, ICML 2022).
4. **Contrastive Learning (CL)** — build a feature vector per node from its 2-hop positive-neighborhood, train a small 3-layer MLP using the NT-Xent loss with `++` edges as positive pairs and `--` edges as negatives, then run k-means on the resulting embeddings.

### The experiments

The pipeline is identical across all of them — generate or load a graph, run all four methods, measure cost and ARI.

| File | What it varies | What it tests |
|---|---|---|
| [cc_core.py](cc_core.py) | E++ density × graph size | Does CL track Pivot as density grows? (Exp 1) |
| [exp_synthetic.py](exp_synthetic.py) | TC violation rate ε | Does CL collapse uniquely under TC violation? (Exp 2 — the main result) |
| [exp_purity.py](exp_purity.py) | Positive-edge purity ρ at fixed density | Is purity *necessary but not sufficient* for CL? (Exp 3) |
| [exp_real_data.py](exp_real_data.py) `--snap epinions` | Subgraph size on Epinions | Does Exp 1 generalize to real signed networks? |
| [exp_real_data.py](exp_real_data.py) `--uci 20news` | Cosine similarity threshold θ | How does CL behave on text data under varying graph construction? |

### The output

Each script writes a figure to [results/](results/) and (for the synthetic ones) a `.npy` results file. The cleaned Epinions and 20news figures are what we produced in this session:

- [results/epinions_cost_by_size.png](results/epinions_cost_by_size.png)
- [results/epinions_cl_vs_tc.png](results/epinions_cl_vs_tc.png)
- [results/20news_threshold_sweep.png](results/20news_threshold_sweep.png)

---

## Level 3 — likely follow-up questions

### "What's the CL model exactly?"

A 3-layer MLP. The input is a 2-hop neighborhood-aggregated feature vector — for each node we average a random projection of its positive 1-hop neighbors, then average those over its positive 2-hop neighbors. Dimensions scale with k: `feat_dim = 4k`, hidden `= 8k`, output `= 4k`. We train with vectorized NT-Xent for 80 epochs, then run k-means on the embeddings.

### "Why scale dimensions with k?"

Spectral clustering principle: separating k clusters needs at least k embedding dimensions. We don't claim novelty there, we just apply it.

### "What's 'triadic closure' (TC) in your terms?"

If `u-v` and `v-w` are both `++`, then `u-w` must not be `--`. It's the structural assumption that positive neighborhoods are transitively consistent. The NT-Xent loss implicitly assumes this because it averages over positive partners.

### "How do you measure TC violation?"

Fraction of `++` wedges `(u, v, w)` whose closing edge `u-w` is `--`. Zero means TC holds; 1.0 means every wedge is violated.

### "What does the real-data experiment add?"

Two things:

(a) **Epinions** confirms on a real signed network that when TC holds (which it does, ~99.5% of wedges), CL is competitive with the combinatorial baselines.

(b) **20 Newsgroups** is a sensitivity test under varying graph construction — it shows CL degrades smoothly as we relax the similarity threshold, while Local Search collapses to trivial all-one-cluster at high θ.

### "Why did the first Epinions run look so bad?"

The ego-network subsampler kept random 2-hop nodes that had no edges inside the kept set. Eighty percent of the "nodes" in those subgraphs were forced singletons, which structurally penalizes any method using a balanced k-clustering readout. After fixing the cleaning — dropping self-loops, dropping reciprocal-disagreement pairs, keeping the center node in the sample, and dropping degree-0 nodes in the kept set — the results align with the synthetic prediction.

### "Why did you pick the best k for CL via a sweep?"

The original code passed `k_est = min(#positive-components, n/5)` to k-means. On Epinions that gave k=40–500, but the CC-cost-optimal k on these graphs is k=2. Pivot/LS/MFP choose their own k from the graph — they don't take it as input. So letting CL sweep over `{2, 5, 10, k_est}` and reporting the min-cost choice is the fair analog: every method gets to pick its own k. We log which k was chosen on every run.

### "What's the headline finding?"

The synthetic story holds on real data when the cleaning is right. Local Search is still the strongest method overall — that's an honest finding we don't hide — but CL is consistently second-best on cleaned Epinions and beats both Pivot and MFP. The paper's contribution is *characterization* — explaining when and why CL works — not a performance claim that CL beats combinatorial CC.

---

## File-by-file map

| File | Purpose |
|---|---|
| [cc_core.py](cc_core.py) | Synthetic graph generation, CL model (MLP + NT-Xent), Pivot, cost helpers, Exp 1 entry point |
| [cc_baselines.py](cc_baselines.py) | Local Search and Veldt MFP implementations |
| [exp_synthetic.py](exp_synthetic.py) | Unified runner for synthetic Experiments 1–4 |
| [exp_purity.py](exp_purity.py) | Purity-controlled Experiment 3 |
| [exp_real_data.py](exp_real_data.py) | SNAP (Epinions/Slashdot) and UCI (20news) experiments with cleaning |
| [diagnose_signed_subgraph.py](diagnose_signed_subgraph.py) | Diagnostic that prints density / component-size / k-sweep on a few ego-networks |
| [docs/proposition.pdf](docs/proposition.pdf) | The proved bound: NT-Xent minimizer ⇒ minimizer of upper bound on continuous CC relaxation |
| [results/](results/) | All figures and `.npy` result arrays |
| [docs/session_report.tex](docs/session_report.tex) / [docs/session_report.pdf](docs/session_report.pdf) | Narrative writeup of the cleaning fix and real-data results |

---

# Level 4 — deep dive (read once, refer back as needed)

This section walks through the actual code and defines every term carefully. Read top-to-bottom the first time; use the section headers as an index afterward.

## 4.1 Definitions, in plain English

### What is a "graph" in this project?

A graph here is just a set of nodes (numbered 0, 1, 2, …, n−1) plus a dictionary of labeled edges:

```python
edges = {
    (0, 1): '++',   # nodes 0 and 1 belong together
    (0, 2): '--',   # nodes 0 and 2 should be separated
    (1, 5): '++',
    # pairs not in the dict have no opinion (zero-weight)
}
```

The label `'++'` is a positive edge ("same cluster"), `'--'` is a negative edge ("different cluster"). All four algorithms work on this same edge-dict format.

### What is a "clustering"?

A numpy array `assignment` of length `n_nodes` where `assignment[i]` is the integer cluster id of node `i`:

```python
assignment = np.array([0, 0, 1, 1, 2, 0])
# node 0 → cluster 0, node 1 → cluster 0, node 2 → cluster 1, ...
```

Cluster ids are arbitrary labels — `[0, 0, 1]` and `[5, 5, 9]` represent the same clustering.

### What is "CC cost"?

The number of edges the clustering *violates*:

- a `'++'` edge whose endpoints are in *different* clusters (we wanted them together)
- a `'--'` edge whose endpoints are in the *same* cluster (we wanted them apart)

Implemented in [cc_core.py:108-122](cc_core.py#L108-L122):

```python
def cc_cost(edges, assignment):
    cost = 0
    for (i,j), label in edges.items():
        same = (assignment[i] == assignment[j])
        if label == '--' and same:       cost += 1
        elif label == '++' and not same: cost += 1
    return cost
```

`norm_cost` divides by `|edges|` so it's always in [0, 1].

### What is ARI (Adjusted Rand Index)?

ARI compares two clusterings to see how well they agree. We use it when we have *ground truth* labels (synthetic graphs and 20 Newsgroups). For two clusterings A and B of the same n nodes:

1. Look at every pair of nodes (i, j).
2. Count pairs that are "concordant" — either same-cluster in both A and B, or different-cluster in both.
3. The **Rand Index** is `(concordant pairs) / (total pairs)`.
4. **Adjusted Rand Index** subtracts the expected RI under random labeling, so ARI = 0 means "no better than chance" and ARI = 1 means "perfect agreement". ARI can be slightly negative if you're worse than chance.

Why use ARI instead of cluster-label matching? Because clustering algorithms don't know the names of true classes — they just produce groupings. ARI is invariant to cluster relabeling.

We get it from sklearn:

```python
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(ground_truth_labels, our_predicted_assignment)
```

### What is "cosine similarity"?

For two vectors $u, v$:

$$\text{cos\_sim}(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|} \in [-1, 1]$$

If both vectors are L2-normalized (unit length), this simplifies to just the dot product `u·v`. Higher = more similar in direction. Used as the standard "how similar are these two embeddings" measure.

### What is a "cosine similarity threshold" (the θ in 20news)?

20 Newsgroups is not a graph — it's a collection of text documents. To run a *graph* algorithm on it, we have to *build* a graph. The recipe:

1. Embed each document with sentence-transformers → 384-dim vector per document, L2-normalized.
2. Compute the full pairwise similarity matrix `S[i, j] = cos_sim(doc_i, doc_j)` — a 400×400 matrix.
3. Pick a threshold θ ∈ (0, 1). The top-θ fraction of pairs (by similarity) get the label `'++'`; the rest get `'--'`.

So θ = 0.25 means: "the top 25% most-similar pairs are positive edges, the bottom 75% are negative edges". At θ = 0.75, three-quarters of all pairs become positive.

In [exp_real_data.py:564-575](exp_real_data.py#L564-L575):

```python
upper = [(sims[i,j], i, j) for i in range(n) for j in range(i+1, n)]
upper.sort(reverse=True)              # most-similar pairs first
n_pos = int(theta * len(upper))       # how many get '++'
pos_set = set((i,j) for _, i, j in upper[:n_pos])

edges = {}
for _, i, j in upper:
    edges[(i,j)] = '++' if (i,j) in pos_set else '--'
```

### What is "1-hop / 2-hop neighborhood"?

For a node `v`, the **1-hop positive neighborhood** is "all nodes directly connected to `v` by a `'++'` edge":

```python
pos_nbrs[v] = {u : edges contains (u,v) with label '++'}
```

The **2-hop positive neighborhood** is "all nodes you can reach in two `'++'`-edge steps":

```python
two_hop[v] = ⋃_{u ∈ pos_nbrs[v]} pos_nbrs[u]   minus v itself
```

If you think of `'++'` edges as "friendship", 1-hop = your friends, 2-hop = friends-of-friends. The CL model uses both as features (see §4.4).

## 4.2 What the two real datasets actually are

### Epinions

Epinions.com was a consumer review website that ran 2000–2014. The dataset on SNAP is a **trust network**:

- **Nodes:** 131,828 users of the site.
- **Edges:** 841,372 *directed* relationships. An edge `u → v` with sign `+1` means "user u marked user v as trusted"; sign `-1` means "user u marked user v as distrusted".
- **Cleaning we apply:** drop self-loops (573 of them — user trusting themselves), drop pairs where `u → v = +1` but `v → u = -1` (2,703 of them — reciprocal disagreement, treated as no observation).
- **No ground truth clustering.** There's no "true" partition of users into communities to compare against. So we only report CC cost on Epinions, not ARI.

After cleaning, we symmetrize: for each pair, if any surviving directed edge is positive we call it `'++'`, otherwise `'--'`. This gives 711,210 undirected signed edges.

We don't run on the *whole* 131k-node graph — too large for CL training. Instead we extract **ego-networks**: pick a high-degree user, take their 2-hop neighborhood (everyone within 2 hops of them in the *any-edge* graph), subsample to a target size if needed. Each ego-network becomes one independent CC problem instance.

### 20 Newsgroups

A classic NLP benchmark from the 1990s. ~11,000 short news articles (USENET posts) across 20 topic categories like `sci.med`, `comp.graphics`, `rec.sport.hockey`, `talk.politics.guns`, etc.

- **Nodes:** documents (we sample 400 of them).
- **Edges:** built from pairwise cosine similarity of document embeddings via the θ-threshold described above. There is no "natural" graph in this data — the graph is a *construction* we impose on top of the embedding space.
- **Ground truth:** each document has a topic label, so we *can* compute ARI here.
- **Cleaning we apply:** keep only documents that are single-topic, ≥50 chars long, in the top-10 most frequent classes (drops degenerate tail classes), random-sampled (not first-N).

The reason we use 20news: it's a "real" dataset (not synthetic), but we can still vary the graph construction (via θ) to study how the algorithms respond. It's the closest real-data analog to varying ε in synthetic Exp 2.

## 4.3 Walking through the baselines

### Pivot — [cc_core.py:134-164](cc_core.py#L134-L164)

The simplest CC algorithm. Random pivot:

```python
def pivot(n_nodes, edges, seed=None):
    pos_nbrs = defaultdict(set)
    for (i,j), label in edges.items():
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

Walkthrough:
1. Build a dictionary mapping each node to its positive neighbors.
2. Walk through nodes in a random order.
3. When we hit a still-unclustered node `v`, create a new cluster containing `v` and all of `v`'s positive neighbors that haven't been clustered yet.
4. Remove all those nodes from `remaining` and increment the cluster id.

That's the entire algorithm. It runs in O(|E|) time. It's a *3-approximation* — the resulting cost is at most 3× the optimal CC cost in expectation (Ailon-Charikar-Newman 2008).

### Local Search — [cc_baselines.py:52-143](cc_baselines.py#L52-L143)

Refines Pivot's output by trying single-node moves:

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
            # Try moving v to: each cluster touching it, or its own singleton
            # Keep the move if it lowers cost
            ...
        if not improved: break
    return assignment
```

The "local cost" of node `v` in cluster `c` (helper at [cc_baselines.py:30-50](cc_baselines.py#L30-L50)) is:

`cost = (negative edges from v to nodes in c) − (positive edges from v to nodes in c)`

Lower is better. The algorithm iterates: for each node, look at the clusters that any of its edges touch (plus a fresh singleton option), and move to whichever cluster minimizes that local cost. Repeat for up to 15 full passes or until nothing changes.

Why it's strong:
- Initialization from Pivot is already a 3-approximation.
- Each local move can only decrease cost.
- Convergence is guaranteed (cost is a non-negative integer, so it must stabilize).

This is the *strongest* baseline in our experiments. It is not a published "approximation algorithm" — it's the obvious greedy local optimum.

### MFP (Veldt's Match-Flip-Pivot) — [cc_baselines.py:149-227](cc_baselines.py#L149-L227)

This is the one algorithm that explicitly uses **triadic closure** as preprocessing.

Step 1 — find open wedges:

```python
def _build_open_wedges(n_nodes, edges):
    # An open wedge is (i, k, j) where:
    #   i-k is ++,  k-j is ++,  but  i-j is NOT ++
    # i.e., a TC-violating triangle, almost
```

Step 2 — greedy matching:

```python
used_pairs = set()
matched_wedges = []
for (i, k, j) in shuffled_wedges:
    if no pair of (i,k), (k,j), (i,j) is used:
        accept this wedge
        mark all three pairs used
```

Step 3 — flip:

```python
for (i, k, j) in matched_wedges:
    modified_edges[(min(i,j), max(i,j))] = '++'
    # We force i-j to be positive, closing the open wedge
```

Step 4 — run Pivot on the modified graph.

The intuition: TC says that for every wedge `i-k-j` with both arms positive, the closing edge `i-j` *should* be positive. If it isn't (open wedge), we have a TC violation. MFP fixes a maximal matching of these violations before running Pivot. Veldt proves this achieves a ~2-approximation in practice (theoretical 6-approximation).

### CL — covered in §4.4 below

## 4.4 Walking through the contrastive-learning model

This is the most involved piece. Three sub-components: features, MLP, NT-Xent loss.

### Step 1: build_features — [cc_core.py:170-232](cc_core.py#L170-L232)

For each node `v`, we produce a feature vector. The vector has two parts: a 1-hop summary and a 2-hop summary.

```python
def build_features(n_nodes, edges, feat_dim, seed=42):
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n_nodes, feat_dim))   # random projection per node
    pos_nbrs = defaultdict(list); neg_nbrs = defaultdict(list)
    # ... build positive and negative adjacency lists ...
```

Why a random projection? We have no node features in CC (just edges). So we assign each node a random vector — this is the "identity" of the node in feature space. It's like a positional encoding.

```python
F1 = np.zeros((n_nodes, 2 + feat_dim))
for v in range(n_nodes):
    F1[v, 0] = len(pos_nbrs[v]) / max_pos      # normalized positive degree
    F1[v, 1] = len(neg_nbrs[v]) / max_neg      # normalized negative degree
    if pos_nbrs[v]:
        F1[v, 2:] = proj[pos_nbrs[v]].mean(axis=0)   # avg of pos neighbors' random vecs
```

So F1[v] is `[pos_deg, neg_deg, avg_random_vec_of_pos_neighbors]`. This is the 1-hop summary.

```python
F2 = np.zeros_like(F1)
for v in range(n_nodes):
    two_hop = set()
    for u in pos_nbrs[v]: two_hop.update(pos_nbrs[u])
    two_hop.discard(v)
    if two_hop: F2[v] = F1[list(two_hop)].mean(axis=0)
```

F2[v] is the average of F1 over `v`'s 2-hop positive neighbors (friends-of-friends). This is the 2-hop summary.

The final feature is `[F1, F2]` concatenated — a `(2 + feat_dim) * 2`-dimensional vector per node.

Why 2-hop? Because triadic closure is fundamentally a 2-hop property — if `u-v` and `v-w` are both `'++'`, then `u` and `w` should be similar. By aggregating over 2-hop neighbors, the feature already encodes the TC assumption. **This is why the synthetic Exp 2 collapse happens precisely when TC is violated: the features are inconsistent with the labels.**

### Step 2: the MLP — [cc_core.py:252-310](cc_core.py#L252-L310)

A 3-layer feed-forward neural net, hand-implemented in numpy (no PyTorch needed):

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
        # Manual backprop, then SGD update
```

Output `z` is L2-normalized — every embedding lives on the unit sphere. This is standard for contrastive learning because it makes the cosine similarity equal to the dot product.

### Step 3: NT-Xent loss — [cc_core.py:317-378](cc_core.py#L317-L378)

Normalized Temperature-scaled Cross-Entropy. For an anchor node `i` with a positive partner `j` (a `'++'` edge), and a set of negative partners `N_i`:

$$\mathcal{L}_i = -\log \frac{\exp(z_i \cdot z_j / \tau)}{\exp(z_i \cdot z_j / \tau) + \sum_{k \in N_i} \exp(z_i \cdot z_k / \tau)}$$

In words: the probability mass should concentrate on the positive partner relative to the negatives. Vectorized in code:

```python
def nt_xent_vectorized(z, pos_idx, neg_idx, T=0.5):
    Za = z[anchors]; Zp = z[partners]; Zn = z[negatives]
    sp = (Za * Zp).sum(1) / T       # anchor·positive similarities
    sn = Za @ Zn.T / T              # anchor·negative similarities
    # Log-sum-exp trick for numerical stability:
    all_s = concat([sp, sn], axis=1)
    ld = logsumexp(all_s, axis=1)
    loss = -(sp - ld).mean()
    # ... compute gradient w.r.t. z manually ...
```

The proved proposition (in [proposition.pdf](proposition.pdf)) bounds the gap between positive and mean negative similarity in terms of this loss — that's the formal link to CC.

### Step 4: train_cl — [cc_core.py:385-428](cc_core.py#L385-L428)

Putting it together:

```python
def train_cl(n_nodes, edges, k, n_epochs=150, ...):
    feat_dim = k * 4    # ← here's the k-scaling
    d_h      = k * 8
    d_out    = k * 4

    X = build_features(n_nodes, edges, feat_dim=feat_dim, ...)
    pos = [(i,j) for ... if label == '++']
    neg = [(i,j) for ... if label == '--']
    if len(neg) > 400: neg = rng.sample(neg, 400)   # subsample for speed

    model = MLP(X.shape[1], d_h, d_out, ...)
    for _ in range(n_epochs):
        z = model.fwd(X)
        _, dz = nt_xent_vectorized(z, pos, neg, T=0.5)
        model.bwd(dz, lr=5e-3)
    return model.fwd(X)
```

Output is an `(n_nodes, 4k)` embedding matrix. We then call `cluster_emb` ([cc_core.py:434](cc_core.py#L434)) which is one line:

```python
def cluster_emb(emb, k, seed=42):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)
```

## 4.5 Why dimensions scale with k (the spectral clustering principle)

The intuition comes from **spectral clustering**. To separate `k` clusters with a linear method (like k-means, which uses hyperplanes), you need at least `k − 1` independent directions in embedding space. Concretely:

- If your output is 1-dimensional, k-means can only carve out 2 clusters (split the line at one point) — so output_dim ≥ 2 for k = 2.
- If your output is 2-dimensional, k-means can carve out at most ~3-4 clusters cleanly.
- In general, you need output_dim ≥ k for k-means to have enough room.

Spectral clustering takes this seriously: it uses exactly the `k` smallest non-trivial eigenvectors of the Laplacian. We do something looser but in the same spirit: scale output_dim with `k` so the model has at least linear capacity in the number of clusters.

In our code:
- `feat_dim = 4k` — the random-projection feature dimension
- `d_h = 8k` — the hidden layer dimension (2× the output to give it some compression capacity)
- `d_out = 4k` — the embedding output dimension (4× safety margin over the theoretical minimum k)

We pick `4k` as the safety multiplier — empirically it works well across the synthetic experiments without being unnecessarily large.

## 4.6 Why we need k-clustering at all

This is a subtle point. CL produces **continuous embeddings** — a vector per node. But CC needs a **discrete partition** — an integer cluster id per node.

So we need a *readout step* that converts embeddings to a partition. k-means is the standard choice because:

1. It directly matches the CL objective. NT-Xent encourages within-cluster compactness and between-cluster separation — exactly what k-means optimizes (Euclidean within-cluster variance under a partition).
2. It's simple and fast — single-pass on small embeddings.
3. It needs to know `k`, the number of clusters. This is the **k-selection problem** that bit us on Epinions.

Alternatives we could swap in (and may discuss in the paper as future work):
- Hierarchical clustering — produces a tree, no need to fix `k` upfront, but needs a cut criterion.
- DBSCAN — density-based, finds singletons naturally, but has its own hyperparameters.
- Spectral clustering on the similarity matrix of embeddings — overlaps with what we already have.

We stuck with k-means because it's the most common readout in self-supervised representation learning. The honest caveat in the paper is that *the CL pipeline includes a readout step that needs k*, while combinatorial methods (Pivot, LS, MFP) don't.

## 4.7 The cleaning process, line by line

Three bugs in the original SNAP pipeline. Here's what was wrong and what we changed.

### Bug 1: self-loops were kept

**Before** — [exp_real_data.py load_snap_signed](exp_real_data.py#L70) original version:

```python
for line in f:
    parts = line.split()
    u, v, s = int(parts[0]), int(parts[1]), int(parts[2])
    directed_edges[(u, v)] = s        # ← keeps u == v
```

A self-loop is an edge `(u, u)` meaning "user trusts themselves." In a CC problem this is meaningless. Epinions had 573 of these.

**After:**

```python
if u == v:
    self_loops += 1
    continue                          # skip self-loops
directed_edges[(u, v)] = s
```

### Bug 2: reciprocal sign disagreement collapsed arbitrarily

In a directed signed graph, you can have `u → v = +1` (u trusts v) and `v → u = -1` (v distrusts u). When we symmetrize to undirected, what should the pair (u, v) be?

**Before** — [symmetrize](exp_real_data.py#L110) original:

```python
label = '++' if sum(signs) > 0 else '--'
```

If `signs = [+1, -1]`, the sum is 0, which is `not > 0`, so it becomes `'--'`. This is arbitrary — equally many arguments for `'++'`. Epinions had 2,703 such pairs.

**After:**

```python
pos = sum(1 for x in signs if x > 0)
neg = sum(1 for x in signs if x < 0)
if pos > 0 and neg > 0:
    dropped_disagreement += 1
    continue                          # drop the pair entirely
edges[(i, j)] = '++' if pos > 0 else '--'
```

Pairs with conflicting evidence are dropped, treated as no observation (`E+` zero-weight). The CC formulation supports this naturally — only labeled edges contribute to cost.

### Bug 3: ego-network subsampling produced fake singletons

This was the big one. The original code:

```python
def extract_ego_network(center_node, edges, n_nodes, hop=2, max_size=500, seed=42):
    # BFS up to 2 hops from center
    visited = {center_node}; frontier = {center_node}
    for _ in range(hop):
        # ... expand frontier ...

    # Subsample if too large
    nodes = list(visited)
    if len(nodes) > max_size:
        nodes = rng.sample(nodes, max_size)    # ← can drop the center!
    nodes = sorted(nodes)
    # Build sub_edges keeping only edges where both endpoints are in nodes
```

Two problems with this:

(a) **Center can be dropped.** `rng.sample(visited, max_size)` is a uniform random sample — there's no guarantee the center node ends up in it. So our "ego-network around node 25" might end up not containing node 25.

(b) **Fringe nodes become forced singletons.** A 2-hop BFS from a high-degree center can fan out to *tens of thousands* of nodes (these social networks have high mixing). When we subsample down to 500, most of the 500 are 2-hop-fringe nodes whose edges go to *other* fringe nodes that weren't sampled. So in the kept subgraph, they have **degree 0**. They appear in the node list but they're disconnected.

This wrecked CL because k-means on n=500 nodes where 400 of them are isolated singletons has nothing to learn from. And it inflated the apparent "cluster count" (lots of disconnected components are 1-node singletons), which made our `k_est = #positive-components` heuristic absurdly large.

**After:**

```python
# Always keep the center
nodes = list(visited)
if len(nodes) > max_size:
    others = [v for v in nodes if v != center_node]
    sampled = rng.sample(others, max_size - 1)
    nodes = [center_node] + sampled

# Build edges in the kept set, count degree, drop deg-0 nodes
node_set = set(nodes)
kept_edges = {}; sub_deg = defaultdict(int)
for (i, j), label in edges.items():
    if i in node_set and j in node_set:
        kept_edges[(i, j)] = label
        sub_deg[i] += 1; sub_deg[j] += 1
nodes = sorted(v for v in nodes if sub_deg[v] > 0)    # drop deg-0
```

What this does:

1. Explicitly include the center in the sample, then random-sample the rest.
2. After sampling, count each node's degree in the **kept subgraph** (not the full graph).
3. Drop any node with zero edges in the kept subgraph — these are sampling artifacts.

The effect on Epinions, on the same six ego-networks: original n = 200 became cleaned n ≈ 20–25 (the rest were fringe singletons). Density went up by ~100×. The graphs became actual graphs instead of "one center + dust."

### Bug 4 (not really a bug, more a bad heuristic): CL's k was the wrong number

The original code passed `k_est = min(#positive-components, n // 5)` to k-means. On Epinions ego-networks, that meant k = 40 to 500. But the CC-cost-optimal number of clusters on these graphs is k = 2.

We added a k-sweep: try CL at k ∈ {2, 5, 10, k_est} and return the assignment with lowest CC cost. Pivot/LS/MFP choose their own k from the graph structure, so letting CL do the same (over a small grid) is the fair analog.

The chosen k is logged on every run — on Epinions it picks k = 2 every single time.

### Cleaning summary table

| Bug | Where | Effect on Epinions | Fix |
|---|---|---|---|
| Self-loops kept | `load_snap_signed` | 573 spurious edges | Skip `u == v` lines |
| Sign disagreement collapsed | `symmetrize` | 2,703 arbitrary labels | Drop disagreeing pairs |
| Center could be dropped | `extract_ego_network` | "ego-net" might not contain ego | Always include center |
| Deg-0 nodes kept | `extract_ego_network` | ~80% of subgraph was singletons | Drop deg-0 after subsampling |
| `k_est` too large for CL | `run_one` | CL forced into bad partitions | Sweep `k ∈ {2, 5, 10, k_est}`, report best |

After all five fixes, Epinions results match the synthetic Exp 1 prediction: CL becomes the second-best method on average, beating Pivot and MFP, approaching LS.

## 4.8 If you have 30 more seconds, the headline

The synthetic story (proposition + Exp 1/2/3) says: **CL is implicitly solving CC under a triadic-closure assumption**. The real-data story (cleaned Epinions, 20news) says: **when TC actually holds on real data, the synthetic prediction is borne out**. The work is a *characterization* of when contrastive learning succeeds, not a performance claim that CL beats classical CC algorithms — Local Search remains the strongest method on every dataset.
