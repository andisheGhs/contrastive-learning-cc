# Contrastive Learning as an Implicit Solver for Correlation Clustering

Code accompanying the paper *Contrastive Learning as an Implicit Solver for
Correlation Clustering* (target venue: ICLR 2027).

The repo studies the structural link between the **NT-Xent contrastive loss**
(SimCLR-style) and the **correlation clustering (CC) objective**. The central
claim is that the NT-Xent loss is structurally equivalent to a soft relaxation
of CC under a *triadic-closure* assumption, and this can be tested both
theoretically (see `docs/proposition.pdf`) and empirically (the experiments in
this repo).

For a deeper conceptual walk-through of the codebase — what each algorithm
does, what the datasets actually are, why dimensions scale with k, and what
the cleaning steps do — see [CODE_EXPLAINER.md](CODE_EXPLAINER.md).

---

## Repository layout

```
.
├── README.md                       ← you are here
├── CODE_EXPLAINER.md               ← deep dive on the code
├── cc_core.py                      ← library: graph gen, Pivot, CL model
├── cc_baselines.py                 ← library: Local Search, Veldt MFP
├── exp_synthetic.py                ← Experiments 1, 2, 3, 4 (synthetic)
├── exp_purity.py                   ← Experiment 3, controlled-ρ variant
├── exp_real_data.py                ← SNAP signed networks + 20 Newsgroups
├── diagnose_signed_subgraph.py     ← per-ego-net diagnostic
├── data/                           ← SNAP raw .txt files
│   ├── soc-sign-epinions.txt
│   └── soc-sign-Slashdot090221.txt
├── results/                        ← .png figures and .npy result arrays
└── docs/                           ← proposition.pdf, session_report.pdf, ...
```

Files split into two roles:

- **Library modules (no `__main__`).** Imported by every experiment.
  - `cc_core.py` — graph generation, CC cost, Pivot, CL model
    (`build_features` + `MLP` + `nt_xent_vectorized` + `train_cl`),
    and `cluster_emb` k-means readout.
  - `cc_baselines.py` — `local_search` and Veldt `mfp`.

- **Experiment scripts (run directly).** Each writes figures + .npy to
  `results/`.
  - `exp_synthetic.py` — the four synthetic sweeps (density, TC violation,
    purity, scaling).
  - `exp_purity.py` — the cleaner controlled-purity variant of Exp 3.
  - `exp_real_data.py` — SNAP signed networks (Epinions, Slashdot) and the
    20 Newsgroups similarity-graph experiment.
  - `diagnose_signed_subgraph.py` — verbose per-ego-net dump (density,
    component sizes, CL across k) for sanity-checking CL behavior on
    real signed data.

---

## Installation

The code is plain Python plus `numpy`, `scikit-learn`, and `matplotlib`. The
20 Newsgroups experiment also needs `sentence-transformers`.

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scikit-learn matplotlib sentence-transformers
```

(20 Newsgroups itself is bundled with scikit-learn — no separate download.)

---

## Running the experiments

### Synthetic experiments (Experiments 1, 2, 3, 4)

```bash
python exp_synthetic.py             # run all four
python exp_synthetic.py --exp 2     # run only Experiment 2 (TC violation)
python exp_synthetic.py --exp 1 4   # run Experiments 1 and 4
```

Saves to `results/`:
- `exp1_all_baselines.png`, `exp1_results.npy`     (density × size)
- `exp2_n160.png`, `exp2_n500.png`, `exp2_results.npy` (TC violation)
- `exp3_purity.png`, `exp3_results.npy`             (purity via noise)
- `exp4_scaling.png`, `exp4_results.npy`            (scaling at fixed k)

### Controlled-purity Experiment 3

```bash
python exp_purity.py
```

Cleaner variant that constructs instances with exact target ρ rather than
varying noise. Saves `exp_purity_main.png`, `exp_purity_advantage.png`,
`exp_purity_results.npy` to `results/`.

### Real-data experiments

```bash
# Epinions signed network (data/soc-sign-epinions.txt)
python exp_real_data.py --snap epinions

# Slashdot signed network (data/soc-sign-Slashdot090221.txt)
python exp_real_data.py --snap slashdot

# 20 Newsgroups (auto-downloads via sklearn on first run)
python exp_real_data.py --uci 20news
```

Optional flags for SNAP:
- `--sizes 1000 2500 5000` — target ego-network sizes (raw, before cleaning).
- `--n-ego-nets 5` — number of distinct ego-net centers per size.
- `--n-trials 2` — trials per (size, center).
- `--file <path>` — override the default data-file location.

For 20 Newsgroups:
- `--max-docs 400` — number of documents to sample for the threshold sweep.

### Diagnostic

```bash
python diagnose_signed_subgraph.py
```

Prints per-ego-net stats (density, positive-component distribution, TC
violation rate) and a CL-cost sweep across k. Use this whenever CL behaves
unexpectedly on real signed data to confirm whether the issue is structural
or just k-selection.

---

## Data

SNAP signed networks live in `data/`:

| File | Source |
|---|---|
| `soc-sign-epinions.txt` | https://snap.stanford.edu/data/soc-sign-epinions.html |
| `soc-sign-Slashdot090221.txt` | https://snap.stanford.edu/data/soc-sign-Slashdot090221.html |

20 Newsgroups is downloaded automatically by scikit-learn on first run.

---

## Cleaning policies applied to real data

These cleaning steps are documented in detail in
[CODE_EXPLAINER.md §4.7](CODE_EXPLAINER.md). Briefly:

**SNAP signed networks** (in `exp_real_data.py`):
- Drop self-loops (`u == v` lines).
- Drop reciprocal-disagreement pairs (`u→v=+1` and `v→u=-1`).
- In ego-net extraction: always include the center node; drop nodes with
  degree 0 within the kept subgraph (sampling artifacts).
- For CL, sweep `k ∈ {2, 5, 10, k_est}` and report the assignment with the
  lowest CC cost (the chosen k is logged on every run).

**20 Newsgroups** (in `exp_real_data.py`):
- Single-topic docs only.
- Keep top-K most frequent classes (default `top_k_classes=10`).
- Drop texts shorter than 50 characters.
- Random subsample (seeded) instead of taking the first N documents.

---

## Headline finding

Across both synthetic and (cleaned) real data:

> **CL is competitive with combinatorial CC baselines when triadic closure
> holds; it collapses uniquely when TC is violated.** Local Search dominates
> on cost overall, but CL beats Pivot and MFP and approaches LS whenever the
> triadic-closure assumption is satisfied. This is a *characterization*
> result — describing when and why CL succeeds — not a performance claim
> that CL beats classical algorithms.

See [CODE_EXPLAINER.md](CODE_EXPLAINER.md) for the full walk-through and
[docs/session_report.pdf](docs/session_report.pdf) for the writeup of the
real-data cleaning iteration.
