# TabMixer

> A gated token-mixing neural architecture for tabular data — no self-attention required.

TabMixer is a from-scratch research project exploring whether an MLP-Mixer-style architecture, augmented with adaptive feature gating and highway-style residuals, can model feature interactions in tabular data as effectively as attention-based approaches — at a fraction of the computational cost.

The project isn't just the architecture: it's a full experimental pipeline that trains and compares **five model families** (a hand-written NumPy MLP, a standard PyTorch MLP, TabMixer, and gradient-boosted trees via XGBoost/CatBoost) on **three datasets** with genuinely different characteristics — a small clean dataset, a numeric regression-turned-classification dataset, and a large mixed-type dataset with missing values.

---

## Motivation

Tabular data remains one of the hardest domains for deep learning to win convincingly — gradient-boosted trees (XGBoost, CatBoost) still tend to outperform neural networks on most tabular benchmarks. Classical MLPs struggle to model feature interactions explicitly, while transformer-style self-attention over features is computationally heavier than the problem often warrants.

TabMixer investigates a middle ground: can a lightweight, attention-free mixer architecture, with an explicit *adaptive gating* mechanism deciding how much each feature update should matter, get closer to tree-based performance while staying cheap to train?

To answer that honestly, this repo doesn't just implement the architecture — it benchmarks it against a hand-rolled MLP (to sanity-check that the training pipeline itself is correct), a standard PyTorch MLP (a fair neural baseline), and gradient-boosted trees (the actual competition on tabular data), across three datasets of increasing difficulty.

---

## Architecture

```
Input Features
      │
      ▼
Numerical Embedding
(sin/cos periodic embedding
 + learnable positional embedding)
      │
      ▼
─────────────────────────────
Mixer Block × N
─────────────────────────────

LayerNorm
      │
Token Mixer
      │
Feature Gate
      │
Highway Residual

      ↓

LayerNorm
      │
Channel Mixer
      │
Feature Gate
      │
Highway Residual

─────────────────

      ▼
  LayerNorm
      ▼
   Pooling
      ▼
Classification Head
```

---

## Key Components

### Numerical Embedding

Each numerical feature is converted into a `D`-dimensional representation via a **periodic sin/cos embedding** with learnable frequency and phase, plus a learnable per-feature positional embedding — instead of a single learned linear projection. This lets the network represent non-linear, periodic relationships in a feature's value that a plain linear embedding cannot.

> Note: this embedding assumes standardized numeric input (mean 0, unit variance) — all dataset scripts in this repo scale features before feeding them into TabMixer.

### Token Mixer

Mixes information *across features* (the `F` axis) with a small bottleneck MLP, in place of self-attention. Cheaper, with no quadratic cost in the number of features.

### Channel Mixer

Mixes information *within* each feature's embedding (the `D` axis), independently per feature — the standard MLP-Mixer channel-mixing step.

### Feature Gate

The core contribution of this architecture. Each feature gets an adaptive, learned gate combining:

- a **local** representation of that feature, and
- a **global context** aggregated across all features

```
gate = σ(local_proj(x) + context_proj(x))
```

The gate controls how much of the newly mixed representation should actually overwrite the previous one, feature-by-feature — rather than blending in a fixed amount everywhere.

### Highway Residual Update

Instead of a standard residual connection `x + F(x)`, each mixer sub-block performs:

```
x + gate · (new − old)
```

During training, the gated update is also randomly dropped for an entire sample at a time (stochastic depth), scaled to keep the expected update magnitude unchanged at eval time — an extra regularizer on top of dropout inside the mixer MLPs.

---

## Repository Structure

```
.
├── models/
│   ├── TabMixer.py          # TabMixer architecture (PyTorch)
│   └── numpy_MLP.py         # MLP implemented from scratch in NumPy 
│
├── ML algorythms/            # Classical ML baselines (KNN, Random Forest)
│   ├── data_cancer.py
│   ├── data_california.py
│   └── data_income.py
│
├── boosting/                  # Gradient boosting baselines (XGBoost, CatBoost)
│   ├── boosting_cancer.py
│   ├── boosting_california.py
│   └── boosting_income.py
│
├── tests/                      # Full comparisons: NumPy MLP vs PyTorch MLP vs TabMixer
│   ├── cancer.py
│   ├── california.py
│   └── income.py
│
└── README.md
```

Each dataset therefore has **three independent scripts** covering the full model lineup: a classical-ML baseline (`ML algorythms/`), a gradient-boosting baseline (`boosting/`), and the deep-learning comparison (`tests/`).

---

## Datasets

| Dataset | Task | Source | Features | Notes |
|---|---|---|---|---|
| **Breast Cancer Wisconsin** | Binary classification | `sklearn.datasets.load_breast_cancer` | 30 numeric | Small, clean, no missing values — sanity-check dataset |
| **California Housing** | Binary classification | `sklearn.datasets.fetch_california_housing` | 8 numeric | Originally a regression dataset; binarized around the median house value (`cheap` vs `expensive`) |
| **Adult Income (Census)** | Binary classification | `fetch_openml(name="adult", version=2)` | ~14 mixed (numeric + categorical) | Real-world messiness: missing values, high-cardinality categorical columns (e.g. `native-country`) |

The three datasets are chosen deliberately to stress different things: Cancer checks whether the models learn *at all*; California checks behaviour on a mostly-continuous, low-dimensional feature space; Adult checks how the pipeline handles missing data and categorical features at a larger scale (~48K rows).

**A deliberate architectural note on categoricals:** TabMixer's numerical embedding is a periodic sin/cos transform designed for genuinely continuous values. For the `tests/` (deep learning) pipeline, categorical columns in Adult Income are therefore **ordinal-encoded** rather than one-hot encoded — one-hot would inflate the feature count from ~14 to 100+ mostly-binary columns fed through an embedding that isn't designed for them. This keeps the comparison between NumPy MLP / PyTorch MLP / TabMixer fair (all three see the same representation), at the cost of giving categorical columns an arbitrary numeric ordering. The `boosting/` scripts avoid this compromise entirely — XGBoost and CatBoost are given categorical columns **natively** (`enable_categorical=True` / `cat_features=...`), since native categorical handling is one of the reasons those libraries are strong tabular baselines in the first place.

---

## Getting Started

### Installation

```bash
pip install torch numpy scikit-learn matplotlib xgboost catboost
```

### Running an experiment

The `tests/` scripts import directly from `models/` (`from TabMixer import TabMixer`, `from numpy_MLP import MLP`), so `models/` needs to be importable when you run them — add it to your path first:

```bash
# Deep learning comparison: NumPy MLP vs PyTorch MLP vs TabMixer
PYTHONPATH=models python "tests/cancer.py"
PYTHONPATH=models python "tests/california.py"
PYTHONPATH=models python "tests/income.py"

# Classical ML baselines (self-contained, no PYTHONPATH needed)
python "ML algorythms/data_cancer.py"
python "ML algorythms/data_california.py"
python "ML algorythms/data_income.py"

# Gradient boosting baselines (self-contained, no PYTHONPATH needed)
python "boosting/boosting_cancer.py"
python "boosting/boosting_california.py"
python "boosting/boosting_income.py"
```

On Windows (PowerShell), set the path instead: `$env:PYTHONPATH="models"`.

The Adult Income scripts download data via `fetch_openml` on first run (requires an internet connection); Breast Cancer and California Housing are fetched by scikit-learn directly / cached locally after the first run.

---

## Results

> Fill in after running the scripts above — numbers below are placeholders.

### Breast Cancer Wisconsin

| Model | Accuracy |
|---|---|
| KNN | XX.XX% |
| Random Forest | XX.XX% |
| XGBoost | XX.XX% |
| CatBoost | XX.XX% |
| NumPy MLP (from scratch) | XX.XX% |
| PyTorch MLP | XX.XX% |
| **TabMixer** | XX.XX% |

### California Housing (binarized)

| Model | Accuracy |
|---|---|
| KNN | XX.XX% |
| Random Forest | XX.XX% |
| XGBoost | XX.XX% |
| CatBoost | XX.XX% |
| NumPy MLP (from scratch) | XX.XX% |
| PyTorch MLP | XX.XX% |
| **TabMixer** | XX.XX% |

### Adult Income

| Model | Accuracy |
|---|---|
| KNN | XX.XX% |
| Random Forest | XX.XX% |
| XGBoost | XX.XX% |
| CatBoost | XX.XX% |
| NumPy MLP (from scratch) | XX.XX% |
| PyTorch MLP | XX.XX% |
| **TabMixer** | XX.XX% |

---

## Baselines

TabMixer is benchmarked against five other models:

- **NumPy MLP** — a fully manual implementation (forward pass, backprop, dropout with correct gradient masking, Adam with bias correction) used primarily as a from-scratch correctness check on the training pipeline itself.
- **PyTorch MLP** — a standard 2-hidden-layer MLP with the same width, dropout, and optimizer settings as TabMixer, to isolate what the mixer/gating architecture adds over a plain feedforward network.
- **KNN** and **Random Forest** — classical, non-boosted baselines.
- **XGBoost** and **CatBoost** — gradient-boosted trees, the standard strong baseline on tabular data and the real bar TabMixer needs to clear to be interesting.

---


## Tech Stack

- Python
- PyTorch
- NumPy
- scikit-learn
- XGBoost / CatBoost
- matplotlib

---

Developed as an independent deep learning research project exploring efficient, attention-free neural architectures for tabular data.
