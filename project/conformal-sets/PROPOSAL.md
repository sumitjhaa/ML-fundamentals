# Project Proposal — ES 678

**Title:** How Small Can a Trustworthy Prediction Set Be? Score Design and Class-Conditional Coverage in Conformal Classification

**Course:** ES 678 (Graduate)
**Student:** `<Your Name>`
**Roll Number:** `<Roll Number>`
**Institution:** Indian Institute of Technology Gandhinagar
**Date:** 15 October 2026
**Final submission due:** 17 November 2026

> This document is written to be pasted into the course portal proposal form (the portal fields
> map 1:1 to the numbered sections below). Section 6 is the required ES 678 depth component and
> is deliberately the most technical part.

---

## Abstract

We study how to make a fixed image classifier produce **trustworthy prediction sets** at a
controlled error rate. Conformal prediction wraps any classifier to output a set of labels that
contains the true class with a user-chosen probability `1 − α`, without retraining and with a
finite-sample, distribution-free guarantee. The guarantee, however, is only **marginal**: it
averages over the data and says nothing about coverage for a specific class, a specific
difficulty level, or an out-of-distribution input. We implement and compare conformal scores
(naive, adaptive APS, regularized RAPS) and a proposed **temperature-scaled, distance-augmented
score** that combines calibrated confidence with a Mahalanobis feature-space distance. Across
MNIST and CIFAR-10 we measure set size (efficiency) at fixed coverage, audit class-conditional
coverage, and analyse failures under distribution shift. Our depth component derives the
coverage theorem, establishes the finite-sample bound, explains why marginal and conditional
coverage must differ, and quantifies the efficiency cost of restoring per-class coverage using
Mondrian conformal prediction.

---

## 1. Problem Formulation and Scope

**Question.** Given a fixed, already-trained image classifier, how do the choice of conformity
score and the addition of feature-space distance information affect (i) the **size** of conformal
prediction sets at a fixed, guaranteed coverage level, and (ii) the **class-conditional** coverage
that marginal guarantees fail to control?

**Target.** A prediction **set** `C(x) ⊆ {1, …, K}` for each test image, satisfying the marginal
guarantee `P(Y ∈ C(X)) ≥ 1 − α`. We evaluate efficiency (mean set size) and conditional coverage,
not raw accuracy. Prediction is made at **inference time**, when no label is available and the
input may be unfamiliar.

**What one row represents.** One image together with its ground-truth class label. Features are
the raw pixels; the model is a convolutional classifier producing `p̂(y | x)`.

**When a prediction is made.** At test time, on inputs drawn from the same distribution as the
calibration set (the exchangeable regime), and separately on out-of-distribution inputs.

**Why it matters.** High-accuracy classifiers are increasingly used where a wrong confident answer
is costly (medical triage, safety systems). A set-valued, error-rate-controlled prediction is more
honest than a single label: it says "these are the plausible answers, and this claim is valid
with probability at least `1 − α`." The practical question is not whether a set can be valid, but
how *small* and how *useful* it can be while still being valid.

**Scope of the claim.** We claim distribution-free *marginal* coverage under exchangeability, and
*empirical* class-conditional coverage. We do **not** claim conditional coverage for every input
`x` (this is provably impossible in general, Section 6.2), and we do **not** claim the guarantee
holds under distribution shift. All comparisons are conditional on a fixed training procedure.

---

## 2. Data

| Dataset | Role | Size | Access |
|---|---|---|---|
| MNIST | Primary in-distribution | 60,000 train / 10,000 test, 28×28 grayscale | `torchvision`, public, no limits |
| CIFAR-10 | Second in-distribution | 50,000 train / 10,000 test, 32×32 RGB | `torchvision`, public, no limits |
| FashionMNIST, KMNIST | OOD for MNIST | 10,000 each | `torchvision`, public |
| SVHN, CIFAR-100 | OOD for CIFAR-10 | 10,000+ each | `torchvision`, public |

**Important variables.** All pixels (784 or 3,072 per row); the label `y ∈ {1, …, K}` with `K = 10`.

**Data-quality checks.** Verify class balance, pixel-value ranges, duplicated/near-duplicate images,
and that MNIST/CIFAR-10 standard splits are used unmodified so results are comparable to the
literature. No personally identifying information; no access or sharing restrictions.

**Multiple datasets.** Two in-distribution datasets test whether conclusions about score design
generalise beyond a single image domain (grayscale digits vs natural images).

---

## 3. Preprocessing and Statistical Reasoning

**Preprocessing.** Normalise inputs with statistics computed on the **training split only**.
Standard augmentation for CIFAR-10 (random crop, horizontal flip) applied during training only.
The classifier is trained with cross-entropy; no preprocessing touches calibration or test data.

**Leakage discipline (critical for conformal validity).** We use a four-way, disjoint partition:

| Split | Fraction | Used for | Never used for |
|---|---|---|---|
| `D_train` | 80% of train | fitting the classifier | tuning conformal parameters |
| `D_design` | 10% of train | early stopping, temperature `T`, RAPS `λ, k_reg`, our `γ`, method selection | computing `q̂` |
| `D_cal` | 10% of train | computing the conformal quantile `q̂` | any model/score selection |
| `D_test` | official test | final reported metrics | anything |

**Why this matters.** Tuning a conformal score on the same set used to compute `q̂` **invalidates
the coverage guarantee**: the calibration scores are no longer exchangeable with the test score.
We therefore separate *design* (`D_design`) from *calibration* (`D_cal`). This is stated explicitly
because it is the single most common way conformal results are silently wrong.

**Exchangeability.** The guarantee requires that `D_cal` and `D_test` are drawn i.i.d. from the same
distribution. We enforce this for the in-distribution evaluation and deliberately violate it for the
OOD analysis, where we *report* the breakdown rather than claim a guarantee.

**Random seeds.** All splits and randomised scores use fixed seeds; every headline number is
reported as mean ± std over 3–5 seeds.

---

## 4. Methods

Let `f` be the trained classifier producing `p̂(y|x)`, and let `s(x, y)` be a **conformity score**
(smaller = more conforming). Fixing `α`, split conformal computes a threshold from `D_cal` and
returns `C(x) = { y : s(x, y) ≤ q̂ }` (Section 6.1 derives `q̂`).

### 4.1 Baselines
1. **Top-1 point prediction** — no set, no guarantee (reference for accuracy).
2. **Naive conformal score** `s(x,y) = 1 − p̂(y|x)`: a fixed probability threshold. Valid but
   **not adaptive** — ambiguous inputs get large sets.

### 4.2 Adaptive scores
3. **APS** (Adaptive Prediction Sets, Romano et al. 2020). Sort `p̂` descending; with `k_y` the rank
   of `y`,

   ```
   s_APS(x, y) = Σ_{j < k_y} p̂_(j)(x) + u · p̂_(k_y)(x),     u ~ Uniform(0,1)
   ```

   (non-randomised variant omits the `u` term). The set accumulates top classes until the
   cumulative probability mass reaches `q̂`. Sets are small when the model is confident.
4. **RAPS** (Angelopoulos et al. 2021). `s_RAPS = s_APS + λ · (k_y − k_reg)^+`, with `λ, k_reg`
   fit on `D_design` to shrink sets, especially for rare classes.

### 4.3 Calibrated confidence: temperature scaling
`p̂_T(y|x) = softmax(z(x) / T)`, where `z` are logits and `T > 0` is fit on `D_design` by minimising
negative log-likelihood. Temperature scaling preserves `argmax` but rescales score magnitudes,
changing set construction.

### 4.4 Feature-space distance: Mahalanobis
On penultimate features `φ(x)`, fit per-class Gaussians with a shared covariance `Σ`:

```
d_M(x, y) = sqrt( (φ(x) − μ_y)ᵀ Σ⁻¹ (φ(x) − μ_y) ),      Σ = (1/N) Σ_i (φ(x_i) − μ_{y_i})(φ(x_i) − μ_{y_i})ᵀ
```

Standardise `d_M` with `D_design` statistics to obtain `d̃_M`.

### 4.5 Proposed score (the tweak)
```
s_γ(x, y) = [ 1 − p̂_T(y | x) ] + γ · d̃_M(x, y),          γ ≥ 0 fit on D_design
```
and the adaptive variant `s_γ^APS(x,y) = s_APS(p̂_T) + γ · d̃_M(x,y)`.
**`γ = 0` recovers the temperature-scaled baseline**, giving a clean ablation. The hypothesis is
that the distance term inflates scores for inputs far from a class's feature manifold, so
confusable and OOD points receive larger, honest sets, while confident in-distribution points keep
small sets — improving coverage where it currently fails, at a controlled cost in mean size.

### 4.6 Class-conditional (Mondrian) conformal
To *guarantee* per-class coverage, calibrate a separate quantile `q̂_c` from the calibration points
with label `c` and build `C(x) = { y : s(x,y) ≤ q̂_y }`. This controls
`P(Y ∈ C(X) | Y = y) ≥ 1 − α` per class, at the cost of larger sets. We use it as both a
**baseline for conditional coverage** and the upper-bound reference on efficiency.

### 4.7 Optional robustness extension
**Cross-conformal / CV+** (Barber et al. 2021) replaces the single split with `K`-fold
out-of-fold scores, reducing the variance induced by one finite calibration split. Included only if
time allows; treated as an ablation.

---

## 5. Evaluation Design and Metrics

**Protocol (fixed before comparison).**
- Train once per seed on `D_train`; never touch `D_test` until the final table.
- Tune `T`, `λ`, `k_reg`, `γ`, and select among methods on `D_design` only.
- Compute `q̂` on `D_cal`; evaluate on `D_test`.
- Report mean ± std over 3–5 seeds.

**Metrics.**
1. **Marginal empirical coverage** `(1/m) Σ 1{Y_j ∈ C(X_j)}` — must be `≥ 1 − α`; validates the theorem.
2. **Average set size** — efficiency; lower is better at equal coverage.
3. **Median / 90th-percentile set size** — tail behaviour.
4. **Singleton rate** and **empty-set rate**.
5. **Class-conditional coverage** per class `c`: `(1/m_c) Σ_{j: y_j = c} 1{y_j ∈ C(X_j)}`, plus the
   **worst-class coverage gap** `max_c |cov_c − (1 − α)|`. This is where marginal coverage misleads.
6. **Subgroup coverage** by model-confidence decile.
7. **OOD behaviour** — coverage and set size on the OOD sets (guarantee void; reported as a failure
   mode).
8. **Context** — classifier accuracy and ECE.

**Sweeps.** `α ∈ {0.01, 0.05, 0.1, 0.2}`; `γ` over a log-grid; `T` and RAPS parameters on `D_design`.
Report the finite-sample slack `1/(n+1)` so coverage is interpreted against `[1−α, 1−α+1/(n+1)]`.

**Where the claims are made.** Efficiency claims are at matched *empirical* coverage; conditional
claims are empirical (audited), not guaranteed, except for Mondrian, whose guarantee we verify.

---

## 6. Graduate Depth Component

This section is the ES 678 research/depth deliverable. It contains **(A)** a derivation of the
coverage guarantee with its finite-sample bound, **(B)** a precise statement of why marginal
coverage cannot imply conditional coverage, and **(C)** the analysis plan that turns the project
from "run three scores" into a study of the efficiency–conditional-coverage frontier.

### 6.1 Derivation of the coverage guarantee

**Setup.** Let `Z_i = (X_i, Y_i)`, `i = 1, …, n+1` be exchangeable. Let `S_i = s(X_i, Y_i)` for
`i = 1, …, n` (calibration) and `S_{n+1} = s(X_{n+1}, Y_{n+1})`. Define the order statistic

```
q̂ = S_(k),     k = ⌈(n+1)(1 − α)⌉,
```

where `S_(1) ≤ … ≤ S_(n)` are the sorted calibration scores, with the convention `q̂ = +∞`
if `k > n`. The prediction set is `C(X_{n+1}) = { y : s(X_{n+1}, y) ≤ q̂ }`.

**Theorem (marginal coverage).** Under exchangeability,

```
1 − α  ≤  P( Y_{n+1} ∈ C(X_{n+1}) )  ≤  1 − α + 1/(n+1).
```

**Proof.** Non-coverage is the event `E = { S_{n+1} > q̂ }`. If `S_{n+1} > S_(k)`, then
`S_{n+1}` is strictly greater than the `k` smallest calibration scores, so its rank among the
`n + 1` scores satisfies `rank(S_{n+1}) ≥ k + 1`. By exchangeability the rank is uniform on
`{1, …, n+1}`, hence

```
P(E) ≤ P(rank ≥ k + 1) = (n + 1 − k)/(n + 1) = 1 − k/(n + 1) ≤ 1 − (1 − α) = α,
```

using `k ≥ (n+1)(1−α)`. For the upper bound, coverage implies `rank(S_{n+1}) ≤ k`, so
`P(coverage) ≤ k/(n+1) ≤ (⌈(n+1)(1−α)⌉)/(n+1) ≤ 1 − α + 1/(n+1)`. ∎

**Consequences we will exploit.**
- Coverage is sandwiched in a window of width `1/(n+1)`; with `n = 5,000` the slack is `≈ 2×10⁻⁴`,
  so observed coverage should sit just above `1 − α`.
- The guarantee is **distribution-free** and holds for *any* score `s` — so the research question
  is necessarily about **efficiency and conditional coverage**, not validity.
- Validity is preserved under any score, which is exactly why we are free to design `s_γ`.

### 6.2 Why marginal coverage does not imply conditional coverage

Marginal coverage averages over `X`. It can hold exactly while some regions have coverage `1` and
others `1 − 2α`. Conditional coverage `P(Y ∈ C(X) | X = x) ≥ 1 − α` for (almost) every `x` is
unattainable in general: obtaining it requires that the set be conservative enough that no
individual `x` is under-covered, which forces sets to grow toward the full label space for the
hardest inputs (a standard impossibility in the conformal literature).

A weaker, achievable target is coverage **conditional on the label**,
`P(Y ∈ C(X) | Y = y) ≥ 1 − α`, which Mondrian conformal delivers by calibrating a separate `q̂_y`
per class. Our proposed score `s_γ` is a **cheap heuristic** that pushes coverage in the right
direction (raising set size for inputs far from a class manifold) *without* the full per-class cost
of Mondrian — so the central empirical question is: **how close does `s_γ` get to the conditional
coverage of Mondrian, and at what efficiency premium?**

### 6.3 Analysis plan (what the depth component demonstrates)

1. **Efficiency frontier.** Average set size versus coverage, for each score, at matched `α`. Does
   `s_γ` (with `γ>0`) reduce the worst-class coverage gap without inflating mean size?
2. **Conditional-coverage audit.** Per-class coverage and the worst-class gap, comparing marginal
   conformal vs Mondrian vs `s_γ`. Quantifies the marginal-vs-conditional tension concretely.
3. **Sensitivity.** `α × γ` grid; effect of temperature `T`; how much of the gain comes from
   calibration versus distance.
4. **Ablation.** `s_APS`, `s_APS(T)`, `s_APS(T)+distance` — isolating the contribution of each term.
5. **Failure-mode audit.** Which classes and which inputs break coverage; the behaviour on OOD sets
   where exchangeability fails; the effect of a small calibration set (`n`) on the `1/(n+1)` slack.

---

## 7. Hypotheses and Expected Results

| # | Hypothesis | Test | Falsifier |
|---|---|---|---|
| H1 | Empirical marginal coverage lies in `[1−α, 1−α+1/(n+1)]` for all scores | Coverage table | Coverage outside the window |
| H2 | APS yields smaller sets than the naive score at equal coverage | Mean set size | Naive ≤ APS |
| H3 | Temperature scaling improves efficiency of APS on CIFAR-10 | Set size at matched coverage | No improvement |
| H4 | `s_γ` with `γ>0` reduces the worst-class coverage gap vs temperature-only | Worst-class gap | Gap unchanged/worse |
| H5 | Mondrian achieves the best worst-class gap but the largest mean set size | Coverage vs size | Mondrian not best on conditional coverage |
| H6 | All methods degrade under OOD (coverage drops despite large sets) | OOD table | Coverage stays nominal |

A **negative result is acceptable and informative**: if `s_γ` does not beat temperature scaling,
the paper reports the efficiency–conditional-coverage frontier honestly and explains why.

---

## 8. Challenges and Mitigations

| Challenge | Why it is hard | Mitigation |
|---|---|---|
| Marginal coverage is trivially satisfied | Examines validity instead of contribution | Frame the contribution as **efficiency + conditional coverage**; never claim the guarantee as novel |
| Conditional coverage is unattainable for `X` | Fundamental impossibility | Target the achievable **class-conditional** guarantee (Mondrian) and audit it empirically |
| Leakage invalidates the guarantee | Tuning on `D_cal` breaks exchangeability | Separate `D_design` from `D_cal`; document explicitly |
| Weak classifier on CIFAR-10 | Sets become uninformative | Use a small pretrained backbone (disclosed) or lead with MNIST and present CIFAR-10 as secondary |
| Small effect sizes | Schedules/distance may barely help | Report the `α × γ` sweep, the `1/(n+1)` slack, and error bars so a modest result is still complete |
| OOD breaks the guarantee | Exchangeability fails | Report as an anticipated failure mode, not a bug |

---

## 9. Tools and AI-Use Disclosure

**Tools.** Python 3.11; PyTorch (+ MPS on Apple Silicon); torchvision; NumPy; SciPy; pandas;
Matplotlib. LaTeX (NeurIPS 2026 course style, as supplied) for the paper.

**AI use (to be finalised and disclosed in the paper).** AI assistance will be used for
brainstorming, boilerplate code, debugging, and editing. Every mathematical statement and every
reported number will be verified independently by the author; external code or material will be
cited. No credentials, private data, or API keys will be published. The exact tools and the kinds
of assistance will be listed in the final paper's disclosure section.

---

## 10. Timeline

| Week | Dates | Deliverable |
|---|---|---|
| 1 | Oct 16–22 | Data + four-way splits + baseline classifier + **naive conformal** with a coverage sanity check |
| 2 | Oct 23–29 | APS + RAPS + temperature scaling; efficiency comparison; plots |
| 3 | Oct 30–Nov 5 | Distance-augmented score; `γ`/`T` sweeps; conditional-coverage audit; OOD; failure audit (depth) |
| 4 | Nov 6–13 | Assemble the 4-page paper; `reproduce.sh`; data-access notes; AI disclosure; defence prep |
| — | Nov 17 | Final submission (PDF + LaTeX source + bibliography + code + reproducibility notes) |

---

## 11. Reproducibility Plan

- **Code layout** mirrors the pipeline (`data → model → calibrate → scores → conformal → evaluate`),
  with a single `scripts/run_all.py` entry point that regenerates every figure and table.
- **Configs** (YAML) pin dataset, seed, `α`, `γ`, `T`, RAPS parameters, and split fractions.
- **Outputs** (CSV/JSON + figures) are written to `results/` and versioned.
- **Data access**: all datasets via `torchvision`; a `python -c "import torchvision; …"` snippet
  documents exact versions and splits. No restricted data.
- **Environment**: `requirements.txt` pins versions; seed control for Python/NumPy/PyTorch.

---

## References

1. Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
2. Shafer, Vovk. *A Tutorial on Conformal Prediction.* JMLR, 2008.
3. Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA, 2018.
4. Romano, Sesia, Candès. *Classification with Valid and Adaptive Coverage.* NeurIPS, 2020.
5. Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control (RAPS).* 2021.
6. Sadinle, Lei, Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA, 2019.
7. Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021.
8. Vovk. *Conditional Validity of Inductive Conformal Predictors.* ACML, 2012. (Mondrian conformal.)
9. Barber, Candès, Ramdas, Tibshirani. *Predictive Inference with the Jackknife+.* Annals of Statistics, 2021.
10. Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017.
11. Lee, Lee, Lee, Shin. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS, 2018.
12. Hendrycks, Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples.* ICLR, 2017.
