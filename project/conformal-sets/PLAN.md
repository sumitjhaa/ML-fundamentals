# Execution Plan — ES 678 Conformal Prediction Sets

**Project:** How Small Can a Trustworthy Prediction Set Be? Score Design and Class-Conditional Coverage in Conformal Classification

**Author:** `<Your Name>` · `<Roll Number>` · ES 678 · IIT Gandhinagar
**Proposal due:** Oct 15, 2026 · **Final due:** Nov 17, 2026
**Reference:** [`PROPOSAL.md`](./PROPOSAL.md) · [`../project-ideas.md`](../project-ideas.md) · [`../s-tier-explained.md`](../s-tier-explained.md)

---

## 0. How to use this plan

Work top to bottom. Each phase has a **Goal**, **Tasks** (checkboxes), **Definition of Done**, and
**Rubric mapping**. Run the commands from the repo root (`conformal-sets/`). Do not move the
`D_cal` / `D_test` boundary — Section 3 of the proposal explains why that invalidates the guarantee.

---

## 1. Repository layout

```
conformal-sets/
├── PROPOSAL.md            # the proposal document
├── PLAN.md                # this file
├── README.md              # quick index + how to reproduce
├── requirements.txt       # pinned deps
├── configs/
│   └── default.yaml       # dataset, seed, alpha, gamma, T, split fractions
├── src/
│   ├── data.py            # loaders + four-way split (train/design/calib/test)
│   ├── models.py          # LeNet (MNIST) / SmallResNet (CIFAR-10)
│   ├── calibrate.py       # temperature scaling
│   ├── scores.py          # naive, APS, RAPS, distance-augmented
│   ├── conformal.py       # quantile + set construction + Mondrian
│   ├── mahalanobis.py     # per-class Gaussians, shared covariance
│   ├── evaluate.py        # coverage, set size, class-conditional coverage, OOD
│   └── plots.py           # coverage-vs-size, size hist, class-coverage gaps
├── scripts/
│   ├── run_all.py         # end-to-end, regenerates every figure/table
│   └── sweep.py           # alpha x gamma grid + seeds
├── results/               # CSV/JSON + figures (generated)
└── paper/
    ├── es678_conformal.tex  # 4-page paper (NeurIPS 2026 course style)
    ├── neurips_2026.sty
    └── refs.bib
```

---

## 2. Environment setup

```sh
cd conformal-sets
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

- [ ] `.venv` created, deps installed, MPS reported `True`.
- [ ] `python scripts/run_all.py --dry-run` prints the planned config (no training yet).

**Compute note:** train MNIST on MPS (seconds/epoch). CIFAR-10 with a small ResNet is a few
minutes/epoch on MPS; use Colab T4 only if needed. Mahalanobis + conformal are near-free.

---

## 3. The math reference (keep this open while coding)

**Conformal quantile.** `k = ⌈(n+1)(1−α)⌉`, `q̂ = S_(k)` (k-th smallest calibration score);
`q̂ = +∞` if `k > n`.

**Set.** `C(x) = { y : s(x, y) ≤ q̂ }`.

**Guarantee.** `1 − α ≤ P(Y ∈ C(X)) ≤ 1 − α + 1/(n+1)` under exchangeability.

**Scores.**
- Naive: `s = 1 − p̂(y|x)`
- APS: `s = Σ_{j<k_y} p̂_(j)(x) + u · p̂_(k_y)(x)`
- RAPS: `s = s_APS + λ (k_y − k_reg)^+`
- Temperature: `p̂_T = softmax(z/T)`
- Mahalanobis: `d_M(x,y) = sqrt((φ(x) − μ_y)ᵀ Σ⁻¹ (φ(x) − μ_y))`
- **Proposed:** `s_γ(x,y) = (1 − p̂_T(y|x)) + γ · d̃_M(x,y)`, and `s_γ^APS = s_APS(p̂_T) + γ · d̃_M`
- Mondrian: per-class quantile `q̂_c` from calibration points with label `c`.

---

## 4. Phases

### Phase 1 — Data, splits, baseline, coverage sanity (Week 1)
**Goal:** a correct split pipeline and a first valid conformal set.

- [ ] `src/data.py`: load MNIST + CIFAR-10; produce disjoint `train / design / calib / test`; fixed seeds.
- [ ] `src/models.py`: LeNet (MNIST) and SmallResNet (CIFAR-10); train with cross-entropy on `D_train`; early stop on `D_design`.
- [ ] `src/conformal.py`: `quantile(scores, alpha, n)` with the `⌈(n+1)(1−α)⌉` index and the `∞` case; `build_sets(...)`.
- [ ] `src/scores.py`: naive score only (for now).
- [ ] `src/evaluate.py`: marginal coverage + mean set size.
- [ ] Sanity: naive conformal coverage on `D_test` sits in `[1−α, 1−α+1/(n+1)]` for `α ∈ {0.01,0.05,0.1,0.2}`.

**Definition of Done:** a table showing empirical coverage ≈ `1−α` for the naive score on MNIST.
**Rubric:** Problem formulation (10), Data (10), Preprocessing/no-leakage (10).

### Phase 2 — Adaptive scores + calibration (Week 2)
**Goal:** the efficiency comparison.

- [ ] `scores.py`: APS (randomised + non-randomised), RAPS (`λ, k_reg` tuned on `D_design`).
- [ ] `calibrate.py`: temperature scaling by LBFGS on logits using `D_design`; report `T` and ECE before/after.
- [ ] `scores.py`: temperature-scaled variants of naive and APS.
- [ ] `plots.py`: coverage-vs-set-size curve; set-size histograms.
- [ ] Add 3–5 seeds; report mean ± std.

**Definition of Done:** naive vs APS vs RAPS vs temp-APS table at matched coverage, with plots.
**Rubric:** Baseline & method selection (15), Evaluation design & metrics (20).

### Phase 3 — The tweak + depth analysis (Week 3)
**Goal:** the research contribution and the depth section.

- [ ] `mahalanobis.py`: fit per-class Gaussians + shared `Σ` on `D_train` features; normalise `d_M`.
- [ ] `scores.py`: `s_γ` (naive+distance) and `s_γ^APS`; tune `γ` on `D_design`.
- [ ] `conformal.py`: Mondrian (class-conditional) conformal.
- [ ] `evaluate.py`: class-conditional coverage, worst-class gap, subgroup coverage by confidence decile.
- [ ] OOD evaluation: FashionMNIST/KMNIST (MNIST) and SVHN/CIFAR-100 (CIFAR-10); report coverage/size breakdown.
- [ ] `scripts/sweep.py`: `α × γ` grid; temperature ablation.
- [ ] Ablation table: `APS` → `APS(T)` → `APS(T)+distance`.
- [ ] Failure-mode audit: which classes/inputs break coverage.

**Definition of Done:** the efficiency–conditional-coverage frontier figure + the worst-class gap table.
**Rubric:** Graduate depth analysis (10), Results/errors/limitations (10).

### Phase 4 — Paper, reproducibility, defence (Week 4)
**Goal:** submission-ready package.

- [ ] Fill `paper/es678_conformal.tex` (≤ 4 pages incl. references); keep the supplied style.
- [ ] `paper/refs.bib` complete; every external method/data cited.
- [ ] `scripts/run_all.py` regenerates all figures/tables from scratch into `results/`.
- [ ] `README.md` documents setup + data access + the exact commands.
- [ ] AI-use disclosure section written (tools + kinds of assistance + what was verified).
- [ ] Defence prep: 5 slides — question, method, main figure, depth analysis, one limitation.

**Definition of Done:** proposal structure reproduced as a 4-page PDF + source + code + repro notes.
**Rubric:** LaTeX & communication (5), Reproducibility (5), Defence (5).

---

## 5. Experiment grid (configs/default.yaml)

```yaml
datasets: [mnist, cifar10]
seeds: [0, 1, 2]           # use 5 for the final run
split: {train: 0.80, design: 0.10, calib: 0.10}
alphas: [0.01, 0.05, 0.10, 0.20]
scores: [naive, aps, raps, temp_naive, temp_aps, dist_naive, dist_aps, mondrian]
gamma_grid: [0.0, 0.01, 0.1, 0.3, 1.0]
temperature: auto          # fit on D_design
ood: {mnist: [fashion_mnist, kmnist], cifar10: [svhn, cifar100]}
```

**Rule:** `T`, `λ`, `k_reg`, `γ`, and method selection use `D_design` only. `q̂` uses `D_cal` only.
`D_test` is opened once, for the final table.

---

## 6. Report template (numbers to produce)

| Score | α | Coverage | Mean size | 90th-pct size | Worst-class gap |
|---|---|---|---|---|---|
| naive | 0.10 | | | | |
| APS | 0.10 | | | | |
| RAPS | 0.10 | | | | |
| temp-APS | 0.10 | | | | |
| **dist-APS (`s_γ`)** | 0.10 | | | | |
| Mondrian | 0.10 | | | | |

Report the same table for `α ∈ {0.01, 0.05, 0.1, 0.2}`, for MNIST and CIFAR-10, mean ± std over
seeds, plus an OOD variant of the table.

---

## 7. Risk log

| Risk | Trigger | Response |
|---|---|---|
| Coverage outside `[1−α, 1−α+1/(n+1)]` | Phase 1 sanity fails | Check quantile index, split disjointness, score sign convention |
| `γ` gives no gain | Phase 3 | Report the null result honestly; the frontier + audit still carry the depth component |
| CIFAR-10 classifier too weak | Phase 2 | Switch to a disclosed small pretrained backbone; lead with MNIST |
| OOD coverage collapses | Phase 3 | Expected — report as a failure mode, not a defect |
| Paper over 4 pages | Phase 4 | Move ablations to a compact table; keep one main figure |

---

## 8. Rubric self-check (before submitting)

- [ ] Problem formulation & scope (10) — question, target, row, prediction time, claim limits.
- [ ] Data understanding (10) — sources, row meaning, size, access, quality checks.
- [ ] Preprocessing & statistical reasoning (10) — `D_design` vs `D_cal` separation stated; training-only fits.
- [ ] Baseline & method selection (15) — honest baseline; methods justified.
- [ ] Evaluation design & metrics (20) — split fixed before comparison; metrics justified; `D_test` untouched.
- [ ] Results, errors, limitations (10) — errors, uncertainty, no overclaiming.
- [ ] Reproducibility (5) — code, data access, split details, seeds.
- [ ] ES 678 depth (10) — derivation + conditional-coverage audit + sensitivity + failure audit.
- [ ] LaTeX paper (5) — 4 pages, supplied style, complete references.
- [ ] Defence (5) — can explain the theorem, a limitation, and a small change.
