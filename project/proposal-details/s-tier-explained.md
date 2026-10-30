# S-Tier Project Deep Dives — ES 335 / ES 678 Course Project

Companion to [`project-ideas.md`](./project-ideas.md). These are the six **Tier S** ("maximum wow")
projects, explained properly: the core idea from the paper, why it's fancy, exactly what the
algorithmic tweak is, data/baseline/metrics, the depth component (ES 678) and interaction
(ES 335), and the honest risks.

> Difficulty was deliberately ignored when ranking these. The binding constraint is now
> **evaluation quality** — 20 of the 80 shared rubric points. So: fix the split before tuning,
> run multiple seeds with error bars, and keep test data out of model selection.

| If you want… | Pick |
|---|---|
| The most "research contribution" feel | **#2 TabPFN** (mini-PFN) or **#25 SAM/ASAM** |
| The best interactive website | **#17 Conformal sets** |
| Trendiness / LLM energy | **#43 Self-consistency** (accept noisier eval) |
| Bang-for-buck compute | **#17** or **#25** (both run on an M5) |
| The most impressive plots | **#22 SimCLR** (embeddings) or **#25 SAM** (landscapes) |
| Time-series drama | **B1 Anomaly Transformer** |

---

## 1. #2 — TabPFN prior-family swap

**What it actually is.** TabPFN is a *prior-data fitted network* (PFN). Instead of training a model
on your dataset, it was pretrained once on **millions of synthetic tabular datasets** drawn from a
hand-designed generative "prior" (a mix of random neural nets, random decision trees, and GP-like
functions). At inference you feed the entire training set **plus** the test rows into the transformer
as one sequence, do a single forward pass, and out come predictions for the test rows. There is no
gradient training on your data at all.

**The key insight.** The transformer is amortizing **Bayesian inference**: if the synthetic prior
matches the real data distribution, the network's in-context prediction approximates the true
posterior predictive. So the *prior defines the inductive bias* — and this is the thing nobody has
studied well. That is the research gap to poke at.

**Why it's S-tier.**
- It's a **foundation model for tabular data** — the trendiest idea in the catalog, and tabular data
  is where most real ML happens.
- "Predicts in a few seconds with zero training" is a spectacular demo.
- Changing the prior is a genuine, defensible research contribution rather than a hyperparameter.

**The tweak.** Modify the synthetic prior's functional families (e.g. add heavy-tailed features,
strongly correlated features, or a new structural family) and measure how few-shot accuracy on *real*
datasets changes. This directly answers: "is TabPFN's prior well-matched to real tabular data, or is
it leaving accuracy on the table?"

**Data / baseline / metric.** Small OpenML datasets (≤1000 rows, ≤100 features, ≤10 classes —
TabPFN v1's intended regime). Baseline: logistic regression / XGBoost → TabPFN variant.
Metric: ROC-AUC, accuracy, ECE.

**Depth component (ES 678).** A *prior-sensitivity study* — a 2-D grid of prior family × dataset
property.

**Reality check (important).** Fully retraining TabPFN is enormous (massive GPU-days in the paper),
so you **cannot** replicate the full pretraining. Two viable paths:
1. Use the released TabPFN weights and study prior matching via *controlled evaluation datasets*
   (heavy-tailed, correlated, many-irrelevant-features) — cheap and honest.
2. Train a **miniature PFN** (small transformer, cheap prior) on a custom synthetic distribution to
   demonstrate the prior's effect at small scale — the "real research" version, and it fits Colab.

State clearly in the paper that you are studying a scaled-down PFN.

---

## 2. #17 — Conformal prediction sets

**What it actually is.** Instead of one predicted label (or one probability), conformal prediction
outputs a **set** of labels that provably contains the true label with probability ≥ 1 − α, for *any*
model, with **no distributional assumptions** (only exchangeability of calibration/test data). It is
distribution-free uncertainty quantification.

**The mechanism.**
1. Define a conformity score `s(x, y)` — e.g. `1 − p̂(y|x)` (APS instead uses cumulative probability
   mass, which makes the set adaptive).
2. On a held-out **calibration set**, compute scores for the true labels and take the
   `⌈(n+1)(1−α)⌉/n` empirical quantile, call it `q`.
3. At test time, include every label whose score is ≤ `q`:
   `C(x) = { y : s(x, y) ≤ q }`.

The coverage guarantee holds *for any score function* — which is the beautiful part.

**Why it's S-tier.**
- Mathematically elegant and genuinely modern (conformal prediction is a hot research area).
- The guarantee is a perfect "wow" line and a perfect **derivation** for the depth component.
- The interaction is unbeatable: **slide the target coverage α and watch the sets resize live**,
  while empirical coverage tracks the target.

**The tweak.** Since coverage is guaranteed for *any* score, the research question becomes
**efficiency and conditional coverage**: design a better conformity score (temperature-scaled APS, a
RAPS-style regularized score, or a score blended with a Mahalanobis/OOD term) to get **smaller sets
at the same coverage**, and study **per-class / per-difficulty conditional coverage** (marginal
coverage is guaranteed; conditional coverage is *not*, and that is the interesting subtlety).

**Data / baseline / metric.** CIFAR-10 (+ an OOD set). Baseline: softmax thresholding → APS → RAPS →
your score. Metrics: **empirical coverage** and **average set size**, plus class-wise coverage.

**Depth component (ES 678).** Empirical coverage validation + set-size comparison + a
conditional-coverage audit (the failure mode where class-conditional coverage breaks despite marginal
coverage holding).

**Interaction (ES 335).** Slide the target coverage → sets resize live with coverage tracking the
target.

**Reality check.** The method is well established, so novelty lives entirely in the **score design
and the conditional-coverage analysis** — which is plenty. No heavy compute; runs on an M5.

---

## 3. #22 — Annealed SimCLR temperature

**What it actually is.** SimCLR is self-supervised contrastive learning: take an image, make two
augmented views, encode both, and pull the positive pair together while pushing apart all other
images in the batch using the **NT-Xent** loss. For a positive pair `(i, j)`:

```
ℓ(i,j) = − log [ exp( sim(z_i, z_j) / τ )  /  Σ_{k≠i} exp( sim(z_i, z_k) / τ ) ]
```

where `sim` is cosine similarity and **τ is the temperature**. Then freeze the encoder and train a
**linear probe** to measure representation quality.

**The role of τ.** τ controls the sharpness of the softmax: small τ → the loss concentrates on the
**hardest negatives** (peakier distribution); large τ → it spreads attention smoothly across all
negatives. SimCLR used a small fixed τ (≈0.1) and found smaller τ generally helped.

**Why it's S-tier.**
- Self-supervised learning **from scratch** — the most "modern ML" thing you can build.
- The t-SNE/embedding visualizations update over training and look fantastic.
- The τ-schedule is a clean, definable tweak with a clear hypothesis.

**The tweak.** **Anneal τ over training** — start *high* (softer loss, since early representations are
garbage and hard negatives are noisy) and decay *low* (exploit hard negatives once representations
mature). Alternatives: a **learned** temperature (like CLIP's learned logit scale) or a
**per-dimension** temperature. Hypothesis: a curriculum from easy → hard negatives improves the
linear-probe accuracy.

**Data / baseline / metric.** CIFAR-10 / STL-10. Baseline: fixed-τ SimCLR → annealed-τ. Metrics:
**linear-probe accuracy** and kNN accuracy. Add **alignment & uniformity** (Wang & Isola 2020) as
representation-quality diagnostics.

**Depth component (ES 678).** τ-schedule ablation + alignment/uniformity analysis (research-grade).

**Interaction (ES 335).** t-SNE of learned embeddings updating over epochs.

**Reality check.** Training SSL from scratch is the heaviest item here — ResNet-18 on CIFAR-10 for
~200 epochs on a Colab T4 is doable but slow; shrink the backbone/epochs and be transparent. This is
the S-tier pick most likely to hit compute walls.

---

## 4. #25 — SAM with adaptive ρ

**What it actually is.** Sharpness-Aware Minimization targets **flat minima**, which generalize
better. It solves a min–max problem:

```
min_w   max_{‖ε‖ ≤ ρ}   L(w + ε)
```

The inner max is approximated (first-order) by a single normalized gradient-ascent step:

```
ε*(w) = ρ · ∇L(w) / ‖∇L(w)‖
```

then you take a gradient step **at the perturbed weights**: `w ← w − η ∇L(w + ε*)`. Cost: two
forward–backward passes per step. **ρ is the neighborhood radius** — fixed at ~0.05 in the paper.

**Why it's S-tier.**
- Optimizer research is a prestigious genre, and SAM is a genuinely modern method (used for
  state-of-the-art generalization).
- The **loss-landscape visualizations** are gorgeous.
- Deriving the SAM objective and its first-order approximation is an ideal ES 678 derivation.

**The tweak.** Make ρ **adaptive** instead of constant. The clean, well-motivated version is
**ASAM** (Kwon et al. 2021): scale the perturbation per-layer by the weight norm, `ρ_l ∝ ‖w_l‖`, so
the method is invariant to reparameterization. Alternatives: a **schedule** for ρ, per-parameter ρ,
or ρ tied to the gradient norm. Compare against vanilla SAM, Adam, and SGD.

**Data / baseline / metric.** CIFAR-10. Baselines: SGD/Adam → SAM → ASAM. Metrics: accuracy,
generalization gap, and **sharpness** (e.g. largest Hessian eigenvalue, or the loss at the perturbed
point).

**Depth component (ES 678).** ρ sensitivity sweep + filter-normalized **2-D loss-landscape plots**
(Li et al. 2018) showing your variant is flatter.

**Interaction (ES 335).** 2-D loss-landscape visualization.

**Reality check.** Two forward–backward passes per step ≈ 2× cost, but CIFAR-10 is small so it is
fine on Colab. ρ tuning can be finicky — report the sweep.

---

## 5. #43 — Self-consistency voting (small LLM)

**What it actually is.** Self-Consistency (Wang et al. 2022) is an **inference-time** method: instead
of greedy decoding (one answer), you **sample many diverse reasoning paths** at temperature > 0 and
take the **majority answer**. The intuition: correct reasoning paths tend to *converge* on the same
answer, while errors are *diverse* and get outvoted. It notably improves chain-of-thought reasoning
accuracy with zero training.

**Why it's S-tier.**
- LLM reasoning is the hottest topic in ML right now.
- The **compute-vs-accuracy curve** (accuracy as a function of the number of samples N) is a clean,
  publishable-looking plot.
- "Sample 20 times and vote" is easy to explain to a grader and easy to demo.

**The tweak.** Tune the **voting scheme**: number of samples N, temperature T, **weighted voting** by
the model's own token log-probabilities/confidence, or self-consistency vs best-of-N with a verifier.
Research question: how much accuracy per unit of compute, and **where does it saturate?**

**Data / baseline / metric.** A small open LLM you can run on Colab (e.g. Qwen2.5-0.5B/1.5B,
Llama-3.2-1B/3B, Phi-3-mini) on SST-2 / AG News or a small multiple-choice reasoning set (ARC /
CommonsenseQA subset). Baseline: greedy → self-consistency. Metrics: **accuracy vs N× compute**,
plus calibration.

**Depth component (ES 678).** A voting-budget audit + variance across random seeds + a "why does it
work" analysis (diversity of wrong answers vs convergence of correct ones).

**Interaction (ES 335).** Vote-count slider → accuracy.

**Reality check.** This is the S-tier pick with the **noisiest evaluation** — small models, sampling
variance, and cost. Evaluation is the rubric's biggest bucket (20 pts), so budget time for multiple
seeds and error bars, and keep the test set small and fixed.

---

## 6. B1 — Anomaly Transformer (association-discrepancy tweak)

**What it actually is.** A transformer for **time-series anomaly detection** built on a specific
insight about attention. Every time point's attention has two components:
- a **prior-association** — a learnable Gaussian kernel over *adjacent* points (a locality bias),
- a **series-association** — the *learned* attention that can reach distant, informative points.

**The insight.** Anomalies are characterized by an **association discrepancy** — anomalous points'
attention stays locally concentrated instead of linking to distant informative points. So the **KL
divergence** between the series-association and the prior-association becomes the anomaly signal.

They also add a **minimax strategy**: minimize the discrepancy for normal points while *maximizing*
it for anomalies (via a stop-gradient on the prior term), which sharpens the separation. The final
anomaly score multiplies the reconstruction error by the association discrepancy.

**Why it's S-tier.**
- Attention-internals research + a dramatic time-series demo.
- The anomaly scores are easy to plot and inspect (spikes vs ground-truth labels).
- The association-discrepancy mechanism is genuinely clever and worth explaining.

**The tweak.** Modify the components: change the **prior** (Gaussian → a learned or Laplace kernel),
change the **discrepancy measure** (KL → symmetric JS or a learnable divergence), or change the
**minimax weighting**. Compare detection quality.

**Data / baseline / metric.** Public benchmarks: **SMD, SWaT, PSM, MSL, SMAP**. Baselines:
reconstruction methods (Autoencoder, USAD, LSTM-AE) → Anomaly Transformer. Metrics: **F1**,
precision/recall, and **VUS** (volume under the ROC surface, the range-aware metric from the paper).

**Depth component (ES 678).** Prior ablation + a failure-mode audit of missed anomalies.

**Interaction (ES 335).** Time-series anomaly score plot with a threshold slider.

**Reality check.** Moderate PyTorch implementation effort (you write the dual-association attention
yourself), and SMD has ~24k timesteps × 38 features — fine on a GPU, slow on CPU. The minimax
training (with stop-gradients) is the fiddly part.

---

## References (all S-tier)

- Hollmann, Müller, Eggensperger, Hutter. *TabPFN: A Transformer That Solves Small Tabular
  Classification Problems in a Few Seconds.* ICLR 2023.
- Müller, Hollmann, Arango, Grabocka, Hutter. *Transformers Can Do Bayesian Inference.* ICLR 2022.
- Romano, Sesia, Candès. *Classification with Valid and Adaptive Coverage (APS).* NeurIPS 2020.
- Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test / RAPS.* 2021.
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* 2005.
- Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual
  Representations (SimCLR).* ICML 2020.
- Wang, Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity
  on the Hypersphere.* ICML 2020.
- Foret, Kleiner, Mobahi, Neyshabur. *Sharpness-Aware Minimization for Efficiently Improving
  Generalization (SAM).* ICLR 2021.
- Kwon, Kim, Park, et al. *ASAM: Adaptive Sharpness-Aware Minimization.* ICML 2021.
- Li, Xu, Taylor, Studer, Goldstein. *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018.
- Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. *Self-Consistency Improves Chain of
  Thought Reasoning in Language Models.* ICLR 2023.
- Xu, Wu, Wang, Long. *Anomaly Transformer: Time Series Anomaly Detection with Association
  Discrepancy.* ICLR 2022.
