# ES 335 / ES 678 · Course Project — 50 Project Ideas + Fanciest-First Ranking

> Built directly against the rubric: every idea has a **real research-paper anchor**, a genuine
> **algorithmic tweak** (a modification to the method, not just hyperparameter tuning), public
> data, a **simple baseline**, a **justified metric**, and a built-in **depth component**
> (the ES 678 20-point deliverable) or an **interaction** angle (the ES 335 20-point deliverable).

**Hard constraints from the PDFs**

- Proposal due **Thursday, Oct 15, 2026, 11:59 PM IST** (in the course portal, no upload).
- Final project due **Tuesday, Nov 17, 2026** → ~4 weeks of work.
- Shared ML reasoning = **80 pts**. Biggest buckets: **Evaluation design & metrics (20)**,
  **Baseline & method selection (15)**, Problem formulation (10), Data (10), Preprocessing &
  statistical reasoning (10), Results/errors/limitations (10), Reproducibility (5).
- ES 335 extra 20: clear website (10) + **one genuinely useful interaction** (5) + defense (5).
- ES 678 extra 20: **depth component** (10) — derivation, sensitivity/ablation,
  calibration/uncertainty, failure-mode audit, or deployment-risk analysis — + defense (5).
- A deep-learning model is **not required**; a modest or negative result can earn full credit.

**Reading the tags**

- 🟢 runs on a MacBook M5 (CPU/MPS) · 🟡 wants the Colab GPU
- ⭐ = easiest *and* fanciest (kept for reference, but ignore for the rankings below)

---

## Fanciest-First Ranking (difficulty deliberately ignored)

"Fancy" here = visually striking demo, trendy/novel method, surprising result, or high
wow-factor for a live website and a research paper.

### Tier S — maximum wow
| Rank | Idea | Why it's the fanciest |
|---:|---|---|
| 1 | **#2 TabPFN prior-family swap** | A *foundation model for tabular data* — the single trendiest idea in the list. Swapping the synthetic prior is a real research contribution, and "predicts in seconds, no training" demos beautifully. |
| 2 | **#17 Conformal prediction sets** | Elegant, distribution-free guarantees; "here is a set that contains the truth 90% of the time" is a killer interaction. |
| 3 | **#22 Annealed SimCLR temperature** | Self-supervised representation learning from scratch; the τ-schedule is a clean novel tweak and linear-probe results look impressive. |
| 4 | **#25 SAM adaptive ρ** | Sharpness-Aware Minimization + loss-landscape visualizations = very "modern ML" energy. |
| 5 | **#43 Self-consistency voting (small LLM)** | LLM inference-time reasoning; sample-and-vote is trendy and the compute-vs-accuracy curve is a great story. |
| 6 | **B18 Anomaly Transformer (bonus)** | Attention-based anomaly detection with an association-discrepancy KL prior; time-series anomaly demos are dramatic. |

### Tier A — very fancy
| Rank | Idea | Why |
|---:|---|---|
| 7 | **#16 Evidential deep learning** | Predicts a Dirichlet, giving "I don't know" *for free*; the uncertainty funnel plots are gorgeous. |
| 8 | **#15 Deep Deterministic Uncertainty (DDU)** | Single-pass uncertainty that rivals ensembles; spectral-normalized features are advanced and clean. |
| 9 | **#14 SWAG** | A Gaussian posterior over weights from SGD iterates — Bayesian deep learning with a real derivation for the depth component. |
| 10 | **#19 Energy-based OOD detection** | Thermodynamic framing; "free energy beats softmax" is counter-intuitive and memorable. |
| 11 | **#41 LoRA rank & placement** | Parameter-efficient fine-tuning of a transformer; adapter placement heatmaps are a beautiful ablation. |
| 12 | **#46 Concept Bottleneck Models** | Interpretable-by-design networks; interactive concept interventions are the fanciest possible website demo. |
| 13 | **#39 Sequential-recommendation masking objective** | BERT4Rec/SASRec masking-objective redesign; sequences + rankings feel like a production recsys. |
| 14 | **#28 EfficientNet compound scaling** | Deriving a scaling law and fitting it is a genuinely research-flavored deliverable. |
| 15 | **#33 PatchTST patch design** | Patch-based transformers are SOTA-adjacent and the patch-length ablation is very publishable-looking. |
| 16 | **#45 LLM confidence calibration** | Verbalized vs token-probability confidence; hot topic, weird results, great discussion. |

### Tier B — fancy, high ceiling
| Rank | Idea | Why |
|---:|---|---|
| 17 | **#35 GCNII adaptive decay** | Over-smoothing curves and deep graph nets; the depth-vs-accuracy plot writes itself. |
| 18 | **#24 MixUp/CutMix hybrid** | Image-mixing policy; mixed-sample visualizations are eye-catching and the tweak is easy to defend. |
| 19 | **#50 DoubleML vs causal forests** | Causal inference with refutation tests; feels far more advanced than "classification accuracy". |
| 20 | **#47 Grad-CAM vs IG vs SHAP faithfulness** | Explaining the explainers; deletion/insertion curves + a sanity-check audit. |
| 21 | **#23 Anomaly — Deep SVDD on pretrained features** | One-class detection with an embedding-space boundary; elegant geometry story. |
| 22 | **#42 SetFit few-shot** | Contrastive few-shot that beats fine-tuning with 8 examples/class; the sample-efficiency curve is a great plot. |
| 23 | **#31 N-BEATS interpretable bases** | Decomposition into trend/seasonality basis functions; clean, interpretable forecasts. |
| 24 | **#27 Knowledge distillation T/α** | Teacher-student; soft-label visualizations and a distillation grid. |
| 25 | **#44 Text augmentation (EDA/back-translation)** | Back-translation pipelines and augmentation ablations look sophisticated. |
| 26 | **#37 Graph-diffusion kernels** | Heat vs personalized-PageRank propagation; a principled spectral story for the derivation. |

### Tier C — solid and impressive if executed well
| Rank | Idea |
|---:|---|
| 27 | **#11 Temperature scaling + Mahalanobis** (best calibration *and* OOD in one pipeline) |
| 28 | **#12 Focal loss as a calibrator** |
| 29 | **#13 Beta/Dirichlet calibration** (derivation-friendly) |
| 30 | **#6 Adaptive focal loss on fraud** |
| 31 | **#7 LDAM + deferred reweighting** |
| 32 | **#9 Asymmetric loss for multi-label** |
| 33 | **#1 PLR/periodic embeddings** |
| 34 | **#4 Soft feature gating for tabular** |
| 35 | **#10 Label smoothing × calibration under imbalance** |
| 36 | **#49 Adversarial debiasing** |
| 37 | **#48 Equalized-odds post-processing** |
| 38 | **#26 Stochastic-depth schedule** |
| 39 | **#29 Confidence-weighted TTA** |
| 40 | **#40 BPR vs NCF objective** |
| 41 | **#34 Probabilistic interval calibration** |
| 42 | **#30 Robust RevIN** |
| 43 | **#32 DLinear + decomposition** |
| 44 | **#8 GHM vs Focal** |
| 45 | **#20 ReAct adaptive clipping** |
| 46 | **#18 ODIN ε + temperature** |
| 47 | **#3 TabNet sparsemax→entmax** |
| 48 | **#36 Annealed DropEdge** |
| 49 | **#5 Ordered-boosting shrinkage** |
| 50 | **#38 LightGCN learned layer weights** |

---

# The Full Catalog (all 50)

## A. Tabular & structured data

### 1. PLR / periodic numerical embeddings
- **Papers:** *On Embeddings for Numerical Features in Tabular Deep Learning* (Gorishniy et al., 2022); *Revisiting Deep Learning Models for Tabular Data* / FT-Transformer (Gorishniy et al., 2021).
- **The tweak:** replace the linear per-feature projection inside FT-Transformer with **periodic (PLR) activations**.
- **Data:** OpenML CC18 / UCI (adult, covertype subset). One row = one instance.
- **Baseline → Method:** XGBoost/CatBoost → FT-Transformer + PLR embeddings.
- **Metric:** ROC-AUC, log-loss (justify why not accuracy).
- **Depth (ES 678):** embedding-type × sample-size ablation + calibration.
- **Interaction (ES 335):** toggle embedding type and watch AUROC/loss bars shift.

### 2. TabPFN prior-family swap
- **Papers:** *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Few Seconds* (Hollmann et al., 2023).
- **The tweak:** modify the **synthetic prior** (e.g. add heavy-tailed / correlated features) used to generate the training distribution and measure few-shot accuracy.
- **Data:** small OpenML datasets (≤1000 rows, ≤10 classes, ≤100 features) — TabPFN's intended regime.
- **Baseline → Method:** logistic regression / GBDT → TabPFN variant.
- **Metric:** ROC-AUC, accuracy, ECE.
- **Depth (ES 678):** prior-family sensitivity analysis.
- **Interaction (ES 335):** drop in a tiny CSV, compare TabPFN vs GBDT live.

### 3. TabNet sparsemax → entmax attention
- **Papers:** *TabNet: Attentive Interpretable Tabular Learning* (Arik & Pfister, 2019).
- **The tweak:** swap the sparse-attention function `sparsemax` for `entmax-α` (learn/tune α).
- **Data:** UCI adult / covertype.
- **Baseline → Method:** plain MLP → TabNet.
- **Metric:** accuracy, ECE.
- **Depth (ES 678):** sparsity ablation (α vs selected features vs accuracy).
- **Interaction (ES 335):** show per-row feature masks that TabNet selects.

### 4. Soft feature gating for DL-tabular
- **Papers:** *Why do tree-based models still outperform deep learning on typical tabular data?* (Grinsztajn et al., 2022).
- **The tweak:** learn a **soft feature mask** before the MLP and test whether it closes the tree-vs-DL gap.
- **Data:** OpenML / UCI.
- **Baseline → Method:** CatBoost → gated MLP.
- **Metric:** ROC-AUC, log-loss.
- **Depth (ES 678):** failure audit on rotated / uninformative features (the paper's key claim).
- **Interaction (ES 335):** rotate a feature and watch both models degrade.

### 5. Ordered-boosting shrinkage tweak
- **Papers:** *CatBoost: unbiased boosting with categorical features* (Prokhorenkova et al., 2018).
- **The tweak:** modify the ordered target-statistic computation / add per-fold shrinkage.
- **Data:** KDD Cup / Amazon categorical datasets.
- **Baseline → Method:** one-hot GBDT → ordered boosting.
- **Metric:** accuracy / AUC.
- **Depth (ES 678):** leakage sensitivity (show target leakage without ordering).
- **Interaction (ES 335):** toggle "ordered" on/off and see train-vs-test gap.

## B. Imbalanced data & loss functions

### 6. Class-frequency-adaptive focal γ  ⭐
- **Papers:** *Focal Loss for Dense Object Detection* (Lin et al., 2017); *Class-Balanced Loss Based on Effective Number of Samples* (Cui et al., 2019).
- **The tweak:** per-class adaptive focusing parameter `γ_c = γ₀ · (1 − f_c)^p` instead of a fixed γ.
- **Data:** ULB Credit-Card Fraud (0.17% positives) or CIFAR-10-LT.
- **Baseline → Method:** cross-entropy, XGBoost with `scale_pos_weight` → adaptive focal.
- **Metric:** **PR-AUC** (not accuracy), F1, balanced accuracy.
- **Depth (ES 678):** γ₀/p sensitivity sweep + false-negative failure audit.
- **Interaction (ES 335):** decision-threshold slider showing the precision/recall trade-off live.

### 7. LDAM + deferred reweighting
- **Papers:** *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* (Cao et al., 2019); *Decoupling Representation and Classifier for Long-Tailed Recognition* (Kang et al., 2020).
- **The tweak:** tune the margin schedule and the "deferred" reweighting point.
- **Data:** CIFAR-10-LT, iNaturalist subset.
- **Baseline → Method:** CE → LDAM-DRW.
- **Metric:** balanced accuracy, many/medium/few-shot split accuracy.
- **Depth (ES 678):** margin ablation + two-stage decoupling analysis.
- **Interaction (ES 335):** per-class accuracy heatmap as you slide the imbalance ratio.

### 8. Gradient Harmonized (GHM) vs Focal
- **Papers:** *Gradient Harmonized Single-stage Detector* (Li et al., 2019).
- **The tweak:** replace focal weighting with **gradient-density** harmonizing; compare/combine.
- **Data:** fraud or medical imbalanced sets.
- **Baseline → Method:** CE → GHM → hybrid.
- **Metric:** PR-AUC.
- **Depth (ES 678):** gradient-density distributions.
- **Interaction (ES 335):** histogram of gradient density per class.

### 9. Asymmetric loss for multi-label
- **Papers:** *Asymmetric Loss For Multi-Label Classification* (Ridnik et al., 2021).
- **The tweak:** asymmetric focusing `γ₋, γ₊` + probability shifting.
- **Data:** VOC/COCO subset or ChestX-ray (multi-label).
- **Baseline → Method:** BCE → ASL.
- **Metric:** mAP.
- **Depth (ES 678):** γ₋/γ₊ ablation.
- **Interaction (ES 335):** per-label probability bars with a threshold slider.

### 10. Label smoothing × calibration under imbalance
- **Papers:** *When Does Label Smoothing Help?* (Müller et al., 2019); *Calibrating Deep Neural Networks using Focal Loss* (Mukhoti et al., 2020).
- **The tweak:** test how smoothing strength interacts with calibration under class imbalance.
- **Data:** CIFAR-10-LT.
- **Baseline → Method:** CE → CE+smoothing → focal.
- **Metric:** ECE, NLL, balanced accuracy.
- **Depth (ES 678):** ECE-vs-smoothing sweep.
- **Interaction (ES 335):** reliability diagram that updates with the smoothing slider.

## C. Calibration & uncertainty

### 11. Temperature scaling + Mahalanobis  ⭐
- **Papers:** *On Calibration of Modern Neural Networks* (Guo et al., 2017); *A Simple Unified Framework for Detecting OOD and Adversarial Samples* (Lee et al., 2018).
- **The tweak:** `score = α · calibrated_confidence − β · Mahalanobis_distance`, with α, β fit **on validation only**.
- **Data:** MNIST → FashionMNIST + KMNIST (OOD). One row = one 28×28 image.
- **Baseline → Method:** max-softmax, MC-Dropout → the combined score.
- **Metric:** AUROC, FPR@95, ECE, accuracy.
- **Depth (ES 678):** α/β sensitivity sweep + OOD failure audit.
- **Interaction (ES 335):** drag/choose an image → raw vs calibrated confidence + OOD gauge.

### 12. Focal loss as a calibrator
- **Papers:** *Calibrating Deep Neural Networks using Focal Loss* (Mukhoti et al., 2020).
- **The tweak:** sweep focal γ as a *calibration* knob and pair with temperature scaling.
- **Data:** CIFAR-10 / CIFAR-100.
- **Baseline → Method:** CE → focal-trained net.
- **Metric:** ECE, NLL.
- **Depth (ES 678):** reliability diagrams + γ analysis.
- **Interaction (ES 335):** compare reliability diagrams side by side.

### 13. Beta / Dirichlet calibration
- **Papers:** *Beta calibration* (Kull et al., 2017); *Beyond temperature scaling: Dirichlet calibration* (Kull et al., 2019).
- **The tweak:** replace single-parameter temperature with richer calibration families; add regularization.
- **Data:** CIFAR-10.
- **Baseline → Method:** uncalibrated → temperature → beta/Dirichlet.
- **Metric:** ECE, class-wise ECE.
- **Depth (ES 678):** the derivation of the calibration family.
- **Interaction (ES 335):** class-wise ECE matrix heatmap.

### 14. SWAG vs Deep Ensembles at low compute
- **Papers:** *A Simple Baseline for Bayesian Uncertainty in Deep Learning* (Maddox et al., 2019); *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (Lakshminarayanan et al., 2017).
- **The tweak:** SWAG (Gaussian posterior from SGD iterates) vs ensembling; tune the moment-collection window.
- **Data:** CIFAR-10.
- **Baseline → Method:** single net → SWAG → deep ensemble.
- **Metric:** NLL, ECE, OOD AUROC.
- **Depth (ES 678):** posterior-approximation comparison + calibration.
- **Interaction (ES 335):** uncertainty bands per sample.

### 15. Deep Deterministic Uncertainty (DDU)
- **Papers:** *Deep Deterministic Uncertainty* (Mukhoti et al., 2021/2023).
- **The tweak:** spectral-normalized residual features + feature-space density; tune the normalization.
- **Data:** MNIST/CIFAR-10.
- **Baseline → Method:** softmax → DDU.
- **Metric:** ECE, OOD AUROC.
- **Depth (ES 678):** feature-space audit (cluster separation vs error).
- **Interaction (ES 335):** 2-D feature embedding scatter colored by error.

### 16. Evidential deep learning
- **Papers:** *Evidential Deep Learning to Quantify Classification Uncertainty* (Sensoy et al., 2018).
- **The tweak:** predict Dirichlet evidence; tune the evidence-regularization coefficient.
- **Data:** MNIST / FashionMNIST.
- **Baseline → Method:** softmax → evidential net.
- **Metric:** ECE, OOD AUROC, "vacuous" fraction.
- **Depth (ES 678):** evidence ablation.
- **Interaction (ES 335):** simplex visualization of the predicted Dirichlet.

### 17. Conformal prediction sets
- **Papers:** *Classification with Valid and Adaptive Coverage* (Romano et al., 2020); Conformal Prediction (Vovk et al.).
- **The tweak:** design/tune the **conformal score** (e.g. APS vs RAPS vs a temperature-scaled variant).
- **Data:** CIFAR-10 + an OOD set.
- **Baseline → Method:** softmax thresholding → conformal sets.
- **Metric:** empirical coverage, average set size.
- **Depth (ES 678):** empirical coverage validation.
- **Interaction (ES 335):** slide target coverage → watch set size change live.

### 18. ODIN ε + temperature
- **Papers:** *A Baseline for Detecting Misclassified and OOD Examples* (Hendrycks & Gimpel, 2017); *Enhancing The Reliability of OOD Detectors* / ODIN (Liang et al., 2018).
- **The tweak:** jointly tune the input perturbation ε and temperature.
- **Data:** CIFAR-10 → SVHN.
- **Baseline → Method:** max-softmax → ODIN.
- **Metric:** AUROC, FPR@95.
- **Depth (ES 678):** ε sensitivity.
- **Interaction (ES 335):** ε slider showing the score separation.

## D. Out-of-distribution & anomaly detection

### 19. Energy score vs max-softmax
- **Papers:** *Energy-based Out-of-distribution Detection* (Liu et al., 2020).
- **The tweak:** free-energy objective with a margin term; compare to temperature scaling.
- **Data:** CIFAR-10 → SVHN / TinyImageNet.
- **Baseline → Method:** max-softmax → energy.
- **Metric:** AUROC, FPR@95.
- **Depth (ES 678):** ablation + calibration analysis.
- **Interaction (ES 335):** energy histogram for in- vs out-distribution.

### 20. ReAct adaptive clipping
- **Papers:** *ReAct: Out-of-distribution Detection With Rectified Activations* (Sun et al., 2021).
- **The tweak:** choose the activation-clipping quantile adaptively from validation.
- **Data:** CIFAR-10 → SVHN.
- **Baseline → Method:** energy score → ReAct.
- **Metric:** AUROC.
- **Depth (ES 678):** quantile sweep.
- **Interaction (ES 335):** clip-percentile slider.

### 21. Deep SVDD on pretrained features
- **Papers:** *Deep One-Class Classification* (Ruff et al., 2018).
- **The tweak:** run SVDD in a pretrained embedding space; tune the hypersphere init/radius.
- **Data:** MNIST/CIFAR-10 one-class splits.
- **Baseline → Method:** autoencoder reconstruction → Deep SVDD.
- **Metric:** AUROC.
- **Depth (ES 678):** feature-choice ablation.
- **Interaction (ES 335):** 2-D embedding with the learned boundary drawn.

## E. Vision & representation learning

### 22. Annealed SimCLR temperature
- **Papers:** *A Simple Framework for Contrastive Learning of Visual Representations* (Chen et al., 2020).
- **The tweak:** anneal the NT-Xent temperature τ over training instead of fixing it.
- **Data:** CIFAR-10 / STL-10.
- **Baseline → Method:** fixed-τ SimCLR → annealed-τ.
- **Metric:** linear-probe accuracy.
- **Depth (ES 678):** τ-schedule ablation.
- **Interaction (ES 335):** t-SNE of learned embeddings updating over epochs.

### 23. Augmentation-strength curriculum
- **Papers:** *RandAugment* (Cubuk et al., 2020); *AugMix* (Hendrycks et al., 2020).
- **The tweak:** ramp augmentation strength over training (curriculum) instead of a constant policy.
- **Data:** CIFAR-10.
- **Baseline → Method:** fixed RandAugment → curriculum.
- **Metric:** accuracy + robustness to corruption.
- **Depth (ES 678):** augmentation ablation + calibration.
- **Interaction (ES 335):** show augmented samples at different training stages.

### 24. MixUp / CutMix hybrid policy
- **Papers:** *mixup* (Zhang et al., 2018); *CutMix* (Yun et al., 2019).
- **The tweak:** a new convex mixing rule blending the two (or an adaptive λ).
- **Data:** CIFAR-10/100.
- **Baseline → Method:** mixup → CutMix → hybrid.
- **Metric:** accuracy, ECE.
- **Depth (ES 678):** policy sweep.
- **Interaction (ES 335):** λ slider showing mixed images + resulting predictions.

### 25. SAM adaptive ρ
- **Papers:** *Sharpness-Aware Minimization for Efficiently Improving Generalization* (Foret et al., 2021).
- **The tweak:** make the perturbation radius ρ adaptive per layer/iteration.
- **Data:** CIFAR-10.
- **Baseline → Method:** SGD/Adam → SAM.
- **Metric:** accuracy + loss-landscape flatness.
- **Depth (ES 678):** ρ sensitivity + landscape plots.
- **Interaction (ES 335):** 2-D loss-landscape visualization.

### 26. Stochastic-depth schedule
- **Papers:** *Deep Networks with Stochastic Depth* (Huang et al., 2016).
- **The tweak:** linear vs uniform vs cosine survival-probability schedules.
- **Data:** CIFAR-10.
- **Baseline → Method:** no drop → stochastic depth variants.
- **Metric:** accuracy vs depth.
- **Depth (ES 678):** depth + schedule ablation.
- **Interaction (ES 335):** layer-usage heatmap over training.

### 27. Knowledge distillation T/α
- **Papers:** *Distilling the Knowledge in a Neural Network* (Hinton et al., 2015).
- **The tweak:** tune the temperature/α trade-off; add feature-matching loss.
- **Data:** CIFAR-10.
- **Baseline → Method:** student-only → KD.
- **Metric:** student accuracy, ECE.
- **Depth (ES 678):** T/α grid + calibration.
- **Interaction (ES 335):** soft-label bar charts at different T.

### 28. EfficientNet compound scaling
- **Papers:** *EfficientNet: Rethinking Model Scaling* (Tan & Le, 2019).
- **The tweak:** rebalance width/depth/resolution for a fixed compute budget and fit the scaling relation.
- **Data:** CIFAR-100 / ImageNette.
- **Baseline → Method:** manual scaling → compound scaling.
- **Metric:** accuracy per FLOP.
- **Depth (ES 678):** fit and validate the scaling law.
- **Interaction (ES 335):** accuracy-vs-FLOPs Pareto curve.

### 29. Confidence-weighted test-time augmentation
- **Papers:** *Pitfalls of In-Domain Uncertainty Estimation and Ensembling* (Ashukha et al., 2020) + TTA literature.
- **The tweak:** weight each TTA view by its calibrated confidence instead of averaging equally.
- **Data:** CIFAR-10.
- **Baseline → Method:** single view → uniform TTA → confidence-weighted TTA.
- **Metric:** accuracy, ECE.
- **Depth (ES 678):** TTA ablation.
- **Interaction (ES 335):** toggle TTA and see confidence shift.

## F. Time series

### 30. Robust RevIN
- **Papers:** *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift* (Kim et al., 2022).
- **The tweak:** replace RevIN's mean/std with robust statistics (median/IQR) or a learned normalization.
- **Data:** ETTh1 / Electricity.
- **Baseline → Method:** no-norm → RevIN → robust RevIN.
- **Metric:** MASE, sMAPE (scale-free — justify).
- **Depth (ES 678):** normalization ablation.
- **Interaction (ES 335):** pick a series + horizon → forecast vs actual.

### 31. N-BEATS interpretable bases
- **Papers:** *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting* (Oreshkin et al., 2019).
- **The tweak:** add/exchange trend and seasonality basis blocks.
- **Data:** M4.
- **Baseline → Method:** seasonal-naïve → N-BEATS.
- **Metric:** sMAPE, MASE.
- **Depth (ES 678):** basis ablation.
- **Interaction (ES 335):** decomposed trend/seasonality plots.

### 32. DLinear + decomposition
- **Papers:** *Are Transformers Effective for Time Series Forecasting?* (Zeng et al., 2023).
- **The tweak:** add a season-trend decomposition branch to a linear model and beat/serve-as-baseline for a transformer.
- **Data:** ETTh / Electricity.
- **Baseline → Method:** transformer → DLinear/decomposition.
- **Metric:** MSE, MAE.
- **Depth (ES 678):** honest failure analysis of where attention helps.
- **Interaction (ES 335):** compare a transformer vs linear forecast interactively.

### 33. PatchTST patch design
- **Papers:** *A Time Series is Worth 64 Words* / PatchTST (Nie et al., 2023).
- **The tweak:** patch length/stride/channel-independence sensitivity.
- **Data:** ETTh1 / Electricity.
- **Baseline → Method:** point-wise transformer → PatchTST variants.
- **Metric:** MSE, MAE.
- **Depth (ES 678):** patch-design ablation.
- **Interaction (ES 335):** patch-window slider.

### 34. Probabilistic interval calibration
- **Papers:** *DeepAR* (Salinas et al., 2020); *Temporal Fusion Transformers* (Lim et al., 2021).
- **The tweak:** quantile-loss tweak + post-hoc interval calibration.
- **Data:** Electricity.
- **Baseline → Method:** point forecast → quantile model → calibrated intervals.
- **Metric:** CRPS, empirical coverage.
- **Depth (ES 678):** interval calibration / reliability.
- **Interaction (ES 335):** coverage slider showing widening prediction bands.

## G. Graphs & recommenders

### 35. GCNII adaptive decay
- **Papers:** *Simple and Deep Graph Convolutional Networks* (Chen et al., 2020); *Simplifying Graph Convolutional Networks* (Wu et al., 2019).
- **The tweak:** initial residual + adaptive decay λ (vs the fixed λ and α).
- **Data:** Cora / Citeseer / ogbn-arxiv.
- **Baseline → Method:** 2-layer GCN → GCNII.
- **Metric:** node-classification accuracy (+ calibration).
- **Depth (ES 678):** over-smoothing curve vs depth.
- **Interaction (ES 335):** visualize a misclassified node's neighborhood.

### 36. Annealed DropEdge
- **Papers:** *DropEdge: Towards Deep GCN* (Rong et al., 2020).
- **The tweak:** schedule the edge-dropout rate (start high, anneal down).
- **Data:** Cora / Citeseer.
- **Baseline → Method:** GCN → DropEdge.
- **Metric:** accuracy.
- **Depth (ES 678):** rate-schedule ablation.
- **Interaction (ES 335):** show the graph thinning as rate increases.

### 37. Graph-diffusion kernels
- **Papers:** *Diffusion Improves Graph Learning* (Klicpera et al., 2019); *GRAND* (Chamberlain et al., 2021).
- **The tweak:** heat-kernel vs personalized-PageRank propagation, or a learned diffusion.
- **Data:** Cora / Citeseer.
- **Baseline → Method:** GCN → graph diffusion.
- **Metric:** accuracy.
- **Depth (ES 678):** kernel ablation (derivation-friendly).
- **Interaction (ES 335):** diffusion-time slider.

### 38. LightGCN learned layer weights  ⭐
- **Papers:** *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (He et al., 2020).
- **The tweak:** learn the layer-combination weights α_k instead of uniform averaging.
- **Data:** MovieLens-1M / 100K. One row = one (user, item) interaction.
- **Baseline → Method:** popularity, BPR/ALS → LightGCN variant.
- **Metric:** Recall@K, NDCG@K.
- **Depth (ES 678):** depth + weighting ablation.
- **Interaction (ES 335):** enter a user → top-K recommendations.

### 39. Sequential-recommendation masking objective
- **Papers:** *SASRec* (Kang & McAuley, 2018); *BERT4Rec* (Sun et al., 2019).
- **The tweak:** redesign the masking/next-item objective (e.g. mixed cloze + next-item).
- **Data:** Amazon / MovieLens.
- **Baseline → Method:** SASRec → BERT4Rec → hybrid objective.
- **Metric:** NDCG@10, HR@10.
- **Depth (ES 678):** masking-objective ablation.
- **Interaction (ES 335):** type a history → next-item predictions.

### 40. BPR vs NCF objective
- **Papers:** *Neural Collaborative Filtering* (He et al., 2017); *BPR: Bayesian Personalized Ranking* (Rendle et al., 2009).
- **The tweak:** implicit-feedback loss tweak (BPR vs pointwise vs hybrid).
- **Data:** MovieLens.
- **Baseline → Method:** BPR → NCF.
- **Metric:** Recall@K.
- **Depth (ES 678):** cold-start / sparsity failure audit.
- **Interaction (ES 335):** sparsity slider → recall curve.

## H. NLP / text

### 41. LoRA rank & placement
- **Papers:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2022).
- **The tweak:** where to inject low-rank adapters and at what rank.
- **Data:** SST-2 / AG News (small transformer backbone).
- **Baseline → Method:** full fine-tune → LoRA variants.
- **Metric:** accuracy, ECE.
- **Depth (ES 678):** rank/placement ablation.
- **Interaction (ES 335):** adapter-placement heatmap.

### 42. SetFit pair-generation tweak
- **Papers:** *Efficient Few-Shot Learning Without Prompts* / SetFit (Tunstall et al., 2022); *Sentence-BERT* (Reimers & Gurevych, 2019).
- **The tweak:** change how contrastive pairs are generated/curated.
- **Data:** AG News with N examples per class.
- **Baseline → Method:** fine-tuned head → SetFit.
- **Metric:** accuracy vs N.
- **Depth (ES 678):** sample-efficiency curve.
- **Interaction (ES 335):** samples-per-class slider → accuracy curve.

### 43. Self-consistency voting
- **Papers:** *Self-Consistency Improves Chain of Thought Reasoning* (Wang et al., 2022).
- **The tweak:** sample-and-vote vs greedy; tune the voting budget and temperature.
- **Data:** SST-2 / AG News / small QA set.
- **Baseline → Method:** greedy → self-consistency.
- **Metric:** accuracy vs compute.
- **Depth (ES 678):** voting-budget audit.
- **Interaction (ES 335):** vote-count slider → accuracy.

### 44. Text augmentation (EDA / back-translation)
- **Papers:** *EDA: Easy Data Augmentation* (Wei & Zou, 2019).
- **The tweak:** measure where augmentation helps and whether it leaks.
- **Data:** SST-2 / AG News.
- **Baseline → Method:** no-aug → EDA → back-translation.
- **Metric:** accuracy, variance across seeds.
- **Depth (ES 678):** augmentation ablation + leakage check.
- **Interaction (ES 335):** show augmented sentence variants.

### 45. LLM confidence calibration
- **Papers:** *Language Models (Mostly) Know What They Know* (Kadavath et al., 2022); *Calibrated Language Models Must Hallucinate* / LLM calibration (Zhao et al., 2021).
- **The tweak:** verbalized confidence vs token-probability confidence vs temperature scaling.
- **Data:** small QA / classification sets.
- **Baseline → Method:** token-prob → verbalized → recalibrated.
- **Metric:** ECE, AUROC.
- **Depth (ES 678):** method comparison.
- **Interaction (ES 335):** confidence-vs-correctness scatter.

## I. Interpretability

### 46. Concept Bottleneck Models
- **Papers:** *Concept Bottleneck Models* (Koh et al., 2020); *TCAV* (Kim et al., 2018).
- **The tweak:** concept set + intervention strategy; measure the accuracy–interpretability trade-off.
- **Data:** CIFAR-10 or a medical subset with interpretable attributes.
- **Baseline → Method:** black-box CNN → CBM.
- **Metric:** accuracy, concept accuracy.
- **Depth (ES 678):** intervention analysis.
- **Interaction (ES 335):** adjust a concept value → watch the prediction change.

### 47. Grad-CAM vs Integrated Gradients vs SHAP
- **Papers:** *Grad-CAM* (Selvaraju et al., 2017); *Integrated Gradients* (Sundararajan et al., 2017); *SHAP* (Lundberg & Lee, 2017); *Attention is not Explanation* (Jain & Wallace, 2019).
- **The tweak:** a faithfulness metric (deletion/insertion AUC) + sanity checks.
- **Data:** CIFAR or tabular.
- **Baseline → Method:** compare attribution methods.
- **Metric:** deletion/insertion AUC.
- **Depth (ES 678):** sanity-check audit.
- **Interaction (ES 335):** hover a pixel → attribution heatmap.

## J. Fairness & causal (higher-risk, high-reward)

### 48. Equalized-odds post-processing
- **Papers:** *Equality of Opportunity in Supervised Learning* (Hardt et al., 2016); *Data preprocessing techniques for classification without discrimination* (Kamiran & Calders, 2012).
- **The tweak:** tune the post-processing randomizer for the fairness–accuracy frontier.
- **Data:** UCI Adult / COMPAS.
- **Baseline → Method:** unconstrained → post-processed.
- **Metric:** DP/EOD gap + accuracy.
- **Depth (ES 678):** trade-off-curve audit.
- **Interaction (ES 335):** fairness-vs-accuracy slider.

### 49. Adversarial debiasing
- **Papers:** *Mitigating Unwanted Biases with Adversarial Learning* (Zhang et al., 2018).
- **The tweak:** adjust the adversary's loss weighting / gradient-reversal strength.
- **Data:** UCI Adult / COMPAS.
- **Baseline → Method:** baseline → adversarial debiasing.
- **Metric:** fairness–accuracy pair.
- **Depth (ES 678):** sensitivity analysis.
- **Interaction (ES 335):** adversary-strength slider.

### 50. DoubleML vs causal forests
- **Papers:** *Double/Debiased Machine Learning* (Chernozhukov et al., 2018); *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests* (Wager & Athey, 2018); *CFRNet* (Shalit et al., 2017).
- **The tweak:** compare estimators; tune the nuisance models / propensity clipping.
- **Data:** IHDP / Lalonde, or a Kaggle A/B dataset.
- **Baseline → Method:** naive difference → DoubleML / causal forest.
- **Metric:** ATE + CI coverage.
- **Depth (ES 678):** refutation tests.
- **Interaction (ES 335):** subgroup treatment-effect plot.

---

# Bonus ideas (beyond the 50)

### B1. Anomaly Transformer association-discrepancy tweak  🟡
- **Papers:** *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* (Xu et al., 2022).
- **The tweak:** modify the KL prior over attention; alternative discrepancy measure.
- **Data:** SMD / SWaT / PSM.
- **Baseline → Method:** reconstruction → Anomaly Transformer.
- **Metric:** F1, VUS.
- **Depth:** prior ablation.

### B2. CSI: contrastive self-supervised anomaly detection  🟡
- **Papers:** *CSI: Novelty Detection via Contrastive Learning* (Tack et al., 2020).
- **The tweak:** augmentation-set design for the contrastive pretext task.
- **Data:** CIFAR-10 / MVTec.
- **Baseline:** autoencoder → CSI. **Metric:** AUROC.

### B3. VICReg / Barlow Twins (collapse-free SSL)  🟡
- **Papers:** *VICReg* (Bardes et al., 2022); *Barlow Twins* (Zbontar et al., 2021).
- **The tweak:** variance–covariance / redundancy-reduction weighting.
- **Data:** CIFAR-10. **Metric:** linear-probe accuracy.

### B4. DropBlock block-size tuning  🟢
- **Papers:** *DropBlock: A regularization method for convolutional networks* (Ghiasi et al., 2018).
- **The tweak:** block-size schedule. **Data:** CIFAR. **Metric:** accuracy.

### B5. Grad-CAM sanity-check replication  🟢
- **Papers:** *Sanity Checks for Saliency Maps* (Adebayo et al., 2018).
- **The tweak:** model-parameter randomization test. **Data:** MNIST/CIFAR.

### B6. BatchNorm vs GroupNorm under small batch  🟢
- **Papers:** *Group Normalization* (Wu & He, 2018).
- **The tweak:** batch-size sensitivity + a hybrid norm. **Metric:** accuracy, ECE.

### B7. Attention temperature in ViT  🟡
- **Papers:** *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2021); *Pay Less Attention with Lightweight and Dynamic Convolutions* (Wu et al., 2019).
- **The tweak:** scale attention logits / learn a temperature. **Data:** CIFAR-10. **Metric:** accuracy.

### B8. Temporal Fusion Transformer variable-selection tweak  🟡
- **Papers:** *Temporal Fusion Transformers* (Lim et al., 2021).
- **The tweak:** modify variable-selection network regularization. **Metric:** MSE + interpretation.

### B9. Random Forest vs TabPFN on tiny data  🟢
- **Papers:** TabPFN (Hollmann et al., 2023).
- **The tweak:** sample-size crossover analysis. **Metric:** AUROC vs n.

### B10. Conformal prediction for regression  🟢
- **Papers:** *Conformalized Quantile Regression* (Romano et al., 2019).
- **The tweak:** adaptivity of the interval width. **Metric:** coverage, width.

---

# Rubric checklist (applies to every project)

- [ ] **Frame the problem** — state the question, what one row represents, the target, and *when* a prediction is made.
- [ ] **Data** — source, size, key variables, access limits, quality issues.
- [ ] **Preprocessing** — fit imputation/scaling/feature-selection on **training only** (the 10-pt leakage trap).
- [ ] **Baseline** — a genuinely simple one, compared fairly.
- [ ] **Method + tweak** — objective, assumptions, why it fits.
- [ ] **Evaluation** — fix the split/validation scheme *before* comparing; keep test data out of tuning; justify the metric.
- [ ] **Results, errors, limitations** — representative errors, uncertainty, no overclaiming, no causality from prediction.
- [ ] **Reproducibility** — code, data access, split details.
- [ ] **Track deliverable** — ES 335: useful interaction. ES 678: derivation / ablation / calibration / failure audit / deployment risk.
