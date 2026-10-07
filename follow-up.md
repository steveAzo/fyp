# Follow-Up Paper 1: Multivariate & Deep Learning Drift Detection
## Literature Background + Proposed Scope

Prepared for: Stephen Azongo (+ team)
Base paper: "When Configuration Matters More Than Algorithm" (Azongo & Sarpong, UG)
Purpose: get the team fully briefed before implementation begins.

---

## 0. Read this section first — prior art check

Before scoping anything, the team needs to know that **two papers already occupy a large chunk of the territory** the original "New Directions" document called novel. This doesn't kill the project, but it changes what we can honestly claim, so everyone should read this before opening a single dataset.

### 0.1 The closest neighbor: image domain

**Rabanser, S., Günnemann, S., & Lipton, Z. (2019). "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift." NeurIPS 2019.**
(arXiv: 1810.11953)

What it already did:
- Dimensionality reduction (PCA, autoencoders, and **Black-Box Shift Detection** — feeding data through a pretrained classifier and using its outputs as reduced features) **followed by** two-sample tests (MMD, KS, chi-squared) on MNIST and CIFAR-10.
- Explicitly varied the number of target/current samples available (10 up to 10,000) — a resource-constraint axis conceptually close to ours.
- Found that the pretrained-classifier + two-sample-test combination performs best across the shifts they tested.

**Implication for us:** the "first constraint-aware evaluation of multivariate detectors and embedding-space monitoring" claim in the New Directions doc is not accurate as stated. This exact combination (dimensionality reduction → two-sample test, on image data, under a sample-size constraint) has 65+ citations already. If we go into the vision domain, this is the paper every reviewer will hold us against, and we need to explain what we add beyond it.

### 0.2 The closest neighbor: text domain

**Feldhans, R. et al. (2021). "Drift Detection in Text Data with Document Embeddings." IDEAL 2021.**

What it already did:
- Compared drift detectors (MMD, Least-Squares Density Difference, KS test) directly on document embeddings (BoW at two dimensionalities, and BERT-768) on real datasets including Twitter data.
- Found multivariate detectors (MMD, LSDD) outperform univariate KS on high-dimensional embeddings, and that detectors perform better on smaller embedding dimensions.

A closely related, more recent paper worth skimming too:

**"Detecting covariate drift in text data using document embeddings and dimensionality reduction." (arXiv: 2309.10000, 2023).**
Runs TF-IDF, doc2vec, and BERT embeddings through KS and MMD, with and without PCA/LSA dimensionality reduction — essentially the exact pipeline the New Directions doc proposes for the NLP arm, already executed.

**Implication for us:** the NLP-drift sub-direction is not a green field either. Our angle, if we pursue it, has to be the **constraint grid** (monitoring frequency × window size × reference size — not just "does MMD beat KS on embeddings"), because that specific factorial design is not present in either paper above.

### 0.3 The honest bottom line

Nobody has run **your specific constraint grid** (F × W × R, full factorial, with the priority-hierarchy framing) on multivariate detectors or on embedding spaces. That is a legitimate, defensible novelty claim. "First to evaluate multivariate/embedding drift detection at all" is not — say that in a submission and a reviewer will find these papers in one search. Frame the contribution as: *does the "configuration beats algorithm" finding from Paper 1 replicate once the detector becomes multivariate / the feature space becomes a learned embedding?* That is a clean, answerable, novel question sitting directly on top of work the team already understands.

---

## 1. Core statistical methods to learn

### 1.1 Maximum Mean Discrepancy (MMD)
**Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). "A Kernel Two-Sample Test." Journal of Machine Learning Research, 13, 723–773.**

The foundational paper. Maps both samples into a reproducing kernel Hilbert space (RKHS) via a kernel function (usually RBF/Gaussian) and measures the distance between their mean embeddings. This is the standard multivariate analogue of what KS does for a single feature — the single most important paper to actually understand, not just cite. You only need: the core test statistic, the idea of kernel choice (bandwidth), and roughly how the permutation/bootstrap test for significance works. The full RKHS theory in the early sections can be skimmed.

### 1.2 Classifier-based two-sample tests
**Lopez-Paz, D., & Oquab, M. (2017). "Revisiting Classifier Two-Sample Tests." ICLR 2017.**

Train a classifier to distinguish "reference" from "current" samples. If it does meaningfully better than chance, the distributions differ; if it can't beat a coin flip, they're indistinguishable. Cheap, intuitive, and directly usable as one of our multivariate detectors. Much more readable than the Gretton paper — good second read.

### 1.3 HDDDM (Hellinger Distance Drift Detection Method)
**Ditzler, G., & Polikar, R. (2011). "Hellinger distance based drift detection for nonstationary environments." 2011 IEEE Symposium on Computational Intelligence in Dynamic and Uncertain Environments (CIDUE), pp. 41–48.**

A specific, named multivariate detector using Hellinger distance on binned histograms of multiple features jointly, with an adaptive threshold. This is one of the concrete detectors the New Directions doc names — we'd implement this one directly (it is not in scikit-learn; there are community re-implementations on GitHub worth checking before writing from scratch, e.g. search "HDDDM python implementation").

### 1.4 Black-Box Shift Detection / Estimation (BBSD/BBSE)
**Lipton, Z., Wang, Y.-X., & Smola, A. (2018). "Detecting and Correcting for Label Shift with Black Box Predictors." ICML 2018.**

Introduces the trick of using an arbitrary pretrained ("black box") predictor purely as a dimensionality-reduction step before running shift detection — feed data through the model, use its output layer (or an internal layer) as the reduced representation, then test that. This is the mechanism Rabanser et al. (2019) build on and is directly relevant to our embedding-space arm.

**Read order for this section: 1.1 → 1.2 → 1.4 → 1.3.** Read 0.1 (Rabanser) either just before or just after 1.4, since it's the direct extension of the BBSD idea.

---

## 2. Embedding-space / representation drift monitoring

- **Rabanser et al. (2019)** — already covered above, read in full.
- **Feldhans et al. (2021)** and **arXiv:2309.10000 (2023)** — already covered above, both short and directly actionable as method templates for the NLP arm.
- **McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426** (companion software paper: McInnes, Healy, Saul & Großberger, 2018, *Journal of Open Source Software*, 3(29), 861). UMAP is one of the two standard dimensionality-reduction options (alongside PCA) for shrinking an embedding vector before running a distribution test on it. You do not need the full Riemannian-geometry derivation — the practical usage (via the `umap-learn` Python package) is what matters here.

Optional, for context on how far the field has gone (skim only, not required reading before starting):
- Greco, S. & Cerquitelli, T. — "DriftLens," a BERT-embedding-based drift framework.
- "Concept Drift Adaptation in Text Stream Mining Settings: A Comprehensive Review" (arXiv:2312.02901, 2023) — broad survey, useful if we need a related-work paragraph on text-stream drift specifically.

---

## 3. Vision domain (if pursued)

**Hendrycks, D., & Dietterich, T. (2019). "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR 2019.**

Introduces CIFAR-10-C and ImageNet-C: the standard test sets with synthetic corruptions (blur, noise, weather effects, compression artifacts) applied at 5 severity levels. This is not a drift-detection paper itself — it's the standard mechanism the field uses to simulate distribution shift in images, and it would replace our current σ-shift injection approach if we take this arm. If pursued, our detectors run on embeddings produced by a pretrained ResNet or CLIP fed CIFAR-10 vs. CIFAR-10-C at increasing severity, mirroring our sudden/gradual/recurring injection logic by treating corruption severity as the "shift magnitude" axis.

---

## 4. Reading order (priority tiers)

**Tier 1 — everyone on the team reads before writing any code:**
1. Rabanser et al. (2019) — "Failing Loudly" (§0.1)
2. Gretton et al. (2012) — MMD (§1.1), core sections only
3. Feldhans et al. (2021) — text embeddings (§0.2)

**Tier 2 — read before implementing your assigned detector:**
4. Lopez-Paz & Oquab (2017) — classifier two-sample test
5. Ditzler & Polikar (2011) — HDDDM
6. Lipton, Wang & Smola (2018) — BBSD/BBSE

**Tier 3 — read only if your sub-team is assigned that domain arm:**
7. Hendrycks & Dietterich (2019) — vision corruption benchmarks (vision arm only)
8. arXiv:2309.10000 (2023) — text embeddings + dimensionality reduction (NLP arm only)
9. McInnes, Healy & Melville (2018) — UMAP (whoever owns dimensionality reduction)

---

## 5. Proposed scope (revised for a team, not a solo author)

Given more hands are now available, the full three-pronged extension from the New Directions doc becomes realistic **if split across sub-teams working in parallel on a shared experimental backbone**, rather than sequentially. Structure below assumes 3–4 people; adjust group sizes to whoever's actually on board.

### 5.1 Shared backbone (build first, before any domain work starts)

One person/pair owns this, and everyone else's arm depends on it:
- Port the existing constraint-grid harness (F × W × R factorial loop, statistical testing pipeline: Kruskal-Wallis, Mann-Whitney U + Bonferroni, rank-biserial r, Spearman's ρ) into a detector-agnostic form, so any detector (univariate or multivariate) can be dropped in without rewriting the experiment loop.
- Implement the three new multivariate detectors as interchangeable modules: MMD (§1.1), classifier two-sample test (§1.2), HDDDM (§1.3).
- Decide up front: same 3 tabular datasets (Credit Default, Telco Churn, Predictive Maintenance) but drift injected jointly across a correlated feature pair, instead of one feature — this is the multivariate tabular baseline and should be the *first* result produced, since it's the cheapest and most directly comparable to Paper 1's findings.

### 5.2 Arm A — Multivariate tabular (core contribution, do this regardless of team size)

- Run MMD, classifier-test, and HDDDM against the existing KS/PSI/Wasserstein baseline, on **jointly-injected** drift (e.g., correlated shift in two features at once) across the same 3 datasets and same constraint grid.
- Central question: does detection delay still show the ~37-fold constraint-driven range once the detector must track a joint distribution instead of a marginal? Does the "configuration > algorithm" ranking hold, or does detector choice start to matter more once detectors are multivariate?
- This alone is a complete, defensible, submittable paper even if Arms B and C don't ship in time — treat it as the minimum viable version of Paper 1.

### 5.3 Arm B — Embedding-space extension (NLP)

- Follow the Feldhans et al. (2021) template: BERT sentence embeddings on a public sentiment or topic-shift Twitter/news dataset, projected via PCA or UMAP, tested with MMD/KS.
- Add the constraint grid on top: vary monitoring frequency, window size, reference size — this is the piece Feldhans et al. and the 2023 follow-up don't do, and is the actual novel contribution of this arm.
- Scope guard: pick **one** NLP dataset, not several. Sentiment drift on a Twitter-style stream is the most standard choice and has the most prior work to compare against.

### 5.4 Arm C — Embedding-space extension (vision)

- Follow the Rabanser et al. (2019) / Hendrycks & Dietterich (2019) template: ResNet or CLIP embeddings on CIFAR-10 vs. CIFAR-10-C at increasing corruption severity, standing in for sudden/gradual/recurring drift.
- Same constraint grid overlay as Arm B.
- Scope guard: pick **one** corruption type family (e.g., noise corruptions) rather than all 15+ in the benchmark, unless the team has significant compute headroom.

### 5.5 What to explicitly cut, even with more hands

- The reinforcement-learning adaptive-monitoring paper (Proposed Paper 2 in the New Directions doc) and the causal-ablation robustness paper (Proposed Paper 3) are separate papers, not part of this scope. Don't let them creep in.
- Skip the "embedding dimension" and "projection method" as full factorial constraint-grid factors on top of F/W/R — that multiplies the grid size sharply for marginal insight. Fix one reasonable embedding dimension and one projection method per arm; treat those as a design choice, not an experimental variable, unless a first pass shows something surprising there.
- Don't chase every multivariate detector in the literature (Energy Distance, Learned Kernel Tests, etc. — mentioned in the New Directions doc). MMD, HDDDM, and the classifier two-sample test are enough to support the "does multivariate change the finding" question; more detectors add implementation time without changing the core answer.

### 5.6 Suggested sequencing

1. Backbone + Arm A (multivariate tabular) — this is the load-bearing result, do it first and completely.
2. Once Arm A results are in hand, decide as a team whether to pursue Arm B, Arm C, both, or neither, based on time remaining and how interesting Arm A's findings are. A paper with just Arm A plus a clear write-up of *why* embedding-space extension is future work is entirely legitimate and lower-risk than an unfinished three-arm paper.
3. If pursuing B and/or C, run them in parallel once the backbone is stable, since they don't depend on each other.

---

## 6. One framing note for the writeup, whenever you get there

Lead the related work section with §0.1 and §0.2 (Rabanser; Feldhans) stated plainly, the same way this document does — naming the closest prior work up front and clearly stating what's different is what makes a reviewer trust the novelty claim in the rest of the paper, rather than trying to find the gap themselves and possibly deciding it doesn't exist.