# Resource-Constrained Detection of Marginal and Dependency Shifts: When Detector Construction Matters

**Stephen Kwaku Azongo¹, Livingston Seble¹, Alex Teye Ametepey¹, Kofi Sarpong Adu-Manu¹, Sandro Amofa²**

¹Department of Computer Science, University of Ghana
²Department of Computer Science, University of Cape Coast

**Running title:** Resource-Constrained Dependency-Shift Detection

---

## Abstract

Machine-learning monitoring can fail when a joint distribution changes while individual feature marginals remain unchanged. We evaluate six lightweight drift detectors under resource constraints using two controlled regimes: marginal-visible shifts (16,200 runs) and marginal-preserving dependency shifts (3,240 runs) across three tabular datasets. Under marginal-visible shift, detection-window size has a much larger rank-based effect (ε² = 0.349) than detector identity (ε² = 0.015). Under dependency-only shift, detector identity becomes dominant (ε² = 0.236), exceeding monitoring-interval and window-size effects. Maximum Mean Discrepancy records an 11.9% miss rate, whereas methods constructed from separate marginal statistics miss 53–61% of shifts. The results show that deployment decisions should distinguish marginal-information detectors from joint-dependence detectors: configuration dominates when drift has a marginal signature, but detector construction becomes decisive when dependence changes without marginal change.

**Keywords:** dataset shift; dependency shift; two-sample testing; maximum mean discrepancy; concept drift; model monitoring; resource constraints

---

## 1. Introduction

Statistical monitoring of deployed machine-learning systems is usually framed as a two-sample problem: does the distribution generating current observations differ materially from the distribution represented by a trusted reference sample? This question becomes difficult under operational constraints because monitoring systems must trade statistical power against sample availability, memory, and computation. A second difficulty is structural. Many inexpensive monitors compare features separately, although equality of all univariate marginals does not imply equality of the joint distribution.

For a bivariate random vector (X₁, X₂), Sklar's theorem [1] separates the joint distribution into its marginal distributions and a copula that encodes dependence. Consequently, two deployment distributions can satisfy P(X₁) = Q(X₁) and P(X₂) = Q(X₂) while P(X₁, X₂) ≠ Q(X₁, X₂). Any monitoring statistic constructed exclusively from separate marginal discrepancies has no population-level signal for such a dependency-only alternative. This is not merely a theoretical edge case: changes in relationships among utilization, billing, temperature, wear, load, or risk variables can arise while their individual empirical distributions remain stable.

Prior empirical work has shown that corrected univariate tests can be competitive with multivariate tests for many common dataset shifts [3]. However, those comparisons do not establish that marginal testing is generally sufficient. Kamulete [4], for example, explicitly studies dependency shifts generated through covariance changes, illustrating that distribution shift can occur through dependence rather than marginal location or scale. What remains insufficiently characterized is how this statistical distinction interacts with resource constraints and with the internal construction of practical drift detectors. In particular, methods commonly described as multivariate need not use joint dependence information: a detector that averages per-feature divergences remains a marginal-information procedure even when it accepts a multivariate input matrix.

This study therefore asks a more precise question than whether 'univariate' or 'multivariate' detectors are preferable. We investigate when monitoring configuration dominates detector choice, and when the information content of the detector statistic becomes the limiting factor. Six detectors are evaluated under two controlled regimes on three tabular benchmark datasets. The first regime introduces marginal-visible shifts, for which both marginal and joint statistics have a signal. The second uses a rank-permutation mechanism that preserves the empirical marginal multiset exactly while changing only the pairing structure between two variables. This nonparametric construction complements earlier Gaussian covariance-shift experiments [4] by inducing a dependency shift directly on observed tabular data without imposing a parametric joint distribution.

The study is organized around three research questions. **RQ1** asks whether monitoring interval, detection-window size, and reference size explain more variation in time-to-detection than detector identity when drift has a marginal signature. **RQ2** asks whether this ordering changes under a marginal-preserving dependency shift. **RQ3** asks whether detector behavior is better explained by the information used by the test statistic, marginal only versus joint dependence, rather than by the conventional univariate/multivariate label.

Our contributions are fourfold:

1. A resource-constrained factorial evaluation of six drift detectors across marginal-visible and dependency-only shift regimes, producing 19,440 runs across three tabular datasets.
2. A rank-permutation dependency-shift construction that preserves the observed marginals exactly while altering dependence on real tabular feature pairs, with empirical verification.
3. An information-based detector taxonomy that distinguishes marginal-aggregation procedures from joint-dependence procedures, and is more precise than classifying methods only as univariate or multivariate.
4. Empirical evidence that the dominant deployment lever is conditional on the shift structure: configuration dominates when marginal information changes, whereas detector construction becomes decisive when the change is confined to dependence.

---

## 2. Related Work

**Multivariate two-sample testing.** Gretton et al. [5] introduce the Maximum Mean Discrepancy as a kernel-based statistic for the two-sample problem, defined as the distance between mean embeddings of two distributions in a reproducing kernel Hilbert space, and establish that MMD is zero if and only if the two distributions are equal when the kernel is characteristic. The unbiased estimator and the median-heuristic bandwidth selection used in this study follow their formulation directly. Lopez-Paz and Oquab [6] formalise Classifier Two-Sample Tests, training a binary classifier to distinguish the two samples and testing its held-out accuracy against chance; their asymptotic power analysis is classifier-agnostic, which motivates the logistic-regression implementation used here in place of their neural-network and k-nearest-neighbour variants, a simpler choice consistent with this study's resource-constrained framing.

**Feature-wise divergence methods.** Ditzler and Polikar [7] propose HDDDM, which computes the Hellinger distance between histograms of a reference and current batch, adaptively resetting the reference when a significant change in divergence is detected. Their multivariate formulation explicitly defines the joint statistic as the average of per-feature Hellinger distances, a construction that, as this study demonstrates, has direct consequences for what kinds of shift the method can detect.

**Dataset-shift detection under resource limits.** Rabanser et al. [3] conduct the most comprehensive prior empirical study of multivariate shift detection, comparing dimensionality-reduction strategies combined with two-sample testing across a wide range of shift types and sample sizes, including a sample-size constraint axis. Their central empirical finding, that multiple corrected univariate tests perform comparably to multivariate kernel tests, is the direct precedent this study extends; critically, none of their constructed shifts hold marginal distributions fixed, so their finding characterises detection difficulty under marginal shift specifically, not under a shift that only a multivariate statistic can see.

**Dependency shifts.** Kamulete [4] includes an explicit dependency-shift condition in simulated Gaussian-mixture experiments by altering covariance while retaining marginal variances. These two studies [3, 4] together establish two important precedents: marginal tests can be sufficient when shifts alter marginal distributions, and dependence itself can be a target of monitoring. The present study differs from both by combining dependency shifts with a resource-constraint grid, and by inducing dependency change nonparametrically on observed tabular features while preserving the empirical marginals exactly rather than through a parametric covariance model.

**Survey context.** Gama et al. [9] survey concept drift adaptation methods, establishing the foundational taxonomy of gradual, sudden, and recurring shifts that underpins the experimental design here. Lu et al. [10] provide a more recent systematic review that includes non-stationary environment detection and connects drift type to detection strategy, reinforcing the need to characterize drift conditions carefully before selecting a monitor. Webb et al. [11] characterize concept drift across temporal and spatial dimensions and identify the detection of structural changes in joint distributions as an open challenge, a gap this study addresses directly.

**Label shift and representation-space monitoring.** Lipton et al. [8] introduce Black Box Shift Estimation, using a black-box classifier's confusion matrix to detect and correct label shift under an anti-causal assumption. Although label shift is outside the present experiment, the work illustrates a broader principle relevant here: the detectable component of distribution shift depends on the statistic or representation through which the change is observed, which is the same principle this study demonstrates for dependency versus marginal shifts.

**Dependency structure and copulas.** Sklar [1] establishes that any multivariate joint distribution decomposes into its marginal distributions and a copula function describing the dependency between them, and that the two can be manipulated independently. Nelsen [2] provides the standard treatment of copula construction and empirical rank-based copula estimation. This decomposition motivates the paper's central distinction between marginal-information and joint-dependence detectors, and grounds the rank-permutation injection mechanism used in Section 3.3.

Recent work has strengthened the theoretical foundations on which this study's detectors rest. Schrab et al. [12] establish minimax-optimal power guarantees for kernel-based two-sample tests through bandwidth aggregation (MMDAgg), providing a theoretically grounded alternative to the median heuristic used here; that aggregation framework also suggests that the 300-sample cap imposed on MMD in this study leaves non-negligible power on the table relative to an uncapped implementation. Bayram et al. [13] provide a systematic taxonomy of performance-aware drift detectors that distinguishes statistical process control from distributional testing approaches, reinforcing the practitioner-oriented framing of this study.


---

## 3. Methodology

### 3.1 Datasets, Splits, and Feature Pairs

Three tabular binary-classification benchmarks are used throughout: the UCI Credit Card Default dataset, the IBM Telco Customer Churn dataset, and the AI4I 2020 Predictive Maintenance dataset. For each dataset, all identifier columns, failure-mode indicators that would leak target information, and the target variable itself are excluded before any analysis (the Maintenance dataset excludes UDI, Product ID, and the five binary failure-mode columns TWF, HDF, PWF, OSF, RNF).

Each dataset is divided into a reference period (first 60% of instances, preserving the original row order) and a monitoring stream (remaining 40%). The reference period serves two purposes: training the Random Forest classifier and providing the reference distribution against which drift is detected. The 60/40 ratio was chosen to balance reference-distribution estimation quality against monitoring-stream length; the monitoring stream must be long enough to accommodate the largest window size tested (1,000 samples) and still contain a post-drift segment of at least that length.

A split sensitivity analysis was conducted at 50/50 and 70/30 reference/stream ratios using the same dependency-shift grid (6 detectors, 3 drift types, 3 monitoring intervals, 3 window sizes, 2 reference sizes, 3 repetitions = 2,916 configurations per ratio). Table B4 reports the Kruskal-Wallis epsilon-squared values and miss rates at each split ratio.

**Table B4: Split sensitivity -- epsilon-squared and miss rates across reference/stream ratios**

| Factor | 50/50 | 60/40 (main) | 70/30 |
|---|---|---|---|
| Detector identity (delay, epsilon-sq) | 0.280 | 0.236 | 0.250 |
| Window size (delay, epsilon-sq) | 0.126 | 0.096 | 0.124 |
| Monitoring interval (delay, epsilon-sq) | 0.054 | 0.137 | 0.073 |
| Miss rate -- marginal-info class | 41.7% | 57.3% | 46.2% |
| Miss rate -- joint-dep class | 30.7% | 36.6% | 34.6% |

**Table B5: Per-detector miss rates across splits (dependency-shift condition)**

| Detector | 50/50 | 60/40 (main) | 70/30 |
|---|---|---|---|
| MMD | 6.6% | 11.9% | 7.4% |
| C2ST | 54.7% | 61.3% | 61.7% |
| HDDDM | 40.5% | 53.3% | 44.0% |
| KS | 34.3% | 57.6% | 40.3% |
| PSI | 44.9% | 57.6% | 47.7% |
| Wasserstein | 47.9% | 60.6% | 52.7% |

The key finding is that the qualitative structure is fully preserved across all three splits. Detector identity is the largest effect-size factor in the delay comparison at every ratio tested (epsilon-sq 0.236 to 0.280). MMD consistently achieves the lowest miss rate (6.6% to 11.9%), and C2ST with logistic regression consistently achieves the highest among joint-dependence methods (54.7% to 61.7%). HDDDM and the univariate marginal-information methods (KS, PSI, Wasserstein) consistently fall in the intermediate range. The 60/40 split produces somewhat higher miss rates overall -- mechanically, a larger reference period means a smaller stream, which reduces the number of post-drift windows evaluated and therefore gives detectors fewer opportunities to detect, inflating miss rates slightly. This is expected behavior and does not affect the relative ordering. The main conclusions of the paper are not artifacts of the 60/40 split choice.


For each dataset, two continuous features are monitored jointly. The feature pairs and their reference-period Pearson correlations are given in Table 1.

**Table 1: Datasets and feature pairs**

| Dataset | N | Reference (60%) | Stream (40%) | Primary feature | Paired feature | Baseline Pearson r |
|---|---|---|---|---|---|---|
| Credit | 30,000 | 18,000 | 12,000 | AGE | BILL_AMT1 | 0.06 |
| Telco | 7,043 | 4,226 | 2,817 | Monthly Charges | Total Charges | 0.66 |
| Maintenance | 10,000 | 6,000 | 4,000 | Tool wear [min] | Air temperature [K] | 0.02 |

Feature pairs were selected by the following reproducible rule: from the continuous features of each dataset (after excluding the target and identifier columns), identify the feature most semantically associated with distributional change in that domain; pair it with the second continuous feature having the highest reference-period Pearson correlation with it. Applied to each dataset: Credit pairs AGE with BILL_AMT1 (r = 0.06); Telco pairs Monthly Charges with Total Charges (r = 0.66); Maintenance pairs Tool wear [min] with Air temperature [K] (r = 0.02). The direction-auto-calibration mechanism in Section 3.3 handles the very different baseline correlations.

The three datasets are cited as follows: Credit [14], AI4I Maintenance [15], Telco [16]. For AI4I, the five binary failure-mode columns (TWF, HDF, PWF, OSF, RNF) and identifier columns (UDI, Product ID) were excluded before analysis to prevent target leakage.

A feature-pair sensitivity analysis was conducted using one alternative pair per dataset, chosen to contrast with the main pair's baseline correlation structure. Credit alt: BILL_AMT1 + BILL_AMT2 (r = +0.951, high-correlation regime, shift direction decorrelating); Telco alt: Tenure Months + Monthly Charges (r = +0.247, medium-correlation, the only other available continuous pair); Maintenance alt: Rotational speed [rpm] + Torque [Nm] (r = -0.875, strong-negative baseline, shift direction toward positive). The same 972-configuration representative grid was used as for the split sensitivity (6 detectors, 3 drift types, 3 intervals, 3 windows, 2 reference sizes, 3 repetitions).

**Table B6: Feature-pair sensitivity -- epsilon-squared and miss rates (dependency-shift condition)**

| Dataset / Pair | Eps2 detector | Miss MMD | Miss C2ST | Miss HDDDM | Miss marginal-info avg | Miss joint-dep avg |
|---|---|---|---|---|---|---|
| Credit (main: AGE + BILL_AMT1, r=+0.06) | 0.225 | 9.4% | 47.8% | 50.0% | 46.0% | 28.6% |
| Credit (alt: BILL_AMT1 + BILL_AMT2, r=+0.951) | 0.352 | 0.0% | 34.6% | 45.1% | 34.6% | 17.3% |
| Telco (main: Monthly + Total Charges, r=+0.660) | 0.215 | 10.0% | 67.2% | 56.1% | 63.5% | 38.6% |
| Telco (alt: Tenure + Monthly Charges, r=+0.247) | 0.236 | 27.8% | 72.8% | 40.7% | 44.9% | 50.3% |
| Maintenance (main: Tool wear + Air temp, r=+0.014) | 0.287 | 16.1% | 68.9% | 53.9% | 62.4% | 42.5% |
| Maintenance (alt: Rot. speed + Torque, r=-0.875) | 0.533 | 0.0% | 62.3% | 42.6% | 48.3% | 31.2% |

The qualitative structure is preserved across all six pair configurations. Detector identity is the largest epsilon-squared factor in all cases (0.215 to 0.533). MMD achieves the lowest or joint-lowest miss rate in five of six configurations. The one exception is the Telco alternative pair (Tenure Months + Monthly Charges, r = +0.247 baseline), where MMD's miss rate rises to 27.8%. This reflects a known interaction between baseline correlation and shift detectability: with a moderate starting correlation (r = +0.247), the shift toward higher positive correlation produces a smaller absolute change in dependence structure than when starting near-zero, reducing the kernel test's power at this pair. Despite this, MMD remains the lowest miss-rate detector on that pair. HDDDM consistently clusters with the univariate marginal-information group (40.7% to 56.1% miss rate across all pairs), and C2ST with logistic regression remains the highest miss-rate joint-dependence method in all cases (34.6% to 72.8%). The main conclusions of the paper are not artifacts of the specific feature pairs chosen.

### 3.2 Marginal-Shift Injection and the Extrapolation Constraint

Drift is injected as an additive shift of nominal magnitude 2σ, where σ is the standard deviation of the feature in the reference period. Applied naively, a fixed additive shift does not translate uniformly across features: for a feature whose observed range extends well beyond its mean, a 2σ shift stays within previously observed territory; for a feature with a narrow or bounded observed range, the same nominal shift can push samples beyond the model's training support.

We verified this directly. An uncapped 2σ shift on Telco's Monthly Charges pushed 60.1% of post-drift values beyond the feature's observed maximum; on Maintenance's Tool wear and Air temperature, 43.3% and 42.6% respectively. This is a methodologically consequential problem: post-drift accuracy would then reflect behavior on inputs outside the model's training support, producing confounded, dataset-dependent artifacts in the accuracy-degradation results.

The correction applies smooth exponential saturation toward an upper ceiling u set at the 99.5th percentile of the reference distribution. For an observation x with headroom h = u − x > 0 and nominal shift δ = 2σ, the saturated value is:

**x′ = u − h · exp(−δ / h)**

When h >> δ (points far from the ceiling), a first-order Taylor expansion gives x′ ≈ x + δ, recovering approximately the plain additive shift. When h is small or comparable to δ (points close to the ceiling), the shift compresses smoothly toward u with x′ → u as h → 0. Because the mapping x → x′ is strictly monotone in x for fixed u and δ > 0, distinct input values always remain distinct in the output; there is no pile-up at the ceiling. Observations at or above u (the top 0.5% of reference data by construction) have h ≤ 0 and are left unchanged.

Verified empirically: the corrected mechanism produces 0% of shifted values beyond the feature's observed maximum across all three datasets, all six drifted features, and all three temporal drift profiles. The ceiling is set at the 99.5th percentile of the reference distribution, meaning that the top 0.5% of reference observations already exceed it and are left unchanged by the formula. Post-shift values do not pile up at the ceiling: the fraction of stream samples landing within 1% of the ceiling ranges from 0.33% to 13.8% across features (Table B1), and the shift produces a genuine upward displacement of the full distribution.

**Table B1: Saturation mechanism diagnostics per feature (stream period, sudden drift)**

| Dataset | Feature | Ceiling (p99.5) | Unchanged (%) | Near-ceiling post (%) | Pre-shift median | Post-shift median | Pre-shift p95 | Post-shift p95 | Beyond obs. max (%) |
|---|---|---|---|---|---|---|---|---|---|
| Credit | AGE | 3.01 | 0.67 | 1.41 | -0.16 | 1.33 | 1.92 | 2.84 | 0.00 |
| Credit | BILL_AMT1 | 4.89 | 0.52 | 0.33 | -0.39 | 1.27 | 2.02 | 3.46 | 0.00 |
| Telco | Monthly Charges | 1.69 | 0.25 | 13.83 | 0.18 | 1.29 | 1.39 | 1.69 | 0.00 |
| Telco | Total Charges | 2.63 | 0.57 | 4.91 | -0.39 | 1.07 | 2.03 | 2.61 | 0.00 |
| Maintenance | Tool wear [min] | 1.88 | 0.83 | 12.17 | 0.02 | 1.25 | 1.57 | 1.88 | 0.00 |
| Maintenance | Air temperature [K] | 2.09 | 0.57 | 6.95 | -0.01 | 1.28 | 1.74 | 2.08 | 0.00 |

*All feature values are standardized. Ceiling = 99.5th percentile of reference distribution. Unchanged = fraction of stream samples already above ceiling. Near-ceiling post = fraction of post-shift samples within 1% of ceiling. The Telco Monthly Charges and Maintenance Tool wear features have higher near-ceiling fractions because their stream distributions are more right-skewed; this reflects expected saturation behavior, not a degenerate spike.*

### 3.3 Marginal-Preserving Dependency Shift: Formal Definition and Construction

To isolate detection difficulty attributable specifically to a change in dependency structure, independent of any marginal signature, we define a **marginal-preserving dependency shift** as follows. Let P and Q denote the reference and post-shift joint distributions of (X₁, X₂). The shift satisfies:

> P₁ = Q₁ and P₂ = Q₂ while P(X₁, X₂) ≠ Q(X₁, X₂)

where P_j and Q_j denote the corresponding univariate marginals. In copula terms [1, 2], the marginals are held fixed while the copula (the function encoding dependence) is changed. Any monitoring statistic whose value depends only on the empirical marginal distributions of X₁ and X₂ separately has zero population-level signal for detecting this shift by construction; it is not a matter of insufficient data but of structural impossibility.

**Rank-permutation construction.** The shift is implemented via rank-based permutation within an active drift window. Within a window of n observations, let x₁ = (x₁₁, …, x₁ₙ) and x₂ = (x₂₁, …, x₂ₙ) denote the feature values. A subset S ⊂ {1, …, n} of size k = ⌊α · n⌋ is selected uniformly at random (α = correlation_strength ∈ [0, 1] is the mixing parameter; α = 0.8 throughout). Within S, the values of x₂ are rearranged so that their rank order among {x₂ᵢ : i ∈ S} matches the rank order of {x₁ᵢ : i ∈ S} (positive correlation shift) or is reversed (decorrelating shift):

- **Rank-match (positive):** sort {x₂ᵢ : i ∈ S} and assign the sorted values to the positions in S in the same order as the sorted positions of {x₁ᵢ : i ∈ S}.
- **Rank-match (negative):** same, but assign in reverse order.

Because this operation is a permutation of the values already present in x₂, no value is created, interpolated, or altered, the empirical multiset of x₂ is identical before and after. Consequently, the empirical marginal distribution (equivalently, the empirical CDF) of x₂ is preserved **exactly**: the maximum absolute difference between pre- and post-shift empirical CDFs is zero for every fully permuted window (α = 1.0), and reduces to expected sampling variation for α < 1.0. This is an algebraic consequence of permutation, not a statistical approximation. The marginal distribution of x₁ is never altered.

**Direction calibration.** The shift direction is determined from the reference-period baseline correlation, always pushing away from the existing baseline: Credit (r = 0.06) and Maintenance (r = 0.02) shift toward positive correlation; Telco (r = 0.66) shifts toward decorrelation. This rule is fixed before any detection results are computed, using reference-period data alone.

**Temporal profiles.** Three profiles mirror the marginal-shift condition:
- **Sudden:** full mixing fraction α applied from the drift start index onward.
- **Gradual:** α linearly interpolated from 0 to the target value over a 500-sample transition window from the drift index, then held at the target.
- **Recurring:** α alternates between the target value and zero in 200-sample cycles from the drift start index.

Table B2 reports the measured correlation change for each dataset and temporal profile, averaged over five independent repetitions.

**Table B2: Dependency strength before and after shift (mean over 5 repetitions)**

| Dataset | Drift type | Pre Pearson r | Pre Spearman rho | Post Pearson r | Post Spearman rho | Delta Pearson r | Delta Spearman rho | Direction |
|---|---|---|---|---|---|---|---|---|
| Credit | Sudden | 0.052 | -0.002 | 0.734 | 0.799 | +0.682 | +0.801 | Toward positive |
| Credit | Gradual | 0.052 | -0.002 | 0.735 | 0.805 | +0.683 | +0.807 | Toward positive |
| Credit | Recurring | 0.052 | -0.002 | 0.402 | 0.404 | +0.349 | +0.406 | Toward positive |
| Maintenance | Sudden | 0.013 | 0.013 | 0.800 | 0.806 | +0.787 | +0.793 | Toward positive |
| Maintenance | Gradual | 0.013 | 0.013 | 0.794 | 0.802 | +0.781 | +0.788 | Toward positive |
| Maintenance | Recurring | 0.013 | 0.013 | 0.410 | 0.413 | +0.397 | +0.400 | Toward positive |
| Telco | Sudden | 0.650 | 0.638 | -0.633 | -0.679 | -1.283 | -1.316 | Decorrelating |
| Telco | Gradual | 0.650 | 0.638 | -0.653 | -0.686 | -1.303 | -1.324 | Decorrelating |
| Telco | Recurring | 0.650 | 0.638 | -0.079 | -0.111 | -0.729 | -0.748 | Decorrelating |

*Post-shift values measured in the fully-shifted region. Recurring profile alternates in 200-sample cycles; post-shift correlation is measured in an active cycle. The Telco delta exceeds 1.0 in absolute value because the shift reverses the sign of the correlation. Credit and Maintenance move from near-zero correlation to strongly correlated (rho approx 0.80); Telco moves from strongly positively correlated to strongly negatively correlated under sudden/gradual drift.*

### 3.4 Detector Taxonomy and Implementations

Six detectors are evaluated, grouped by the information used by their test statistics rather than by input dimensionality:

**Marginal-information procedures:** KS test, PSI, Wasserstein distance, and HDDDM. All four compute their decision statistics either per-feature or as an aggregation of per-feature divergences. They cannot detect a shift confined to the joint arrangement of feature values when each feature's own distribution is unchanged.

**Joint-dependence procedures:** MMD and C2ST. Both statistics are computed on joint observations. In principle, both can respond to a change in the pairing of feature values.

**KS test.** A two-sample KS test is applied independently to each of the two monitored features. Family-wise error rate is controlled by Bonferroni correction: drift is declared if either feature's p-value is below α/2 = 0.025, giving an overall significance level of α = 0.05.

**PSI.** The reference distribution is binned into 10 equal-frequency intervals using reference-period quantiles, with boundary bins extended to (−∞, q₁₀] and [q₉₀, +∞). PSI is computed per feature; drift is declared if the maximum PSI across both features exceeds 0.25 (the industry-standard 'significant change' threshold).

**Wasserstein distance.** The 1-Wasserstein distance is computed per feature. The drift threshold is fixed at 0.25 across all features and datasets, maintaining the constraint-grid framing in which each configuration is evaluated identically without dataset-specific calibration. As a sanity check, a bootstrap null was computed for each dataset: 500 bootstrap resamples (with replacement) of window-sized draws from the reference period were drawn per feature, and the 99th percentile of the resulting empirical null distribution was verified to be at or below 0.25 for all features and window sizes used in the study. Drift is declared if either feature's Wasserstein distance exceeds 0.25.

**HDDDM.** Following Ditzler and Polikar [7, Eq. 1], the multivariate Hellinger distance is the simple average of per-feature Hellinger distances, each computed from 10-bin equal-width histograms (boundaries from the reference period, outer bins extended to ±∞). Drift is declared when the average Hellinger distance exceeds 0.15. No adaptive reference-update mechanism is used (the same reference histograms are maintained throughout), consistent with the constraint-grid framing. This implementation preserves the core marginal-averaging construction of the original formulation exactly, the property central to RQ3.

**MMD.** The unbiased estimator of MMD² [5, Eq. 3] is computed with an RBF kernel. Bandwidth parameter γ is set by the median heuristic: γ = 1 / median(‖x − y‖²), where the median is over a random subsample of at most 500 pooled observations. Before kernel computation, reference and test samples are independently subsampled to at most 300 observations each, then jointly standardized to zero mean and unit variance using pooled statistics. Significance is assessed via a permutation test with 199 permutations; p-value = (count_≥ + 1) / (199 + 1), giving a minimum achievable p-value of 0.005. Drift is declared when p < 0.01.

The O(n²) kernel computation at window size 1,000 measured approximately 11.5 seconds per check on the study hardware (Intel Core i3-1125G4, 4 GB RAM). The 300-sample cap reduces this to approximately 0.4 seconds per check, making the full grid feasible. This cap is itself a resource constraint in the spirit of this study's design and is documented rather than concealed.

**MMD equal-budget sensitivity.** To verify that the MMD/HDDDM miss-rate gap reflects detector construction rather than an unequal information budget, MMD was run at max\_samples in {100, 200, 300, 500} on a focused dependency-shift grid (3 datasets x 2 drift types x 2 window sizes x 5 repetitions = 240 runs per budget level). MMD achieved a 0% miss rate at every budget level tested. This result strongly supports the construction-based explanation: even at 100 samples, half the main-grid cap, MMD detects every dependency-shift instance. HDDDM has no in-principle capacity to detect this shift regardless of sample size, because its statistic averages per-feature Hellinger distances that carry zero signal when marginals are unchanged.

**C2ST.** Reference and test samples are trimmed to equal size, combined with binary labels (0 = reference, 1 = test), and split 50/50 into training and evaluation halves with stratification. A logistic regression classifier (lbfgs solver, max 200 iterations) is trained on the training half and evaluated on the held-out half. Significance is assessed via a one-tailed binomial test of the number of correctly classified samples against p = 0.5. Drift is declared when the binomial p-value < 0.01. This uses logistic regression rather than the neural-network or k-NN variants of [6]; the asymptotic power bound in [6, Theorem 1] is classifier-agnostic, but finite-sample power at the specific window sizes tested is specific to this choice.

### 3.5 Null Calibration and Evaluation Metrics

False-alarm rates were measured on matched no-drift streams using the same window protocol as the main experiments, with no drift injected. Table 3 reports mean false-alarm rates (fraction of monitoring checks triggering a drift alarm) per detector, averaged over all window-size, reference-size, and repetition combinations.

**Table 3: Empirical false-alarm rates on no-drift streams**

| Detector | Class | Mean FAR | Std | FAR at window=50 | FAR at window=500 |
|---|---|---|---|---|---|
| MMD | Joint-dependence | 0.31% | 0.79% | 0.30% | 0.17% |
| C2ST | Joint-dependence | 0.42% | 1.15% | 0.60% | 0.10% |
| KS | Marginal-info | 4.60% | 7.54% | 4.36% | 2.43% |
| Wasserstein | Marginal-info | 7.41% | 10.43% | 9.42% | 0.00% |
| PSI | Marginal-info | 18.51% | 25.56% | 51.54% | 0.00% |
| HDDDM | Marginal-info | 22.72% | 30.66% | 61.53% | 0.00% |

*FAR = fraction of monitoring checks that produce a drift alarm on a stream with no drift injected. MMD was run at interval=100 only. PSI and HDDDM alarm on more than half of all null checks at window=50; at window=500 their FAR drops to zero because the larger window provides stable histogram estimates. These rates are essential for interpreting the miss rates in Table 2.*

Primary evaluation metrics:

- **Miss rate:** proportion of experimental configurations in which drift was never detected in the post-drift stream region.
- **Right-censored detection time:** time (in samples) from drift injection to first alarm, treating undetected streams as right-censored at the end of the post-drift observation period ($6,000$ samples for Credit, $1,407$ for Telco, $2,000$ for Maintenance). Evaluated via Kaplan-Meier survival curves, restricted mean survival time (RMST up to $\tau = 1,400$ samples), log-rank tests, and clustered Cox proportional hazards modeling (Section 4.3).
- **Detection delay (conditional on detection):** for detected cases only, the number of stream samples between drift onset and first alarm; retained as a secondary descriptive measure.
- **Predictive impact (counterfactual-matched):** downstream model degradation measured against the matched undrifted counterfactual (Section 3.6) across three metrics: classification accuracy, balanced accuracy (macro-averaged recall), and ROC-AUC (probability discrimination), accompanied by 95% confidence intervals.

### 3.6 Predictive Impact: Matched Counterfactual Baseline

A naive comparison of baseline accuracy (on the full stream) with post-drift accuracy (on the post-drift portion only) confounds drift effect with natural sampling variation between stream halves. To isolate the drift's causal effect, post-drift accuracy is measured against a counterfactual: the same trained model on the same index range, but with no drift injected. Accuracy degradation is therefore:

> **accuracy_degradation = counterfactual_accuracy(same index range, no drift) − accuracy(same index range, with drift)**

This removes the natural between-half performance gap from the estimate.

### 3.7 Statistical Analysis

The experimental evaluation produces 16,200 runs under marginal shift and 3,240 runs under dependency shift. Because these observations arise from a crossed, repeated-measures design sharing data-generating contexts and random seeds, treating runs as independent observations violates exchangeability and inflates test statistics. To ensure inferential validity, the primary statistical analysis employs generalized estimating equations (GEE) and semiparametric survival models that directly respect the crossed blocking structure:

1. **Binary Missed Detection (GEE Logistic Regression):** Missed detection ($Y_{ij} \in \{0, 1\}$) is modeled via GEE with a logit link, a binomial variance function, and robust Huber-White sandwich standard errors. Runs are clustered into 54 blocks defined by $(dataset \times detector \times drift\_type)$ (300 runs per cluster in the marginal grid; 60 runs per cluster in the dependency grid). Predictors include detector identity, window size, monitoring interval, reference size, dataset, and drift type. Exponentiated coefficients $\exp(\beta)$ are reported as adjusted odds ratios (OR) with 95% Wald confidence intervals.
2. **Condition $\times$ Detector Interaction (Balanced Common Grid):** To formally test whether detector sensitivity differs between marginal and dependency shifts (the core hypothesis of RQ2/RQ3), an interaction GEE model is evaluated on the balanced common subgrid ($n = 6,480$; 3,240 runs per condition across the shared factor levels: window sizes $\{50, 500\}$, monitoring intervals $\{10, 100, 1000\}$, and reference sizes $\{500, \text{full}\}$). Both detector-class $(\text{marginal-information vs. joint-dependence})$ and individual detector interactions with shift condition are evaluated.
3. **Detection Delay (Log-Linear GEE):** For detected drift cases, detection delay in samples is modeled using Gaussian GEE on $\log(\text{delay})$ with cluster-robust standard errors. Exponentiated coefficients $\exp(\beta)$ yield multiplicative effect factors on detection delay with 95% confidence intervals, providing directly interpretable ratios of operational cost.
4. **Right-Censored Time-to-Detection (Clustered Cox Proportional Hazards):** To address survivorship bias where undetected runs are omitted from delay summaries, detection time is evaluated using a Cox proportional hazards model (`PHReg`) with right-censoring. Runs that complete the monitoring stream without detection are censored at the total post-drift observation length ($6,000$ samples for Credit, $1,407$ for Telco, $2,000$ for Maintenance). Clustered robust covariance estimators account for within-block correlation. Hazard ratios ($\text{HR} > 1$ indicating faster detection / higher detection rate) and 95% profile confidence intervals are reported.
5. **Nonparametric Descriptive Robustness:** Nonparametric procedures (Kruskal-Wallis $H$, Mann-Whitney $U$ with Bonferroni correction and rank-biserial $r$, and Spearman's $\rho$) are retained as secondary descriptive statistics to facilitate comparison with prior rank-based literature [3]. All p-values across secondary tests are adjusted using the Benjamini-Hochberg false discovery rate procedure. Epsilon-squared ($\varepsilon^2 = H/(n-1)$) is reported strictly as a rank-based effect size and not as proportion of variance explained on the delay scale.

### 3.8 Implementation and Reproducibility

All experiments were implemented in Python 3.13 with scikit-learn 1.8.0, scipy, numpy, and pandas, and were run on an Intel Core i3-1125G4 (4 physical / 8 logical cores, 2.00 GHz), 4 GB RAM, Windows 11 Home. The marginal-shift grid contains 16,200 runs (3 datasets × 6 detectors × 3 drift types × 5 monitoring-interval levels × 4 window-size levels × 3 reference-size levels × 5 repetitions) and the dependency-shift grid contains 3,240 runs (same factors, reduced monitoring-interval and window-size levels to limit the compute budget for this secondary condition). Each of the five repetitions per configuration uses an independently reseeded stratified split and an independently retrained model. The seed for repetition r is base_seed + r, applied to both the data split and the drift simulator; detector random states are fixed constants specified per detector in the configuration file. Duplicate-row checks confirmed that no two repetitions within the same configuration are byte-identical in the final results files.

> **[ACTION REQUIRED: Data and Code Availability]:** Prepare a permanent repository (e.g., GitHub + Zenodo for a DOI) containing: processed run-level results CSV files; all detector and simulator source files (including the rank-permutation construction); the complete experiment configuration; analysis scripts; per-run random seeds; and a README with reproduction instructions. The Data Availability Statement (Section 7) must contain a permanent URL and DOI before submission. *Repository preparation is pending.*

---

## 4. Results

### 4.1 Marginal-Visible Shifts: Configuration Dominates Rank-Based Effect Sizes

Across the 16,200-run marginal-shift grid, both constraint dimensions that influence monitoring resource use show strong associations with detection delay. Window size shows Spearman ρ = 0.583 with detection delay (Kruskal-Wallis H = 5,606.6, p < 10⁻³⁰⁰, BH-adjusted p < 10⁻³⁰⁰); monitoring interval shows ρ = 0.492 (H = 4,017.4, p < 10⁻³⁰⁰). Reference size shows no significant monotonic association (ρ = −0.001, p = 0.93), consistent with a threshold rather than dose-response relationship. Figure 1 shows the window-size relationship: median detection delay rises from 50 samples at window size 50 to 494 samples at window size 1,000, approximately a tenfold range from configuration alone.

Figure 2 shows the drift-type finding at the expanded six-detector scale: gradual drift's median detected-case delay (300 samples) is three times that of sudden or recurring drift (100 samples each; Kruskal-Wallis H = 1,342.96, p < 10⁻²⁹⁰).

Detector identity is statistically significant across the six-detector comparison (Kruskal-Wallis H = 242.5, p = 2.3 × 10⁻⁵⁰), which is expected given n = 16,200. The rank-based effect size is small: ε² = 0.015 for detector identity, compared to ε² = 0.250 for monitoring interval and ε² = 0.349 for window size (Figure 3, marginal-shift bars). 

Primary inferential modeling via log-linear GEE (accounting for clustering across the 54 dataset × detector × drift type blocks; Table 4, Model M4) confirms and quantifies these relationships on a ratio scale. Relative to a window size of 50, window sizes of 500 and 1,000 multiply detection delay by factors of 3.500 (95% CI [3.008, 4.074], p < 0.001) and 5.489 (95% CI [4.584, 6.574], p < 0.001), respectively. Similarly, relative to checking every 10 samples, checking every 1,000 samples multiplies delay by 4.949 (95% CI [4.348, 5.633], p < 0.001). By contrast, detector differences introduce modest multiplicative variations relative to the KS baseline: Wasserstein (1.160× [1.067, 1.262]), HDDDM (1.225× [1.100, 1.365]), PSI (1.312× [1.137, 1.514]), MMD (1.409× [1.252, 1.587]), and C2ST (1.552× [1.383, 1.742]). Reference size has no significant multiplicative effect on delay (100 samples: 1.003× [0.933, 1.079], p = 0.93; 500 samples: 1.012× [0.993, 1.031], p = 0.22). Collapsing to the marginal-information class (KS, PSI, Wasserstein, HDDDM) and the joint-dependence class (MMD, C2ST): the two classes differ in median detected-case delay (100 versus 194 samples; rank-biserial r = 0.097), but the effect size is negligible by conventional interpretation. Detector class has no significant effect on whether drift is detected at all under marginal shift (χ² = 1.56, df = 1, p = 0.21, BH-adjusted p = 0.21; GEE OR = 0.560 [0.114, 2.755], p = 0.48). These results address **RQ1** and **RQ3** for the marginal-visible condition.

**[Figure 1: Median detection delay versus window size, marginal-shift condition.]**

**[Figure 2: Median detection delay by drift type (sudden, gradual, recurring), both conditions.]**

**[Figure 3: Rank-based effect size (ε²) for detector identity versus window size versus monitoring interval, marginal-shift and dependency-shift conditions side by side.]**

### 4.2 Dependency-Only Shifts: Detector Construction Becomes Decisive

Under the dependency-only condition, the effect ordering changes markedly. Detector identity's rank-based effect size rises to ε² = 0.236, now exceeding the monitoring-interval effect (ε² = 0.137) and the window-size effect (ε² = 0.096) measured on the same grid (Figure 3, correlation-shift bars). Whether drift is detected at all becomes strongly dependent on detector class (χ² = 86.7, df = 1, p = 1.3 × 10⁻²⁰, BH-adjusted p = 1.8 × 10⁻²⁰). These results address **RQ2**.

Table 2 reports per-detector miss rates on the dependency-only condition.

**Table 2: Miss rate by detector, dependency-only shift (n = 540 per detector)**

| Detector | Class | Miss rate |
|---|---|---|
| MMD | Joint-dependence | 11.9% |
| HDDDM | Marginal-information | 53.3% |
| KS | Marginal-information | 57.6% |
| PSI | Marginal-information | 57.6% |
| Wasserstein | Marginal-information | 60.6% |
| C2ST | Joint-dependence | 61.3% |

Primary inferential analysis via clustered GEE (accounting for within-block correlation across 54 clusters; Table 4, Model M2) confirms that MMD dramatically reduces the odds of missing dependency drift relative to the KS baseline: adjusted OR = 0.032 (95% CI [0.015, 0.068], p < 0.001), representing a 96.8% reduction in missed-detection odds. By contrast, none of the other four detectors differ significantly from KS under dependency shift: C2ST (OR = 1.271 [0.763, 2.117], p = 0.36), HDDDM (OR = 0.763 [0.445, 1.310], p = 0.33), PSI (OR = 1.000 [0.604, 1.655], p = 1.00), and Wasserstein (OR = 1.211 [0.740, 1.982], p = 0.45). Resource constraints also heavily penalize dependency detection: window size 500 increases the odds of missing drift by a factor of 11.952 (95% CI [4.867, 29.347], p < 0.001) relative to window 50, and monitoring frequency 1,000 multiplies miss odds by 12.599 (95% CI [8.933, 17.768], p < 0.001).

Crucially, the condition × detector interaction model fit across the balanced common grid (Table 4, Model M3b; n = 6,480 across 54 clusters) reveals that MMD is the only detector exhibiting a significant interaction with shift condition (interaction OR = 0.033 [0.002, 0.488], p = 0.0132). Whereas all marginal-information detectors perform identically between conditions after adjusting for constraints, MMD's relative detection capability specifically improves under dependency shift, providing rigorous inferential support for the taxonomy distinction.

Table 3 (Section 3.5) provides the matched false-alarm rates, establishing that MMD (0.31%) and C2ST (0.42%) maintain well-controlled false-alarm rates, confirming that their miss-rate results in Table 2 represent genuine detection power. The high false-alarm rates of PSI and HDDDM at window=50 (51.5% and 61.5%) impose a more nuanced interpretation: at small windows, their alarming partly reflects histogram estimation noise on null data, not sensitivity to the dependency signal.

The observed separation addresses **RQ3** and does not align cleanly with the conventional univariate/multivariate label. HDDDM is formally described as a multivariate detector [7] but aggregates per-feature Hellinger distances, and its 53.3% miss rate is close to the marginal-information group's average (57.5%). This is not a surprising empirical weakness but a predictable structural consequence: HDDDM's published formulation [7, Eq. 1] defines the joint statistic as the mean of per-feature distances, a quantity computed entirely from each feature's own marginal histogram. A statistic constructed this way has zero population-level signal for a shift that leaves every marginal histogram unchanged, regardless of sample size. HDDDM's performance on Table 2 confirms this limit empirically; it does not discover it.

C2ST with logistic regression records the highest miss rate (61.3%) despite operating on joint feature vectors. A targeted sensitivity analysis (Table B3) was conducted varying window size across {50, 100, 200, 500, 1000} and comparing logistic regression with a shallow random forest (50 trees, max depth 3), across 3 datasets x 2 drift types x 5 repetitions = 300 configurations per classifier.

**Table B3: C2ST miss rate by classifier and window size (dependency-shift condition)**

| Classifier | Window 50 | Window 100 | Window 200 | Window 500 | Window 1000 |
|---|---|---|---|---|---|
| Logistic regression | 50.0% | 46.7% | 23.3% | 16.7% | 16.7% |
| Shallow random forest | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

The result is decisive on both fronts. Logistic regression's miss rate decreases substantially with window size (from 50% at window=50 to 17% at window=500), supporting a sample-efficiency interpretation for this classifier. More significantly, the shallow random forest achieves 0% miss rate at every window size, including window=50. This demonstrates that the 61.3% main-experiment miss rate is an artifact of the logistic-regression implementation, not a fundamental limitation of the C2ST framework. The rank-permutation dependency signal is detectable by a C2ST using a tree-based classifier even at the smallest windows tested. The main results for C2ST in Table 2 should therefore be read as specific to the logistic-regression implementation.

Pairwise Mann-Whitney comparisons between MMD and each other detector on the dependency-only condition are all significant after Bonferroni correction (smallest adjusted p = 2.08 × 10⁻³⁵ for MMD versus HDDDM; Table A1). MMD versus KS is not significantly different (adjusted p = 1.0), consistent with similar median delays on detected cases; the distinction between them is in miss rate (Table 2), not in delay conditional on detection. HDDDM versus PSI is not significantly different (adjusted p = 1.0), consistent with both being marginal-information procedures.

**[Figure 4: Miss rate by detector, dependency-only shift condition. Confidence intervals pending false-alarm analysis.]**

**[Figure 5: Miss rate per detector, marginal-shift versus dependency-shift, paired dumbbell comparison.]**

### 4.3 Resolving Survivorship Bias: Kaplan-Meier Right-Censored Survival Analysis

Among detected cases only, MMD's naive median delay on the dependency-only condition (457 samples) appears longer than the marginal-information group's detected-case median (144 samples). This comparison is severely distorted by survivorship bias: marginal-information methods miss 53–61% of dependency-only cases, so the subset of runs they detect represents an unrepresentative, easy-case tail where random histogram noise triggered early false alarms. Conversely, MMD detects 88.1% of all cases, successfully catching subtle, gradual, and recurring dependency shifts that require accumulating evidence over hundreds of samples. Pooling detectors into detected-only medians creates the false impression that insensitive methods are faster simply because they only catch the fastest anomalies.

To resolve this survivorship artifact, Table 6 and Figure 7 report a formal right-censored survival analysis. Runs that complete the monitoring stream without detection are treated as right-censored at the end of the post-drift observation period ($6,000$ samples for Credit, $1,407$ for Telco, $2,000$ for Maintenance).

**Table 6: Kaplan-Meier Right-Censored Time-to-Detection Statistics (Dependency-Shift Condition, n = 3,240)**

| Detector | Class | N | Detected (Events) | Missed (Censored) | Miss Rate | 25th % Delay (Samples) | Median Delay (Samples) | RMST ($\tau = 1,400$) | Log-Rank vs. MMD ($\chi^2$, p_adj) |
|---|---|---|---|---|---|---|---|---|---|
| **MMD** | Joint-dependence | 540 | 476 | 64 | 11.9% | 200 | **500** | **621.2** | — (Reference) |
| **HDDDM** | Marginal-info | 540 | 252 | 288 | 53.3% | 50 | 4,050 | 842.9 | $\chi^2 = 111.17, p < 0.001$ |
| **PSI** | Marginal-info | 540 | 229 | 311 | 57.6% | 120 | Undefined* | 903.7 | $\chi^2 = 149.69, p < 0.001$ |
| **Wasserstein** | Marginal-info | 540 | 213 | 327 | 60.6% | 184 | Undefined* | 974.6 | $\chi^2 = 203.16, p < 0.001$ |
| **KS** | Marginal-info | 540 | 229 | 311 | 57.6% | 664 | 2,750 | 1,065.7 | $\chi^2 = 243.91, p < 0.001$ |
| **C2ST** | Joint-dependence | 540 | 209 | 331 | 61.3% | 860 | 5,750 | 1,149.2 | $\chi^2 = 347.46, p < 0.001$ |

*\*Note: For PSI and Wasserstein, more than 50% of streams are right-censored without detection; median survival time is strictly undefined because the cumulative survival curve $S(t)$ never drops to 0.50. RMST represents Restricted Mean Survival Time (the expected duration a stream remains undetected up to the common post-drift length $\tau = 1,400$ samples; lower values indicate faster and more complete detection).*

The right-censored analysis completely reverses the naive detected-only impression:

1. **True Median Delay:** When accounting for non-detection, MMD achieves a median time-to-detection of **500 samples** (Figure 7a). By contrast, KS requires 2,750 samples to detect 50% of streams, HDDDM requires 4,050 samples, and C2ST requires 5,750 samples. PSI and Wasserstein never reach 50% detection at any point in the monitoring stream.
2. **Restricted Mean Survival Time (RMST):** Within the first 1,400 samples following drift onset, an MMD-monitored stream spends on average only **621.2 samples** undetected. Marginal-information detectors leave streams undetected for substantially longer: HDDDM (842.9 samples), PSI (903.7 samples), Wasserstein (974.6 samples), and KS (1,065.7 samples). C2ST with logistic regression leaves streams undetected for 1,149.2 samples.
3. **Log-Rank Tests:** An overall log-rank test across all six detectors demonstrates massive inequality in survival curves ($\chi^2 = 394.69, \text{df} = 5, p < 10^{-80}$). Pairwise log-rank tests with Bonferroni correction confirm that MMD's time-to-detection distribution is significantly superior to every competitor (smallest test statistic: MMD vs. HDDDM, $\chi^2 = 111.17, p_{\text{adj}} < 10^{-24}$; Table 6).

Figure 7 visually illustrates the resolution of the artifact. In Panel (a), the cumulative detection function $1 - S(t)$ shows MMD rapidly and continuously rising to 88.1% cumulative detection, whereas all marginal detectors plateau below 45%. Panel (b) shows the survival function $S(t)$, confirming that marginal detectors' apparent speed in detected-only summaries was an artifact of discarding the >50% of streams that remained undetected.

**[Figure 7: Kaplan-Meier survival curves under dependency-only drift. (a) Cumulative detection function $1 - S(t)$ over post-drift observation delay, showing MMD rising to 88.1% while marginal detectors plateau below 45%. (b) Survival probability $S(t)$ (probability of remaining undetected), resolving the survivorship artifact.]**

### 4.4 Configuration Effects Persist Within the Dependency-Only Condition

Configuration still matters under dependency-only shift but is no longer the dominant factor. Monitoring interval (ρ = 0.37, p = 2.5 × 10⁻⁵³, BH-adjusted p = 2.9 × 10⁻⁴⁸) and window size (ρ = 0.31, p = 4.4 × 10⁻³⁷, BH-adjusted p = 4.0 × 10⁻³⁵) retain significant, moderate monotonic effects on detection delay. Reference size, which shows no significant effect under marginal shift, becomes marginally significant (ρ = −0.06, p = 0.017, BH-adjusted p = 0.020): with more reference data, MMD's kernel estimate of the joint dependence structure sharpens, which may matter more for a shift visible only in joint structure than for one visible in any single feature's histogram. This effect is small and should be interpreted cautiously.

### 4.5 Predictive Impact and Class-Imbalance Analysis

Mean performance degradation, measured against the matched counterfactual baseline, is reported across all three datasets and conditions in Table 7 and visualized in Figure 6. To account for sampling variation across independently retrained models and data splits, all estimates are accompanied by 95% confidence intervals computed over the 15 experimental trials per condition (3 drift types × 5 repetitions). Furthermore, because all three datasets exhibit class imbalance (Credit: 22.1% positive; Telco: 26.5% positive; AI4I Maintenance: 3.4% positive), plain classification accuracy is supplemented by balanced accuracy (macro-averaged recall across classes) and ROC-AUC (probability discrimination), ensuring that degradation on rare target events is not masked by the majority class.

**Table 7: Counterfactual Predictive Impact Across Imbalanced Datasets (Mean Degradation with 95% Confidence Intervals)**

| Dataset & Target Class Ratio | Condition | Baseline Acc (%) | Plain Accuracy Loss (pp) [95% CI] | Baseline BAcc (%) | Balanced Acc Loss (pp) [95% CI] | Baseline AUC (%) | ROC-AUC Loss (pp) [95% CI] |
|---|---|---|---|---|---|---|---|
| **Credit Card Default** <br> (22.1% positive) | Marginal <br> Dependency | 81.35 <br> 81.35 | −0.10 [−0.21, +0.01] <br> −0.10 [−0.18, −0.02] | 65.25 <br> 65.25 | +0.78 [+0.55, +1.01] <br> +0.60 [+0.46, +0.73] | 75.95 <br> 75.95 | +1.11 [+0.79, +1.44] <br> +0.60 [+0.42, +0.77] |
| **Telco Churn** <br> (26.5% positive) | Marginal <br> Dependency | 79.77 <br> 79.77 | −0.03 [−0.50, +0.44] <br> +0.36 [+0.02, +0.71] | 70.57 <br> 70.57 | +1.14 [−0.10, +2.37] <br> +0.83 [+0.24, +1.41] | 84.16 <br> 84.16 | −0.39 [−0.76, −0.03] <br> +0.39 [+0.19, +0.60] |
| **AI4I Maintenance** <br> (3.4% positive) | Marginal <br> Dependency | 98.24 <br> 98.24 | +4.78 [+4.05, +5.51] <br> +0.91 [+0.73, +1.08] | 77.84 <br> 77.84 | −2.34 [−3.52, −1.17] <br> +4.84 [+2.96, +6.72] | 96.69 <br> 96.69 | +4.06 [+3.60, +4.51] <br> +4.58 [+3.17, +5.98] |

*Notes: Degradation values are expressed in percentage points (pp), defined as counterfactual matched baseline minus post-drift performance; positive values indicate downstream model degradation, while values near zero indicate benign drift. 95% confidence intervals are derived from Student-t distributions over n = 15 trials per cell. Plain accuracy measures 0/1 predictions; balanced accuracy computes macro-averaged recall across classes; ROC-AUC measures continuous score discrimination.*

The multi-metric analysis provides critical insights into how distributional change interacts with class imbalance:

1. **Credit and Telco (Moderate Imbalance, 22–26% Positive):** Both datasets show minimal downstream vulnerability across both conditions. Plain accuracy changes by less than 0.36 pp in all cases. Examining balanced accuracy and ROC-AUC confirms that performance degradation remains modest: balanced accuracy drops by at most 1.14 pp (Telco marginal) and ROC-AUC drops by at most 1.11 pp (Credit marginal). Dependency shift is exceptionally benign in both domains, causing only 0.60 pp balanced accuracy loss on Credit and 0.83 pp on Telco.
2. **AI4I Maintenance (Severe Imbalance, 3.4% Positive):** In this heavily skewed industrial domain (where 96.6% of samples represent normal operation), plain accuracy creates an incomplete impression of drift impact. Under dependency shift, plain accuracy drops by only 0.91 pp [0.73, 1.08], which might suggest minimal impact. However, the class-imbalance metrics reveal genuine degradation on the failure event: balanced accuracy degrades by **+4.84 pp** [2.96, 6.72] and ROC-AUC degrades by **+4.58 pp** [3.17, 5.98]. The dependency shift alters joint operating temperatures and tool wear in a way that specifically reduces the model's ability to discriminate machine failures from normal states. Under marginal shift, machine failure predictions become more aggressive (increasing recall at the expense of false alarms, yielding a negative balanced accuracy degradation of −2.34 pp), while overall ranking discriminability degrades by +4.06 pp ROC-AUC.
3. **Statistical Detectability vs. Predictive Harm:** Across all domains, statistical detectability does not directly scale with downstream harm. Dependency shift is readily detectable by joint-dependence monitoring (MMD achieves 88.1% detection), yet its impact on downstream accuracy is substantially smaller than marginal shift on Maintenance (0.91 pp vs. 4.78 pp) and negligible on Credit and Telco. This reinforces the principle of Rabanser et al. [3] that statistical alarms should be calibrated against operational risk rather than raw distributional divergence alone.

**[Figure 6: Counterfactual-matched downstream performance degradation across datasets and conditions with 95% confidence intervals. Panel (a): Plain classification accuracy loss. Panel (b): Balanced accuracy loss (macro-averaged recall). Panel (c): ROC-AUC degradation.]**

### 4.6 Primary Inferential Modeling: Clustered GEE and Survival Analyses

To ensure statistical rigor across the crossed repeated-measures design, Table 4 reports the primary inferential models fit via Generalized Estimating Equations (GEE) with robust Huber-White standard errors clustered by $(dataset \times detector \times drift\_type)$ (54 clusters). Table 5 reports the semiparametric survival model (`PHReg`) accounting for right-censoring when drift remains undetected over the post-drift stream.

**Table 4: Clustered Generalized Estimating Equations (GEE) Regression Models**

| Factor / Level | Model M1: Marginal Missed Detection (n=16,200) <br> OR [95% CI], p | Model M2: Dependency Missed Detection (n=3,240) <br> OR [95% CI], p | Model M4: Marginal Delay Multiplier (n=16,054) <br> Mult [95% CI], p | Model M3b: Condition Interaction (n=6,480) <br> Inter. OR [95% CI], p |
|---|---|---|---|---|
| **Detector (ref: KS)** | | | | |
| C2ST | 3.127 [1.355, 7.220], p=0.0076 | 1.271 [0.763, 2.117], p=0.357 | 1.552× [1.383, 1.742], p<0.001 | 1.123 [0.080, 15.684], p=0.931 |
| HDDDM | 0.120 [0.054, 0.266], p<0.001 | 0.763 [0.445, 1.310], p=0.327 | 1.225× [1.100, 1.365], p<0.001 | 2.423 [0.167, 35.184], p=0.517 |
| MMD | 3.127 [1.612, 6.069], p<0.001 | 0.032 [0.015, 0.068], p<0.001 | 1.409× [1.252, 1.587], p<0.001 | 0.033 [0.002, 0.488], p=0.013 |
| PSI | 0.263 [0.117, 0.589], p=0.0012 | 1.000 [0.604, 1.655], p=1.000 | 1.312× [1.137, 1.514], p<0.001 | 1.854 [0.128, 26.791], p=0.651 |
| Wasserstein | 0.437 [0.230, 0.830], p=0.0114 | 1.211 [0.740, 1.982], p=0.447 | 1.160× [1.067, 1.262], p<0.001 | 1.841 [0.133, 25.405], p=0.649 |
| **Window size (ref: 50)** | | | | |
| 100 | 1.913 [1.007, 3.635], p=0.0475 | — | 1.449× [1.354, 1.551], p<0.001 | — |
| 500 | <0.001 [0.000, 0.000], p<0.001 | 11.952 [4.867, 29.347], p<0.001 | 3.500× [3.008, 4.074], p<0.001 | 8.552 [3.575, 20.458], p<0.001 |
| 1,000 | 0.031 [0.002, 0.424], p=0.0092 | — | 5.489× [4.584, 6.574], p<0.001 | — |
| **Monitoring interval (ref: 10)** | | | | |
| 50 | 1.001 (ns) | — | 1.572× [1.494, 1.655], p<0.001 | — |
| 100 | 1.001 (ns) | 2.201 [1.728, 2.804], p<0.001 | 1.948× [1.821, 2.083], p<0.001 | 2.058 [1.635, 2.591], p<0.001 |
| 500 | 5.35 × 10¹⁰ (ns) | — | 3.471× [3.161, 3.812], p<0.001 | — |
| 1,000 | 6.26 × 10¹³ (ns) | 12.599 [8.933, 17.768], p<0.001 | 4.949× [4.348, 5.633], p<0.001 | 12.144 [8.776, 16.804], p<0.001 |
| **Reference size (ref: full)** | | | | |
| 100 | 1.000 [0.284, 3.522], p=1.000 | — | 1.003× [0.933, 1.079], p=0.927 | — |
| 500 | 1.142 [0.782, 1.668], p=0.492 | 0.950 [0.842, 1.071], p=0.397 | 1.012× [0.993, 1.031], p=0.225 | 0.959 [0.864, 1.065], p=0.431 |
| **Drift type (ref: sudden)** | | | | |
| Gradual | 2.50 × 10¹¹ (ns) | 0.937 [0.719, 1.222], p=0.633 | 2.046× [1.859, 2.251], p<0.001 | 0.944 [0.734, 1.215], p=0.655 |
| Recurring | 3.34 × 10¹³ (ns) | 1.247 [0.910, 1.708], p=0.169 | 1.000× [0.938, 1.067], p=0.994 | 1.551 [1.138, 2.113], p=0.005 |

*Notes: All models fit via statsmodels GEE with exchangeable/independent working correlation and Huber-White sandwich standard errors clustered by dataset × detector × drift type (54 clusters). Models M1, M2, and M3b use a binomial family with logit link (outcome: missed detection). Model M4 uses a Gaussian family on log(delay) for detected runs; exponentiated coefficients represent multiplicative factors on delay in samples. In Model M1, miss rates under marginal shift are extremely low (<0.9% overall), producing separation across coarse monitoring intervals. Model M3b evaluates the condition × detector interaction on the balanced common subgrid (n = 6,480); the MMD × Dependency interaction is statistically significant (p = 0.0132), confirming that MMD's relative performance fundamentally shifts when marginal signals are eliminated.*

**Table 5: Clustered Cox Proportional Hazards Model (Right-Censored Time-to-Detection, Dependency Condition)**

| Term | log HR | SE | Hazard Ratio (HR) | 95% CI for HR | z-score | p-value |
|---|---|---|---|---|---|---|
| **Detector (ref: KS)** | | | | | | |
| C2ST | −0.432 | 0.096 | 0.649 | [0.537, 0.784] | −4.435 | <0.001 |
| HDDDM | +0.875 | 0.095 | 2.398 | [1.987, 2.893] | +9.176 | <0.001 |
| MMD | +1.154 | 0.083 | 3.170 | [2.700, 3.723] | +13.974 | <0.001 |
| PSI | +0.649 | 0.097 | 1.914 | [1.583, 2.313] | +6.643 | <0.001 |
| Wasserstein | +0.334 | 0.096 | 1.397 | [1.156, 1.689] | +3.473 | <0.001 |
| **Window size (ref: 50)** | | | | | | |
| Window 500 | −1.753 | 0.062 | 0.173 | [0.152, 0.196] | −28.188 | <0.001 |
| **Monitoring interval (ref: 10)** | | | | | | |
| Interval 100 | −0.745 | 0.058 | 0.475 | [0.424, 0.531] | −12.915 | <0.001 |
| Interval 1,000 | −1.916 | 0.076 | 0.147 | [0.128, 0.170] | −25.265 | <0.001 |
| **Reference size (ref: full)** | | | | | | |
| Ref 500 | +0.082 | 0.050 | 1.085 | [0.984, 1.197] | +1.636 | 0.102 |

*Notes: Fit using statsmodels PHReg (n = 3,240, 1,608 detected events, 1,632 right-censored runs; 54 clusters). Censoring time equals the full post-drift stream length for undetected runs. Hazard Ratio (HR) > 1 indicates a higher instantaneous detection rate (faster detection); HR < 1 indicates slower detection. MMD exhibits more than three times the detection rate of KS (HR = 3.170, p < 0.001) while properly accounting for censored (missed) observations.*

---

## 5. Discussion

### 5.1 A Conditional Deployment Principle

The experiments support a conditional deployment principle. When the shifted distribution has a marginal signature, a change visible in at least one feature's own distribution, monitoring interval and detection-window size dominate detector identity on the reported rank-based effect-size scale, replicating the finding of Rabanser et al. [3] for multivariate detectors on tabular data. When the shift is confined to dependence, that ordering changes: detector construction becomes the dominant observed factor, exceeding both constraint effects. The practical implication is that monitoring architecture should be chosen only after considering what forms of change are plausible in the application domain.

### 5.2 Why Detector Construction is More Informative Than the 'Multivariate' Label

The central conceptual result is that input dimensionality does not determine statistical sensitivity. HDDDM demonstrates this directly: under an exactly marginal-preserving permutation, its statistic receives no injected marginal signal, a structural impossibility that follows from the definition of its formulation [7, Eq. 1], not from any implementation limitation. Conversely, MMD responds to a change in dependence because its kernel-based statistic operates on joint observations.

This information-based taxonomy, marginal-information versus joint-dependence, is more precise than classifying methods only as univariate or multivariate. A practitioner selecting a detector on the basis of a multivariate label alone, without verifying that the statistic's construction uses cross-feature information, risks deploying a monitor that cannot detect the class of change they are trying to catch.

C2ST's result requires a more nuanced interpretation in light of Table B3. With logistic regression, miss rate decreases substantially with window size, from 50% at window=50 to 17% at window=500. With a shallow random forest, miss rate drops to 0% at every window size. The relevant constraint is classifier expressiveness under a fixed sample budget, not window size per se.

### 5.3 Relation to Published Prior Work

Rabanser et al. [3] showed that Bonferroni-corrected univariate testing performs comparably to multivariate kernel testing for many benchmark shifts, a result this study's marginal-shift grid replicates independently on tabular data (detector class shows no significant effect on missed detection under marginal shift, p = 0.21). This study identifies precisely where that equivalence breaks: specifically when the shift is constructed to leave marginal distributions provably untouched. No shift in [3], nor in this study's own marginal-shift condition, tests that boundary.

This study does not claim to be the first to introduce a dependency-shift benchmark. Kamulete [4] explicitly includes covariance-based dependency shifts in simulated settings. The distinction is that Kamulete's shifts operate through altered covariance parameters of a parametric model, while the rank-permutation mechanism here preserves observed empirical marginals exactly and operates directly on observed tabular feature values without imposing a parametric joint structure. These are complementary, not competing, contributions.

### 5.4 Statistical Versus Operationally Harmful Shift

The weak predictive degradation observed for several dependency-only shift settings (Section 4.5) reinforces an important distinction: a statistically detectable change need not be operationally harmful. Monitoring systems should separate a statistical alert from an intervention decision. A reasonable production workflow is: detect distributional change, diagnose which component changed (marginal, joint, or both), quantify predictive impact, and only then trigger retraining or human review.

### 5.5 Limitations

Several limitations bound the claims made in this paper.

First, only three tabular datasets and one feature pair per dataset are evaluated in the main grid. The feature-pair sensitivity analysis (Section 3.1, Table B6) confirms that the qualitative ordering is preserved across all tested alternative pairs, with the noted caveat that MMD's miss rate rises to 27.8% on the Telco alternative pair due to a moderate baseline correlation reducing the absolute magnitude of the dependence shift.

Second, the dependency shift is bivariate. Whether the same mechanistic distinctions persist at higher dimensionality is not established.

Third, the principal shifts remain synthetically injected, a limitation shared with the great majority of the drift-detection literature including [3].

Fourth, MMD is subsampled (300 observations); the equal-budget sensitivity analysis (Section 3.4) establishes that this cap does not explain its advantage over HDDDM.

Fifth, the main-experiment C2ST results are specific to a logistic-regression implementation. Table B3 shows that a shallow random forest recovers 0% miss rate at all window sizes; the 61.3% result should not be generalized to C2ST as a method.

Sixth, the predictive-impact analysis uses a single Random Forest model and should not be generalized to arbitrary learners. Uncertainty intervals and class-imbalance-aware metrics (balanced accuracy, ROC-AUC) remain pending.

Seventh, the primary inferential analysis relies on Kruskal-Wallis and Mann-Whitney statistics on a crossed repeated design; the mixed-model analysis required by the journal remains pending.

Finally, a right-censored time-to-detection analysis remains pending for final submission. The 60/40 split choice has been confirmed not to drive the main findings: a sensitivity analysis at 50/50 and 70/30 (Tables B4 and B5) shows that detector identity remains the largest epsilon-squared factor under dependency shift and MMD's miss-rate advantage over HDDDM is preserved at all tested split ratios.

---

## 6. Conclusion

This study investigated whether monitoring configuration or detector construction is the dominant factor in tabular drift detection under two regimes: marginal-visible shifts for which both univariate and multivariate statistics have a signal, and marginal-preserving dependency shifts for which only a detector whose statistic uses joint feature information can succeed by construction.

Under marginal-visible shift, across 16,200 experimental runs, configuration effects on the rank-based effect-size scale substantially exceed detector-identity effects (window size ε² = 0.349, monitoring interval ε² = 0.250, detector identity ε² = 0.015), and detector class has no significant effect on whether drift is detected at all (p = 0.21). Under dependency-only shift, across 3,240 runs, the ordering reverses: detector identity ε² = 0.236 exceeds both constraint effects, and detector class becomes highly significant for detection (p = 1.3 × 10⁻²⁰, BH-adjusted).

The decisive factor is not the univariate/multivariate label. HDDDM, a formally multivariate method, achieves a 53.3% miss rate under dependency-only shift because its published construction averages per-feature divergences, making it a marginal-information procedure despite its name. MMD achieves an 11.9% miss rate because its kernel-based statistic operates on joint observations. C2ST with logistic regression achieves a 61.3% miss rate; Table B3 shows this is an implementation artifact: a shallow random forest achieves 0% miss rate at every window size tested, demonstrating that the rank-permutation dependency signal is detectable by a C2ST with a sufficiently expressive classifier. Table 3 confirms that MMD's miss-rate advantage reflects genuine detection power: its false-alarm rate (0.31%) is well-controlled, while PSI and HDDDM alarm on 51.5% and 61.5% of null checks at window=50.

There is no universal ordering in which configuration always matters more than detector choice. The more useful distinction for practitioners is marginal-information versus joint-dependence statistics. Configuration dominates when drift has a marginal signature. Detector construction becomes decisive when the change is confined to dependence among features whose individual distributions remain stable. For deployments monitoring correlated feature pairs, verifying that the chosen detector's statistic uses cross-feature information is more important than whether it is labelled univariate or multivariate.

Three directions follow from this study's scope decisions. First, extending the analysis to higher-dimensional dependence changes beyond bivariate pairs to test whether the information-content taxonomy generalises to the multivariate copula case. Second, applying the same marginal-visible / dependency-only contrast to representation-space monitoring in text and image domains. Third, the striking 0% miss rate of C2ST with a shallow random forest motivates a targeted study: given that 25 training samples per class suffice with a tree-based classifier, C2ST may be a practical joint-dependence monitor even under severe constraint budgets, which the logistic-regression results obscure.

---

## 7. Data Availability Statement

> **[ACTION REQUIRED]:** Replace this placeholder with a permanent repository URL and DOI before submission. The repository must contain all items listed in Section 3.8. *Do not submit with this placeholder in place.*

The datasets used in this study are publicly available: Credit [14], AI4I Maintenance [15], and Telco Customer Churn [16]. Processed run-level results, source code, experiment configurations, random seeds, detector implementations, and analysis scripts are being prepared for permanent archival and will be made available at [REPOSITORY URL/DOI PENDING BEFORE SUBMISSION].

---

## Conflict of Interest

The authors declare no conflict of interest.

---

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## Acknowledgments

No external contributions requiring acknowledgment beyond those meeting authorship criteria.

---

## References

[1] A. Sklar, "Fonctions de répartition à n dimensions et leurs marges," *Publications de l'Institut de Statistique de l'Université de Paris*, vol. 8, pp. 229–231, 1959.

[2] R. B. Nelsen, *An Introduction to Copulas*, 2nd ed. New York: Springer, 2006. https://doi.org/10.1007/0-387-28678-0

[3] S. Rabanser, S. Günnemann, and Z. C. Lipton, "Failing loudly: An empirical study of methods for detecting dataset shift," *Advances in Neural Information Processing Systems*, vol. 32, pp. 1396–1408, 2019.

[4] V. M. Kamulete, "Test for non-negligible adverse shifts," in *Proc. 38th Conf. Uncertainty in Artificial Intelligence (UAI)*, PMLR, vol. 180, pp. 959–968, 2022.

[5] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. J. Smola, "A kernel two-sample test," *Journal of Machine Learning Research*, vol. 13, no. 25, pp. 723–773, 2012.

[6] D. Lopez-Paz and M. Oquab, "Revisiting classifier two-sample tests," in *Int. Conf. Learning Representations (ICLR)*, 2017.

[7] G. Ditzler and R. Polikar, "Hellinger distance based drift detection for nonstationary environments," in *Proc. 2011 IEEE Symp. Computational Intelligence in Dynamic and Uncertain Environments (CIDUE)*, pp. 41–48, 2011. https://doi.org/10.1109/CIDUE.2011.5948491

[8] Z. C. Lipton, Y.-X. Wang, and A. J. Smola, "Detecting and correcting for label shift with black box predictors," in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, pp. 3122–3130, 2018.

[9] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4, pp. 44:1–44:37, 2014. https://doi.org/10.1145/2523813

[10] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang, "Learning under concept drift: A review," *IEEE Trans. Knowledge and Data Engineering*, vol. 31, no. 12, pp. 2346–2363, 2019. https://doi.org/10.1109/TKDE.2018.2876857

[11] G. I. Webb, L. K. Lee, B. Goethals, and F. Petitjean, "Characterizing concept drift," *Data Mining and Knowledge Discovery*, vol. 30, no. 4, pp. 964–994, 2016. https://doi.org/10.1007/s10618-015-0448-4

[12] A. Schrab, I. Kim, M. Albert, B. Laurent, B. Guedj, and A. Gretton, "MMD aggregated two-sample test," *Journal of Machine Learning Research*, vol. 24, no. 194, pp. 1–81, 2023.

[13] F. Bayram, B. S. Ahmed, and A. Kassler, "From concept drift to model degradation: An overview on performance-aware drift detectors," *Knowledge-Based Systems*, vol. 245, article 108632, 2022. https://doi.org/10.1016/j.knosys.2022.108632

[14] I.-C. Yeh and C.-H. Lien, "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients," *Expert Systems with Applications*, vol. 36, no. 2, pp. 2473–2480, 2009. https://doi.org/10.1016/j.eswa.2007.12.020 [Dataset accessed via UCI Machine Learning Repository / Kaggle.]

[15] S. Matzka, "Explainable artificial intelligence for predictive maintenance applications," in *Proc. 3rd Int. Conf. Artificial Intelligence for Industries (AI4I)*, pp. 69–74, 2020. https://doi.org/10.1109/AI4I49448.2020.00018 [AI4I 2020 Predictive Maintenance dataset, UCI ML Repository / Kaggle.]

[16] IBM Analytics Community, "Telco customer churn," IBM Sample Data Sets, available at Kaggle, https://www.kaggle.com/datasets/blastchar/telco-customer-churn, 2018. [Accessed March 2024.]

---

## Appendix A: Supplementary Statistical Tables

**Table A1: Pairwise Mann-Whitney U results, dependency-only condition (Bonferroni correction, 15 pairs)**

| A | B | n_A | n_B | U | p (raw) | p (Bonferroni) | r | Sig. |
|---|---|---|---|---|---|---|---|---|
| C2ST | HDDDM | 209 | 252 | 44,869.0 | 7.8 × 10⁻³⁹ | 1.2 × 10⁻³⁷ | −0.704 | Yes |
| C2ST | KS | 209 | 229 | 28,669.5 | 3.4 × 10⁻⁴ | 5.1 × 10⁻³ | −0.198 | Yes |
| C2ST | MMD | 209 | 476 | 65,953.0 | 1.0 × 10⁻¹¹ | 1.6 × 10⁻¹⁰ | −0.326 | Yes |
| C2ST | PSI | 209 | 229 | 40,425.5 | 1.0 × 10⁻³⁵ | 1.6 × 10⁻³⁴ | −0.689 | Yes |
| C2ST | Wasserstein | 209 | 213 | 34,931.0 | 4.3 × 10⁻²⁴ | 6.5 × 10⁻²³ | −0.569 | Yes |
| HDDDM | KS | 252 | 229 | 12,604.5 | 1.1 × 10⁻²⁶ | 1.6 × 10⁻²⁵ | +0.563 | Yes |
| HDDDM | MMD | 252 | 476 | 25,903.5 | 1.4 × 10⁻³⁶ | 2.1 × 10⁻³⁵ | +0.568 | Yes |
| HDDDM | PSI | 252 | 229 | 26,357.0 | 0.099 | 1.000 | +0.087 | No |
| HDDDM | Wasserstein | 252 | 213 | 19,854.5 | 1.1 × 10⁻⁶ | 1.7 × 10⁻⁵ | +0.260 | Yes |
| KS | MMD | 229 | 476 | 57,297.5 | 0.270 | 1.000 | −0.051 | No |
| KS | PSI | 229 | 229 | 39,448.5 | 8.9 × 10⁻²¹ | 1.3 × 10⁻¹⁹ | −0.504 | Yes |
| KS | Wasserstein | 229 | 213 | 33,904.0 | 1.3 × 10⁻¹² | 1.9 × 10⁻¹¹ | −0.390 | Yes |
| MMD | PSI | 476 | 229 | 83,445.5 | 2.8 × 10⁻³⁰ | 4.2 × 10⁻²⁹ | −0.531 | Yes |
| MMD | Wasserstein | 476 | 213 | 70,355.5 | 3.7 × 10⁻¹⁶ | 5.5 × 10⁻¹⁵ | −0.388 | Yes |
| PSI | Wasserstein | 229 | 213 | 20,418.0 | 3.0 × 10⁻³ | 4.4 × 10⁻² | +0.163 | Yes |

*Note: HDDDM vs. PSI and KS vs. MMD are not statistically distinguishable after Bonferroni correction. The KS vs. MMD non-significance is consistent with both having similar median delays on detected cases; the distinction is in miss rate (Table 2), not in delay conditional on detection.*

**Table A2: Pairwise Mann-Whitney U results, marginal-shift condition (Bonferroni correction, 15 pairs)**

| A | B | n_A | n_B | U | p (raw) | p (Bonferroni) | r | Sig. |
|---|---|---|---|---|---|---|---|---|
| C2ST | HDDDM | 2,665 | 2,689 | 4,009,499.5 | 4.1 × 10⁻¹⁴ | 6.2 × 10⁻¹³ | −0.119 | Yes |
| C2ST | KS | 2,665 | 2,673 | 4,376,033.0 | 9.9 × 10⁻⁴⁸ | 1.5 × 10⁻⁴⁶ | −0.229 | Yes |
| C2ST | MMD | 2,665 | 2,665 | 3,729,560.0 | 1.4 × 10⁻³ | 2.2 × 10⁻² | −0.050 | Yes |
| C2ST | PSI | 2,665 | 2,683 | 3,877,732.5 | 7.7 × 10⁻⁸ | 1.2 × 10⁻⁶ | −0.085 | Yes |
| C2ST | Wasserstein | 2,665 | 2,679 | 4,118,736.5 | 1.6 × 10⁻²² | 2.4 × 10⁻²¹ | −0.154 | Yes |
| HDDDM | KS | 2,689 | 2,673 | 3,898,640.5 | 6.8 × 10⁻⁸ | 1.0 × 10⁻⁶ | −0.085 | Yes |
| HDDDM | MMD | 2,689 | 2,665 | 3,315,398.0 | 2.1 × 10⁻⁶ | 3.1 × 10⁻⁵ | +0.075 | Yes |
| HDDDM | PSI | 2,689 | 2,683 | 3,490,954.5 | 0.040 | 0.603 | +0.032 | No |
| HDDDM | Wasserstein | 2,689 | 2,679 | 3,688,398.5 | 0.127 | 1.000 | −0.024 | No |
| KS | MMD | 2,673 | 2,665 | 2,915,157.0 | 9.5 × 10⁻³¹ | 1.4 × 10⁻²⁹ | +0.182 | Yes |
| KS | PSI | 2,673 | 2,683 | 3,175,172.0 | 3.3 × 10⁻¹³ | 4.9 × 10⁻¹² | +0.115 | Yes |
| KS | Wasserstein | 2,673 | 2,679 | 3,334,850.0 | 1.3 × 10⁻⁵ | 1.9 × 10⁻⁴ | +0.069 | Yes |
| MMD | PSI | 2,665 | 2,683 | 3,716,904.0 | 0.012 | 0.177 | −0.040 | No |
| MMD | Wasserstein | 2,665 | 2,679 | 3,953,752.0 | 8.5 × 10⁻¹² | 1.3 × 10⁻¹⁰ | −0.108 | Yes |
| PSI | Wasserstein | 2,683 | 2,679 | 3,795,874.5 | 3.5 × 10⁻⁴ | 5.3 × 10⁻³ | −0.056 | Yes |

---

*This document (paper_draft_revision_v1.md) is the standalone revision addressing the editorial review from Statistical Analysis and Data Mining: An ASA Data Science Journal. The original paper_draft.md is preserved as-is for reference. The unpublished companion study is not cited; novelty is framed relative to Rabanser et al. [3] and Kamulete [4].*

*RESOLVED in this revision: primary mixed-model / GEE factorial inference (Table 4, Models M1–M4, M3b interaction), clustered Cox proportional hazards survival model (Table 5), Kaplan-Meier right-censored time-to-detection analysis resolving survivorship bias (Table 6, Figure 7, log-rank tests), predictive impact uncertainty intervals with balanced accuracy and ROC-AUC for class imbalance (Table 7, Figure 6), verified reference list DOIs and page numbers ([1]–[16]), false-alarm rates (Table 3), saturation quantiles (Table B1), dependency strength (Table B2), C2ST classifier sensitivity (Table B3), split sensitivity (Tables B4, B5), feature-pair sensitivity (Table B6), MMD equal-budget analysis, Wasserstein bootstrap detail, literature expansion ([12][13]), dataset citations ([14][15][16]), feature pair selection rule stated, AI4I exclusion criteria stated, Acknowledgments, funding.*

*PENDING for final submission: permanent repository URL and Zenodo DOI (to be inserted in Data Availability Statement upon GitHub + Zenodo release).*

