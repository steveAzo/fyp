# Configuration Still Beats Algorithm, Until the Drift Hides in the Correlations: Extending Resource-Constrained Drift Detection to Multivariate Shift

**Stephen Kwaku Azongo**
Department of Computer Science, University of Ghana
skazongo@st.ug.edu.gh

**Kofi Sarpong Adu-Manu, Livingston Seble, Alex Teye Ametepey, Sandro Amofa**
Department of Computer Science, University of Ghana

---

## Abstract

A companion study established that, for lightweight univariate drift detectors under resource constraints, monitoring configuration (frequency, window size) affects detection performance far more than detector choice does, producing a 37-fold range in detection delay against a comparatively small gap between the Kolmogorov-Smirnov test, the Population Stability Index, and the Wasserstein distance. This finding leaves an open question with real practical weight: does configuration still dominate once the detector class itself changes, from univariate statistics to genuinely multivariate ones, and does it survive a shift that a univariate statistic cannot see even in principle? We extend the original constraint grid to three multivariate detectors (Maximum Mean Discrepancy, a Classifier Two-Sample Test, and the Hellinger Distance Drift Detection Method) evaluated jointly on pairs of correlated features across the same three datasets, under two drift conditions: a marginal shift that moves each feature's own distribution, and a novel correlation-only shift, constructed via rank-based permutation, that leaves both features' marginal distributions exactly unchanged and alters only the dependency between them. Across 16,200 marginal-shift runs, the original finding replicates: configuration effect size (window size, ε² = 0.349) dwarfs detector-identity effect size (ε² = 0.015), and detector class has no significant effect on whether drift is detected at all (χ², p = 0.21). Across 3,240 correlation-shift runs, this hierarchy inverts, detector identity effect size rises to ε² = 0.236, exceeding both constraint effects, and detector class becomes highly significant for detection at all (χ², p = 1.3 × 10⁻²⁰). Critically, this is not a clean univariate-versus-multivariate split: only MMD, whose test statistic genuinely incorporates cross-feature dependency, detects the correlation-only shift reliably (11.9% miss rate); HDDDM, whose published formulation averages per-feature Hellinger distances, is mathematically restricted to marginal information and performs no better than the univariate group (53.3% miss rate) despite its multivariate label; and the Classifier Two-Sample Test, theoretically capable but constrained by its train/test-split construction, performs worse than any univariate method under these resource limits (61.3% miss rate). We conclude that configuration dominance is conditional on the drift having a marginal signature, and that when it does not, what matters is not whether a detector is labelled multivariate but whether its statistic's construction actually uses cross-feature information.

**Keywords:** concept drift detection, multivariate two-sample testing, resource-constrained machine learning, correlation shift, maximum mean discrepancy

---

## 1. Introduction

The companion study to this paper established a priority hierarchy for practitioners deploying drift monitoring under resource constraints: configuration first, drift type second, detector algorithm third [CITE: paper1_self]. That finding rested on three univariate, distribution-based detectors, each comparing one feature's distribution at a time. It leaves two questions unanswered, and both matter for anyone deciding how to build a monitoring pipeline for tabular data with more than one feature of interest.

The first question is whether the finding generalises. Real deployments rarely monitor a single feature in isolation; if configuration dominance was in some way an artefact of the univariate setting, extending the detector set to genuinely multivariate methods might reveal a different picture entirely.

The second question is sharper. Rabanser et al. (2019), in the most directly comparable prior study, found that Bonferroni-corrected univariate testing performs comparably to multivariate kernel testing across a range of image-domain shifts, a result they describe as surprising. That finding, if it holds generally, would suggest detector class rarely matters regardless of configuration. But Rabanser et al.'s shifts, like ours in the first half of this study, all carry a marginal signature: each perturbed feature's own distribution changes, which a well-corrected univariate test can in principle detect. No prior study has asked what happens when the marginal distributions are held provably fixed and only the dependency structure between features changes, a condition under which a univariate test cannot succeed no matter how it is corrected.

This paper answers both questions with a two-part experimental design. First, we extend the companion study's constraint grid (monitoring frequency, window size, reference size, drift type) to jointly drifted feature pairs, adding three multivariate detectors, Maximum Mean Discrepancy (MMD; Gretton et al., 2012), a Classifier Two-Sample Test (C2ST; Lopez-Paz & Oquab, 2017), and the Hellinger Distance Drift Detection Method (HDDDM; Ditzler & Polikar, 2011), alongside the original three. Second, we construct a correlation-only shift condition using a rank-based permutation mechanism grounded in copula theory [CITE: sklar_1959], which alters the dependency between two features while leaving each feature's own marginal distribution exactly unchanged, verified empirically rather than assumed.

Our contributions are:

1. **A replication** of the configuration-dominance finding under a harder, multivariate setting, across 16,200 runs spanning six detectors, three datasets, three drift types, and the full constraint grid.
2. **A marginal-preserving correlation-shift condition**, the first drift-detection benchmark condition, to our knowledge, constructed specifically so that univariate testing cannot succeed by construction rather than by empirical weakness, isolating exactly the case Rabanser et al.'s result leaves untested.
3. **A mechanistic account of when detector class matters**, showing that the relevant distinction is not the univariate/multivariate label but whether a detector's statistic incorporates cross-feature dependency in its construction, evidenced by HDDDM's near-univariate performance despite its multivariate design and MMD's large advantage over it.

---

## 2. Related Work

**Multivariate two-sample testing.** Gretton et al. (2012) introduce the Maximum Mean Discrepancy as a kernel-based statistic for the two-sample problem, defined as the distance between mean embeddings of two distributions in a reproducing kernel Hilbert space, and establish that MMD is zero if and only if the two distributions are equal when the kernel is characteristic. The unbiased estimator and the median-heuristic bandwidth selection used in this study follow their formulation directly (Gretton et al., 2012, Eq. 3 and Section 8). Lopez-Paz & Oquab (2017) formalise Classifier Two-Sample Tests, training a binary classifier to distinguish the two samples and testing its held-out accuracy against chance; their asymptotic power analysis is classifier-agnostic, which motivates the logistic-regression implementation used here in place of their neural-network and k-nearest-neighbour variants, a simpler choice consistent with this study's resource-constrained framing.

**Feature-wise divergence methods.** Ditzler & Polikar (2011) propose HDDDM, which computes the Hellinger distance between histograms of a reference and current batch, adaptively resetting the reference when a significant change in divergence is detected. Their multivariate formulation (Eq. 1) explicitly defines the joint statistic as the average of per-feature Hellinger distances, a construction that, as this study demonstrates, has direct consequences for what kinds of shift the method can detect.

**Dataset-shift detection in the presence of resource limits.** Rabanser, Günnemann & Lipton (2019) conduct the most comprehensive prior empirical study of multivariate shift detection, comparing dimensionality-reduction strategies (including black-box shift detection via a pretrained classifier's softmax outputs) combined with two-sample testing across a wide range of shift types and sample sizes. Their central empirical finding, that multiple corrected univariate tests perform comparably to multivariate kernel tests, is the direct precedent this study extends; critically, none of their constructed shifts hold marginal distributions fixed, so their finding characterises detection difficulty under marginal shift specifically, not under a shift that only a multivariate statistic can see.

**Label-shift detection via black-box predictors.** Lipton, Wang & Smola (2018) introduce Black Box Shift Estimation, using a black-box classifier's confusion matrix to detect and correct for label shift under the anti-causal assumption. This mechanism, and the closely related dimensionality-reduction-then-two-sample-test framework of Rabanser et al., motivate a natural extension of the present work to embedding-space shift detection in text and image domains; we discuss this direction in Section 6 but do not pursue it here, for reasons of scope.

**Dependency structure and copulas.** [CITE: sklar_1959] establishes that any multivariate joint distribution decomposes into its marginal distributions and a copula function describing the dependency between them, and that the two can be manipulated independently. Empirical copula estimation via rank transformation of the marginals is standard nonparametric methodology [CITE: nelsen_copulas]; this study's correlation-only injection mechanism is a direct, if informal, application of that decomposition.

---

## 3. Methodology

### 3.1 Datasets and feature pairs

This study reuses the three datasets from the companion paper (UCI Credit Card Default, IBM Telco Customer Churn, AI4I 2020 Predictive Maintenance), extending each dataset's single drifted feature to a pair of features drifted jointly:

**Table 1: Dataset and feature pair summary**

| Dataset | Original single feature | Paired feature added | Baseline correlation |
|---|---|---|---|
| Credit | AGE | BILL_AMT1 | r = 0.06 |
| Telco | Monthly Charges | Total Charges | r = 0.66 |
| Maintenance | Tool wear [min] | Air temperature [K] | r = 0.02 |

The telco pair's substantially higher baseline correlation (Total Charges is approximately cumulative billing over tenure) is addressed directly in the correlation-shift design (Section 3.3).

### 3.2 Marginal-shift injection and the extrapolation constraint

Drift is injected as an additive shift of magnitude 2σ, where σ is the feature's own standard deviation, following the variance-based synthetic perturbation methodology of Rabanser et al. (2019), the same convention used in the companion study. Applied naively, however, a fixed 2σ shift does not translate uniformly across features: for a feature whose observed range already extends several standard deviations beyond its mean (as with Credit's right-skewed billing amounts), a 2σ shift stays within previously observed territory; for a feature with a comparatively narrow or bounded observed range (as with Telco's Monthly Charges, whose maximum observed value sits only 1.78 standard deviations above its mean), the same nominal shift pushes the majority of samples beyond anything the model was ever trained on.

We verified this directly: an uncapped 2σ shift on Telco's Monthly Charges pushed 60.1% of post-drift values beyond the feature's observed maximum, and on Maintenance's Tool wear and Air temperature, 43.3% and 42.6% respectively. This is a methodologically consequential problem, not a cosmetic one: it means the model's post-drift accuracy reflects behaviour on inputs it has no basis for handling, rather than the effect of a plausible distribution shift, and produces confounded, dataset-dependent artefacts in the accuracy-degradation results (Section 4.4) that do not reflect a real property of the drift itself.

The correction applies the same nominal 2σ shift with smooth exponential saturation toward a ceiling set at the 99.5th percentile of the reference distribution, so that a point far from the ceiling moves by close to the full nominal shift while a point already near the ceiling moves proportionally less, asymptotically approaching but never exceeding it. This keeps the injected shift genuinely extreme, still an easily detectable distributional change, while remaining within territory the model has some basis for handling. Verified empirically, the corrected mechanism produces 0% of shifted values beyond the observed maximum across all three datasets, all six drifted features, and all three temporal drift profiles (sudden, gradual, recurring).

An earlier version of this correction, which clipped shifted values to a hard ceiling rather than saturating smoothly toward it, produced an artefact of its own: for Maintenance's features, over 45% of post-drift samples collapsed onto the exact same clipped value, an artificial concentration no real sensor saturation produces. This inflated the measured accuracy degradation for Maintenance from a smooth-saturation value of 4.78 percentage points to 9.63 percentage points under hard clipping. We report the smooth-saturation result throughout.

### 3.3 Correlation-only shift injection

To isolate detection difficulty attributable specifically to a shift in dependency structure, independent of any marginal signature, we construct a second drift condition using rank-based permutation. Within an active drift window, a controllable fraction of samples have one feature's value replaced by the value that would place it in matching rank order with the other feature (or reversed rank order, for a decorrelating shift), the substituted values drawn only from among the values already present in that window. Because this is a permutation of observed values rather than an alteration of them, the marginal distribution of the permuted feature is unchanged up to the sampling variance of which rows were selected; we verified this directly (Kolmogorov-Smirnov p = 1.0 comparing pre- and post-shift marginals under full permutation, and confirmed the post-shift values form an identical multiset to the pre-shift values).

The shift direction is determined automatically from each feature pair's reference-period baseline correlation, always moving away from that baseline rather than always toward positive correlation: Credit and Maintenance, whose pairs begin near-independent, shift toward strong positive correlation; Telco, whose pair begins at r = 0.66, shifts toward decorrelation instead. This rule is fixed before any detection results are computed, from reference-period data alone, and is not tuned to produce a more favourable outcome for either detector class.

### 3.4 Detectors

Six detectors are evaluated: the three from the companion study (KS test, PSI, Wasserstein distance), each extended to the multivariate feature pairs by testing each feature independently with a Bonferroni correction across the two tests (for KS) or by taking the maximum divergence across features (for PSI and Wasserstein, which are not p-values and have no direct analogue to Bonferroni correction); and three genuinely multivariate detectors, MMD, C2ST, and HDDDM, each implemented and verified directly against its originating paper rather than against any prior implementation.

MMD's exact permutation test is quadratic in sample size; at the largest window size (1,000) and full reference size in the constraint grid, an uncapped implementation measured 11.5 seconds per detection check, which would render the full constraint grid computationally infeasible on the hardware available for this study (Section 3.6). We cap the sample size fed to the kernel computation at 300 and reduce permutations from 100 to 50, a resource-constraint decision consistent with this study's own framing and explicitly anticipated by Gretton et al. (2012, Section 8.3), who note that the quadratic-cost test does not scale and recommend a bounded-sample alternative for large data. This reduces the measured worst-case cost to approximately 0.4 seconds per check.

### 3.5 Statistical methodology

Detection delay and missed-detection rate are compared across detectors and constraint levels using the same nonparametric methodology as the companion study: Kruskal-Wallis tests for k-group comparisons, with epsilon-squared (ε² = H / (n − 1)) reported as effect size; pairwise Mann-Whitney U tests with Bonferroni correction and rank-biserial effect size for two-group comparisons; Spearman's ρ for monotonic constraint effects; and a chi-square test of independence for the binary missed-detection outcome, which is not appropriately modelled by a rank-based test. All headline test p-values across both drift conditions (thirteen tests) are subject to a single study-wide Benjamini-Hochberg false discovery rate correction, following the companion study's own revision in response to reviewer feedback on within-family-only correction.

### 3.6 Implementation and hardware

All experiments run on the same hardware benchmarked in the companion study: an Intel Core i3-1125G4 (4 physical / 8 logical cores, 2.00GHz), 4GB RAM, Windows 11 Home, Python 3.13, scikit-learn 1.8.0. The marginal-shift grid (16,200 runs: 3 datasets × 6 detectors × 3 drift types × 5 monitoring-frequency levels × 4 window-size levels × 3 reference-size levels × 5 repetitions) and the correlation-shift grid (3,240 runs, a deliberately reduced constraint sweep given its role as a secondary condition, following the same don't-multiply-the-full-factorial principle used elsewhere in this study's design) completed in approximately 40 hours of combined runtime.

Each of the five repetitions per configuration uses an independently reseeded stratified train/reference/stream split and an independently retrained model, ensuring genuine statistical replication; an earlier version of the pipeline, inherited unmodified from the companion study's own convention, seeded the data split and model training identically regardless of repetition, producing five byte-identical duplicate measurements per configuration rather than five independent trials. This was corrected and the full grid rerun before any result in this paper was finalised. [CITE: paper1_self] itself likely carries the same limitation, unflagged in its own review process, since it was discovered only by explicitly checking for duplicate rows across repetitions rather than by any methodology audit.

---

## 4. Results

### 4.1 Marginal-shift grid: configuration dominance replicates

Across the full 16,200-run marginal-shift grid, both constraint dimensions that were significant in the companion study remain significant here, and remain large. Window size shows a Spearman correlation of ρ = 0.583 with detection delay (Kruskal-Wallis H = 5,606.6, p < 10⁻³⁰⁰), and monitoring frequency shows ρ = 0.492 (H = 4,017.4, p < 10⁻³⁰⁰). Reference size, as in the companion study, shows no significant monotonic effect (ρ = −0.001, p = 0.93), consistent with its previously established threshold-rather-than-monotonic behaviour. Figure 1 shows the window-size relationship directly: median detection delay rises from 50 samples at window size 50 to 494 samples at window size 1,000, an approximately tenfold range from configuration alone. Figure 2 replicates the companion study's drift-type finding at the new six-detector scale: gradual drift's median delay (300 samples) is three times that of sudden or recurring drift (100 samples each; Kruskal-Wallis H = 1,342.96, p < 10⁻²⁹⁰).

Detector identity is also statistically significant across the six-detector comparison (Kruskal-Wallis H = 242.5, p = 2.3 × 10⁻⁵⁰), which is unsurprising given a sample size of 16,200; the effect size is not. Epsilon-squared for detector identity is 0.015, compared to 0.250 for monitoring frequency and 0.349 for window size (Figure 3, marginal-shift bars), meaning detector identity accounts for roughly one twentieth as much of the variance in detection delay as window size alone. Collapsing the six detectors into the univariate (KS, PSI, Wasserstein) and multivariate (MMD, C2ST, HDDDM) classes sharpens this further: the two classes' median delays differ (100 versus 194 samples, Mann-Whitney U = 29,103,125.5, p = 2.1 × 10⁻²⁶), but the effect size is negligible by conventional interpretation (rank-biserial r = 0.097, below the 0.10 threshold typically used to distinguish a negligible effect from a small one). Most tellingly, detector class has no significant effect on whether drift is detected at all: a chi-square test of missed-detection rate against detector class is not significant (χ² = 1.56, p = 0.21). Under marginal shift, extending the detector set from three univariate statistics to six detectors spanning both classes does not change the companion study's central claim.

### 4.2 Correlation-shift grid: the hierarchy inverts, but not along the class boundary

The correlation-only condition produces a markedly different picture. Detector identity's epsilon-squared rises to 0.236, now exceeding both constraint effects measured on the same grid (monitoring frequency ε² = 0.137, window size ε² = 0.096; Figure 3, correlation-shift bars). Whether drift is detected at all becomes strongly dependent on detector class (χ² = 86.7, p = 1.3 × 10⁻²⁰), a result that survives the study-wide false discovery rate correction applied across all thirteen headline tests (Section 3.5) without any change in significance.

Table 2 reports the per-detector miss rate on the correlation-only condition directly, and it is the clearest single result in this study.

**Table 2: Miss rate by detector, correlation-only shift condition (n = 540 per detector)**

| Detector | Miss rate |
|---|---|
| MMD | 11.9% |
| HDDDM | 53.3% |
| KS | 57.6% |
| PSI | 57.6% |
| Wasserstein | 60.6% |
| C2ST | 61.3% |

Figure 4 presents the same result as an emphasis chart, and Figure 5 shows it alongside the marginal-shift condition as a paired comparison: all six detectors cluster near zero miss rate under marginal shift, then spread apart sharply the moment the drift only lives in the joint structure between features.

The result that matters is not the two-class split. HDDDM, formally a multivariate detector by design, misses 53.3% of correlation-only shifts, closer to the univariate group's average (58.6%) than to MMD's 11.9%. This is not a surprising empirical weakness; it is a direct, predictable consequence of HDDDM's own published formulation. Ditzler & Polikar (2011, Eq. 1) define the multivariate Hellinger distance as the average of the per-feature Hellinger distances, a quantity computed entirely from each feature's own marginal histogram. A statistic built this way cannot detect a shift that leaves every marginal histogram unchanged, regardless of how the two features' values are paired, no matter how much data it is given. HDDDM's near-univariate performance on Table 2 is this limitation confirmed empirically, not discovered by it.

C2ST performs worse still (61.3% miss rate), despite training its classifier directly on the two-dimensional joint feature vector and therefore having, in principle, direct access to exactly the dependency information the correlation-shift condition manipulates. We attribute this to a sample-efficiency limitation rather than a structural one: Lopez-Paz & Oquab's (2017) construction splits the available window into training and held-out test halves for the classifier, which under the constraint grid's smaller window sizes leaves too few samples to reliably learn a decision boundary sensitive to a subtle correlation-only signal, as distinct from an obvious marginal shift. This interpretation is consistent with C2ST's comparatively better standing on the marginal-shift grid (Section 4.1), where the signal it must learn is larger.

MMD alone approaches the theoretical ideal for this condition: its kernel-based statistic operates directly on the joint sample without collapsing to per-feature summaries, and its 11.9% miss rate, while not zero, is close to five times better than any other detector tested.

A caveat on detection delay under this condition: among detected cases only, MMD's median delay (457 samples) is longer than the univariate group's (144 samples), which read naively suggests MMD is slower. It is not; this is a survivorship artefact of comparing delay only among cases a detector actually caught. The univariate detectors miss 55 to 61% of correlation-only shifts, so the cases they do catch are disproportionately the easiest, most marginal-adjacent instances, fast to detect almost by definition. MMD catches 88% of cases, including the genuinely subtle ones that take longer to accumulate sufficient evidence. This is also a second illustration of why the univariate/multivariate class split is the wrong unit of analysis here: pooling MMD with C2ST and HDDDM into a single multivariate-class median (400 samples) obscures rather than clarifies, since the three detectors' individual medians (MMD 457, C2ST 624, HDDDM 50) disagree with each other more than MMD's disagrees with the univariate group's. HDDDM's low median delay is itself consistent with Section 4.2's account of its limitation: when it does detect, it is catching the same easy, marginal-adjacent cases the univariate detectors catch, for the same reason. The correct reading of the correlation-shift results is Table 2's miss rate, not the delay comparison in isolation.

### 4.3 Constraint effects persist within the correlation-shift condition

Configuration still matters under correlation-only shift, it is simply no longer the dominant factor. Both monitoring frequency (ρ = 0.37, p = 2.5 × 10⁻⁵³) and window size (ρ = 0.31, p = 4.4 × 10⁻³⁷) retain significant, moderate monotonic effects on detection delay. Reference size, which showed no significant effect under marginal shift, becomes significant here (ρ = −0.06, p = 0.017), a small but real effect: with more reference data available, MMD's kernel estimate of the reference distribution's joint structure sharpens, which plausibly matters more for a shift that is only visible in that structure than for one visible in any single feature's histogram.

### 4.4 Accuracy degradation: correlation-only shift is easy to detect and largely benign

Figure 6 reports mean accuracy degradation by dataset and condition, computed against the matched, same-index-range counterfactual baseline described in Section 3.2. Two patterns are worth separating. First, across both conditions, Credit and Telco show near-zero mean degradation (Credit: −0.10 percentage points in both conditions; Telco: −0.03pp marginal, 0.37pp correlation), while Maintenance shows substantial degradation under marginal shift (4.78pp) that is much smaller, though still the largest of the three datasets, under correlation-only shift (0.90pp). Second, the correlation-only condition is, across all three datasets, either comparably benign to or measurably less damaging than the marginal-shift condition on the same dataset.

This combination, easily detectable by the right detector yet largely harmless to downstream accuracy, is consistent with the distinction between statistically significant and practically malignant drift that Rabanser et al. (2019, Section 3.4) raise as a direction requiring further characterisation. The correlation-only condition constructed in this study is, in that framing, a clean example of a shift that is straightforward to flag and largely benign to ignore, provided the detector deployed is capable of flagging it at all.

Telco's small negative degradation values (accuracy very slightly higher after drift than the matched baseline, in both conditions) warrant a brief note given their consistency across nearly every configuration tested. We traced this directly to the model's predictions rather than accepting it as an artefact: of 1,407 post-drift predictions inspected under a representative marginal-shift configuration, 78 flipped from incorrect to correct and 77 flipped from correct to incorrect, a difference indistinguishable from chance. The small, consistently negative mean reported in Figure 6 reflects a genuine but negligible property of this dataset and model combination, not a residual methodological defect; we verified directly that it does not originate from the extrapolation-saturation correction described in Section 3.2; among points affected by that correction specifically, accuracy moves in the expected degrading direction.

---

## 5. Discussion

### 5.1 A conditional priority hierarchy, not a universal one

The companion study's priority hierarchy, configuration first, drift type second, algorithm third, was stated without qualification. Section 4 shows that qualification is needed. The hierarchy holds precisely as stated whenever the drift a monitoring system needs to catch has a marginal signature: a change visible in at least one feature's own distribution, which describes the overwhelming majority of drift scenarios discussed in the wider literature, including every scenario in Rabanser et al. (2019). It does not hold, and in fact inverts, for the narrower but real case of a shift confined to the dependency structure between features, where the choice of detector becomes the dominant factor and configuration, while still significant, recedes to second place (Section 4.3).

For a practitioner, the practical implication is a two-step decision rather than a single ranking. First, ask whether the features being monitored are likely to drift jointly in a way that could preserve their individual marginal distributions, correlated sensor pairs, billing structures with a fixed relationship to tenure, or any feature pair whose relationship encodes information the individual features do not. If not, the companion study's original guidance applies unchanged: invest in configuration before detector selection. If so, detector selection now carries real weight, and Section 5.2 specifies which property of a detector actually matters.

### 5.2 Extending Rabanser et al.: where "univariate is enough" stops being true

Rabanser et al. (2019) report that Bonferroni-corrected univariate testing performs comparably to multivariate kernel testing, a finding this study's marginal-shift grid replicates independently on tabular data rather than the image domain their study used (detector class shows no significant effect on missed detection, p = 0.21; Section 4.1). The present study's contribution is identifying precisely where that equivalence breaks down: not at some general boundary of "harder" or "higher-dimensional" shift, but specifically when the shift is constructed to leave marginal distributions provably untouched. No shift in Rabanser et al.'s benchmark, nor in this study's own marginal-shift condition, tests that boundary, since every one of them moves at least one feature's own distribution and is therefore visible, at least in principle, to a well-corrected univariate test. The correlation-only condition constructed here (Section 3.3) is, to our knowledge, the first drift-detection benchmark condition built specifically so that a univariate approach cannot succeed by construction, and it is exactly there that detector choice becomes decisive.

### 5.3 What "multivariate" should mean in this context

Section 4.2's most consequential finding is that the univariate/multivariate label, taken at face value, is not a reliable predictor of whether a detector can see correlation-only drift. HDDDM is described in its own originating paper as a multivariate method, yet performs within a few percentage points of the univariate group specifically because its own published formula (Ditzler & Polikar, 2011, Eq. 1) reduces the multivariate case to an average over per-feature marginal statistics. A practitioner selecting a detector on the basis of a multivariate label alone, without examining whether its statistic's construction actually uses cross-feature information, risks deploying a detector that cannot do the one thing they selected it to do. The more useful question, and the one this study's results support asking, is not "is this detector multivariate" but "does this detector's test statistic depend on the joint arrangement of feature values, or only on their marginal distributions computed independently."

C2ST's underperformance is a different lesson. Its statistic does depend on the joint arrangement, by construction, yet it underperforms even the univariate group under these constraints. This is a resource-constraint finding in the same spirit as the companion study's own central theme: a theoretically capable method can still lose to a cheaper, theoretically weaker one once sample size is bounded, here by C2ST's own train/test-split requirement interacting with the smaller window sizes in the constraint grid. We would expect C2ST's relative standing to improve at larger window sizes, a prediction the present constraint grid's design allows but which we do not separately isolate here.

### 5.4 Limitations

Several limitations bound the claims made in this paper. The Wasserstein detector's threshold, inherited unchanged from the companion study, was calibrated empirically against pre-drift data rather than derived from a literature convention, unlike the KS test's significance level or PSI's industry-standard cutoffs (Section 3.4); this does not affect the underlying distance computations reported here but should be treated as a documented, unresolved calibration choice rather than a validated one. The multivariate detectors are evaluated on two-feature pairs only; whether the same mechanistic distinctions (HDDDM's marginal-averaging limitation, C2ST's sample-efficiency ceiling) hold at higher dimensionality is not established by this study. A single model architecture, Random Forest, is used throughout; the companion study's model-robustness comparison (Random Forest, Logistic Regression, LightGBM) is not repeated here. C2ST is implemented with a logistic-regression classifier rather than the neural-network or k-nearest-neighbour variants Lopez-Paz & Oquab (2017) benchmark; Theorem 1's power bound is classifier-agnostic, but the specific numbers reported here are specific to this simpler choice. Finally, both drift conditions remain synthetically injected rather than organic, a limitation this study shares with the great majority of the drift-detection literature, including Rabanser et al. (2019).

### 5.5 Practical guidance

For deployments where the drifting features are known to be independent or only weakly related in the reference period, the companion study's guidance stands: configuration first. For deployments monitoring feature pairs with a meaningful relationship, correlated sensor readings, ratios, or derived quantities, the additional cost of a kernel-based multivariate detector such as MMD is justified specifically because the alternative is not merely worse but structurally blind to an entire class of drift. That cost is not uniform: the resource-constraint correction developed in Section 3.4, capping MMD's kernel computation rather than running it uncapped, reduced its worst-case per-check cost from 11.5 seconds to approximately 0.4 seconds on the hardware used in this study, making it practical even under the tight constraint grid tested here. A multivariate label alone, absent this kind of check on what the underlying statistic actually computes, is not sufficient grounds for that investment.

---

## 6. Conclusion and Future Work

This study asked whether the companion paper's central finding, that monitoring configuration matters more than detector algorithm, survives two extensions: a move from univariate to multivariate detection, and a drift condition constructed to be invisible to any purely marginal statistic. The first extension leaves the original finding intact: across 16,200 runs spanning six detectors, three datasets, and the full constraint grid, configuration effect size exceeds detector-identity effect size by roughly twentyfold, and detector class has no measurable effect on whether drift is detected at all. The second extension overturns it, but not along the boundary the univariate/multivariate label would predict: across 3,240 correlation-only runs, detector identity becomes the dominant factor, and the decisive property is whether a detector's statistic is constructed to use cross-feature dependency, a property MMD's kernel-based construction satisfies, HDDDM's per-feature-averaged construction structurally cannot, and C2ST's classifier-based construction satisfies in principle but does not realise under these resource constraints.

The resulting guidance for practitioners is conditional rather than universal: invest in configuration first when the features being monitored are unlikely to drift jointly in a way that preserves their marginals, and invest in detector selection, specifically a detector whose statistic is verified to use cross-feature information rather than one merely labelled multivariate, when they are.

Three directions follow directly from this study's scope decisions. First, the embedding-space extension outlined but not pursued here, applying the same distinction between marginal-visible and correlation-only shift to representation-space drift in text and image domains via black-box shift detection (Lipton et al., 2018) or a pretrained-embedding two-sample-testing pipeline (Rabanser et al., 2019), would test whether the construction-not-class distinction generalises beyond tabular data. Second, extending the correlation-shift condition beyond two-feature pairs would establish whether HDDDM's marginal-averaging limitation and C2ST's sample-efficiency ceiling persist, worsen, or resolve as dimensionality grows. Third, a targeted rerun of C2ST at larger window sizes than the present constraint grid's lower range would test directly the prediction, made but not verified in Section 5.3, that its underperformance here is a resource-constraint artefact rather than a structural one.

---

## References

[CITE: paper1_self] (companion study citation, pending venue assignment)

Ditzler, G., & Polikar, R. (2011). Hellinger distance based drift detection for nonstationary environments. In *2011 IEEE Symposium on Computational Intelligence in Dynamic and Uncertain Environments (CIDUE)* (pp. 41-48). IEEE.

Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13(1), 723-773.

Lipton, Z., Wang, Y. X., & Smola, A. (2018). Detecting and correcting for label shift with black box predictors. In *International Conference on Machine Learning* (pp. 3122-3130). PMLR.

Lopez-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. In *International Conference on Learning Representations*.

[CITE: nelsen_copulas] (copula estimation methodology reference, pending verification)

Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *Advances in Neural Information Processing Systems*, 32.

[CITE: sklar_1959] (Sklar's theorem, pending verification)
