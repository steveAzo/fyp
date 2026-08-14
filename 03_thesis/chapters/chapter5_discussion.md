# Chapter 5: Discussion

## 5.1 Overview

This chapter situates the empirical findings reported in Chapter 4 within the broader landscape of the drift detection literature, provides mechanistic explanations for the patterns observed, and draws out their practical implications for resource-constrained ML deployment. The discussion is organised around four themes that emerge from the aggregated findings: (1) the detection trade-off between delay, reliability, and computational cost; (2) why gradual drift remains the unresolved hard case; (3) the primacy of the monitoring regime as a design variable; and (4) the relationship between classifier architecture and drift robustness. The chapter closes with a critical appraisal of the study's limitations and a set of evidence-based recommendations for practitioners.

---

## 5.2 Detector Performance in Context (RQ1)

### 5.2.1 The Wasserstein-KS Equivalence and What It Implies

The most structurally significant finding from RQ1 is not that any one detector is best, but that the KS test and Wasserstein distance are statistically indistinguishable in detection delay under the conditions tested (p = 0.186 after Bonferroni correction, r = -0.029). Both operate on the same fundamental principle, comparing the empirical distribution of a reference window against a stream window, but via different mathematical pathways. The KS test measures the maximum absolute difference between the two empirical cumulative distribution functions, whereas the Wasserstein distance (equivalently, the Earth Mover's Distance) measures the minimum cost of transforming one distribution into the other (Villani, 2009).

The equivalence in delay is not trivial. Theoretically, the Wasserstein distance is more sensitive to the magnitude and geometry of distributional shift because it accounts for the full transport cost rather than the single worst-case deviation captured by the KS statistic (Rabanser et al., 2019). In practice, at the drift magnitudes and sample sizes used here, this additional sensitivity does not translate into an earlier alarm. The two detectors draw from the same information structure, a paired reference and stream window with the same sample sizes, and the drift signals introduced were large enough to be detectable by either test long before the transport-cost advantage of Wasserstein becomes material. This suggests the marginal value of the Wasserstein distance's geometric sensitivity becomes relevant primarily at the boundary between detectable and undetectable drift, a regime that future work could explore by systematically varying drift magnitude.

The practical implication is liberating rather than restrictive: practitioners choosing between KS and Wasserstein for tabular data do not trade delay for something else. The choice can therefore be made on secondary criteria: computation budget, interpretability, and false positive tolerance, rather than on detection latency.

### 5.2.2 PSI Under Scrutiny: Empirical Confirmation of Theoretical Concerns

PSI's significantly longer detection delays (|r| ≈ 0.15–0.18 versus both KS and Wasserstein) confirm, in an empirical and statistically controlled setting, the concerns raised analytically by Taplin and Hunt (2019). Their central critique of PSI is that it measures any change in the feature distribution, not specifically those changes that degrade model utility, and that its reliance on discretisation introduces information loss that makes it less sensitive to the precise location and magnitude of shift. The present findings add a concrete quantification: under representative tabular drift conditions, PSI's expected delay is approximately 35% longer than Wasserstein's (291.73 versus 205.52 windows) and approximately 35% longer than KS's (291.73 versus 216.28 windows).

The false positive pattern extends this picture. PSI generated a mean of 10.92 false positives per run versus 5.56 for KS, despite also detecting drift later. This apparent contradiction (both slower and noisier) is explained by the binning mechanism. PSI uses equal-frequency bins defined on the reference distribution. When the stream distribution differs from the reference, it can produce unusual bin frequencies in any number of bins regardless of whether true drift is occurring; noise in stream sampling produces greater PSI volatility than it does in the rank-based KS statistic or the transport-based Wasserstein. Taplin and Hunt (2019) identified the bin-sensitivity issue (PSI values depend on the number and placement of bins and are undefined when a bin is empty) as a principled limitation; the present results provide empirical evidence that this sensitivity translates into worse operating characteristics even under well-controlled conditions with a fixed 10-bin design.

One respect in which PSI does not confirm the Taplin and Hunt (2019) critique, however, is in missed detections. PSI's missed detection rate was 0.56%, lower than KS's 1.11%, suggesting that PSI's broader sensitivity to any distributional change does catch some events that the more conservative KS threshold misses. This is consistent with PSI's position as an industry standard in banking model monitoring: in a domain where regulators require evidence of population stability (Taplin & Hunt, 2019), false negatives (undetected drift) carry greater institutional risk than false positives (unnecessary investigation). Under that objective function, PSI's operating point may be appropriate. Under a different objective function, minimising unnecessary retraining triggers in a constrained IoT deployment, KS or Wasserstein is the better choice.

### 5.2.3 Computation as a First-Class Constraint

The three-way cost structure revealed by this study (Table 4.1) deserves explicit articulation. PSI is cheapest computationally (77.53 ms), Wasserstein is intermediate (114.44 ms), and KS is most expensive (378.51 ms). The reasons are algorithmic. PSI's computation is O(B) where B is the number of bins (here fixed at 10), making it essentially O(1) with respect to sample size given a pre-computed bin structure. Wasserstein requires sorting both distributions and computing the integral of the absolute difference of their CDFs, giving O(n log n) complexity. KS similarly requires sorting, plus computing the supremum of the CDF difference. In practice, the KS test's overhead reflects both sorting and repeated scipy calls across all monitored features, while PSI's bin-assignment is vectorised and sample-size-independent once bins are fixed.

The significance of this for resource-constrained systems is context-dependent. Yang and Shami (2021) explicitly demonstrate that IoT systems cannot sustain detection methods that scale poorly with stream length; in their OASW-LightGBM framework for IoT streams, compute efficiency is a design requirement, not a secondary concern. Preuveneers et al. (2020) similarly identify computational overhead as a limiting factor in deploying drift monitoring at the edge. Under these constraints, PSI's 5× compute advantage over KS is material: enough to run multiple features' checks within a single monitoring cycle, whereas KS might require prioritising which features to monitor. The design choice, then, is not simply "which detector is best" but "which detector's operating point is compatible with the deployment's resource budget while satisfying acceptable delay and false positive thresholds."

---

## 5.3 Why Gradual Drift is the Hard Problem (RQ2)

### 5.3.1 The Detection Mechanism Under Gradual Shift

The finding that gradual drift produces significantly longer detection delays than both sudden and recurring drift (|r| ≈ 0.30, medium effect) is the largest effect in the study by effect size and is theoretically expected. The mechanism operates through the interaction between the drift rate and the detector's window-based comparison.

All three detectors compare a fixed reference window against a recent stream window. When drift is sudden, the stream window is drawn from a fully shifted distribution from the moment of onset; the distributional gap between reference and stream is at its maximum immediately. When drift is gradual, the stream window is a mixture: early samples after onset are nearly identical to the reference, and only late samples approach the full magnitude of the shift. For a window of size W beginning at time t after the gradual onset, the expected distribution shift detectable by the window is proportional to t/T, where T is the total transition length. At t = 0, the detectable shift is zero; at t = T, it reaches full magnitude. This means gradual drift can only be reliably detected once a sufficient number of shifted observations have accumulated in the stream window, creating a structural lag that is absent for sudden drift.

This mechanism is well-understood theoretically. Lu et al. (2019) identify gradual drift as the most challenging pattern for window-based detectors precisely because of this accumulation requirement. Suárez-Cetrulo et al. (2022), in their survey of methods for recurring concept drift, note that detectors calibrated on sudden drift benchmarks routinely underperform on gradual drift in empirical studies. The present findings provide a controlled quantification of this gap under varying resource constraints, adding to the empirical evidence base.

The practical consequence is significant: a system configured to detect sudden drift adequately may systematically fail to detect gradual drift within an operationally acceptable timeframe, even at liberal monitoring frequencies. Monitoring regime optimisation alone (increasing frequency or reducing window size) cannot fully compensate, because the fundamental limitation is the rate at which shifted observations accumulate in the window, a property of the drift itself, not of the monitoring configuration.

### 5.3.2 Recurring Drift: the Null Result and What It Teaches

The non-significant difference between sudden and recurring drift (p = 1.000, r = 0.012) is as informative as the significant findings. Recurring drift, in this study's design, applies a full-magnitude shift during its active cycle phase, the same shift as sudden drift, applied periodically. When the detector checks during an active phase, it encounters the same distributional gap it would encounter under sudden drift. The detectability per-alarm is therefore identical; what differs is that the stream cycles between drifted and non-drifted states, generating multiple detection opportunities (and, correspondingly, multiple opportunities for missed detections between cycles or during the return to baseline).

The finding contradicts a simplistic reading of Suárez-Cetrulo et al. (2022), who argue that recurring drift introduces special challenges due to the need to maintain memory of past concepts and the instability-plasticity trade-off in adaptive models. Those challenges are real but operate at the model adaptation layer: the question of whether a model that adapts to one concept should retain the previous concept for use when it recurs. At the detection layer, which is the focus of this study, recurring drift is no harder than sudden drift because the per-cycle detection problem is the same. The implication for system design is that recurring drift does not require special handling at the detection stage; the challenge is downstream in the adaptation and retraining logic.

### 5.3.3 Dataset-Level Accuracy Degradation: Ceiling Effects and Class Imbalance

The pronounced variation in accuracy degradation across datasets (Table 4.4), from near-zero in Credit Default and Telco Churn to 5.11 percentage points in Predictive Maintenance under sudden drift, is attributable to two compounding factors: proximity to the accuracy ceiling and class imbalance.

Predictive Maintenance achieves 98.05% baseline accuracy, operating near the ceiling of what is achievable on the test set. A model at this level has, by construction, memorised many near-boundary cases that contribute to the last percentage points of performance. When the feature distribution shifts, these boundary-adjacent cases are disproportionately affected: the model's confidence in the decision region near the true boundary deteriorates faster than its confidence in clearly separated instances. Credit Default (81.45% baseline) and Telco Churn (78.97% baseline) operate further from the ceiling; their models are less tightly fitted to the training distribution, and more of the classification relies on robust, widely-separated features that are resistant to moderate distributional shift.

The implication for practitioners is that models operating near peak performance on imbalanced datasets are the most fragile under drift, a counterintuitive finding, since high accuracy is typically interpreted as a sign of model quality. This is consistent with observations from medical imaging drift research (Kore et al., 2024), who found that high-performing classifiers trained on specific patient populations could exhibit sharp accuracy drops when distribution shifted due to population changes, while more moderate-baseline models were more resilient. The lesson is that drift robustness testing is most critical, not least critical, for systems achieving high baseline accuracy, especially when class imbalance is present.

---

## 5.4 The Primacy of the Monitoring Regime (RQ3)

### 5.4.1 Monitoring Frequency: The Dominant Variable

The 37-fold range in mean detection delay achieved by varying monitoring frequency and window size alone, from 22.0 windows in the most-monitored configuration to 827.6 in the least, is the study's most consequential quantitative finding. It establishes that the monitoring regime is the primary design variable governing detection latency, superseding both the choice of detector (which produces small-to-negligible effect sizes, |r| ≤ 0.18 for any pairwise comparison) and drift type (which is a fixed property of the environment, not a design choice).

This finding has not, to the authors' knowledge, been quantified in this form in the existing literature. Yang and Shami (2021) acknowledge that monitoring frequency is a key parameter in their IoT drift framework and propose adaptive window sizing as a solution, but do not systematically evaluate the delay consequences of specific frequency configurations. Preuveneers et al. (2020) discuss the cost-of-monitoring tradeoff in federated and edge settings but focus on computation overhead rather than detection latency. The present results give practitioners a concrete number: increasing monitoring frequency from 1,000 to 10 samples reduces expected detection delay by a factor of 37. No detector switch achieves anything comparable.

The missed detection threshold observed at monitoring\_frequency = 1,000 (5.56% missed for KS, 2.78% for PSI, 1.85% for Wasserstein) marks the point where the stream length provides too few check opportunities for reliable detection. This is a structural failure mode, not a calibration failure, and cannot be corrected by adjusting thresholds or choosing a different detector. It can only be corrected by increasing monitoring frequency.

### 5.4.2 Window Size: The Dilution Effect and Its Inversion for PSI

Window size's strong positive relationship with detection delay (ρ = 0.543, H = 2,381.184) reflects the dilution effect described in Section 4.4.3: a larger stream window contains more pre-drift observations at the moment drift begins, reducing the test's ability to detect a shift that has only recently commenced. This is a direct consequence of the way windowed two-sample tests work: they compare an aggregate over the window against an aggregate over the reference, so a recent sharp change is averaged across a long window and appears weaker than the same change compared across a shorter window.

The inversion for PSI at small window sizes (missed detection rate 0.74% at window\_size = 50, zero at window\_size ≥ 500) is a separate phenomenon, not dilution but statistical instability. PSI's binning mechanism requires enough samples per bin to produce reliable density estimates. With window\_size = 50 and 10 equal-frequency bins, each bin receives on average 5 samples; random fluctuations at this scale produce bin proportions that are noisy even in the absence of drift. The consequence is that PSI's detection threshold (0.25) can be masked by noise in either direction: the test may fail to cross the threshold when drift is present because the bins are so noisy that they mask the distributional shift. This is exactly the sparse-bucket failure mode identified by Taplin and Hunt (2019), now confirmed empirically under controlled conditions.

The KS test shows the opposite: missed detection at small window sizes due to low statistical power (fewer samples → less power to detect the same effect size), and zero missed detections at larger windows as power increases. This asymmetry points to a fundamental incompatibility between the two detectors' failure modes: the conditions that make KS unreliable (small windows, low power) are the conditions under which PSI is most reliable for detection, and vice versa. An ensemble approach combining KS and PSI signals, triggering on union or intersection, could exploit this complementarity, though designing the correct combination rule without inflating false positive rates is a non-trivial task.

### 5.4.3 Reference Size: A Threshold, Not a Scale

The non-monotonic effect of reference size is the study's most nuanced resource finding. The Kruskal-Wallis result confirms that group differences exist (H = 50.015, p < 0.001), but the negligible Spearman correlation (ρ = 0.012) establishes that those differences do not follow a consistent ordering: more reference data does not consistently yield shorter or longer delays.

The table of means (Table 4.8) reveals the actual structure: 100-sample references produce the longest delays and highest missed detection rates across all three detectors, while 500-sample and unlimited references are broadly comparable. The functional form is therefore a step function, not a monotone: there is a minimum threshold of approximately 500 samples below which reference quality degrades meaningfully, and above which additional reference data yields diminishing returns. This is consistent with the statistical theory of non-parametric two-sample tests: the power of the KS test and the reliability of Wasserstein estimates both stabilise once the reference distribution is characterised well enough to distinguish from the stream, and the characterisation stabilises well before the reference size reaches the order of the stream.

The implications for system design are practical and encouraging. Memory budgets in resource-constrained environments can be bounded at 500 reference samples without meaningful performance penalty, provided the reference is drawn from the pre-drift baseline period. The critical protection is against falling below 100 samples: at that level, detection delays lengthen noticeably and PSI missed detection rates rise. A minimum reference budget of 500 samples is a deployable heuristic with empirical support from this study.

### 5.4.4 Design Priority Hierarchy

The combined RQ3 findings establish a clear design priority hierarchy for practitioners configuring drift monitoring under resource constraints:

1. **Protect monitoring frequency first.** The magnitude of the monitoring frequency effect (H = 1,948.767) and the structural missed detection failure at monitoring\_frequency = 1,000 make this the most consequential parameter. If resources must be conserved, sacrifice window size or reference size before reducing monitoring frequency below 100 samples.

2. **Window size is the secondary lever.** A smaller window detects drift faster (at the cost of noisier PSI estimates); a larger window is more stable statistically but dilutes recent shift signals. A window of 100–500 samples represents a reasonable operating range for the drift types and magnitudes studied here.

3. **Reference size has a floor, not a scale.** Maintain at least 500 reference samples. Once above this threshold, allocating further memory to the reference set has marginal benefit.

This hierarchy has not, to this author's knowledge, been stated empirically in the literature. Yang and Shami (2021) and Preuveneers et al. (2020) identify these parameters as important but do not rank their relative impact. The present study provides that ranking.

---

## 5.5 Model Robustness Under Drift (RQ4)

### 5.5.1 Why Random Forest Is More Robust

The finding that Random Forest suffers significantly less accuracy degradation under drift than LightGBM (r = -0.383, p < .001, medium effect) and significantly less than Logistic Regression (r = -0.185, p = .025, small effect) is consistent with, rather than contradictory to, the established understanding of ensemble variance reduction. Breiman (2001) demonstrated that Random Forest's averaging of many deep trees, each trained on a bootstrap sample and a randomly restricted feature subset at every split, produces an ensemble with lower variance than any individual constituent tree. The present results suggest that the same decorrelation mechanism that reduces variance under a fixed training distribution also confers a degree of robustness when the input distribution shifts at deployment time.

The mechanism operates through decorrelation rather than regularisation. Because each tree in the Random Forest is grown from a different bootstrap sample and considers only a random subset of features at each split, no two trees rely on identical decision boundaries. When a given feature's distribution shifts, only the subset of trees whose splits depend heavily on that feature are strongly affected; the remaining trees, having partitioned the input space using different features or different thresholds, continue to classify displaced instances correctly. The ensemble average absorbs the shift-induced error in a subset of trees rather than propagating it uniformly across the whole ensemble.

LightGBM's boosting procedure works differently, and this difference is the more plausible source of its greater fragility. Each tree is fit sequentially to the residual error of the ensemble built so far, and its leaf-wise growth strategy (Ke et al., 2017) greedily pursues the split that yields the largest reduction in training loss at each step. This concentrates the ensemble's decision capacity on the small set of features and thresholds that were most informative on the reference training distribution. Because later trees correct the residual of earlier ones, a distributional shift that degrades the earliest trees' predictions is not independently absorbed by later trees: it is instead propagated as a mis-estimated residual and compounded across boosting rounds. Regularisation (min\_data\_in\_leaf, lambda\_l1, lambda\_l2, all at their default non-zero values here) constrains overfitting to sparse leaves, but it does not decorrelate the trees from one another, so it does not reproduce the ensemble-averaging protection that Random Forest exhibits.

This reframes, rather than confirms, the pattern reported by Kore et al. (2024) in medical imaging drift, where higher-expressivity models showed sharper performance drops under distribution shift. Expressivity alone is not the determining factor here: Random Forest is at least as expressive as LightGBM (both are non-parametric tree ensembles capable of fitting complex, non-linear boundaries), yet it is the more robust of the two. What distinguishes them is the *structure* of the ensemble, bagged and decorrelated versus boosted and sequentially dependent, not the raw capacity of the model class.

### 5.5.2 Random Forest as the Most Resilient Choice

The non-significant difference between LightGBM and Logistic Regression (p = 0.160, |r| = 0.136) means neither can be declared superior in drift robustness within this study's power. Random Forest, by contrast, is significantly more robust than both. Mean baseline accuracy is essentially indistinguishable across the three models on the datasets tested (Random Forest 86.15%, Logistic Regression 86.23%, LightGBM 86.36%), so Random Forest's robustness advantage is not purchased at a meaningful cost in baseline performance. Random Forest therefore represents a Pareto-favourable choice for drift-prone deployments: baseline accuracy on par with the alternatives, and a substantial reduction in the accuracy lost when drift occurs. For deployments where retraining is expensive or infrequent, a common condition in edge and resource-constrained settings (Yang & Shami, 2021; Preuveneers et al., 2020), Random Forest's combination of competitive baseline accuracy and superior drift tolerance is the most favourable operating profile among those tested.

### 5.5.3 Dataset Sensitivity: When Robustness Findings Generalise

Decomposing the model comparison by dataset shows that Random Forest's advantage over LightGBM is not an artefact of a single dataset, but its magnitude does vary considerably. On Predictive Maintenance, the dataset closest to its accuracy ceiling (Section 5.3.3), both tree ensembles degrade substantially (Random Forest 4.00 percentage points, LightGBM 4.71 points), while Logistic Regression is markedly more stable (0.72 points). On Credit Default, degradation is small across all three models, with LightGBM the least robust (0.81 points), Random Forest intermediate (0.46 points), and Logistic Regression the most robust (0.23 points). On Telco Churn the pattern inverts sharply: Logistic Regression degrades by 7.01 percentage points, more than five times any other model-dataset combination in the study, while Random Forest is essentially unaffected (0.06 points) and LightGBM is moderate (1.02 points).

Two conclusions follow. First, Random Forest's advantage over LightGBM is consistent in direction across all three datasets, which supports the generality of the bagging-versus-boosting mechanism proposed in Section 5.5.1 rather than attributing it to one dataset's idiosyncrasies. Second, Logistic Regression's aggregate degradation is driven almost entirely by its anomalous failure on Telco Churn; on the other two datasets it is competitive with or better than both ensembles. A plausible explanation is that Logistic Regression's robustness is conditional on how well the true decision boundary approximates a linear surface in the drifted feature: where that approximation holds, a fixed linear separator is naturally insensitive to shift; where it does not, the model has no capacity to adapt its boundary shape the way an ensemble method can. Testing this explanation directly would require documenting each drifted feature's marginal relationship to the target label per run, a finer-grained analysis that this study's design does not currently support, and which is noted as a direction for future work in Section 5.7.4.

---

## 5.6 The Central Finding: A Priority Hierarchy for Constrained Deployment

The four research questions together permit a unified answer to the thesis's motivating question: **when, why, and under what constraints do lightweight drift detection approaches fail or succeed?**

**They fail primarily when the monitoring regime is too infrequent or too coarse**: specifically when monitoring intervals exceed 500 samples or window sizes exceed 1,000 samples for the drift magnitudes studied. These failures are structural: no choice of detector or threshold can compensate. The monitoring configuration is the dominant engineering decision, with a 37-fold impact on detection latency that dwarfs the effect of any other variable in the study.

**They fail secondarily when drift is gradual**, because the accumulation of shifted observations in a fixed window is inherently slow. This failure mode is a property of the drift pattern, not of the monitoring configuration, and cannot be fully engineered around. However, maintaining high monitoring frequency (checking every 10–50 samples) and using smaller windows (50–100 samples) reduces the accumulation lag and is the best available mitigation.

**They fail at the margins when the reference baseline is too small** (below 100–500 samples), when PSI bins are sparse (small windows), or when the chosen model is a sequentially-boosted ensemble whose trees are correlated by construction. These are correctable failures: maintain adequate reference size, use KS or Wasserstein when window sizes are small, and prefer bagged, decorrelated ensembles (Random Forest) over sequentially-boosted ones (LightGBM) in high-stakes drift-prone environments.

**They succeed** under frequent monitoring, moderate window sizes, adequate reference baselines, and with well-regularised classifiers. Under these conditions, all three detectors are effective, with Wasserstein offering the lowest missed detection rate, KS offering the best false positive control, and PSI offering the lowest computational cost. The choice among them is a secondary decision once the monitoring regime is correctly configured.

This priority hierarchy, **monitoring regime first, drift type awareness second, detector and model choice third**, is the core practical contribution of the study.

---

## 5.7 Limitations

### 5.7.1 Simulated Drift vs. Real-World Drift

The drift patterns in this study, namely sudden, gradual, and recurring, are clean, analytically defined, single-feature perturbations applied at a known time and magnitude. Real-world drift is typically messier: simultaneous shifts across multiple features, shifts of varying magnitude in different features, noise confounded with genuine drift, and drift that interacts with class label shifts rather than acting purely on features (Lu et al., 2019; Gemaque et al., 2020).

The controlled simulation approach is methodologically necessary for isolating the effects of each variable of interest and for obtaining statistically valid comparisons. But it introduces an idealisation that limits the direct transferability of the specific quantitative thresholds (e.g., the 500-sample reference floor, the 37-fold delay range) to operational settings with naturalistic drift. The direction of the findings, which variables matter most, and in what order, is expected to generalise; the specific magnitudes are study-specific.

### 5.7.2 Univariate Rather Than Multivariate Detection

All three detectors in this study monitor individual features sequentially and report drift if any feature triggers. This is the standard practice for tabular data monitoring (Arora et al., 2024) but does not detect multivariate drift in the interaction structure of features, a limitation directly identified by Taplin and Hunt (2019) as a known failure mode of PSI, and one that extends to KS and Wasserstein when applied univariately. Two features may each show only marginal individual drift while jointly exhibiting a large shift in their correlation structure. The present study cannot speak to whether the detectors succeed or fail under this multivariate regime.

Extending the experimental design to multivariate drift detection (using multivariate versions of the KS test, the Maximum Mean Discrepancy, or the multivariate Wasserstein distance as studied by Rabanser et al. (2019) in their "Failing Loudly" benchmark) is a natural direction for follow-on work.

### 5.7.3 Fixed Detection Thresholds

The thresholds used in this study, namely KS p-value < 0.05, PSI > 0.25, and Wasserstein > 0.25, are the conventional default values from the literature. All three detectors are sensitive to threshold choice; a more liberal KS threshold (e.g., α = 0.10) would reduce missed detections at the cost of more false positives, and a lower PSI threshold (e.g., 0.10) would detect earlier but with substantially more false alarms. The comparative findings reflect a specific point on each detector's ROC curve, not a global comparison across their full operating ranges.

A threshold-free comparison, for example comparing the area under the receiver operating characteristic curve as a function of the detection threshold, would provide a more complete characterisation of relative detector performance, and is noted as a methodological refinement for future work.

### 5.7.4 Dataset Scope and Generalisability

Three datasets, namely Credit Default (30,000 samples, financial domain), Telco Churn (7,043 samples, telecommunications domain), and Predictive Maintenance (10,000 samples, industrial IoT domain), provide breadth across domains but are insufficient to establish generalisability across the full range of ML deployment contexts. High-dimensional datasets (more than 30 features), time-series datasets, multi-class problems with more than two labels, and datasets with strong temporal auto-correlation would each present different detection challenges.

The three datasets used are well-established benchmarks in the drift and monitoring literature and were chosen specifically to represent a range of baseline difficulty and domain characteristics. They provide a more controlled basis for comparison than bespoke datasets, but replication on a wider benchmark suite, such as the STORM benchmark proposed by Lukats et al. (2024) for unsupervised drift detectors, would strengthen confidence in the findings. A related gap specific to the model robustness results (Section 5.5.3) is that the per-run drifted feature's marginal relationship to the target label was not recorded; capturing this would allow the Logistic Regression anomaly on Telco Churn to be tested directly rather than inferred post hoc.

### 5.7.5 Detection Without Adaptation

This study evaluates drift detection in isolation, measuring when and how reliably drift is flagged, but does not evaluate the downstream adaptation response (retraining, model switching, or ensemble updating) triggered by that flag. In a full operational system, what matters is not only the detection delay but the combined latency from drift onset to restored model performance, which is a function of detection delay plus retraining time plus redeployment overhead. The present findings quantify the first component only. Integrating detection with adaptation is a well-studied problem (Lu et al., 2019; Suárez-Cetrulo et al., 2022; Yang & Shami, 2021) but was outside the scope of this study's controlled design.

---

## 5.8 Implications for Practice

The findings translate into a set of evidence-based recommendations for practitioners deploying drift monitoring on tabular ML systems under resource constraints.

**Recommendation 1: Prioritise monitoring frequency over detector sophistication.** The single most impactful lever for reducing detection delay is how often the system checks for drift. A system that runs PSI every 50 samples will detect drift far earlier than a system running Wasserstein every 500 samples. Invest computational resources in frequency before investing in a more sophisticated detector algorithm.

**Recommendation 2: Use window sizes of 100–500 samples as a default operating range.** Smaller windows (50 samples) make PSI statistically unreliable (sparse bins, elevated missed detections) while larger windows (1,000 samples) dilute recent drift signals across all detectors. The 100–500 sample range avoids both failure modes for the drift patterns studied.

**Recommendation 3: Maintain a reference baseline of at least 500 samples.** Below this threshold, all three detectors show elevated delays and some show elevated missed detections (PSI in particular). The cost of storing 500 samples is low relative to the degradation incurred below this floor.

**Recommendation 4: Choose detector based on deployment priority, not detection delay.** KS and Wasserstein are equivalent in delay. If computation is the binding constraint, use PSI. If false positives are costly (spurious retraining is expensive), use KS. If missed detections are most costly (accuracy degradation is operationally severe), use Wasserstein.

**Recommendation 5: Be especially vigilant for gradual drift.** No resource configuration fully compensates for the detection lag inherent to gradual drift. Deployments in environments where gradual shift is expected (seasonal demand changes, demographic shifts, model score drift over time) should lower detection thresholds, increase monitoring frequency, or implement complementary error-based monitoring alongside distribution-based detection. Error-based approaches (monitoring model accuracy or prediction confidence, where labels are available) are sensitive to the cumulative effect of gradual drift even when distribution tests are slow to trigger (Lu et al., 2019).

**Recommendation 6: Prefer Random Forest over LightGBM in drift-prone, high-stakes deployments.** Random Forest's bagged, decorrelated ensemble structure makes it significantly more robust to drift-induced accuracy degradation than LightGBM on the datasets tested (r = 0.383), at essentially no cost in baseline accuracy. In systems where retraining is costly or infrequent, a more drift-resilient initial classifier reduces the frequency and severity of performance failures.

**Recommendation 7: Treat PSI's well-known limitations as operating constraints, not disqualifiers.** PSI remains a viable choice where computation is severely constrained, provided window sizes are kept above 100 samples (to avoid sparse-bin failures), monitoring frequency is kept below 500 samples (to avoid the structural missed detection threshold), and false positive tolerance is adequate. Using PSI below its minimum viable window size is worse than not monitoring at all, as it produces unreliable signals that could trigger unnecessary retraining.

---

## 5.9 Positioning Relative to the Literature

This study occupies a specific position in the drift detection literature that is worth stating explicitly. Most empirical drift detection comparisons evaluate performance under a single, unconstrained monitoring configuration and report which detector performs best on average (Rabanser et al., 2019; Lukats et al., 2024). This study's contribution is different: it evaluates how the interaction between detector choice and resource constraint governs performance, establishing that constraint configuration dominates detector choice as a determinant of detection latency. This question has been raised conceptually (Yang & Shami, 2021; Preuveneers et al., 2020) but not, to this author's knowledge, answered empirically with the systematic factorial design used here.

The finding that monitoring regime is the primary design variable, and that all three tested detectors are approximately equivalent in delay once the regime is fixed, has direct relevance for the growing body of work on edge and IoT drift monitoring. It suggests that research effort focused on developing more sophisticated detection algorithms, while valuable for accuracy under single-configuration conditions, may yield lower practical return than research into adaptive monitoring scheduling: algorithms that dynamically adjust monitoring frequency and window size based on detected or predicted drift risk. This is consistent with the direction pursued by Yang and Shami (2021) in their OASW framework, and suggests that the most impactful next step in this research line is the development and evaluation of adaptive monitoring schedulers, not better static detectors.

---

## References

Arora, S., Rani, R., & Saxena, N. (2024). A systematic review on detection and adaptation of concept drift in streaming data using machine learning techniques. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 14. https://doi.org/10.1002/widm.1536

Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Gemaque, R., Costa, A., Giusti, R., & Santos, E. (2020). An overview of unsupervised drift detection methods. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 10. https://doi.org/10.1002/widm.1381

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.

Kore, A., Bavil, E., Subasri, V., Abdalla, M., Fine, B., Dolatabadi, E., & Abdalla, M. (2024). Empirical data drift detection experiments on real-world medical imaging data. *Nature Communications*, 15. https://doi.org/10.1038/s41467-024-46142-w

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under Concept Drift: A Review. *IEEE Transactions on Knowledge and Data Engineering*, 31, 2346–2363. https://doi.org/10.1109/tkde.2018.2876857

Lukats, D., Zielinski, O., Hahn, A., & Stahl, F. (2024). A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams. *International Journal of Data Science and Analytics*, 19, 1–31. https://doi.org/10.1007/s41060-024-00620-y

Preuveneers, D., Tsingenopoulos, I., & Joosen, W. (2020). Resource usage and performance trade-offs for machine learning models in smart environments. *Sensors*, 20(4), 1176. https://doi.org/10.3390/s20041176

Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *Advances in Neural Information Processing Systems*, 32, 1394–1406.

Suárez-Cetrulo, A., Quintana, D., & Cervantes, A. (2022). A survey on machine learning for recurring concept drifting data streams. *Expert Systems with Applications*, 213, 118934. https://doi.org/10.1016/j.eswa.2022.118934

Taplin, R., & Hunt, C. (2019). The population accuracy index: A new measure of population stability for model monitoring. *Risks*, 7(2), 53. https://www.mdpi.com/2227-9091/7/2/53

Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

Yang, L., & Shami, A. (2021). A Lightweight Concept Drift Detection and Adaptation Framework for IoT Data Streams. *IEEE Internet of Things Magazine*, 4, 96–101. https://doi.org/10.1109/iotm.0001.2100012
