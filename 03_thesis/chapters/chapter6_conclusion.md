# Chapter 6: Conclusion

## 6.1 Overview

This thesis set out to answer a practical question that is underserved by the existing drift detection literature: when lightweight, distribution-based drift detectors are deployed under resource constraints, which constraints matter most, how much do they matter, and do the answers differ by detector, drift type, or model architecture? The study addressed this question through a systematic empirical evaluation spanning 8,100 experimental runs, crossing three drift detectors, three datasets, three drift types, five monitoring frequencies, four detection window sizes, and three reference window sizes, supplemented by 405 model comparison experiments across three machine learning architectures.

This chapter draws together the principal findings from Chapters 4 and 5, states the study's contributions explicitly, identifies the limitations of the work, and proposes directions for future research that build on its foundations.

---

## 6.2 Summary of Principal Findings

### 6.2.1 Findings on Detector Performance (RQ1)

The three distribution-based detectors, namely KS test, PSI, and Wasserstein distance, are not equivalent in performance, but the differences between them are smaller and less consistent than the differences introduced by constraint configuration.

Wasserstein distance achieves the lowest missed detection rate (0.37%) and the lowest detection delay (mean 205.52 monitoring windows). KS test achieves the lowest missed detection rate of the two statistical-test detectors (1.11%, delay 216.28 windows) but the highest computational cost (378.51 ms per evaluation period). PSI achieves the highest missed detection rate (0.56%) and the longest detection delay (291.73 windows) but the lowest computational cost (77.53 ms). All pairwise detector comparisons are statistically significant (Kruskal-Wallis H = 148.739, p < 0.001), but effect sizes are small (r ≈ 0.15), indicating that the differences, while real, are modest in absolute magnitude.

The practical implication is that detector choice should be secondary to constraint configuration in deployment design. A well-configured PSI monitor outperforms a poorly-configured Wasserstein monitor, and the magnitude of this constraint effect dwarfs the between-detector performance gap.

### 6.2.2 Findings on Drift Type Effects (RQ2)

Gradual drift is substantially and consistently harder to detect than sudden or recurring drift. The Kruskal-Wallis test across drift types yields H = 504.546 (p < 0.001), with a medium effect size (r ≈ 0.30), the largest effect in the study. Sudden and recurring drift are statistically indistinguishable from each other (Mann-Whitney p = 1.0, r ≈ 0.00), meaning detectors respond to recurring concept re-emergence with the same sensitivity as to abrupt distributional shifts.

Gradual drift's resistance to detection is explained mechanistically: the co-occurrence of old and new distributions during the transition period dilutes the distributional signal, reducing test statistic values below detection thresholds and extending detection delay. This finding replicates and extends the result reported by Barros & Santos (2018) for supervised detectors, confirming that the difficulty of gradual drift is not specific to error rate-based methods but is a fundamental property of distributional comparison approaches as well.

This result carries a clear warning for practice: a drift monitoring system calibrated to detect sudden changes may systematically fail on the gradual distributional erosion that is arguably more common in real deployment, where environments shift slowly rather than discontinuously.

### 6.2.3 Findings on Resource Constraint Effects (RQ3)

The resource constraint analysis is the core empirical contribution of this study. Across all datasets and detectors, monitoring frequency and detection window size are the dominant constraint parameters; reference window size has a threshold effect rather than a monotone relationship.

**Monitoring frequency** exhibits a strong positive correlation with detection delay (Spearman's ρ = 0.487, Kruskal-Wallis H = 1,948.767, p < 0.001). As monitoring frequency decreases (fewer checks per time period), detection delay increases substantially. A system checking every 10 samples detects drift orders of magnitude faster than one checking every 1,000 samples.

**Detection window size** exhibits the strongest correlation with detection delay in the study (Spearman's ρ = 0.543, Kruskal-Wallis H = 2,381.184, p < 0.001). Larger detection windows increase the average delay because more samples must accumulate before each check can be executed. The effect of window size on missed detection rate is the reverse: larger windows provide more statistical power, reducing missed detections, but at the cost of longer delays.

**Reference window size** shows a statistically significant but weaker relationship with detection delay (Spearman's ρ = 0.012, H = 50.015, p < 0.001), with a non-monotone pattern. The critical finding is that 100 reference samples is insufficient: the reference distribution estimate is too noisy, degrading detection accuracy, while 500 samples is functionally equivalent to an unlimited reference window. This threshold effect suggests that practitioners can maintain high detection quality without storing arbitrarily large reference datasets, provided a minimum threshold of approximately 500 samples is met.

The combined effect of constraint configuration is dramatic: the range of detection delay across constraint combinations spans from 22.0 windows (monitoring every 10 samples, window size 50) to 827.6 windows (monitoring every 1,000 samples, window size 1,000), a 37-fold range for the same detector on the same dataset. This finding establishes that how a monitoring system is configured matters more than which detector is chosen.

The priority order of decisions for resource-constrained drift monitoring deployment is:

1. **First priority, monitoring regime:** Set monitoring frequency as high as the compute budget allows; choose window sizes that balance statistical power against acceptable delay.
2. **Second priority, drift type awareness:** Design the system to be sensitive to gradual drift, not just sudden changes; calibrate thresholds accordingly.
3. **Third priority, detector and model choice:** After the above decisions are made, choose the detector and model that best fit the remaining resource envelope.

### 6.2.4 Findings on Model Robustness (RQ4)

Random Forest is significantly more robust to drift-induced performance degradation than either LightGBM or Logistic Regression. The pairwise comparison between Random Forest and LightGBM yields a medium effect size (rank-biserial r = −0.383, p < 0.001); the comparison between Random Forest and Logistic Regression yields a small but significant effect (r = −0.185, p = 0.025, Bonferroni-corrected). LightGBM and Logistic Regression do not differ significantly from each other (p = 0.160, Bonferroni-corrected).

The pattern, Random Forest clearly most robust, with LightGBM and Logistic Regression statistically indistinguishable from each other, is consistent with the interpretation that Random Forest's bagging and per-split feature randomisation decorrelate its constituent trees, so that a shift affecting one feature's decision boundary is not shared by the whole ensemble. LightGBM's sequential, residual-fitting boosting procedure has no equivalent decorrelation mechanism, which plausibly explains why it does not share Random Forest's robustness advantage (Section 5.5.1 develops this mechanism in full).

This finding complicates the common recommendation to prefer gradient boosting over random forests for tabular data. Baseline accuracy is essentially equal across all three models on the datasets tested, so Random Forest's robustness advantage under drift is not offset by any meaningful accuracy cost in stationary conditions, a consideration that should inform model selection in applications where drift is expected and retraining is expensive.

---

## 6.3 Contributions of the Study

This study makes four primary contributions to the field:

**Contribution 1: Constraint-performance surface characterisation.** The study provides the first systematic empirical characterisation of how three resource constraint parameters (monitoring frequency, detection window size, reference window size) jointly affect the detection delay and missed detection rate of three lightweight distribution-based drift detectors. The 8,100-experiment design yields a constraint-performance surface that is sufficiently dense to support quantitative comparisons and practical recommendations.

**Contribution 2: Priority hierarchy for deployment decisions.** By establishing the relative effect sizes of detector choice, drift type, and each constraint parameter, the study produces a ranked ordering of deployment decisions by their impact on monitoring effectiveness. This priority hierarchy, monitoring regime first, drift type awareness second, detector and model choice third, is directly actionable by practitioners designing monitoring systems.

**Contribution 3: Reference window threshold identification.** The study identifies a functional equivalence threshold for reference window size: 500 samples provides detection performance indistinguishable from an unlimited reference window, while 100 samples is insufficient. This threshold supports memory-efficient reference storage in resource-constrained deployments without sacrificing detection quality.

**Contribution 4: Model robustness characterisation under drift.** The study provides a multi-dataset, multi-drift-type characterisation of how Random Forest, Logistic Regression, and LightGBM differ in their robustness to drift-induced performance degradation. The medium-effect advantage of Random Forest over LightGBM (r = 0.383) is a practically meaningful finding for model selection in non-stationary deployment contexts.

---

## 6.4 Limitations

The following limitations constrain the generalisability and scope of the study's findings:

**1. Tabular binary-classification scope.** All three datasets involve tabular features and binary classification targets. The findings may not generalise to image classification, natural language processing, or multi-class problems, where the structural properties of the feature space and the nature of distributional shift are substantially different.

**2. Synthetic drift injection.** Drift is simulated through controlled statistical transformations of the original data rather than observed as a naturally occurring process. While this enables precise experimental control, it may not capture the full complexity of real-world distributional changes, which often involve correlated shifts across multiple features, gradual changes in data collection processes, and non-stationary noise.

**3. Feature-wise, univariate drift detection.** All three detectors are applied independently to each feature dimension. Multivariate drift, involving changes in feature correlations rather than marginal distributions, may be undetectable by this approach. The extension to multivariate detection methods is beyond the scope of this study.

**4. Labelled comparison only.** The model comparison uses classification accuracy as the performance metric. Calibration, fairness metrics, and operational metrics (precision, recall at specific operating points) are not evaluated; the robustness findings are specific to the accuracy dimension.

**5. Dataset size constraints.** The three datasets range from 7,043 to 30,000 instances. The findings of Barros & Santos (2018) suggest that detector behaviour may change qualitatively at the scale of millions of instances. The present study cannot characterise behaviour at that scale.

---

## 6.5 Directions for Future Research

The findings and limitations of this study suggest several productive directions for future research:

**1. Multivariate drift detection under constraints.** Extending the constraint-aware evaluation framework to multivariate distributional tests, such as the Maximum Mean Discrepancy (MMD), Least-Squares Density Difference (LSDD), or the sliced Wasserstein distance, would determine whether the constraint-performance relationships identified here generalise beyond feature-wise univariate tests.

**2. Adaptive threshold calibration.** All three detectors in this study use fixed thresholds throughout the evaluation. Adaptive thresholding strategies, where the detection threshold is calibrated dynamically based on recent false positive and false negative rates, may substantially reduce missed detections under gradual drift. Empirical comparison of fixed versus adaptive thresholding under constraint conditions is a direct extension of this work.

**3. Recurring drift with memory.** The finding that recurring drift is as detectable as sudden drift raises the question of whether detection performance can be improved further by explicitly maintaining model snapshots from prior concept periods. A comparison of window-based detectors (as used here) against memory-based adaptive ensembles (as reviewed by Gama et al., 2013) under constraint conditions would clarify whether stored memory provides detection benefits beyond what windowed comparison achieves.

**4. Real-world streaming datasets.** Extending the framework to datasets with naturally occurring drift, such as financial time series, IoT sensor streams, or healthcare monitoring data, would test whether the synthetic drift injection results transfer to organic distributional change. The evaluation challenges identified by Lukats et al. (2024) for real-world streams (absence of ground truth drift timestamps) would need to be addressed through annotation or proxy metrics.

**5. Energy and communication constraints.** The present study operationalises resource constraints as monitoring frequency, window size, and reference size, parameters relevant to CPU and memory budgets. Edge and IoT deployments also face energy consumption limits and communication bandwidth constraints. Extending the framework to include energy-aware drift detection policies is a practically important direction.

**6. Ensemble detectors.** Each detector is evaluated independently. An ensemble that combines the outputs of KS, PSI, and Wasserstein, taking a majority vote or a weighted combination, may achieve lower missed detection rates and detection delays than any single detector, particularly under gradual drift where individual test statistics are weak. The resource cost of running three detectors simultaneously could be evaluated against the performance gain.

---

## 6.6 Concluding Remarks

This study began from a simple observation: that the literature on drift detection has extensively characterised *which algorithms work* but has said little about *whether they still work when resources are constrained*. The empirical answer provided by this study is that they do, but that the configuration of the monitoring system matters far more than the choice of algorithm. A well-configured PSI monitor detects drift reliably; a poorly-configured Wasserstein monitor misses it. The 37-fold range in detection delay across constraint configurations, compared to the modest effect sizes for between-detector comparisons, makes this conclusion quantitatively concrete.

The practical guidance distilled from 8,100 experiments is straightforward: prioritise monitoring frequency and window size over detector selection; ensure at least 500 reference samples; build in sensitivity to gradual drift, not just sudden changes; and prefer Random Forest over LightGBM when the deployment environment is known to be non-stationary and retraining is costly. These are not complex recommendations. They are the kind of recommendations that can only be made after systematic empirical investigation of a parameter space that the existing literature has largely ignored.

The broader implication is that the evaluation of drift detection systems must routinely account for the resource conditions under which they will actually be deployed. A detector that performs well in a benchmark with unlimited memory and continuous monitoring is not necessarily the right choice for a system that checks every thousand samples with a 50-sample detection window. The deployment context should inform the algorithm choice, not the other way around. This study provides an initial empirical foundation for making that choice with evidence rather than assumption.

---

## References for This Chapter

Barros, R. S. M., & Santos, S. G. T. C. (2018). A large-scale comparison of concept drift detectors. *Information Sciences*, 451–452, 348–370.

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2013). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.

Lukats, D., Zielinski, O., Hahn, A., & Stahl, F. (2024). A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams. *International Journal of Data Science and Analytics*, 19, 1–31.

Pathi, N. (2025). Model evaluation beyond AUC: A comparative study of Somers' D, KS statistic, and PSI. *Proceedings of ICRTCSIT 2025*.

Prashanth, B. R., Kumar, A., Puneetha, P., & Vinay, S. (2025). Drift detection using Bayesian neural networks and Wasserstein distance. *IEEE International Conference on Advances in Machine Intelligence and Applications (AMATHE 2025)*.

Taplin, R., & Hunt, C. (2019). The population accuracy index: A new measure of population stability for model monitoring. *Risks*, 7(2), 53.

Yang, L., & Shami, A. (2021). A lightweight concept drift detection and adaptation framework for IoT data streams. *IEEE Internet of Things Magazine*, 4(2), 96–101.
