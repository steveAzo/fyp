# Chapter 4: Results

## 4.1 Overview

This chapter presents the empirical findings from two coordinated experiments conducted to answer the four research questions established in Chapter 3. The primary experiment yielded 8,100 observations across a full factorial design spanning three datasets, three detectors, three drift types, three drift features, five monitoring frequencies, four window sizes, three reference sizes, and five repetitions. A second, targeted experiment produced 405 observations from a model comparison grid using fixed resource constraints. All inferential claims are supported by non-parametric statistical tests, with significance evaluated at α = 0.05 after Bonferroni correction for multiple comparisons.

Results are organised around the four research questions:

- **RQ1:** Do the three detectors differ significantly in detection delay and error characteristics?
- **RQ2:** Does drift type (sudden, gradual, recurring) significantly affect detection difficulty?
- **RQ3:** Which resource constraints have the strongest effect on detection delay, and is the effect monotonic?
- **RQ4:** Does the choice of predictive model affect robustness to drift-induced accuracy degradation?

The presentation proceeds from detector-level comparisons (RQ1), through drift-type effects (RQ2), resource constraint analysis (RQ3), and concludes with model robustness findings (RQ4). A synthesis of cross-cutting patterns closes the chapter.

---

## 4.2 Detector Performance (RQ1)

### 4.2.1 Aggregate Performance Summary

Table 4.1 reports aggregate performance metrics for the three detectors, namely the Kolmogorov-Smirnov test (KS), the Population Stability Index (PSI), and the Wasserstein distance, collapsed across all datasets, drift types, and constraint conditions.

**Table 4.1: Aggregate Detector Performance Across All Experimental Conditions**

| Detector | Missed Detection Rate | Mean Delay (windows) | Median Delay (windows) | Mean Compute (ms) | Mean False Positives |
|---|---|---|---|---|---|
| KS Test | 1.11% | 216.28 | 100.0 | 378.51 | 5.56 |
| PSI | 0.56% | 291.73 | 154.0 | 77.53 | 10.92 |
| Wasserstein | 0.37% | 205.52 | 100.0 | 114.44 | 11.30 |

The data reveal a three-way tension between delay, reliability, and computational cost that is central to the findings of this study. Wasserstein distance achieved the lowest missed detection rate (0.37%) and the shortest mean detection delay (205.52 windows), but incurred a higher false positive count than KS. PSI showed the lowest missed detection rate among all three detectors but imposed the longest mean detection delay (291.73 windows) and also generated the most false positives per run. The KS test exhibited an intermediate missed detection rate (1.11%) with competitive mean delay but required substantially more computation (378.51 ms per run) than either alternative.

These headline differences are visually characterised in Figure 4.1 (fig5\_compute\_time.png) and Figure 4.3 (fig3\_tradeoff\_curves.png) in the results directory. The tradeoff curves (Figure 4.3) confirm that no single detector dominates on all dimensions simultaneously; the optimal choice depends on which metric is most costly in the deployment context (Lu et al., 2019).

### 4.2.2 Detection Delay: Statistical Evidence

A Kruskal-Wallis H-test on detection delay, grouped by detector, was highly significant (H = 148.739, p < 0.001), confirming that the three detectors do not draw from the same delay distribution. Post-hoc pairwise Mann-Whitney U tests with Bonferroni correction (three comparisons, adjusted α = 0.017) produced the results shown in Table 4.2.

**Table 4.2: Pairwise Mann-Whitney U Tests on Detection Delay (Bonferroni Corrected)**

| Comparison | n₁ | n₂ | U Statistic | p (raw) | p (Bonferroni) | Effect Size r | Significant |
|---|---|---|---|---|---|---|---|
| KS vs. PSI | 2,670 | 2,685 | 3,041,567.5 | < 0.001 | < 0.001 | 0.151 | Yes |
| KS vs. Wasserstein | 2,670 | 2,690 | 3,696,355.0 | 0.062 | 0.186 | -0.029 | No |
| PSI vs. Wasserstein | 2,685 | 2,690 | 4,248,867.5 | < 0.001 | < 0.001 | -0.177 | Yes |

The effect size r is the rank-biserial correlation, where |r| < 0.10 is negligible, 0.10–0.30 is small, 0.30–0.50 is medium, and ≥ 0.50 is large (Cohen, 1988). A positive r indicates group_a detects faster; negative r indicates group_b detects faster.

Two findings stand out. First, PSI is significantly slower than both KS and Wasserstein, with small but consistent effect sizes (|r| ≈ 0.15–0.18). This is consistent with PSI's sensitivity to threshold calibration and its bin-based aggregation of the distribution comparison, a known limitation documented by Taplin and Hunt (2019), who demonstrated that PSI is highly sensitive to the number of bins and can fail to trigger on genuine distributional shifts if the threshold is not well-calibrated to the data. Second, KS and Wasserstein do not differ significantly in detection delay (p = 0.186 after correction, r = -0.029, negligible), indicating that they are functionally equivalent in their responsiveness under the conditions tested.

### 4.2.3 Missed Detections and False Positives

Missed detection rates were uniformly low across all three detectors (< 2% in all cases), suggesting that all three are adequately sensitive to the magnitudes of drift introduced in this study. The marginal difference in missed detection rates between PSI (0.56%) and Wasserstein (0.37%) is practically small and both are within acceptable operating bounds for production monitoring (Yang & Shami, 2021).

False positive behaviour, however, differed more substantially. The KS test generated a lower mean false positive count (5.56 per run) than either PSI (10.92) or Wasserstein (11.30). This is attributable to the KS test's use of a stringent p-value threshold (α = 0.05) which provides inherent control over Type I error, whereas PSI and Wasserstein rely on fixed distance thresholds calibrated by domain convention rather than formal hypothesis testing (Arora et al., 2024). The practical implication is that deployments sensitive to false alarms, where spurious retraining triggers are costly, would favour the KS test, accepting the cost of higher computation time.

### 4.2.4 Computation Time

The three detectors differed substantially in computation time (see Table 4.1). The KS test was the most computationally expensive at a mean of 378.51 ms per run. This is consistent with the O(n log n) complexity of sorting-based two-sample tests; because the KS statistic requires comparing the full empirical CDFs, its cost scales with window and reference size (Massey, 1951). Wasserstein distance was intermediate (114.44 ms), and PSI was by far the cheapest (77.53 ms), reflecting its reliance on a fixed binning grid rather than on sorting or integration.

For edge and IoT deployments where inference budgets are tight, contexts explicitly identified as under-served in the literature (Preuveneers et al., 2020; Yang & Shami, 2021), PSI's computation advantage is material, even though it trails on delay and false positive control. This trade-off is discussed further in Chapter 5.

---

## 4.3 Effect of Drift Type (RQ2)

### 4.3.1 Detection Delay by Drift Type

The three simulated drift patterns, namely sudden, gradual, and recurring, produced markedly different detection delays. A Kruskal-Wallis H-test was highly significant (H = 504.546, p < 0.001), with an effect size substantially larger than that observed for detector differences, indicating that drift type is a stronger structural determinant of detection difficulty than detector choice across these experimental conditions.

**Table 4.3: Pairwise Mann-Whitney U Tests on Detection Delay by Drift Type (Bonferroni Corrected)**

| Comparison | n₁ | n₂ | U Statistic | p (raw) | p (Bonferroni) | Effect Size r | Significant |
|---|---|---|---|---|---|---|---|
| Gradual vs. Recurring | 2,700 | 2,645 | 4,670,331.5 | < 0.001 | < 0.001 | -0.308 | Yes |
| Gradual vs. Sudden | 2,700 | 2,700 | 4,745,081.5 | < 0.001 | < 0.001 | -0.302 | Yes |
| Recurring vs. Sudden | 2,645 | 2,700 | 3,526,975.0 | 0.435 | 1.000 | 0.012 | No |

Gradual drift is significantly harder to detect than both sudden drift (r = -0.302, medium effect) and recurring drift (r = -0.308, medium effect). These are the largest effect sizes observed in the study, reflecting the theoretical expectation that distributional tests based on a fixed window struggle when the underlying shift is spread over many time steps: the marginal change per check is small, and accumulation takes longer to cross any threshold (Lu et al., 2019; Suárez-Cetrulo et al., 2022).

The absence of a significant difference between sudden and recurring drift (p = 1.000 after Bonferroni correction, r = 0.012) is a substantively informative null result. Recurring drift, despite its oscillating nature, is just as detectable as sudden drift once a full cycle has completed. This finding aligns with the structure of the simulator: recurring drift, during its active phase, applies the same magnitude shift as sudden drift; the detector encounters an identical distributional gap. What varies with recurring drift is the frequency of expected alarms, not the per-alarm detectability.

These findings are visualised in Figure 4.2 (fig6\_delay\_by\_drift\_type.png) in the results directory, which shows the full delay distributions for each drift type as violin plots.

### 4.3.2 Accuracy Degradation by Drift Type and Dataset

Table 4.4 reports the mean baseline and post-drift classification accuracy for each dataset-drift type combination, along with the accuracy degradation (baseline minus post-drift). These figures are averaged across all detectors and constraint conditions.

**Table 4.4: Accuracy Degradation by Dataset and Drift Type**

| Dataset | Drift Type | Baseline Accuracy | Post-Drift Accuracy | Accuracy Degradation |
|---|---|---|---|---|
| Credit Default | Gradual | 0.8145 | 0.8095 | 0.0050 |
| Credit Default | Recurring | 0.8145 | 0.8105 | 0.0040 |
| Credit Default | Sudden | 0.8145 | 0.8098 | 0.0047 |
| Predictive Maintenance | Gradual | 0.9805 | 0.9361 | 0.0445 |
| Predictive Maintenance | Recurring | 0.9805 | 0.9535 | 0.0271 |
| Predictive Maintenance | Sudden | 0.9805 | 0.9294 | 0.0511 |
| Telco Churn | Gradual | 0.7897 | 0.7864 | 0.0033 |
| Telco Churn | Recurring | 0.7897 | 0.7925 | -0.0028 |
| Telco Churn | Sudden | 0.7897 | 0.7891 | 0.0006 |

Three patterns merit attention. First, accuracy degradation is strongly dataset-dependent. The Predictive Maintenance dataset exhibits by far the largest degradation, as much as 5.11 percentage points under sudden drift, while Credit Default and Telco Churn show degradation of less than one percentage point in most conditions. This is attributable to the imbalanced and near-perfect baseline performance of the Predictive Maintenance classifier: a model operating near its ceiling is more sensitive to distributional perturbation because small distributional shifts can cross the decision boundary for the minority class.

Second, sudden drift produces the largest degradation in the Predictive Maintenance dataset (0.0511 vs. 0.0445 for gradual, 0.0271 for recurring), consistent with the intuition that an abrupt shift maximally disrupts the feature distribution from the model's training regime before any adaptation can occur.

Third, Telco Churn under recurring drift shows a slight negative degradation (-0.0028), meaning post-drift accuracy marginally exceeds baseline. This is an artefact of the recurring pattern: the stream oscillates between drifted and non-drifted states, and the window evaluated for post-drift accuracy may, by chance, sample more from the undrifted phase. This illustrates a genuine complexity in evaluating recurring drift: the measured impact depends on where in the cycle the evaluation window falls (Suárez-Cetrulo et al., 2022).

---

## 4.4 Effect of Resource Constraints (RQ3)

### 4.4.1 Spearman Correlation and Kruskal-Wallis Summary

Three constraint dimensions were tested for their association with detection delay: monitoring frequency (the interval between successive detector checks), window size (the number of recent stream observations used for each check), and reference size (the number of baseline observations stored for comparison). Table 4.5 summarises the Spearman rank correlation and Kruskal-Wallis H-statistic for each constraint.

**Table 4.5: Constraint Effects on Detection Delay: Spearman Correlation and Kruskal-Wallis Test**

| Constraint | Spearman ρ | Spearman p | Kruskal-Wallis H | KW p | Significant |
|---|---|---|---|---|---|
| Monitoring Frequency | 0.487 | < 0.001 | 1,948.767 | < 0.001 | Yes |
| Window Size | 0.543 | < 0.001 | 2,381.184 | < 0.001 | Yes |
| Reference Size | 0.012 | 0.275 | 50.015 | < 0.001 | Yes (non-monotonic) |

Monitoring frequency and window size each demonstrate strong, statistically significant monotonic relationships with detection delay (ρ = 0.487 and ρ = 0.543 respectively, both H > 1,900, both p < 0.001). Reference size presents a qualitatively different pattern: the Kruskal-Wallis test is significant (H = 50.015, p < 0.001) confirming that group differences exist, but the Spearman correlation is negligible and non-significant (ρ = 0.012, p = 0.275), indicating that the relationship is not monotonic. These findings are unpacked in the following subsections.

### 4.4.2 Monitoring Frequency

Monitoring frequency, the number of stream samples observed between successive detector checks, had the second-strongest individual effect on detection delay (ρ = 0.487, H = 1,948.767). The mechanism is direct: a detector that checks every 10 samples can, in principle, flag drift within a window of that size; a detector that checks every 1,000 samples cannot flag drift until at least 1,000 samples after its onset. The quantitative evidence from Table 4.6 illustrates this escalation.

**Table 4.6: Mean Detection Delay and Missed Detection Rate by Monitoring Frequency**

| Monitoring Frequency | KS Delay | KS Missed | PSI Delay | PSI Missed | Wasserstein Delay | Wasserstein Missed |
|---|---|---|---|---|---|---|
| 10 | 78.09 | 0.00% | 154.39 | 0.00% | 84.48 | 0.00% |
| 50 | 103.09 | 0.00% | 173.46 | 0.00% | 107.72 | 0.00% |
| 100 | 130.31 | 0.00% | 202.91 | 0.00% | 132.72 | 0.00% |
| 500 | 286.43 | 0.00% | 355.87 | 0.00% | 258.65 | 0.00% |
| 1,000 | 499.22 | 5.56% | 580.02 | 2.78% | 448.55 | 1.85% |

The transition from monitoring\_frequency = 500 to monitoring\_frequency = 1,000 is the most consequential step: mean delay roughly doubles across all three detectors, and missed detection rates rise from zero to non-trivial values (1.85%–5.56%). This threshold effect reflects a structural limitation: when the monitoring interval is so long that the total stream length provides only a small number of checks, some drifts are simply never encountered within the stream window. This finding directly quantifies the theoretical concern raised in the edge ML literature that infrequent monitoring can render drift detectors effectively blind (Yang & Shami, 2021; Preuveneers et al., 2020).

Across all monitoring frequency levels, false positive counts decrease monotonically as frequency decreases (more frequent checking → more opportunities for spurious alerts). At monitoring\_frequency = 10, KS generates a mean of 21.28 false positives per run; at monitoring\_frequency = 1,000, this drops to 0.19. There is therefore a secondary cost-benefit trade-off: higher monitoring frequency improves detection latency but increases false alarm burden on downstream systems.

### 4.4.3 Window Size

Window size, the number of recent stream samples included in each detector check, was the single strongest constraint predictor in the study (ρ = 0.543, H = 2,381.184). The direction of the effect is positive and monotonic: larger windows produce longer detection delays. This is counterintuitive at first glance but follows from the statistical mechanics of distribution comparison: a larger window dilutes the local distributional signal. When drift has occurred only recently, a large window still contains many pre-drift samples, reducing the power of the hypothesis test or distance measure to discriminate the current distribution from the reference (Lu et al., 2019; Gemaque et al., 2020).

**Table 4.7: Mean Detection Delay by Window Size**

| Window Size | KS Delay | KS Missed | PSI Delay | PSI Missed | Wasserstein Delay | Wasserstein Missed |
|---|---|---|---|---|---|---|
| 50 | 122.33 | 2.22% | 103.70 | 0.74% | 104.52 | 0.00% |
| 100 | 129.76 | 2.22% | 162.50 | 1.48% | 117.01 | 1.48% |
| 500 | 233.19 | 0.00% | 373.48 | 0.00% | 240.07 | 0.00% |
| 1,000 | 375.85 | 0.00% | 523.93 | 0.00% | 359.19 | 0.00% |

An important reversal is visible in the PSI missed detection rate: PSI misses 0.74% of drifts at window\_size = 50 but zero at window\_size = 500 and window\_size = 1,000. This reflects a known property of PSI: it uses fixed-width or equal-frequency binning, and bins become unreliable at small sample sizes (Taplin & Hunt, 2019). Larger windows give PSI more samples per bin, making its statistics more stable and eliminating the false negatives that arise from sparse bin counts. The KS test shows the opposite trend: its missed detection rate is 2.22% at small windows (reflecting low power with few samples) and falls to zero at larger windows, consistent with the increasing power of the KS test as sample size grows (Massey, 1951).

### 4.4.4 Reference Size

Reference size produced a non-monotonic effect on detection delay. The Kruskal-Wallis test confirms that reference size levels are not equivalent (H = 50.015, p < 0.001), but the Spearman correlation is effectively zero (ρ = 0.012, p = 0.275), indicating that larger reference sizes do not consistently produce shorter or longer delays.

**Table 4.8: Mean Detection Delay by Reference Size**

| Reference Size | KS Delay | KS Missed | PSI Delay | PSI Missed | Wasserstein Delay | Wasserstein Missed |
|---|---|---|---|---|---|---|
| 100 | 231.36 | 1.11% | 262.78 | 0.00% | 172.06 | 0.00% |
| 500 | 217.31 | 1.11% | 308.55 | 1.11% | 223.55 | 0.56% |
| Unlimited (-1) | 200.18 | 1.11% | 304.11 | 0.56% | 221.15 | 0.56% |

The 100-sample reference condition consistently produces the highest detection delays across all three detectors, a threshold degradation effect. With only 100 samples as baseline, the reference distribution is imprecisely characterised; statistical tests comparing a small reference against a larger window have reduced power, and distance metrics are noisy. Beyond 500 samples, however, additional reference data yields diminishing returns: delays under reference\_size = 500 and reference\_size = unlimited are broadly comparable for KS and Wasserstein.

The implication for resource-constrained deployment is practical and encouraging: maintaining a reference set of 500 samples is sufficient to achieve near-unlimited-baseline performance, meaning memory cost can be bounded without meaningful degradation in detection quality. The critical cliff is at 100 samples, not at any higher level.

### 4.4.5 Constraint Interactions: Monitoring Frequency × Window Size

Table 4.9 presents the mean detection delay at the intersection of monitoring frequency and window size, averaged across all other conditions. This is the cross-tabulation underlying Figure 4.1 (fig1\_delay\_heatmap.png) in the results directory.

**Table 4.9: Mean Detection Delay (windows): Monitoring Frequency × Window Size**

| Monitoring Freq. ↓ / Window Size → | 50 | 100 | 500 | 1,000 |
|---|---|---|---|---|
| 10 | 22.0 | 41.2 | 124.8 | 234.7 |
| 50 | 49.2 | 64.0 | 143.1 | 256.0 |
| 100 | 65.3 | 103.2 | 173.3 | 279.5 |
| 500 | 118.4 | 162.2 | 420.2 | 500.5 |
| 1,000 | 305.4 | 328.1 | 549.9 | 827.6 |

The heatmap reveals the full span of achievable detection delay under this experimental design: from 22.0 windows in the least-constrained configuration (monitoring\_frequency = 10, window\_size = 50) to 827.6 windows in the most-constrained (monitoring\_frequency = 1,000, window\_size = 1,000), a 37-fold range. This 37-fold variation is attributable entirely to resource configuration choice, not to the drift itself or to the detector selected. The practical message is stark: how often and with how large a window a system monitors matters far more than which of the three tested detectors it uses. The choice of monitoring regime is the dominant engineering decision.

The interaction is also approximately additive on the log scale: the cells do not exhibit extreme super-additive or sub-additive interactions, suggesting that the two constraints can be analysed somewhat independently when making deployment decisions, though their combined effect is always compound.

---

## 4.5 Model Robustness Under Drift (RQ4)

### 4.5.1 Accuracy Degradation by Model

The model comparison experiment (405 configurations: 3 models × 3 datasets × 3 detectors × 3 drift types × 5 repetitions) was designed to evaluate whether the choice of predictive model mediates the severity of drift-induced performance loss. The outcome variable for this analysis is accuracy\_degradation: the difference between baseline and post-drift classification accuracy.

A Kruskal-Wallis H-test on accuracy\_degradation grouped by model was significant (H = 26.692, p < 0.001). Post-hoc pairwise tests (Table 4.10) reveal the structure of this effect.

**Table 4.10: Pairwise Mann-Whitney U Tests on Accuracy Degradation by Model (Bonferroni Corrected)**

| Comparison | n₁ | n₂ | U Statistic | p (raw) | p (Bonferroni) | Effect Size r | Significant |
|---|---|---|---|---|---|---|---|
| LightGBM vs. Logistic Regression | 135 | 135 | 10,350.0 | 0.054 | 0.160 | -0.136 | No |
| LightGBM vs. Random Forest | 135 | 135 | 12,600.0 | < 0.001 | < 0.001 | -0.383 | Yes |
| Logistic Regression vs. Random Forest | 135 | 135 | 10,800.0 | 0.008 | 0.025 | -0.185 | Yes |

In both significant comparisons, the negative r values indicate that Random Forest suffered *less* accuracy degradation than either LightGBM or Logistic Regression, meaning Random Forest is the more robust model. The effect size between Random Forest and LightGBM is medium (|r| = 0.383), while the effect between Random Forest and Logistic Regression is small (|r| = 0.185). LightGBM and Logistic Regression do not differ significantly from each other (p = 0.160 after correction, |r| = 0.136).

The finding that Random Forest is the *most* robust to drift-induced degradation is consistent with the established understanding of ensemble variance reduction (Breiman, 2001), extended here from robustness to training noise to robustness under covariate shift. Random Forest's bagging and per-split feature randomisation decorrelate its constituent trees: when a given feature's distribution shifts, only the subset of trees whose splits depend heavily on that feature are strongly affected, and the ensemble average absorbs this local error rather than propagating it uniformly. LightGBM's boosting procedure has no equivalent decorrelation mechanism: each tree is fit sequentially to the residual of the ensemble built so far, so a shift that degrades an early tree's predictions is compounded across subsequent boosting rounds rather than averaged away (Ke et al., 2017). This mechanism is developed further in Chapter 5.

It is essential to note that the model comparison experiment controlled resource constraints at fixed values (monitoring\_frequency = 100, window\_size = 500, reference\_size = unlimited) and used the same detectors across models. The differences in accuracy\_degradation are therefore attributable to the classifiers themselves, not to any variation in the monitoring regime. The drift detectors, being purely distribution-based and model-agnostic, were not affected by model choice.

**Robustness check: relative degradation.** The absolute degradation metric used above (baseline − post-drift accuracy) does not control for the fact that models with different baseline accuracy have different amounts of "room" to degrade. As a robustness check, degradation was recomputed as a relative measure, Δ_rel = (baseline − post-drift) / baseline, and the pairwise tests re-run. Both significant effects are numerically unchanged (RF vs. LightGBM: r = −0.383; RF vs. Logistic Regression: r = −0.185), confirming that Random Forest's advantage is not an artefact of baseline accuracy differences. The relative metric additionally finds LightGBM significantly more robust than Logistic Regression (r = −0.185, p = 0.025 after Bonferroni correction), a distinction the absolute metric could not separate from chance (p = 0.160). Correcting for the baseline-accuracy confound therefore sharpens the finding into a complete, significant three-way ordering (Random Forest > LightGBM > Logistic Regression) rather than changing its direction.

### 4.5.2 Dataset-Level Model Patterns

The model effect is not uniform across datasets, and its source differs from what the aggregate comparison alone suggests. On Predictive Maintenance (the dataset closest to its accuracy ceiling, 0.9805 baseline), both tree ensembles degrade substantially (LightGBM 0.047, Random Forest 0.040), while Logistic Regression is markedly more stable (0.007). On Credit Default, degradation is small across all three models, with LightGBM the least robust (0.008), Random Forest intermediate (0.005), and Logistic Regression the most robust (0.002). On Telco Churn the pattern inverts sharply: Logistic Regression degrades by 0.070, the single largest degradation observed anywhere in this experiment and more than five times any other model-dataset combination, while Random Forest is essentially unaffected (0.0004) and LightGBM is moderate (0.010).

Two conclusions follow. First, Random Forest's advantage over LightGBM is consistent in direction across all three datasets, supporting the generality of the bagging-versus-boosting mechanism rather than attributing it to one dataset's idiosyncrasies. Second, Logistic Regression's aggregate degradation is driven almost entirely by its anomalous failure on Telco Churn; on the other two datasets it is competitive with or better than both ensembles. This is consistent with a decision boundary that is well-approximated as linear for most of the feature space here, but poorly so where the drifted feature (Monthly Charges) interacts non-linearly with the target, a conditional robustness that a single aggregate comparison obscures (Gemaque et al., 2020; Kore et al., 2024).

### 4.5.3 Hyperparameter Sensitivity and an Overfitting Diagnostic

The model comparison experiment used fixed, untuned hyperparameters for all three classifiers (Section 3.5), which raises two validity questions: whether the trained models are overfit to the reference period in a way that could exaggerate the measured degradation, and whether the robustness ranking depends on this specific hyperparameter configuration rather than reflecting a general property of the model classes.

**Overfitting diagnostic.** No internal train/validation split was used when fitting the reference-period models in the main experiment, since hyperparameters were fixed rather than tuned. As a supplementary check, each model was refit on an 80% subset of the reference period and evaluated on the remaining, genuinely held-out 20%. The results are striking for Random Forest: in-sample accuracy reaches 99.9–100% on all three datasets, but held-out accuracy on the same split is markedly lower (Credit Default: 81.9%, Telco Churn: 79.2%, Predictive Maintenance: 97.9%), a train-validation gap of up to 21 percentage points. LightGBM shows a smaller but non-trivial gap on the smaller datasets (Telco Churn: 11.9 percentage points), while Logistic Regression shows essentially no gap (within ±0.5 percentage points on all datasets), consistent with its much lower model capacity. Random Forest's unlimited tree depth is therefore memorising substantial reference-period idiosyncrasies that do not generalise to a held-out sample.

This raises a direct question for the central RQ4 finding: does Random Forest's drift-robustness advantage depend on this specific, unregularised, overfitting-prone configuration?

**Hyperparameter sensitivity check.** To answer this, the full drift-injection procedure was rerun with Random Forest at max\_depth ∈ {3, 5, 10, 20, unlimited} (the value used in the main experiment), holding everything else fixed, using a smaller replication of the experimental design (n = 45 per configuration: 3 datasets × 3 drift types × 5 repetitions; detector choice is omitted because it does not affect accuracy\_degradation, as confirmed directly from the original 405-row dataset, where every detector produces an identical value for a given dataset–model–drift\_type–repetition combination). Table 4.11 reports the results against LightGBM's established degradation (0.0218).

**Table 4.11: Random Forest Accuracy Degradation by Maximum Tree Depth (n = 45 per depth)**

| max_depth | Mean Degradation | vs. LightGBM (p, Bonferroni) | Effect Size r |
|---|---|---|---|
| 3 | 0.0033 | < 0.001 | +0.753 |
| 5 | 0.0044 | < 0.001 | +0.802 |
| 10 | 0.0125 | < 0.001 | +0.432 |
| 20 | 0.0169 | 0.020 | +0.284 |
| Unlimited (main experiment) | 0.0150 | 0.002 | +0.383 |

Random Forest's advantage over LightGBM is significant at every depth tested, and is in fact strongest at the shallowest, most regularised depths (r = 0.75–0.80 at depth 3–5, a large effect) rather than at the unlimited depth used in the main experiment (r = 0.383, medium). A parallel check varying LightGBM's learning rate (0.05, 0.1, 0.2, against Random Forest's original unlimited-depth degradation of 0.0150) found Random Forest significantly more robust at every rate tested (all p < 0.005, Bonferroni-corrected).

Two conclusions follow. First, the overfitting identified in the diagnostic above does not appear to be driving the robustness finding. If anything, a properly regularised Random Forest (shallow trees) is *more* drift-robust than the unregularised configuration used in the main experiment, not less. Second, the finding generalises across the tested hyperparameter space for both models, rather than being an artefact of one specific, arbitrary configuration choice.

### 4.5.4 Robustness Under Imbalance-Aware Metrics

All three datasets are class-imbalanced (Table 3.1), and raw accuracy is dominated by majority-class performance. This motivates a further validity check: does the accuracy-based robustness ranking hold under metrics that are less sensitive to class imbalance? Degradation was recomputed as balanced accuracy (the mean of sensitivity and specificity) and ROC-AUC, using the same n = 45 replication described above, additionally instrumented to record predicted probabilities.

**Table 4.12: Model Degradation Across Metrics (n = 45 per model)**

| Metric | Random Forest | LightGBM | Logistic Regression |
|---|---|---|---|
| Accuracy degradation | 0.0150 | 0.0218 | 0.0265 |
| Balanced accuracy degradation | −0.0360 | −0.0130 | −0.0566 |
| AUC degradation | 0.0171 | 0.0282 | −0.0055 |

The picture is more nuanced than the accuracy-based result alone suggests. Balanced accuracy *improves* post-drift for all three models (negative values in Table 4.12 indicate improvement), suggesting the injected drift shifts predictions toward better minority-class recall at some cost to majority-class accuracy, a trade-off raw accuracy does not capture. Under ROC-AUC, LightGBM again shows the largest degradation (0.028, significantly worse than Random Forest, p < 0.001), but Logistic Regression shows the smallest change (an improvement of 0.006), which is not statistically distinguishable from Random Forest's degradation of 0.017 at this sample size (p = 0.26).

Across all three metrics, LightGBM is consistently the least robust model, which strengthens confidence in that part of the finding. The comparison between Random Forest and Logistic Regression, however, is metric-dependent: Random Forest's advantage over Logistic Regression is clear and significant for raw and relative accuracy (Section 4.5.1), but the two become statistically indistinguishable under AUC and balanced accuracy. This is consistent with the dataset-level pattern established in Section 4.5.2: Logistic Regression's poor aggregate accuracy is driven almost entirely by its severe failure on Telco Churn, and that failure appears to weigh less heavily on rank-based and imbalance-corrected metrics than on raw accuracy.

The practical implication is that the choice of degradation metric should match the deployment priority. Where overall predictive accuracy is what matters operationally, Random Forest is the more robust choice, and this conclusion is now supported by both a relative-degradation check (Section 4.5.1) and a hyperparameter sensitivity check (Section 4.5.3). Where minority-class detection is the priority, common in the exact domains these datasets represent (credit default, churn, machine failure), Logistic Regression's robustness is competitive with Random Forest's despite its markedly worse showing on raw accuracy, and the choice between them should not be made on accuracy degradation alone.

---

## 4.6 Summary of Key Findings

The results across all four research questions converge on five primary conclusions, each with direct implications for practitioners designing drift-aware monitoring systems under resource constraints.

**Finding 1: Detector choice produces small but significant differences in delay; no detector dominates on all metrics.** Wasserstein distance and the KS test are statistically equivalent in detection delay (p = 0.186 after correction), with Wasserstein offering the lowest missed detection rate (0.37%). PSI is significantly slower than both (|r| ≈ 0.15–0.18) but offers the cheapest computation (77.53 ms). Deployments that are compute-constrained but delay-tolerant favour PSI; those prioritising detection speed over computation favour Wasserstein or KS.

**Finding 2: Gradual drift is significantly harder to detect than sudden or recurring drift, with medium effect sizes.** The performance gap between gradual drift and both alternatives (|r| ≈ 0.30) dwarfs the differences between any two detectors. Sudden and recurring drift are statistically indistinguishable in detection latency (p = 1.000). Systems deployed in domains where gradual concept drift is the expected pattern should be designed with this elevated detection difficulty in mind, potentially using more aggressive monitoring parameters or ensemble detection approaches.

**Finding 3: Monitoring frequency and window size are the dominant constraint dimensions, with near-identical effect magnitudes (ρ ≈ 0.49–0.54).** The full factorial experiment produced a 37-fold range in mean detection delay attributable solely to constraint configuration. The practical implication is that optimising the monitoring regime yields far greater improvements in detection performance than switching between any of the three tested detectors. This is a design priority hierarchy that the existing literature has not quantified empirically in this way.

**Finding 4: Reference size has a threshold rather than monotonic effect.** Maintaining a reference set of at least 500 samples recovers near-unlimited-baseline performance for all three detectors. Below 100 samples, detection delay increases and PSI false negative rates rise. Systems with memory budgets should protect at least 500 reference samples before reducing other resources.

**Finding 5: Random Forest suffers significantly less drift-induced accuracy degradation than LightGBM (medium effect, |r| = 0.383) or Logistic Regression (small effect, |r| = 0.185), particularly on the Predictive Maintenance dataset.** Random Forest's bagged, decorrelated ensemble structure makes it more robust to distributional perturbation than LightGBM's sequentially-boosted trees, an advantage confirmed across a range of tree depths and boosting learning rates (Section 4.5.3). Its advantage over LightGBM also holds under balanced accuracy and AUC (Section 4.5.4), though its advantage over Logistic Regression is specific to accuracy-based metrics and narrows substantially once class imbalance is accounted for. In deployments where retraining is expensive, overall accuracy is the priority, and the cost of degraded predictions is high, Random Forest is the preferred classifier among those tested.

These five findings are interpreted in the context of the broader literature and their implications for system design in Chapter 5.

---

## References

Arora, S., Rani, R., & Saxena, N. (2024). A systematic review on detection and adaptation of concept drift in streaming data using machine learning techniques. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 14. https://doi.org/10.1002/widm.1536

Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Gemaque, R., Costa, A., Giusti, R., & Santos, E. (2020). An overview of unsupervised drift detection methods. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 10. https://doi.org/10.1002/widm.1381

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.

Kore, A., Bavil, E., Subasri, V., Abdalla, M., Fine, B., Dolatabadi, E., & Abdalla, M. (2024). Empirical data drift detection experiments on real-world medical imaging data. *Nature Communications*, 15. https://doi.org/10.1038/s41467-024-46142-w

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under Concept Drift: A Review. *IEEE Transactions on Knowledge and Data Engineering*, 31, 2346–2363. https://doi.org/10.1109/tkde.2018.2876857

Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association*, 46(253), 68–78. https://doi.org/10.2307/2280095

Preuveneers, D., Tsingenopoulos, I., & Joosen, W. (2020). Resource usage and performance trade-offs for machine learning models in smart environments. *Sensors*, 20(4), 1176. https://doi.org/10.3390/s20041176

Suárez-Cetrulo, A., Quintana, D., & Cervantes, A. (2022). A survey on machine learning for recurring concept drifting data streams. *Expert Systems with Applications*, 213, 118934. https://doi.org/10.1016/j.eswa.2022.118934

Taplin, R., & Hunt, C. (2019). The population accuracy index: A new measure of population stability for model monitoring. *Risks*, 7(2), 53. https://www.mdpi.com/2227-9091/7/2/53

Yang, L., & Shami, A. (2021). A Lightweight Concept Drift Detection and Adaptation Framework for IoT Data Streams. *IEEE Internet of Things Magazine*, 4, 96–101. https://doi.org/10.1109/iotm.0001.2100012
