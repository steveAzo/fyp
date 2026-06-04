# Chapter 3: Methodology

## 3.1 Research Design

This study employs a **controlled experimental approach** to investigate how resource constraints affect the performance of lightweight data drift detection methods in deployed machine learning systems. The experimental design treats resource constraints as systematically varied independent variables and measures drift detection performance through a set of quantitative dependent variables, enabling direct causal attribution of performance changes to specific constraint types.

The motivation for a controlled experimental approach, rather than observational or simulation-only methods, arises from the nature of the research question. Understanding *when* and *why* drift detectors fail under constraints requires the ability to (a) inject drift at a precisely known point in the data stream, (b) vary constraints independently without confounding, and (c) replicate each configuration multiple times to quantify variability. These requirements are satisfied by laboratory-style controlled experimentation on real-world tabular datasets, a methodology consistent with established practices in empirical machine learning research (Lu et al., 2019; Arora et al., 2024).

The study is structured around two complementary experiments:

**Primary experiment:** A full factorial design crossing three datasets, three drift detection methods, three drift patterns, five monitoring frequencies, four window sizes, and three reference memory configurations, producing 8,100 experimental configurations. Each configuration is replicated five times with different random seeds, resulting in 8,100 observations. This experiment addresses the core research question regarding how resource constraints affect drift detection effectiveness.

**Model comparison experiment:** A supplementary factorial design holding constraints fixed at a single representative configuration and crossing three classifier models against all datasets, detectors, and drift types, producing 405 configurations. This experiment assesses whether the accuracy degradation findings from the primary experiment generalise across classifier types, addressing the concern that results may be model-dependent.

All experiments were conducted on a standard personal laptop without high-performance computing resources, which is intentionally consistent with the resource-constrained deployment context under investigation. The full experimental pipeline is implemented in Python and is made available to support reproducibility.

---

## 3.2 Datasets

Three publicly available classification datasets were selected to provide diversity across domain, sample size, and feature characteristics. Dataset selection followed three criteria: (1) binary classification target to enable consistent model training and evaluation, (2) presence of features likely to undergo natural distributional shift in deployed systems, and (3) sufficient sample size to sustain a sliding-window monitoring protocol over an extended stream.

### 3.2.1 UCI Default of Credit Card Clients

This dataset contains 30,000 records of credit card holders in Taiwan, with 23 features encompassing demographic variables, credit limit, payment history, and billing amounts (Yeh, 2009). The binary target variable indicates whether a client defaulted on the subsequent month's payment.

Credit scoring is a canonical application domain for drift detection: model deployment spans years during which the customer population shifts due to demographic ageing, economic cycles, and changes in credit approval policy. The feature **AGE** was designated as the primary drift target for this dataset, as it represents a demographic variable likely to shift as a deployed credit model's active customer base evolves over time. This choice is consistent with the broader literature identifying demographic covariate shift as a primary source of model degradation in financial applications (Lu et al., 2019; Taplin & Hunt, 2019).

Following removal of the row identifier column, the dataset provides 30,000 usable records with 22 numeric features. No missing values were present. The class distribution is imbalanced (approximately 22% defaults), which motivates the use of ROC-AUC as a supplementary evaluation metric alongside accuracy.

### 3.2.2 IBM Telco Customer Churn

This dataset contains approximately 7,000 customer records from a telecommunications provider with 33 raw features. Following removal of geographic identifiers (Country, State, City, Zip Code, latitude, longitude), administrative columns (CustomerID, Count), and direct leakage columns — variables that encode or summarise the target outcome (Churn Label, Churn Score, Churn Reason, Customer Lifetime Value) — 19 usable features remain. The binary target indicates whether a customer churned.

Removing leakage columns is critical for experimental validity: retaining a column such as *Churn Score*, which directly summarises the outcome, would allow any classifier to achieve near-perfect accuracy trivially and would not reflect realistic deployment conditions. This form of data leakage is a well-documented threat to validity in applied machine learning research.

The feature **Monthly Charges** was selected as the drift target. Subscription charges are subject to pricing changes, promotional campaigns, and service upgrades, making this an operationally realistic variable for distributional shift in a deployed churn prediction system.

Eleven records with missing values in the *Total Charges* field (arising from customers with zero tenure whose total was left blank) were removed prior to splitting, yielding approximately 7,032 usable records.

### 3.2.3 AI4I 2020 Predictive Maintenance

This dataset contains 10,000 records generated by a simulation of an industrial production environment, with four numeric sensor features (air temperature, process temperature, rotational speed, torque) and one ordinal feature (machine type: L, M, H) (Matzka, 2020). The binary target indicates machine failure.

Five sub-failure columns (TWF, HDF, PWF, OSF, RNF) encode the specific failure mode retrospectively and were removed, as they constitute post-hoc diagnostic codes unavailable at prediction time and would constitute leakage if retained. The machine type feature was ordinally encoded (L=0, M=1, H=2) rather than one-hot encoded, as the ordering reflects a genuine low-to-high load level gradient that ordinal representation preserves.

The feature **Tool wear [min]** was designated as the drift target. Cumulative tool wear increases monotonically over operating time, creating a natural, realistic source of distributional shift in this domain. This dataset provides the most physically grounded drift scenario of the three, where the direction and approximate rate of drift are interpretable from domain knowledge.

### 3.2.4 Preprocessing and Reference-Stream Split

Each dataset was partitioned into a **reference set** (60%) and a **stream** (40%) using stratified random splitting, preserving the class proportion in both halves. Stratification ensures that any difference in prediction performance between the reference period and stream period reflects drift-induced degradation rather than a pre-existing class imbalance arising from random chance in the split.

All numerical features were standardised using a StandardScaler fitted *exclusively on the reference set* and applied to both reference and stream using the learned parameters. This constraint replicates real-world deployment practice: at deployment time, only the training-period data distribution is available to parameterise preprocessing, and the incoming stream must be normalised using the same statistics. Fitting the scaler on stream data would constitute temporal leakage, inflating apparent comparability between distributions (Gemaque et al., 2020).

Categorical features were encoded using a LabelEncoder fitted on the union of reference and stream values to prevent unseen-label errors at evaluation time, while ensuring that the integer mapping is consistent across both splits.

---

## 3.3 Drift Detection Methods

Three drift detection methods were selected based on the following criteria: (a) interpretability, enabling clear explanation of detection decisions; (b) low computational overhead, consistent with the lightweight deployment context; (c) widespread adoption in academic and industrial practice, ensuring that findings are relevant to real deployment decisions. All three methods operate as univariate two-sample tests — they compare a reference distribution against a test window drawn from the stream on a single feature at a time.

### 3.3.1 Kolmogorov-Smirnov Test

The two-sample Kolmogorov-Smirnov (KS) test is a nonparametric statistical test that quantifies the distance between two empirical distributions by computing the maximum absolute difference between their cumulative distribution functions (CDFs):

> D = sup_x |F_ref(x) − F_test(x)|

where F_ref and F_test are the empirical CDFs of the reference and test samples respectively (Massey, 1951). The associated p-value gives the probability of observing a distance of at least D under the null hypothesis that both samples were drawn from the same distribution. A p-value below the significance threshold α = 0.05 triggers a drift detection.

The KS test makes no parametric assumption about the form of the underlying distribution and is sensitive to differences in location, scale, and shape — a generality that makes it broadly applicable across feature types and drift patterns (Gemaque et al., 2020; Agrahari & Singh, 2021). Its computational cost is O(n log n) due to the need to sort both samples before CDF comparison, making it practical for the window sizes used in this study. The KS test has been used as a primary drift detection method in numerous empirical evaluations (Lu et al., 2019; Arora et al., 2024).

In this study, drift is declared when the returned p-value falls below **0.05**, the conventional significance level in hypothesis testing.

### 3.3.2 Population Stability Index

The Population Stability Index (PSI) is an industry-standard measure of distributional shift that originated in credit risk model monitoring, where regulatory frameworks require continuous assessment of whether a deployed model remains fit for purpose (Taplin & Hunt, 2019). PSI quantifies shift by partitioning the reference distribution into equal-frequency bins and comparing the proportion of test observations falling in each bin:

> PSI = Σ (P_test,i − P_ref,i) × ln(P_test,i / P_ref,i)

where P_ref,i and P_test,i are the proportions of observations falling in bin i for the reference and test samples respectively.

The conventional interpretation thresholds are: PSI < 0.10 indicates no significant shift; 0.10 ≤ PSI < 0.25 indicates moderate shift warranting investigation; PSI ≥ 0.25 indicates significant shift (Taplin & Hunt, 2019). In this study, a threshold of **0.25** was adopted, calibrated against the empirical noise floor on standardised pre-drift data to maintain a false positive rate below 1%.

PSI is included despite known conceptual limitations identified in the literature. Taplin and Hunt (2019) demonstrate that PSI cannot distinguish between extrapolation (where the test data falls outside the reference distribution — a genuine concern) and interpolation (where the test data is less variable than the reference — a safe condition where the model is actually more reliable). They also show that PSI's binning approach discards information about how extreme outlier values are, and that univariate PSI applied independently to each feature can fail to detect multivariate distributional shift. Including PSI in this evaluation allows these conceptual limitations to be assessed empirically under varying resource constraints — specifically whether small window sizes exacerbate sparse bin problems, and whether infrequent monitoring inflates the cost of PSI's interpolation false positives.

Equal-frequency binning with **10 bins** was adopted, consistent with banking practice and the recommendation in Taplin and Hunt (2019). To prevent undefined values when a bin is empty, a floor of 1×10⁻⁴ was applied to all proportions before computing the logarithm.

### 3.3.3 Wasserstein Distance

The first Wasserstein distance, also known as the Earth Mover's Distance (EMD), measures the minimum total effort required to transform one probability distribution into another, where effort is defined as the product of mass moved and distance moved (Villani, 2009). For two univariate distributions, the first Wasserstein distance equals the area between their CDFs:

> W₁(P, Q) = ∫ |F_P(x) − F_Q(x)| dx

Unlike the KS statistic, which measures only the maximum gap between two CDFs at a single point, the Wasserstein distance integrates the total difference across the entire support. This makes it sensitive to the *magnitude* of shift across the whole distribution, not just its most extreme single-point deviation. As a result, Wasserstein distance tends to respond more gradually to distributional change and is particularly effective at characterising shift magnitude in continuously evolving distributions (Gemaque et al., 2020).

Because all features in this study were standardised to zero mean and unit variance prior to detection, a Wasserstein threshold of **0.25** corresponds to a shift of approximately 0.25 standard deviations in the underlying distribution — a threshold calibrated against the pre-drift noise floor to maintain a false positive rate below 1% on clean data.

The Wasserstein distance was computed using the `scipy.stats.wasserstein_distance` implementation, which operates on empirical samples and requires O(n log n) time due to the internal sort operation. It is marginally more expensive than the KS test for large window sizes, representing a genuine computational trade-off relevant to the resource constraint analysis.

---

## 3.4 Drift Simulation

Controlled drift injection was adopted rather than relying on observed real-world drift events. This design choice is motivated by two requirements that cannot be satisfied with natural drift in static datasets: (a) ground truth availability — detecting drift requires knowing precisely when it occurred to measure detection delay; and (b) systematic controllability — varying drift type, magnitude, and timing independently requires programmatic injection (Lu et al., 2019; Arora et al., 2024).

Drift was injected into a single designated feature per dataset (the drift feature specified in Section 3.2). This univariate injection approach enables direct evaluation of each detector's sensitivity to a single-feature shift under varying constraints. While real-world drift often involves multiple features simultaneously, the univariate approach provides a controlled baseline that isolates detection capability from multivariate complexity — a deliberate simplification appropriate for a systematic empirical study.

Three drift patterns were implemented, reflecting the standard taxonomy of drift types in the literature (Lu et al., 2019; Suárez-Cetrulo et al., 2022):

### 3.4.1 Sudden Drift

Sudden (abrupt) drift introduces an immediate, permanent shift to the feature distribution at the designated drift point. Following the approach of Arora et al. (2024), the shift magnitude is expressed in units of the feature's standard deviation to ensure comparability across datasets with different natural scales:

> x'_i = x_i + 2σ, for all i ≥ drift_start

where σ is the standard deviation of the feature in the stream and 2σ is the shift magnitude. A magnitude of 2 standard deviations was selected as representing a shift that is detectable by statistical tests but not trivially obvious — large enough to test real detection capability while remaining within operationally realistic ranges.

Sudden drift models scenarios such as sensor failure or replacement, abrupt changes in data pipeline configuration, or discontinuous population shifts arising from policy changes.

### 3.4.2 Gradual Drift

Gradual drift introduces a shift that ramps linearly from zero to the full magnitude over a transition window, after which the shift remains constant:

> x'_i = x_i + ramp(i) × 2σ

where ramp(i) increases linearly from 0 to 1 over the first 500 samples of the drift period, and equals 1 thereafter. This pattern models slowly-evolving distributional changes such as demographic ageing in a customer population, incremental sensor calibration drift, or gradual shifts in pricing patterns over months.

Gradual drift is generally considered more challenging to detect than sudden drift, because the distributional distance between reference and test window grows slowly, and many successive checks may observe sub-threshold shifts before the full magnitude is reached (Suárez-Cetrulo et al., 2022). This difficulty is expected to interact with monitoring frequency and window size constraints in interpretable ways.

### 3.4.3 Recurring Drift

Recurring drift alternates the feature between a shifted and unshifted state in regular cycles, using modular arithmetic to determine the active phase at each sample index:

> x'_i = x_i + 2σ × 1[pos(i) < cycle_length]

where pos(i) = (i − drift_start) mod (2 × cycle_length) and cycle_length = 200 samples. This produces a pattern in which 200 consecutive samples are shifted, followed by 200 unshifted samples, repeating for the remainder of the stream. Recurring drift models seasonal or periodic phenomena such as cyclical demand patterns, shift-based operational changes, or periodic maintenance cycles (Suárez-Cetrulo et al., 2022).

Recurring drift poses a particular challenge for resource-constrained detectors: if monitoring is infrequent relative to cycle length, individual checks may fall in the unshifted phase and miss active drift entirely. This interaction between monitoring frequency and cycle structure is a key phenomenon explored in the results.

### 3.4.4 Drift Timing and Magnitude

In all three drift patterns, drift was injected at **50% of the stream length**, ensuring a pre-drift period of equal length to the post-drift period. The pre-drift period provides a baseline against which false positive rates are measured; the post-drift period provides the window within which detection delay is computed. An equal split maximises the statistical power available for measuring both types of errors.

The drift start index was additionally constrained to fall no earlier than one full window-size away from the beginning of the stream, ensuring that at least one complete monitoring check is performed before drift begins. This guarantees a valid pre-drift observation for each experiment regardless of window size.

---

## 3.5 Machine Learning Models

Three classifiers were employed to evaluate the generalisability of accuracy degradation findings across model types:

**Random Forest** is an ensemble classifier that aggregates the predictions of 100 decision trees, each trained on a bootstrap sample of the training data using a random subset of features at each split (Breiman, 2001). The averaging mechanism across diverse trees provides inherent variance reduction and makes Random Forest relatively robust to individual data point variations — a property expected to affect how model accuracy degrades under drift.

**Logistic Regression** is a linear classifier that models the log-odds of class membership as a linear combination of features (Cox, 1958). Its simplicity makes it interpretable and computationally efficient, but its linear decision boundary may be more sensitive to distributional shift that moves samples across that boundary.

**LightGBM** is a gradient boosting framework that uses leaf-wise tree growth with histogram-based split finding, providing efficient training on tabular data with high predictive accuracy (Ke et al., 2017). As a high-capacity, aggressive learner, LightGBM fits the training distribution more precisely than the ensemble-averaging approach of Random Forest, which may make it more sensitive to deployment-time distributional shift.

All classifiers were trained exclusively on the reference set with no retraining during the stream phase. This design intentionally simulates a static deployed model that has not yet been updated in response to drift — the scenario in which drift detection is most consequential, as undetected drift causes accuracy to silently degrade.

The model comparison experiment used a fixed constraint configuration (monitoring frequency = 100 samples, window size = 500 samples, unlimited reference memory) to hold resource constraints constant while varying model type. This isolates the effect of model choice on accuracy degradation, independent of constraint-induced detection delays.

---

## 3.6 Experimental Variables

### 3.6.1 Independent Variables — Resource Constraints

The following resource constraint dimensions were varied systematically in the primary experiment. Each dimension represents a specific type of resource limitation encountered in real-world low-resource deployment scenarios (Yang & Shami, 2021; Preuveneers et al., 2020):

| Constraint | Values | Real-world interpretation |
|---|---|---|
| **Monitoring frequency** | 10, 50, 100, 500, 1000 samples | How often the drift check is invoked; constrained by CPU budget or pipeline scheduling |
| **Window size** | 50, 100, 500, 1000 samples | Number of recent stream samples used as the test distribution; constrained by available memory and latency budget |
| **Reference size** | 100, 500, unlimited | Maximum number of reference samples retained in memory; constrained by device memory capacity |

**Monitoring frequency** represents a computational constraint. In production monitoring systems, drift checks are not free — each invocation consumes CPU time and may block the main inference pipeline. A monitoring frequency of 10 means the detector runs after every 10 samples; a frequency of 1000 means it runs once per thousand samples. Higher frequency enables earlier detection but at greater total computational cost.

**Window size** represents both a memory and a computational constraint. A larger window provides more statistical power per check (reducing noise and false positives) but requires retaining more samples in a rolling buffer and performing more comparisons per invocation. A smaller window requires less memory and completes faster but may have insufficient sample size for reliable statistical testing, particularly for PSI which requires adequate counts in each bin.

**Reference size** represents a memory constraint. In a typical deployment, the reference distribution is the training-time feature distribution, and retaining all training samples may be infeasible on memory-constrained devices. When reference size is limited, only the most recent `k` reference samples are retained, simulating a device that cannot store the full historical distribution. The unlimited condition represents the idealised baseline with no memory constraint. When reference size is constrained, the most recent reference samples are retained in preference to the oldest, reflecting a temporal recency bias appropriate to systems that expect slow background drift in the reference itself.

### 3.6.2 Dependent Variables — Detection Performance Metrics

Five dependent variables were measured for each experimental configuration, capturing different facets of drift detection effectiveness and resource cost:

| Metric | Definition | Interpretation |
|---|---|---|
| **Detection delay** | Samples between drift injection and first detection | Primary effectiveness metric; lower is better |
| **False positive rate** | Pre-drift detections / total pre-drift checks | Reliability metric; higher rates cause unnecessary model retraining |
| **Missed detection** | Binary: drift never flagged post-injection | Worst-case failure; model degrades without any alert |
| **Accuracy degradation** | Baseline accuracy − post-drift accuracy | Quantifies model impact; larger = more harm before detection |
| **Computation time** | Cumulative detector execution time (ms) | Resource cost metric; directly measures computational overhead |

**Detection delay** is the primary outcome of interest and the central dependent variable for the constraint analysis. It operationalises the cost of delayed detection: every sample in the delay window represents a prediction made by a degraded model on out-of-distribution input, without any system-level awareness that the model may no longer be reliable.

**Accuracy degradation** complements detection delay by quantifying the model-level consequence of that delay. Two experiments with identical detection delays but different accuracy degradation values indicate that the same detection latency carries different operational costs depending on how quickly the model degrades under the specific drift pattern.

---

## 3.7 Experimental Protocol

For each of the 8,100 experimental configurations in the primary experiment, the following protocol was executed:

1. **Load and preprocess the dataset** using the configuration's random seed to ensure reproducible splitting and shuffling.

2. **Train the classifier** on the reference set and evaluate baseline accuracy and ROC-AUC on the full stream prior to drift injection. This baseline accuracy represents the model's expected performance in the absence of distribution shift.

3. **Inject drift** into the target feature at 50% of the stream length using the specified drift type and magnitude. A different random seed, derived as (base\_seed + repetition\_index), was used for each of the five repetitions to introduce independent stochastic variation across replications.

4. **Measure post-drift model accuracy** by evaluating the trained classifier on the drifted portion of the stream. The difference between baseline and post-drift accuracy defines accuracy degradation.

5. **Construct the reference window** by extracting the target feature values from the reference set, truncating to the specified reference size if applicable. When reference size is limited, the most recent samples are retained.

6. **Execute the monitoring loop**: a window of the specified size is slid across the stream. At each monitoring checkpoint (every `monitoring_frequency` samples), the detector compares the current window against the reference and records whether drift was flagged, along with the computation time for that check. All detection results are accumulated as a list of (stream\_index, is\_drift) pairs.

7. **Compute metrics** from the accumulated detection flags and the known drift start index: detection delay as the gap between the first post-injection True flag and the drift start; false positive rate as the count of True flags before the drift start; missed detection as the absence of any True flag after the drift start.

8. **Save results** incrementally to a CSV file after each configuration completes. Incremental saving ensures that a partial run can be resumed without discarding completed experiments.

Each configuration was replicated five times with different random seeds. This replication level supports non-parametric statistical comparisons across groups while remaining computationally feasible within the available time and hardware constraints.

---

## 3.8 Statistical Analysis

Given the non-normal distribution of detection delay values — evidenced by large standard deviations relative to means, the presence of censored observations (missed detections producing undefined delay), and the discrete nature of the measurement — non-parametric statistical tests were employed throughout.

### 3.8.1 Kruskal-Wallis H-Test

The Kruskal-Wallis test is a non-parametric analogue of the one-way analysis of variance (ANOVA) that tests whether two or more independent groups differ significantly in their rank distributions (Kruskal & Wallis, 1952). It was used as the primary multi-group test for comparing detectors, drift types, and constraint levels. The test produces an H-statistic and associated p-value under the null hypothesis that all groups share the same distribution.

The Kruskal-Wallis test serves as a screening test: a significant result confirms that at least one group differs, motivating subsequent pairwise comparisons to identify which specific pairs are responsible.

### 3.8.2 Pairwise Mann-Whitney U Tests with Bonferroni Correction

Following a significant Kruskal-Wallis result, pairwise Mann-Whitney U tests were conducted for all pairs of groups (Kruskal & Wallis, 1952). The Mann-Whitney test evaluates whether one group's values tend to rank higher than the other's, making no distributional assumptions.

When conducting multiple simultaneous hypothesis tests, the probability of at least one false positive increases with the number of tests. Bonferroni correction was applied to control the family-wise error rate: each raw p-value was multiplied by the number of pairwise comparisons (three, for three groups). Statistical significance was assessed at α = 0.05 after correction.

Observations with missing detection delay (missed detections) were excluded from pairwise tests, as Mann-Whitney requires numeric values. The missed detection outcome was analysed separately as a binary variable.

### 3.8.3 Effect Size

Statistical significance alone is insufficient for interpreting results with large sample sizes: with 2,600–2,700 observations per detector group, even trivially small differences will produce extremely small p-values. Effect size was therefore reported alongside every pairwise test using the **rank-biserial correlation** r:

> r = 1 − (2U) / (n₁ × n₂)

where U is the Mann-Whitney statistic and n₁, n₂ are group sizes (Cohen, 1988). Effect sizes were interpreted using the thresholds: |r| < 0.10 negligible; |r| < 0.30 small; |r| < 0.50 medium; |r| ≥ 0.50 large.

### 3.8.4 Spearman Rank Correlation

For the continuous constraint dimensions (monitoring frequency, window size, reference size), Spearman rank correlation was computed between the constraint value and detection delay (Spearman, 1904). This provides a directional measure of the monotonic relationship between constraint severity and detection performance, complementing the group-level Kruskal-Wallis tests. A positive Spearman ρ indicates that increasing the constraint value corresponds to longer detection delay; a negative ρ indicates the reverse.

---

## 3.9 Implementation

The experimental framework was implemented in Python 3.10. The primary dependencies were: NumPy and pandas for data manipulation, scikit-learn for model training and preprocessing (Random Forest, Logistic Regression, StandardScaler, LabelEncoder), LightGBM for the gradient boosting classifier, SciPy for statistical tests (Kolmogorov-Smirnov, Wasserstein distance, Mann-Whitney U, Kruskal-Wallis, Spearman correlation), and tqdm for progress monitoring during batch execution.

The codebase is organised into seven modules: data loading and preprocessing (`data/`), model training (`models/`), drift simulation (`simulators/`), drift detection (`detectors/`), experiment orchestration (`experiments/`), metric computation (`evaluation/`), and statistical analysis (`evaluation/significance.py`). Each module exposes a clear interface and is unit-tested against a known-outcome smoke test to verify end-to-end correctness before full batch execution.

Experiments were executed in batch mode using an incremental CSV-writing strategy: each completed experiment is written to disk immediately upon completion, so that a hardware or software failure does not result in loss of previously computed results. A resume mechanism detects previously completed experiment keys on startup and skips them, enabling interrupted runs to continue from the point of failure.

All experiments were conducted on a Windows 11 laptop with a standard Intel CPU, without GPU acceleration or cloud resources. The full primary experiment (8,100 configurations × 5 repetitions = 8,100 total runs, given repetitions are folded into the config count) required approximately six hours of wall-clock time. The model comparison experiment (405 configurations) required approximately 25 minutes. These runtimes confirm that the experimental framework is executable within the resource constraints that characterise the deployment scenarios under investigation.

---

## References

Agrahari, S., & Singh, A. (2021). Concept drift detection in data stream mining: A literature review. *Journal of King Saud University – Computer and Information Sciences*, 34, 9523–9540.

Arora, S., Rani, R., & Saxena, N. (2024). A systematic review on detection and adaptation of concept drift in streaming data using machine learning techniques. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 14.

Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B*, 20(2), 215–232.

Gemaque, R., Costa, A., Giusti, R., & Santos, E. (2020). An overview of unsupervised drift detection methods. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 10.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621.

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, 31, 2346–2363.

Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association*, 46(253), 68–78.

Matzka, S. (2020). Explainable artificial intelligence for predictive maintenance applications. *2020 Third International Conference on Artificial Intelligence for Industries*.

Preuveneers, D., Tsingenopoulos, I., & Joosen, W. (2020). Resource usage and performance trade-offs for machine learning models in smart environments. *Sensors*, 20(4), 1176.

Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72–101.

Suárez-Cetrulo, A., Quintana, D., & Cervantes, A. (2022). A survey on machine learning for recurring concept drifting data streams. *Expert Systems with Applications*, 213, 118934.

Taplin, R., & Hunt, C. (2019). The population accuracy index: A new measure of population stability for model monitoring. *Risks*, 7(2), 53.

Villani, C. (2009). *Optimal transport: Old and new*. Springer.

Yang, L., & Shami, A. (2021). A lightweight concept drift detection and adaptation framework for IoT data streams. *IEEE Internet of Things Magazine*, 4, 96–101.

Yeh, I.-C. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
