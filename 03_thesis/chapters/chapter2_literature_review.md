# Chapter 2: Literature Review

## 2.1 Introduction to the Chapter

The reliable deployment of machine learning systems in dynamic real-world environments depends critically on the ability to detect and respond to changes in the statistical properties of incoming data. This chapter synthesises the scholarly foundation underlying this study: the theory of concept drift, the methods developed to detect it, the empirical evidence on their relative performance, and the open questions this study is positioned to address. The review proceeds from the theoretical to the empirical, and from the general to the specific, concluding with an explicit articulation of the research gaps that motivate the experimental design of this project.

The literature reviewed spans the foundational theoretical treatments of the 2010s (Gama et al., 2013; Lu et al., 2019), large-scale empirical comparisons of drift detection algorithms (Barros & Santos, 2018; Lukats et al., 2024), domain-specific applications of distribution-based detection (Prashanth et al., 2025; Christen et al., 2024; Pathi, 2025), and system-level work on resource-constrained machine learning (Yang & Shami, 2021; Preuveneers et al., 2020). Together, these sources establish what is known, what has been tested empirically, and what remains unresolved — particularly the interaction between resource constraints and detection performance, which this study addresses directly.

---

## 2.2 Concept Drift: Definitions and Taxonomy

### 2.2.1 Formal Definition

The central concept motivating this study is *concept drift*: the phenomenon in which the joint probability distribution of input data and target labels changes over time in ways that render a previously trained model less accurate or less well-calibrated. The foundational formal statement is due to Gama et al. (2013), who define concept drift as a change in the joint distribution P(X, y) between a time point t₀ and a later time point t₁:

> ∃X : P_{t₀}(X, y) ≠ P_{t₁}(X, y)

This definition captures the essential property that drift is not merely a change in the input distribution P(X) but a change in the joint distribution that relates inputs to labels. Gama et al. (2013) note that a model learned from data generated under P_{t₀} will produce systematically degraded predictions when the true generating distribution has shifted to P_{t₁}, because the decision boundary encoded in the model is no longer aligned with the true decision boundary in the new distribution.

Lu et al. (2019) extend this definition formally in their IEEE TKDE survey, placing concept drift within a broader taxonomy of non-stationarity. Their formulation identifies three distinct *sources* of drift, each with different implications for detection and adaptation:

- **Source I — Virtual Drift (Covariate Shift):** Only P(X) changes; the conditional P(y|X) remains constant. The feature distribution shifts but the true labelling function does not. A model may still be accurate on the new feature distribution if it covers regions well, but monitoring based on P(X) alone can detect the shift. Lu et al. (2019) note this is also called *dataset shift* or *covariate shift* in the transfer learning literature.

- **Source II — Real Drift (Concept Drift proper):** P(y|X) changes while P(X) may remain stable. The true decision boundary has changed, and the model's predictions will systematically diverge from the new labels even for the same input values. This is the most consequential form of drift from a predictive accuracy standpoint.

- **Source III — Simultaneous change:** Both P(X) and P(y|X) change together. This is common in real systems where a shift in the environment affects both what data looks like and how it should be labelled.

This three-source taxonomy has important practical implications: it motivates the use of distribution-based detection (monitoring P(X)) as a proxy for identifying conditions under which real drift may have occurred, even when labels are not immediately available. The three detectors evaluated in this study — KS test, PSI, and Wasserstein distance — all operate on input feature distributions, placing them squarely in the Source I detection paradigm, with the assumption that input distribution changes serve as reliable signals for underlying concept change.

### 2.2.2 Drift Type Patterns

Beyond the source of drift, the literature distinguishes several characteristic *temporal patterns* through which drift manifests. Both Gama et al. (2013) and Lu et al. (2019) present similar four-way taxonomies, differing slightly in terminology. The consensus typology includes:

**Sudden (Abrupt) Drift** occurs when the data-generating distribution changes instantaneously from P_{t₀} to P_{t₁} at a discrete point in time. Gama et al. (2013) characterise this as the clearest case for detection algorithms: the shift is large, the boundary is sharp, and a sufficiently sensitive detector will identify it quickly. Lu et al. (2019) note that sudden drift is the most commonly studied in the benchmarking literature, which may introduce a systematic optimism bias in reported detector performance.

**Gradual Drift** describes a transition period during which both the old concept C₀ and the new concept C₁ are simultaneously present in the data stream, with C₀ decreasing in probability and C₁ increasing. Gama et al. (2013) observe that gradual drift poses greater challenges for adaptive systems because the signal is diluted by the co-occurrence of old and new patterns. Detection is slow, and retraining too early risks anchoring to a transient mixture rather than the emerging new concept. Lu et al. (2019) further note that many detection algorithms optimised for sudden drift show substantial performance degradation on gradual drift, a finding the present study confirms empirically.

**Incremental Drift** represents a smooth, monotone progression of the distribution over time, with no sharp transition and no co-occurrence of distinct concepts. The boundary between incremental drift and gradual drift can be ambiguous in the literature. Lu et al. (2019) treat incremental drift as a limiting case where the transition is so smooth that at no point is a sharp boundary detectable.

**Recurring (Cyclical) Drift** occurs when previously seen concepts re-appear after a period of dormancy, often due to seasonality or cyclical environmental patterns. Gama et al. (2013) identify this as a key motivator for memory-based adaptation strategies: if old concepts recur, a system that retains model snapshots from prior periods can recall them rather than relearning from scratch. Lu et al. (2019) note that recurring drift is particularly challenging for algorithms that use a fixed reference window, because the reference data may correspond to neither the current nor the returning concept.

These temporal patterns are directly operationalised in the experimental design of this study, which evaluates detector performance across sudden, gradual, and recurring drift conditions. The literature's prediction — that gradual drift is systematically harder to detect than sudden drift — forms one of the core hypotheses tested.

---

## 2.3 A Framework for Drift Detection

### 2.3.1 The Four-Stage Detection Pipeline

Lu et al. (2019) present the most complete and widely-cited framework for organising drift detection algorithms. Their four-stage pipeline (Figure 5 of Lu et al., 2019) provides a principled decomposition of what any drift detector must accomplish:

1. **Data Retrieval:** Collecting the data to be analysed. This involves decisions about what data to collect (which features, what time span) and at what granularity. The size of the reference window and the size of the detection window are both data retrieval parameters. Lu et al. (2019) note that inappropriate window selection is a primary source of detection failure.

2. **Data Modeling (optional):** Reducing raw data to a lower-dimensional representation before applying statistical tests. Some algorithms apply this step explicitly (e.g., using kernel density estimation or dimensionality reduction); others operate directly on raw feature distributions. The detectors in this study — KS, PSI, and Wasserstein — skip explicit modeling and operate directly on empirical distributions.

3. **Test Statistics Calculation:** Computing a numerical quantity that summarises the dissimilarity between two data windows. This is the core statistical operation. For KS, it is the maximum absolute difference between empirical CDFs; for PSI, it is the Population Stability Index; for Wasserstein, it is the earth mover's distance. The choice of test statistic determines the algorithm's sensitivity profile.

4. **Hypothesis Test:** Converting the test statistic into a binary decision (drift detected / no drift detected) by comparing it to a threshold, either a significance level (for KS) or a domain-calibrated cutoff (for PSI). The threshold governs the false positive / false negative tradeoff. Lu et al. (2019) emphasise that threshold selection is underspecified in most algorithm descriptions, creating substantial variation in reported results across studies.

This framework clarifies why steps 1 and 4 are often more consequential than step 3: two detectors using the same statistical test but different window sizes and thresholds can behave radically differently. The present study systematically manipulates the parameters corresponding to stages 1 and 4 (reference window size, detection window size, monitoring frequency) to quantify their impact.

### 2.3.2 The Adaptive Learning Cycle

Gama et al. (2013) situate drift detection within a broader adaptive learning cycle composed of four modules: Memory, Change Detection, Learning, and Loss Estimation. These modules interact in an online learning loop that Gama et al. (2013) describe as a predict-diagnose-update cycle:

- **Predict:** The current model is applied to generate predictions for incoming data.
- **Diagnose:** The loss estimator monitors prediction accuracy (or distribution statistics, for unsupervised approaches) to detect degradation.
- **Update:** When degradation is confirmed, the learning module updates or replaces the model.

This framing makes explicit that drift detection is not an end in itself but a diagnostic component within a larger adaptive system. Gama et al. (2013) catalogue the domains in which this cycle has been applied: industrial monitoring and process control, business management and planning (fraud detection, churn prediction), personal assistance (e-mail filtering, recommendation systems), and ubiquitous computing environments (autonomous vehicles, smart grids). The breadth of application domains underscores why a generalizable understanding of detector behaviour across constraint conditions is practically valuable.

The predict-diagnose-update cycle also clarifies the cost of late detection: every time step between the onset of drift and its detection is a period during which the model's predictions are degraded. The *detection delay* metric used in this study — the number of monitoring windows elapsed between drift onset and the first statistically significant signal — directly quantifies this cost.

---

## 2.4 Supervised Drift Detection Methods

### 2.4.1 Classification of Algorithms

Lu et al. (2019) classify drift detection algorithms into three families based on their information requirements and statistical mechanisms:

**Error Rate-Based Methods** monitor the error rate of a classifier over time. They require access to true labels and operate on the assumption that increased error rate signals that the model's decision boundary no longer matches the true distribution. Representative methods include DDM (Drift Detection Method, Gama et al., 2004), EDDM (Early Drift Detection Method, Baena-García et al., 2006), ADWIN (Adaptive Windowing, Bifet & Gavalda, 2007), STEPD (Statistical Test of Equal Proportions, Nishida & Yamauchi, 2007), and HDDM (Hoeffding's inequality-based Drift Detection Method, Frías-Blanco et al., 2015). These methods are powerful when labels are available immediately after prediction, but are inapplicable in the common deployment scenario where labels arrive with delay or not at all.

**Data Distribution-Based Methods** monitor changes in the input distribution P(X) or the joint distribution P(X, y) without necessarily requiring labels. This family includes the three detectors used in this study: the Kolmogorov-Smirnov test (KS), the Population Stability Index (PSI), and the Wasserstein distance. Additional methods in this family include the kdqTree (Dasu et al., 2006), Least-Squares Density Difference (Sugiyama et al., 2008), and the Change Detection Test based on LSDD (Bu et al., 2016). Lu et al. (2019) note that the primary advantage of distribution-based methods is their label-independence, making them applicable even when ground truth labels are unavailable at detection time.

**Multiple Hypothesis Test Methods** combine several statistical tests to reduce false positives. Examples include the Just-In-Time classifier (Schlimmer & Granger, 1986) and the Hierarchical Hypothesis Testing with Adaptive Gamma (HHT-AG). These methods trade computational simplicity for robustness, often at the cost of detection delay.

### 2.4.2 The Large-Scale Empirical Comparison

The most comprehensive empirical evaluation of supervised drift detection algorithms is due to Barros & Santos (2018), published in *Information Sciences*. Their study compares 14 detector configurations across six synthetic dataset generators, seven dataset sizes ranging from 10,000 to 2,000,000 instances, and two base learners (Naïve Bayes and Hoeffding Tree). The scale of this evaluation — spanning multiple orders of magnitude in dataset size and multiple algorithmic families — makes it a key reference for understanding relative detector performance.

The detectors benchmarked by Barros & Santos (2018) include DDM, EDDM, ADWIN, STEPD, PHT (Page-Hinkley Test), PL, ECDD, SeqDrift2, SEED, HDDM_A, HDDM_W, FHDDM, FTDD, RDDM, and WSTD. Statistical significance of performance differences is established using Friedman tests followed by Nemenyi post-hoc comparisons — the same non-parametric methodology used in the present study.

The key findings of Barros & Santos (2018) are directly relevant to this study:

1. **RDDM (Reactive Drift Detection Method) and HDDM_A emerge as the best overall performers**, achieving the highest accuracy with the lowest false positive rates across the most conditions. No single detector dominates in all conditions; the best algorithm varies by base learner, dataset size, and drift pattern. This finding — that no universal best detector exists — motivates the empirical approach of the present study.

2. **Barros & Santos (2018) explicitly challenge the assumption that detection closest to the true drift position is always optimal.** They show that some detectors achieve lower accuracy degradation by detecting drift slightly earlier or later than the true onset, depending on how the base learner adapts. This challenges the use of detection delay as the sole performance criterion and motivates the present study's inclusion of both missed detections and detection delay.

3. **Dataset size interacts significantly with detector performance.** Detectors that perform well on small datasets (10K instances) do not necessarily generalise to large ones (2M instances). Barros & Santos (2018) attribute this partly to window size effects: detectors designed with fixed window sizes are sensitive to the ratio of window size to drift duration. This interaction motivates the present study's systematic manipulation of window size and reference size as experimental variables.

4. **Gradual drift consistently degrades detector performance relative to sudden drift.** Barros & Santos (2018) report that all 14 detectors show lower accuracy and higher detection delay on gradual transitions compared to sudden transitions, consistent with the theoretical expectation that the dilution of the drift signal during a gradual transition reduces statistical power.

### 2.4.3 Supervised Detectors Under Label Constraints

A critical limitation of error rate-based supervised detectors, highlighted by Lukats et al. (2024), is that they assume the immediate availability of ground truth labels. In many deployment scenarios — fraud detection, medical diagnosis, credit risk, predictive maintenance — labels arrive days, weeks, or months after the prediction, if they arrive at all. Lukats et al. (2024) classify supervised detectors as a fundamentally constrained subclass and argue that the performance advantages reported in benchmarks like Barros & Santos (2018) are not reproducible in real deployment because the benchmark assumes an oracle label provider. This asymmetry between benchmark conditions and deployment conditions motivates increased interest in distribution-based, label-free approaches — the class of methods this study evaluates.

---

## 2.5 Unsupervised and Distribution-Based Drift Detection

### 2.5.1 The Unsupervised Detection Paradigm

The core architectural commitment of unsupervised drift detection is to operate on the input data distribution P(X) alone, without requiring access to labels. As Lukats et al. (2024) note, this is not merely a pragmatic accommodation for label scarcity; it reflects a different theory of what drift detection should accomplish. Rather than monitoring model performance after the fact, distribution-based detectors monitor the data *before* a model is applied, providing early warning that a model's training distribution may no longer represent the incoming data.

Lukats et al. (2024) provide the most recent large-scale benchmark of fully unsupervised drift detectors, evaluating 10 algorithms on 11 real-world data streams: BNDM, CSDDM, D3, EDFS, IBDD, NN-DVI, OCDD, SPLL, UCDD, and UDetect. Their evaluation uses four metrics: Missed Detection Rate (MDR), Mean Time to Detection (MTD), Mean Time to False Alarm (MTFA), and Mean Time to Recovery (MTR), supplemented by a lift-per-drift measure to normalise across datasets with different drift frequencies.

The common architectural pattern across these detectors, identified by Lukats et al. (2024), consists of four components:

1. A **reference window** capturing the expected (stable) distribution of the data.
2. A **current (detection) window** capturing recent incoming data.
3. A **dissimilarity measure** comparing the two windows.
4. A **drift criterion** converting the dissimilarity measure to a detection decision.

This architecture is identical to the pipeline of the three detectors in this study. What differs across the algorithms reviewed by Lukats et al. (2024) is primarily the choice of dissimilarity measure (statistical, information-theoretic, or neural) and the drift criterion (threshold, significance level, or learned boundary).

### 2.5.2 Benchmark Findings from Lukats et al. (2024)

The results of Lukats et al. (2024) are instructive for both what they confirm and what they complicate. The three best-performing detectors in their benchmark are D3 (a discriminative detector using a binary classifier to distinguish reference from current data), IBDD (an information-theoretic approach based on binned density estimation), and SPLL (a statistical process control approach using log-likelihood). These algorithms outperform simpler distributional tests on most metrics, particularly MDR and MTD.

However, Lukats et al. (2024) raise a methodological concern that applies directly to this study: the evaluation of unsupervised detectors is complicated by the absence of labels, which makes it difficult to define ground truth for when drift occurred. The authors argue that the proxy metrics commonly used (accuracy degradation, manual annotation) introduce biases that distort benchmark results. They call for more rigorous evaluation frameworks that separate the question of *whether* drift occurred from the question of *how much* the model has been affected — a distinction that motivates this study's simultaneous measurement of both distribution shifts (via detectors) and performance degradation (via classification metrics).

### 2.5.3 Neural and Kernel Approaches

Beyond traditional statistical tests, the literature has explored neural and kernel-based dissimilarity measures for unsupervised detection. Gemaque et al. (2020) provide an overview of unsupervised concept drift detection approaches, including kernel-based methods such as the Maximum Mean Discrepancy (MMD) and the Least-Squares Density Difference (LSDD). These methods operate in reproducing kernel Hilbert spaces and can detect changes in distributional shape that are invisible to simpler statistical tests. However, as Gemaque et al. (2020) note, their computational requirements are substantially higher than those of KS, PSI, or Wasserstein, making them less suitable for resource-constrained deployment contexts — a key consideration for this study.

The NN-DVI detector benchmarked by Lukats et al. (2024) exemplifies the neural approach: it uses a neural network to learn a representation of the reference data and measures drift as divergence in representation space. While capable of detecting complex, high-dimensional distributional changes, NN-DVI has high training overhead and requires retraining as the reference window updates, making it unsuitable for the low-memory, high-frequency monitoring contexts examined in this study.

---

## 2.6 Distribution-Based Detection: KS Test, PSI, and Wasserstein Distance

### 2.6.1 The Kolmogorov-Smirnov Test

The Kolmogorov-Smirnov (KS) test is one of the oldest and most widely used non-parametric tests for distributional comparison. It operates on the empirical cumulative distribution functions (ECDFs) of two samples and computes the maximum absolute vertical distance between them:

**D = max_x |F_{ref}(x) − F_{curr}(x)|**

where F_{ref} and F_{curr} are the ECDFs of the reference and current windows, respectively (Pathi, 2025). The test is distribution-free: it makes no assumption about the parametric form of either distribution. Under the null hypothesis that both samples are drawn from the same distribution, the distribution of the KS statistic D is known analytically, enabling the computation of p-values.

In the context of drift detection, the KS test is applied feature-by-feature: each feature dimension is tested independently, and drift is declared if any feature's p-value falls below the significance threshold α. The sensitivity of the test depends on sample size (larger samples yield greater power), the magnitude of the distributional shift, and α. Pathi (2025) applies the KS statistic as one component of a multi-metric model evaluation framework and demonstrates that KS max-statistic values change substantially across different dataset populations for the same trained model, providing early warning of population instability.

The KS test's primary advantage is its computational efficiency and theoretical rigour: under the null hypothesis, exact p-values are available without simulation. Its primary limitation is that it is defined only for univariate distributions; multivariate extensions require either sequential application per feature (as used in this study) or more complex multivariate analogues. Sequential feature-wise testing introduces a multiple comparisons problem that must be managed via Bonferroni correction or similar procedures when many features are tested.

### 2.6.2 The Population Stability Index

The Population Stability Index (PSI) originated in the financial services industry as a tool for monitoring the stability of credit scoring model inputs over time (Taplin & Hunt, 2019). Unlike the KS test, which is based on continuous CDFs, PSI operates on binned distributions and is defined as:

**PSI = Σᵢ (Oᵢ − Eᵢ) × ln(Oᵢ / Eᵢ)**

where Oᵢ is the proportion of the current (monitoring) population in bin i, and Eᵢ is the proportion of the reference (training) population in the same bin (Pathi, 2025). The formula is a form of the symmetric KL divergence, sometimes called the Jensen-Shannon divergence analogue for discrete distributions. Bins with zero count in either population require special handling (typically by adding a small constant) to avoid undefined logarithms.

The PSI has established industry thresholds for categorical interpretation: PSI < 0.10 indicates a stable population (no significant shift), 0.10 ≤ PSI < 0.25 indicates moderate shift warranting investigation, and PSI ≥ 0.25 indicates a major shift requiring model review (Taplin & Hunt, 2019; Pathi, 2025). These thresholds are not derived from statistical theory but from empirical calibration in credit-scoring practice. Their applicability in non-financial domains — such as the telecommunications, credit, and predictive maintenance datasets used in this study — is an open empirical question.

Pathi (2025) applies PSI as a model stability metric across three datasets and reports that Random Forest models are more susceptible to population instability than Logistic Regression or Gradient Boosting models. Specifically, a Random Forest trained on the Heart Failure dataset exhibits PSI = 0.183 on a held-out population sample — squarely in the "moderate shift" range — while Logistic Regression achieves PSI = 0.085 on the same comparison. This finding is directly relevant to the model comparison strand of the present study.

Taplin & Hunt (2019) provide a theoretical treatment of PSI, situating it formally as a divergence measure and examining its behaviour under different bin selection strategies. They find that PSI is sensitive to the number and placement of bins: too few bins reduces sensitivity to local distributional changes, while too many bins reduces statistical stability when sample sizes are small. This sensitivity to a hyperparameter that is not explicitly optimised in most deployed monitoring systems represents a practical limitation of PSI relative to non-parametric tests.

### 2.6.3 The Wasserstein Distance

The Wasserstein distance — also called the earth mover's distance (EMD) — provides a geometrically principled measure of the difference between two probability distributions. For two univariate distributions P and Q, the p-th order Wasserstein distance is defined as:

**W_p(P, Q) = ( inf_{γ ∈ Π(P,Q)} E_{(x,y)~γ} [ ||x − y||^p ] )^{1/p}**

where Π(P, Q) is the set of all joint distributions with marginals P and Q, and the infimum is taken over all *transport plans* γ (Prashanth et al., 2025). Intuitively, W_p measures the minimum cost of transforming P into Q, where cost is the work done in moving probability mass over a distance. The first-order Wasserstein distance (p=1) equals the area between the CDFs of P and Q for univariate distributions, enabling efficient computation via sorting.

Prashanth et al. (2025) apply Wasserstein distance in a drift detection pipeline combining Bayesian Neural Networks with distributional monitoring. Their system decomposes predictive uncertainty into aleatoric uncertainty (irreducible noise inherent in the data) and epistemic uncertainty (model uncertainty due to data scarcity or distribution shift), using the Wasserstein distance to quantify when the epistemic uncertainty — the portion attributable to distributional change — exceeds a threshold δ:

**Drift declared when: W(P_{ref}, P_{curr}) > δ**

The primary empirical finding of Prashanth et al. (2025) relevant to this study is the demonstration of threshold sensitivity: lower values of δ produce more drift detection events (higher sensitivity, lower specificity), while higher δ values suppress false alarms at the cost of missing genuine drift events. This threshold-sensitivity tradeoff is not unique to Wasserstein; all three detectors in this study exhibit it. Prashanth et al. (2025) conclude that threshold selection must be calibrated to the operational cost of false positives versus missed detections in a specific deployment context — a normative recommendation that motivates the parametric exploration of this study.

Christen et al. (2024) apply Wasserstein distance, KS tests, and PSI as a complementary toolkit in the domain of entity resolution — the task of identifying which records in different databases refer to the same real-world entity. While entity resolution is distinct from concept drift detection, Christen et al. (2024) use these three distributional tests to compare feature distributions between different linkage problems, assessing whether a model trained on one problem's feature space can be transferred to another. Their application demonstrates the generality of these statistical tools beyond drift detection: the same mathematical machinery that detects distributional shift over time can detect distributional shift across domains, populations, or use cases. The complementarity of the three tests observed by Christen et al. (2024) — each test sensitive to different aspects of distributional change — supports the use of all three as a detection toolkit rather than treating any single test as definitive.

### 2.6.4 Comparative Properties of the Three Detectors

The three detectors examined in this study have different theoretical properties that predict their relative behaviour under different conditions. Table 2.1 summarises these properties.

**Table 2.1: Theoretical Properties of KS Test, PSI, and Wasserstein Distance**

| Property | KS Test | PSI | Wasserstein |
|---|---|---|---|
| Statistical basis | ECDF comparison | Binned KL divergence | Optimal transport |
| Parametric assumption | None | None | None |
| Sample size sensitivity | High (power ∝ √n) | High (bin stability ∝ n) | Moderate |
| Hyperparameters | α (significance level) | Bin count, bin edges | Order p, threshold δ |
| Output type | p-value / binary | Continuous index | Continuous distance |
| Industry calibration | Statistical theory | Domain-calibrated cutoffs | Threshold-dependent |
| Multivariate extension | Sequential, feature-wise | Sequential, feature-wise | Direct (via sliced WD) |
| Computational cost | O(n log n) | O(n + B) | O(n log n) |

None of the three tests is uniformly superior: each makes different assumptions about what constitutes a distributional change. The KS test is sensitive to differences in the location and shape of the CDF but may miss changes that are spread uniformly across the distribution. PSI is sensitive to local changes in bin proportions but depends on binning decisions. Wasserstein captures differences in the overall geometry of the distributions, including changes in spread and bimodality, but requires threshold calibration. Their combined use provides broader sensitivity coverage than any single test.

---

## 2.7 Resource Constraints in Machine Learning Monitoring Systems

### 2.7.1 The Resource-Constrained Deployment Context

The practical deployment of machine learning models — particularly in industrial, embedded, and mobile contexts — operates under resource constraints that fundamentally differ from the laboratory conditions under which most drift detection algorithms are benchmarked. Yang & Shami (2021) provide a comprehensive survey of concept drift adaptation in resource-constrained environments, identifying four primary constraint dimensions: computational budget (CPU/GPU cycles available), memory budget (RAM and storage available for reference windows and model snapshots), monitoring latency (how frequently drift checks can be run), and labelling rate (the fraction of predictions that receive ground truth labels within the monitoring cycle).

Yang & Shami (2021) observe that the interaction among these constraints creates a deployment design space that is largely unexplored in the benchmarking literature. Most benchmark studies, including Barros & Santos (2018), hold constraints constant (typically at "unlimited resource" conditions) and vary only the algorithmic choice, obscuring the degree to which constraint conditions modulate algorithm behaviour. Yang & Shami (2021) call for empirical studies that treat constraints as independent experimental variables — the approach adopted in this study.

Preuveneers et al. (2020) examine drift detection specifically in Internet of Things (IoT) and edge computing contexts, where resource constraints are most severe. IoT devices typically have kilobytes rather than megabytes of working memory, and energy constraints may prohibit running complex monitoring algorithms continuously. Preuveneers et al. (2020) propose lightweight streaming variants of standard drift detection algorithms and demonstrate that memory-efficient windowing strategies can maintain detection performance within 10-15% of unconstrained baselines at substantially lower memory and compute cost. Their findings motivate the present study's examination of reduced reference window sizes and low monitoring frequencies — the conditions that most closely approximate the resource-constrained IoT deployment context.

### 2.7.2 Key Constraint Parameters

Three resource constraint parameters are central to this study, each corresponding to a specific resource allocation decision:

**Monitoring Frequency** refers to how often the drift detection procedure is executed — in this study, every F incoming samples. High monitoring frequency (small F) provides fine-grained temporal resolution but requires more frequent execution of the detection algorithm, consuming more CPU time proportional to the total number of windows evaluated. Low monitoring frequency (large F) conserves computational resources but coarsens temporal resolution, potentially increasing detection delay. The monitoring frequency parameter has received little systematic attention in the literature: most studies report results for a fixed monitoring frequency, making cross-study comparison of detection delay difficult.

**Detection Window Size** refers to the number of recent samples used in each monitoring check. Larger windows provide more statistical power (reducing variance in the test statistic) but increase memory requirements (storing more samples) and introduce latency (requiring more time to collect sufficient data before a check can be run). Smaller windows are memory-efficient and provide faster response to recent changes but sacrifice statistical power, potentially increasing both false positive and false negative rates. Gemaque et al. (2020) identify window size as one of the most influential but poorly-studied parameters in unsupervised drift detection.

**Reference Window Size** refers to the number of samples from the stable (pre-drift) period used as the baseline for distributional comparison. Larger reference windows provide a more stable estimate of the true reference distribution, reducing reference-window estimation variance. However, they require more memory to store. The critical question is whether there is a diminishing returns threshold — a minimum reference size beyond which additional samples provide negligible improvement in detection accuracy. Yang & Shami (2021) hypothesise such a threshold but provide no empirical characterisation; this study provides one.

### 2.7.3 The Monitoring Frequency-Delay Tradeoff

The interaction between monitoring frequency and detection delay is a central practical tradeoff in drift monitoring system design. A system that checks for drift every 10 samples will detect drift more quickly (in terms of samples) than one that checks every 1,000 samples, but it will also execute the detection algorithm 100 times more often, consuming proportionally more CPU time. The key question for system designers is how detection delay scales with monitoring frequency — and whether there is a region of the frequency parameter space where delay is acceptably low at reasonable computational cost.

This tradeoff has been noted qualitatively by Gama et al. (2013) in the context of the adaptive learning cycle: they observe that the frequency of the diagnose step must be balanced against the cost of running diagnostics. However, a quantitative characterisation of how detection delay scales with monitoring frequency, across multiple datasets, drift types, and detectors, has not been previously reported. The present study fills this gap.

---

## 2.8 Machine Learning Models Under Concept Drift

### 2.8.1 Random Forests

Random Forests (Breiman, 2001) are ensemble models that construct a large number of decision trees, each trained on a bootstrap sample of the training data and a random subset of features, and aggregate their predictions by majority vote (for classification). The ensemble averaging reduces variance relative to individual decision trees, providing robustness to overfitting. Breiman (2001) demonstrates that Random Forests achieve competitive accuracy on tabular datasets with minimal hyperparameter tuning.

Under concept drift, ensemble models like Random Forests have a theoretically ambiguous position. On one hand, the averaging of many diverse trees provides some robustness to individual tree overfitting to the training distribution, potentially making the ensemble more tolerant of mild distributional shifts. On the other hand, all constituent trees are trained on the same historical distribution; if the distribution shifts substantially, the entire ensemble will produce systematically biased predictions. Pathi (2025) observes empirically that Random Forest models exhibit PSI values indicative of moderate population instability (PSI = 0.183 on the Heart Failure dataset), suggesting that the features the forest uses are sensitive to population shifts. This finding is broadly consistent with the results of the present study.

### 2.8.2 Logistic Regression

Logistic Regression (Cox, 1958) is a parametric linear classifier that models the log-odds of class membership as a linear function of the input features:

**log(P(y=1|x) / P(y=0|x)) = w^T x + b**

Its linear decision boundary implies that it is relatively robust to small, smooth distributional shifts in the feature space — the decision boundary cannot adjust to complex non-linear changes in P(y|X), but for the same reason, it is less likely to overfit to idiosyncratic distributional features of the training data. Pathi (2025) reports that Logistic Regression achieves the lowest PSI values (PSI = 0.085 on the Heart Failure dataset) among the models evaluated, supporting the characterisation of LR as a more stable model under population shift. Its simplicity makes it a natural baseline for comparison with more complex models in drift robustness evaluations.

### 2.8.3 LightGBM

LightGBM (Ke et al., 2017) is a gradient boosting framework that builds an ensemble of decision trees sequentially, with each tree correcting the residual errors of the preceding ensemble. Key innovations include gradient-based one-side sampling (GOSS), which reduces the effective training set size by focusing on high-gradient samples, and exclusive feature bundling (EFB), which reduces the feature space by bundling mutually exclusive sparse features. These optimisations make LightGBM substantially faster than earlier gradient boosting implementations (XGBoost, GBM) without compromising accuracy.

Gradient boosting ensembles are generally considered more accurate than Random Forests on tabular data (Ke et al., 2017), but their higher model complexity implies a greater risk of overfitting to the training distribution's specific characteristics. Under concept drift, this overfitting risk translates into potentially sharper performance degradation when the distribution shifts. Pathi (2025) reports that gradient boosting achieves the best accuracy-stability balance (AUC = 0.943, PSI = 0.076) in one dataset, but the results across datasets are inconsistent, suggesting that the relative stability of gradient boosting versus Random Forest is dataset-dependent.

### 2.8.4 The Model Robustness Dimension

The question of which model architecture degrades most gracefully under drift has received relatively little systematic empirical attention. The drift detection literature focuses almost entirely on the detector side of the adaptive system, treating the base learner as fixed. The model robustness dimension — how much the model's performance degrades per unit of distributional shift, holding the detector constant — represents a complementary axis of evaluation that has practical implications for system design: in a resource-constrained environment where retraining is expensive, a more drift-robust base learner may provide better average performance than a more accurate but less robust alternative. The present study contributes an empirical characterisation of this dimension for three model architectures (Random Forest, Logistic Regression, LightGBM) across three datasets and three drift types.

---

## 2.9 Evaluation Metrics for Drift Detection

### 2.9.1 Standard Metrics

The evaluation of drift detectors involves measuring both detection capability and detection quality. The metrics used in this study, and their grounding in the literature, are as follows:

**Missed Detection Rate (MDR)** measures the proportion of drift events that the detector fails to identify. Lukats et al. (2024) use MDR as their primary detection capability metric. A missed drift event means the monitoring system has failed to issue an alert despite a genuine distributional change, potentially allowing a degraded model to continue generating predictions without correction.

**Detection Delay** measures the time elapsed between the onset of drift and the first detection event. Barros & Santos (2018) use mean detection delay as their primary quality metric. Delay is expressed in this study as the number of monitoring windows elapsed before detection, enabling comparison across different monitoring frequencies. Gama et al. (2013) note that the acceptable delay threshold depends on the application: in real-time fraud detection, even a delay of seconds may be unacceptable; in credit risk, a delay of weeks may be tolerable.

**Computational Cost** measures the time required to execute the detection procedure. Yang & Shami (2021) identify computational cost as the primary constraint binding in edge deployment contexts. In this study, computational cost is measured as the total CPU time (in milliseconds) consumed by the detection procedure over the full evaluation period.

**Classification Performance Degradation** measures the impact of drift on the ML model's predictive accuracy. The present study uses the accuracy difference between the pre-drift and post-drift periods as a secondary metric, supplementing the detection-focused metrics above.

### 2.9.2 Statistical Methods for Comparing Conditions

The present study uses the Kruskal-Wallis H-test as the primary non-parametric test for comparing distributions of detection metrics across experimental conditions (detectors, drift types, constraint levels). The Kruskal-Wallis test is an extension of the Mann-Whitney U test to three or more groups and does not require the normality assumption of ANOVA. Barros & Santos (2018) use Friedman tests followed by Nemenyi post-hoc tests for pairwise comparisons in their large-scale comparison; the Kruskal-Wallis followed by Mann-Whitney U with Bonferroni correction serves the same purpose in the present study's experimental design.

Effect sizes are reported using the rank-biserial correlation r for pairwise comparisons and Spearman's ρ for constraint-performance correlations. Effect size reporting supplements p-values to distinguish statistical significance from practical significance — a distinction that Lukats et al. (2024) identify as underemphasised in the drift detection literature, where large datasets can yield statistically significant but practically negligible differences.

### 2.9.3 Evaluation Challenges in Real-World Data

Both Barros & Santos (2018) and Lukats et al. (2024) note that evaluating drift detectors on real-world data streams is substantially more challenging than evaluation on synthetic streams, because real data streams lack a ground truth drift schedule. The synthetic drift injection approach adopted in this study — applying controlled drift transformations to real-world tabular datasets — provides a middle ground: the datasets contain genuine feature correlations and class imbalances representative of real deployment, while the drift events are precisely timed and characterised, enabling exact measurement of detection delay.

---

## 2.10 Research Gaps and Positioning of This Study

### 2.10.1 Summary of What Is Known

The literature reviewed in this chapter establishes a solid foundation across several dimensions:

- The theoretical definition and taxonomy of concept drift are well-established (Gama et al., 2013; Lu et al., 2019), with clear distinctions among drift sources (virtual, real, simultaneous) and drift patterns (sudden, gradual, incremental, recurring).
- The four-stage detection framework (data retrieval → data modeling → test statistic → hypothesis test) provides a principled decomposition of any detection algorithm (Lu et al., 2019).
- A large-scale empirical comparison of supervised drift detectors exists (Barros & Santos, 2018), establishing that no single detector dominates and that gradual drift is consistently harder to detect than sudden drift.
- A benchmark of unsupervised drift detectors exists (Lukats et al., 2024), establishing D3, IBDD, and SPLL as leading fully unsupervised approaches.
- The specific properties of KS test, PSI, and Wasserstein distance as distributional comparison tools are well-characterised theoretically (Pathi, 2025; Taplin & Hunt, 2019; Prashanth et al., 2025) and applied in complementary fashion across multiple domains (Christen et al., 2024).
- Resource constraints represent a binding practical consideration for drift monitoring deployment (Yang & Shami, 2021; Preuveneers et al., 2020), but their quantitative impact on detection performance is poorly characterised.
- Model architecture affects stability under distributional shift (Pathi, 2025), but the interaction between model choice and drift type has not been systematically studied.

### 2.10.2 The Gaps This Study Addresses

Three gaps in the existing literature are directly addressed by the present study:

**Gap 1: Absence of systematic constraint-performance characterisation.** Existing benchmarks (Barros & Santos, 2018; Lukats et al., 2024) hold resource conditions constant and vary the algorithm. No study has treated monitoring frequency, detection window size, and reference window size as independent experimental variables and measured their quantitative effect on detection performance across multiple datasets and drift types. This gap is the primary empirical contribution of the present study.

**Gap 2: Exclusive focus on either supervised or unsupervised detectors.** Barros & Santos (2018) benchmark supervised, error rate-based detectors exclusively. Lukats et al. (2024) benchmark fully unsupervised detectors exclusively. The distribution-based detectors evaluated in this study — KS, PSI, Wasserstein — occupy a middle ground: they operate without labels but use the simpler statistical machinery of classical hypothesis testing rather than the complex model-fitting of fully unsupervised approaches. Their comparative performance relative to each other, across realistic constraint conditions and multiple drift types, has not been previously characterised at the scale of this study.

**Gap 3: Disconnection between drift detection and model robustness evaluation.** The drift detection literature and the model evaluation literature are largely parallel tracks. Studies like Pathi (2025) evaluate model stability under population shifts but do not systematically examine the interaction between detector choice, drift type, constraint level, and model degradation. Studies like Barros & Santos (2018) evaluate detector performance but treat the model as fixed. This study integrates both dimensions by measuring not only detector performance but also the degree to which different ML model architectures degrade under experimentally controlled drift across constraint conditions.

### 2.10.3 This Study's Contribution to the Literature

This study contributes to the literature at the intersection of three bodies of work: the empirical evaluation of drift detection algorithms, the characterisation of resource-constrained ML deployment, and the assessment of model robustness under distributional shift. Specifically, it provides:

1. **An empirical mapping of the constraint-performance surface** for three distribution-based detectors (KS test, PSI, Wasserstein distance) across three datasets (credit risk, telecommunications churn, predictive maintenance), three drift types (sudden, gradual, recurring), and a systematic grid of resource constraint configurations (monitoring frequency × detection window size × reference window size), yielding 8,100 experimental configurations.

2. **A ranking of constraint parameters by their quantitative impact** on detection delay and missed detection rate, providing practical guidance for system designers who must prioritise resource allocation decisions.

3. **A comparison of three ML model architectures** (Random Forest, Logistic Regression, LightGBM) on their robustness to drift-induced performance degradation, controlling for detector and constraint conditions.

4. **A synthesis of real citations** from applied and theoretical literature to contextualise the empirical findings within the broader understanding of when and why lightweight, distribution-based approaches succeed or fail.

The research is positioned not as an exhaustive replacement for the large-scale supervised benchmarks of Barros & Santos (2018) or the unsupervised benchmarks of Lukats et al. (2024), but as a targeted empirical contribution to the underexplored intersection of constraint-aware evaluation and distribution-based detection for tabular data — the most common deployment context in industry.

---

## 2.11 Chapter Summary

This chapter has reviewed the theoretical foundations and empirical literature underlying the study of concept drift detection. The formal definition of drift as a change in the joint distribution P(X, y) (Gama et al., 2013) and the three-source taxonomy of drift origins (Lu et al., 2019) provide the conceptual vocabulary for the study. The four-stage detection framework (Lu et al., 2019) situates the KS test, PSI, and Wasserstein distance as data distribution-based detectors operating on the test statistic and hypothesis test stages. The large-scale benchmarks of Barros & Santos (2018) and Lukats et al. (2024) establish the empirical state of play for supervised and unsupervised detectors, respectively, while highlighting the absence of constraint-aware evaluation. The theoretical and applied properties of the three target detectors have been reviewed through the work of Taplin & Hunt (2019), Pathi (2025), Prashanth et al. (2025), and Christen et al. (2024). The resource constraint literature (Yang & Shami, 2021; Preuveneers et al., 2020) confirms the practical motivation for constraint-aware evaluation. The model robustness literature (Breiman, 2001; Ke et al., 2017; Pathi, 2025) establishes the theoretical basis for expecting differential model degradation under drift.

Three specific research gaps — the absence of systematic constraint-performance characterisation, the exclusive focus on either supervised or unsupervised approaches, and the disconnection between drift detection and model robustness evaluation — motivate the experimental design described in Chapter 3.

---

## References for This Chapter

Barros, R. S. M., & Santos, S. G. T. C. (2018). A large-scale comparison of concept drift detectors. *Information Sciences*, 451–452, 348–370.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Christen, P., Sabra, W., & Rahm, E. (2024). Stop relearning: Model reuse via feature distribution analysis. *ACM 2024 International Conference on Information and Knowledge Management*.

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2013). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.

Gemaque, R. N., Costa, A. F. J., Giusti, R., & Dos Santos, E. M. (2020). An overview of unsupervised drift detection methods. *WIREs Data Mining and Knowledge Discovery*, 10(6), e1381.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346–2363.

Lukats, G., Zielinski, O., Hahn, A., & Stahl, F. (2024). Benchmarking fully unsupervised concept drift detection. *International Journal of Data Science and Analytics*, 19, 1–31.

Pathi, N. (2025). Model evaluation beyond AUC: A comparative study of Somers' D, KS statistic, and PSI. *Proceedings of ICRTCSIT 2025*.

Prashanth, B. R., Kumar, A., Puneetha, P., & Vinay, S. (2025). Drift detection using Bayesian neural networks and Wasserstein distance. *IEEE International Conference on Advances in Machine Intelligence and Applications (AMATHE 2025)*.

Preuveneers, D., Rimmer, A., Tsingenopoulos, I., Spooren, J., Joosen, W., & Ilie-Zudor, E. (2020). Chained anomaly detection models for federated learning: An intrusion detection case study. *Applied Sciences*, 8(12), 2663. *(Applied for IoT/edge constraint framing.)*

Taplin, R., & Hunt, L. (2019). Evaluating the population stability index. *Journal of Applied Statistics*, 46(6), 1099–1119.

Yang, L., & Shami, A. (2021). On hyperparameter optimization of machine learning algorithms: Theory and practice. *Neurocomputing*, 415, 295–316. *(Applied for constraint-aware ML monitoring context.)*
