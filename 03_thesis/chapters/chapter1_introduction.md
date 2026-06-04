# Chapter 1: Introduction

## 1.1 Background and Motivation

Machine learning models are built on a fundamental assumption: that the statistical properties of the data they will encounter in deployment are sufficiently similar to the data they were trained on. In many real-world settings, this assumption fails. Credit applicant populations change as economic conditions shift. Network traffic patterns evolve as user behaviour and attack strategies change. Industrial equipment degrades gradually, altering the sensor signatures that indicate impending failure. In each case, a model trained on historical data becomes progressively less reliable without any change to the model itself — because the world it was trained to represent has changed.

This phenomenon is known as *concept drift*: a change over time in the joint probability distribution of input features and target labels, P(X, y), such that a model trained under the original distribution generates degraded predictions under the new one (Gama et al., 2013; Lu et al., 2019). Concept drift is not a rare edge case. It is the default condition for any machine learning system that remains in production long enough for the world to change — which is to say, for nearly every deployed model of practical importance.

The response to this challenge is *drift monitoring*: the continuous inspection of incoming data or model outputs to detect when a significant distributional shift has occurred, so that appropriate action — retraining, recalibration, or human review — can be triggered. The literature on drift monitoring spans several decades and encompasses dozens of proposed detection algorithms, evaluated across synthetic data streams, benchmark datasets, and real-world applications (Barros & Santos, 2018; Lukats et al., 2024). Yet a consequential gap remains: the overwhelming majority of this work evaluates detectors under idealised conditions — unlimited memory for reference data, high-frequency monitoring, large detection windows — that do not reflect the resource constraints of many real deployment environments.

In practice, machine learning models are frequently deployed in contexts where computational and memory resources are not unlimited. Edge devices, IoT sensors, and mobile systems operate under strict memory and CPU budgets. Enterprise batch monitoring systems must balance the cost of frequent checks against the cost of delayed detection. Cloud-hosted monitoring pipelines must be designed to run efficiently at scale. In all of these contexts, the practitioner must make explicit decisions: how often should drift be checked? How large should the detection window be? How much historical reference data should be retained? The answers to these questions are not merely engineering choices — they are consequential decisions that determine whether the monitoring system will detect drift quickly, miss it entirely, or flood operators with false alarms.

Despite this practical importance, the interaction between resource constraint parameters and detection performance has received limited empirical attention. We do not have a data-driven answer to the question: *which constraints matter most, by how much, and for which detectors and drift patterns?* This study is designed to provide that answer.

## 1.2 Problem Statement

The deployment of lightweight, distribution-based drift detectors in resource-constrained machine learning systems requires practitioners to configure monitoring parameters — monitoring frequency, detection window size, and reference window size — whose performance implications are not well characterised by the existing literature. The three most widely used distribution-based detectors in industry — the Kolmogorov-Smirnov (KS) test, the Population Stability Index (PSI), and the Wasserstein distance — have been applied extensively in credit risk monitoring, fraud detection, and model stability assessment, but their relative detection capability, computational cost, and sensitivity to constraint configurations have not been studied together at the scale and systematicity required to provide actionable deployment guidance.

At the same time, real-world drift takes different forms — sudden step changes, gradual transitions, and recurring seasonal patterns — and different forms may interact differently with both the detector choice and the constraint configuration. The degree to which drift type moderates the effect of constraint parameters is unknown. Similarly, the degree to which the choice of machine learning model architecture (Random Forest, Logistic Regression, LightGBM) influences drift robustness — holding the detector and constraints constant — has not been characterised empirically in a systematic, multi-dataset study.

## 1.3 Research Questions

This study is organised around four research questions:

**RQ1 — Detector Performance:** Which lightweight distribution-based drift detector (KS test, PSI, Wasserstein distance) achieves the best detection performance — defined jointly as lowest missed detection rate, lowest detection delay, and lowest computational cost — across datasets, drift types, and constraint configurations?

**RQ2 — Drift Type Effects:** How does the temporal pattern of drift (sudden, gradual, recurring) affect detection performance, and which drift types are most reliably detected under resource-constrained monitoring conditions?

**RQ3 — Resource Constraint Effects:** How do the three primary resource constraint parameters — monitoring frequency, detection window size, and reference window size — affect detection delay and missed detection rate? Which constraints have the greatest quantitative impact on detection performance?

**RQ4 — Model Robustness:** Which machine learning model architecture (Random Forest, Logistic Regression, LightGBM) is most robust to drift-induced performance degradation, and does model robustness interact with drift type or constraint configuration?

## 1.4 Objectives

The specific objectives pursued to address these research questions are:

1. To design and implement a controlled experimental framework that injects three types of synthetic drift (sudden, gradual, recurring) into three real-world tabular datasets (credit risk, telecommunications churn, predictive maintenance) at specified points in time, enabling precise measurement of detection delay and missed detection rate.

2. To implement three lightweight distribution-based drift detectors (KS test, PSI, Wasserstein distance) and evaluate them across a full factorial grid of resource constraint configurations: three monitoring frequencies (10, 100, 1,000 samples), three detection window sizes (50, 200, 1,000 samples), and three reference window sizes (100, 500, unlimited samples).

3. To apply non-parametric statistical tests (Kruskal-Wallis H-test, Mann-Whitney U with Bonferroni correction) and effect size measures (rank-biserial correlation r, Spearman's ρ) to establish which observed performance differences are statistically significant and practically meaningful.

4. To evaluate the classification performance degradation of three ML model architectures (Random Forest, Logistic Regression, LightGBM) under controlled drift conditions, using the same experimental framework, and to determine whether model choice moderates drift robustness.

5. To synthesise the empirical findings into a priority hierarchy of deployment decisions — which choices matter most, by how much — to provide actionable guidance for practitioners designing drift monitoring systems under resource constraints.

## 1.5 Scope and Delimitations

This study focuses on tabular, binary-classification datasets and lightweight distribution-based detectors. The following delimitations define its scope:

- **Datasets:** Three publicly available tabular datasets are used — the UCI Credit Card Default dataset (30,000 instances), the IBM Telco Customer Churn dataset (7,043 instances), and the UCI AI4I Predictive Maintenance dataset (10,000 instances). Results are specific to tabular binary-classification tasks; generalisability to image, text, or multi-class problems is not claimed.

- **Detectors:** Only three distribution-based, label-free detectors are evaluated. Error rate-based supervised detectors (DDM, EDDM, ADWIN) and complex unsupervised neural detectors (D3, NN-DVI) are excluded; their evaluation is beyond the scope of this study.

- **Drift types:** Only sudden, gradual, and recurring drift are implemented. Incremental (monotone) drift is not evaluated.

- **Constraint parameters:** Three parameters (monitoring frequency, window size, reference size) are systematically varied. Other resource constraints — energy consumption, network bandwidth, parallelism — are not addressed.

- **Models:** Three classification models are evaluated: Random Forest, Logistic Regression, and LightGBM. Deep learning architectures are excluded.

## 1.6 Significance of the Study

The practical significance of this study lies in its contribution to the evidence base for deployment decisions that practitioners must make but for which little quantitative guidance currently exists. The finding that monitoring frequency and detection window size dominate the constraint-performance relationship — while reference window size has a threshold effect — directly informs how system designers should prioritise memory and compute allocation in monitoring pipelines. The finding that gradual drift is systematically harder to detect across all detectors and constraint configurations warns practitioners against optimising their monitoring systems exclusively for sudden-change scenarios, which dominate the published benchmark literature.

The theoretical significance of the study lies in its demonstration that the choice of resource constraint configuration can produce a 37-fold range in detection delay — from 22.0 monitoring windows under favourable configurations to 827.6 windows under unfavourable ones — for the same detector applied to the same dataset with the same drift type. This result establishes that constraint configuration is not a secondary engineering concern but a primary determinant of monitoring effectiveness, comparable in importance to the choice of detection algorithm.

For the broader field, the study contributes a methodology for constraint-aware evaluation of drift detectors that can be replicated and extended with other algorithms, datasets, and constraint types.

## 1.7 Thesis Structure

The remainder of this thesis is organised as follows:

**Chapter 2 — Literature Review** reviews the theoretical foundations of concept drift (Gama et al., 2013; Lu et al., 2019), the empirical evidence on drift detector performance (Barros & Santos, 2018; Lukats et al., 2024), the properties of the three target detectors (Taplin & Hunt, 2019; Pathi, 2025; Prashanth et al., 2025), the resource constraint literature (Yang & Shami, 2021; Preuveneers et al., 2020), and the positioning of this study relative to three identified research gaps.

**Chapter 3 — Methodology** describes the experimental design in full: the three datasets and their preprocessing, the implementation of the three drift types, the three detectors and their parameter configurations, the constraint grid that defines the 8,100 experimental configurations, the statistical analysis procedures, and the model comparison design.

**Chapter 4 — Results** presents the empirical findings for each research question: detector performance comparisons (RQ1), drift type effects (RQ2), resource constraint effects (RQ3), and model robustness comparisons (RQ4). All key findings are supported by statistical significance tests and effect size estimates.

**Chapter 5 — Discussion** interprets the findings in the context of the existing literature, explains the mechanisms behind the key results, articulates the central priority hierarchy emerging from the study, and discusses the limitations of the study design.

**Chapter 6 — Conclusion** summarises the principal findings, states the contributions of the study, identifies directions for future research, and reflects on the implications for the design of resource-constrained drift monitoring systems.
