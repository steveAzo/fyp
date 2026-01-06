# **NAME: STEPHEN KWAKU AZONGO**
**ID:** 11027858  
**COURSE:** COMPUTER SCIENCE  
**UNIVERSITY OF GHANA  

# **Project Title**
**When, Why, and Under What Constraints Do Lightweight Drift Detection Approaches Fail or Succeed? An Empirical Evaluation of Data Drift Detection in Resource-Constrained Machine Learning Systems**

---

## **1. Problem Statement**
Machine learning models deployed in real-world systems are susceptible to performance degradation over time due to changes in data distribution, commonly referred to as **data drift**. To mitigate this, model monitoring and drift detection mechanisms are employed within MLOps pipelines. However, most existing drift detection solutions assume access to abundant computational resources and stable infrastructure, making them unsuitable for deployment in **resource-constrained environments** such as academic institutions, small organizations, edge devices, and low-infrastructure regions.

While lightweight drift detection approaches are often proposed as alternatives, there is limited **empirical understanding** of how resource constraints—such as limited computation, memory, and monitoring frequency—affect the reliability, sensitivity, and timeliness of drift detection. In particular, it remains unclear under what conditions lightweight monitoring strategies succeed or fail, and how trade-offs between resource usage and detection effectiveness manifest in deployed machine learning systems.

This lack of empirical insight leads to poorly informed deployment decisions, potentially resulting in **undetected model degradation** or **unnecessary monitoring overhead**. Therefore, there is a need for a **systematic empirical evaluation** of drift detection strategies under varying resource constraints to better understand their effectiveness and limitations in real-world, low-resource deployment scenarios.

This project seeks to address this gap by designing controlled experimental MLOps setups and evaluating the impact of resource constraints on drift detection performance and model reliability.

---

## **2. Research Question**
**Primary Question:** Under what conditions do data drift detection methods succeed or fail when deployed in resource-constrained machine learning systems?

**Sub-questions:**
- How do different types of resource constraints (monitoring frequency, window size, memory, computational budget) affect drift detection effectiveness?
- What are the quantifiable trade-offs between resource consumption and detection sensitivity across different drift patterns?
- Which drift detection methods exhibit greater robustness under specific constraint profiles?
- What are the minimum resource requirements for maintaining acceptable drift detection performance?

---

## **3. Aim**
To empirically evaluate the effectiveness of data drift detection strategies in machine learning systems deployed under resource-constrained conditions, and to identify the conditions under which these strategies succeed or fail.

---

## **4. Specific Objectives**
1. To review existing data drift detection techniques used in machine learning model monitoring.
2. To design a baseline machine learning deployment pipeline with standard monitoring and drift detection capabilities.
3. To design resource-constrained deployment configurations by systematically limiting computational and monitoring resources.
4. To implement selected drift detection methods (statistical tests, distribution-based approaches) within both baseline and constrained pipelines.
5. To simulate different types of data drift (sudden, gradual, recurring, compound) and observe their impact on model performance.
6. To conduct controlled experiments evaluating drift detection effectiveness across multiple datasets and resource configurations.
7. To analyze the trade-offs between resource usage, detection sensitivity, detection latency, and false positive rates.
8. To identify and characterize conditions under which lightweight drift detection approaches succeed or fail, providing actionable guidelines for practitioners.

---

## **5. Methodology Overview**
### **5.1 Experimental Design**
This project employs a **controlled experimental approach** with the following components:

**Machine Learning Tasks:**
- 3-4 publicly available classification/regression datasets from diverse domains (e.g., credit risk, predictive maintenance, customer churn)
- Simple, interpretable models (Logistic Regression, Random Forest, LightGBM) to isolate drift detection effects from model complexity

**Drift Simulation Module:**
- Controlled simulation of drift types: sudden, gradual, recurring, compound, and varying magnitudes
- Precise control over drift timing and characteristics for measurable evaluation

**Drift Detection Methods (2-3 selected):**
- Statistical tests: Kolmogorov-Smirnov test, Population Stability Index
- Distribution-based methods: Wasserstein Distance or similar
- Selection based on interpretability, computational cost, and widespread adoption

**Experimental Configurations:**
- **Independent Variables (Resource Constraints):**
    - Monitoring frequency: {every 10, 50, 100, 500, 1000 samples}
    - Window size: {50, 100, 500, 1000, 5000 samples}
    - Memory budget: {10MB, 50MB, 100MB, 500MB, unlimited}
    - Computational budget: simulated CPU time limits
- **Dependent Variables (Outcomes):**
    - Detection delay (time between drift occurrence and detection)
    - False positive rate
    - Missed drift events (false negatives)
    - Model accuracy degradation before detection
    - Resource consumption (CPU time, memory usage, monitoring overhead)

### **5.2 Evaluation Strategy**
- Multiple runs per configuration (10+ repetitions) for statistical reliability
- Cross-dataset validation to assess generalizability
- Statistical significance testing (confidence intervals, hypothesis tests)
- Ablation studies to isolate effects of individual constraint dimensions

### **5.3 Analysis Framework**
- Characterization of failure modes and success conditions
- Derivation of resource-effectiveness trade-off curves
- Development of practitioner guidelines for resource-aware drift detection deployment
- Identification of method-specific robustness profiles under constraints

---

## **6. Expected Contributions**
### **6.1 Empirical Contributions**
- Systematic characterization of drift detection behavior under resource constraints
- Quantitative trade-off analysis between resource consumption and detection effectiveness
- Dataset-validated insights into failure modes and success conditions

### **6.2 Practical Contributions**
- Decision framework for selecting drift detection methods under resource constraints
- Minimum resource requirement guidelines for different drift types
- Open-source experimental framework for reproducible drift detection evaluation

### **6.3 Research Contributions**
- Filling a gap in empirical understanding of resource-constrained drift detection
- Foundation for future work on resource-adaptive monitoring strategies
- Potential publication in ML systems or applied ML venues

---

## **7. Deliverables**
1. **Literature Review:** Comprehensive survey of drift detection methods and resource-constrained ML systems
2. **Experimental System:** Reproducible codebase including drift simulation, detection implementations, and evaluation framework
3. **Empirical Results:** Quantitative evaluation across multiple datasets, methods, and constraint configurations
4. **Analysis & Insights:** Deep analysis explaining observed patterns, failure conditions, and success factors
5. **Written Thesis:** Complete documentation following standard thesis structure (60-80 pages)
6. **Conference Paper (Optional):** Condensed findings suitable for workshop or conference submission

---

## **8. Timeline (9 Months)**

| Phase | Duration | Key Activities | Deliverable |
|-------|----------|----------------|-------------|
| **Phase 1: Foundation** | Months 1-2 | Literature review, method selection, dataset acquisition | Literature review chapter |
| **Phase 2: Implementation** | Months 3-4 | System design, drift simulation, detector implementation | Working experimental system |
| **Phase 3: Experimentation** | Months 5-6 | Controlled experiments, data collection across configurations | Complete experimental results |
| **Phase 4: Analysis** | Month 7 | Statistical analysis, pattern identification, insight derivation | Results and analysis chapters |
| **Phase 5: Writing** | Month 8 | Thesis writing, methodology documentation | Complete draft thesis |
| **Phase 6: Finalization** | Month 9 | Revisions, final experiments, polishing | Final thesis submission |

---

## **9. Resources Required**
**Computational:**
- Personal laptop/workstation (sufficient for tabular data experiments)
- No cloud resources required (experiments designed for local execution)

**Software:**
- Python scientific stack (scikit-learn, pandas, numpy)
- Drift detection libraries (scipy for statistical tests, custom implementations)
- Version control (Git/GitHub for code management)

**Data:**
- Publicly available datasets (Kaggle, UCI ML Repository)
- No proprietary or restricted data required

---

## **10. Risk Mitigation**

| Risk | Mitigation Strategy |
|------|---------------------|
| Insufficient time for extensive experiments | Prioritize 2-3 core datasets; expand if time permits |
| Technical implementation challenges | Use well-established libraries; start with simplest methods |
| Inconclusive or "boring" results | Negative results are valid scientific findings; focus on thorough analysis |
| Scope creep | Maintain strict focus on empirical evaluation; avoid building production tools |

---

## **11. Ethical Considerations**
- Use of publicly available, non-sensitive datasets
- No human subjects or personal data involved
- Open-source release of code promotes reproducibility and scientific transparency