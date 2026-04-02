# Thesis Chapter Structure & Writing Guide

## Overall Thesis Arc (60-80 pages total)

Your thesis tells this story:

1. **Context** (Chapters 0-1): Why does this matter?
2. **Background** (Chapter 2): What do we already know?
3. **Problem** (Chapter 3): What are we studying?
4. **Method** (Chapters 4-5): How do we study it?
5. **Results** (Chapter 6): What did we find?
6. **Meaning** (Chapters 7-8): Why does it matter?

---

## Chapter Breakdown (Page counts, Writing Timeline)

### **Chapter 0: Abstract (1 page)**
**Timeline**: Write LAST (Month 8-9)

**Purpose**: 150-250 word summary of entire thesis

**Structure**:
- **Background** (1 sentence): "Machine learning models in production degrade due to data drift."
- **Problem** (2 sentences): "Most drift detection research assumes unlimited resources. Deployment reality is different."
- **Your approach** (2 sentences): "We empirically evaluate how resource constraints affect drift detection effectiveness across 3 methods and 4 datasets."
- **Key findings** (3-4 sentences): "KS-Test fails under X constraints, PSI performs well under Y constraints, minimum monitoring frequency is Z samples."
- **Implication** (1 sentence): "Our findings provide practitioners guidance for deploying drift detection in resource-constrained environments."

**Example**:
```
Machine learning models deployed in production systems degrade over time due to data drift, 
where the distribution of input data changes. Existing drift detection solutions assume access 
to abundant computational and memory resources, but real deployments—especially in edge computing, 
small organizations, and developing regions—operate under significant constraints. 

This thesis presents a systematic empirical evaluation of drift detection methods (Kolmogorov-Smirnov 
test, Population Stability Index, and Wasserstein distance) under varying resource constraints 
(monitoring frequency, window size, memory budget). We conduct 500+ controlled experiments across 
4 datasets and characterize trade-offs between resource consumption and detection effectiveness.

Key findings: (1) Monitoring frequency is more impactful than window size on detection latency; 
(2) PSI achieves 95% of KS-Test accuracy with 60% less memory; (3) Effective drift detection 
is possible with monitoring every 500 samples on windows of size 100. These findings provide 
practitioners an actionable framework for constraint-aware drift detection deployment.
```

---

### **Chapter 1: Introduction (4-5 pages)**
**Timeline**: Write in Month 2 (after lit review conceptually done)

**Structure**:

**1.1 Hook & Context** (0.5 page)
- "In the last 5 years, thousands of machine learning models have been deployed to production."
- "A common problem: models degrade over time."
- "Why? Data changes—a phenomenon called data drift."

**1.2 Problem Statement** (1 page)
- What is drift? (definition + examples)
- Why does it matter? (business impact: fraud detection system accuracy drops from 95% to 85%)
- How do we detect it? (monitoring systems)
- What's the challenge? (detecting requires computation, but many deployments can't afford much)

**1.3 Research Motivation** (1 page)
- Current state: Papers assume unlimited resources
- Reality: IoT devices have 50MB RAM, small startups can't afford constant monitoring
- Gap: No systematic study of this trade-off
- Your study: "We fill this gap"

**1.4 Research Question** (0.5 page)
- Primary: "Under what conditions do lightweight drift detection methods succeed or fail?"
- Sub-questions leading to your experimental design

**1.5 Contributions** (1 page)
- **Empirical**: Quantified trade-offs between frequency, window size, memory, and detection effectiveness
- **Practical**: Decision framework for practitioners
- **Scientific**: Foundation for future resource-aware monitoring research

**1.6 Thesis Structure** (0.5 page)
- "Chapter 2 reviews existing work..."
- "Chapter 3 defines our research problem..."
- etc.

---

### **Chapter 2: Literature Review (10-12 pages)**
**Timeline**: Month 2 (after systematic search complete)

*See `01_literature_review/README.md` for detailed guidance*

**Key sections**:
- 2.1: Data drift concepts (covariate, concept, label)
- 2.2: Detection methods (KS, PSI, Wasserstein)
- 2.3: Resource constraints in practice
- 2.4: Research gaps
- 2.5: Summary & motivation for your work

---

### **Chapter 3: Research Problem & Methodology (6-8 pages)**
**Timeline**: Write in Month 3-4 (as you start implementation)

**Structure**:

**3.1 Research Question** (1 page)
- Formalize: What are you studying?
- Sub-questions: Monitoring frequency vs window size vs memory budget
- Variables: Independent (what you control) and dependent (what you measure)

**3.2 Experimental Approach** (2 pages)
- Why empirical evaluation? (systematic, reproducible, generalizable)
- Why simulation? (ground truth, controlled conditions)
- Why multiple datasets? (generalizability)
- Why multiple detectors? (understand method-specific behavior)

**3.3 Drift Detection Methods** (2 pages)
- KS-Test: Definition, algorithm, computational complexity
- PSI: Definition, algorithm, computational complexity
- Wasserstein: Definition, algorithm, computational complexity
- Justification: Why these 3?

**3.4 Data & Drift Types** (1 page)
- Datasets: Credit, Maintenance, Churn, Bike
- Why these? (diverse domains, public, sufficient size)
- Drift patterns: Sudden, gradual, recurring, compound
- Drift magnitudes: Small, medium, large

**3.5 Experimental Setup** (2 pages)
- Model: Random Forest (why? simple, interpretable)
- Resource constraints tested: (frequency levels, window sizes, memory budgets)
- Metrics: (detection delay, FPR, missed drift, accuracy drop)
- Repetitions: (10 runs per configuration)

---

### **Chapter 4: Experimental Design (8-10 pages)**
**Timeline**: Write in Month 3 (as you design experiments)

**Structure**:

**4.1 Independent Variables** (1 page)
- Monitoring frequency: 10, 50, 100, 500, 1000 samples
- Window size: 50, 100, 500, 1000, 5000 samples
- Memory budget: 10MB, 50MB, 100MB, unlimited
- Drift type: Sudden, gradual, recurring, compound
- Table showing all configurations

**4.2 Dependent Variables** (2 pages)
- Detection delay: Definition + formula + interpretation
- False positive rate: Definition + formula + interpretation
- False negative rate: Definition + interpretation
- Accuracy drop: Definition + formula
- Resource consumption: CPU time, memory usage
- Monitoring overhead percentage

**4.3 Control Variables** (1 page)
- What you keep constant (model, training data, random seeds)
- Why? (isolate constraint effects)

**4.4 Experimental Configuration** (1 page)
- How many total experiments? (example: 250 core configurations × 10 repetitions = 2,500 runs)
- Why this many? (statistical significance)
- How long to run? (100 hours CPU time, can parallelize)
- Data collection: What gets logged each experiment?

**4.5 Evaluation Metrics (Detailed)** (2 pages)
- Metric 1: Detection delay (why it matters, interpretation)
- Metric 2: False positive rate (why it matters, interpretation)
- Metric 3: Accuracy degradation (why it matters, interpretation)
- Metric 4: Resource consumption (why it matters, interpretation)
- Statistical analysis: Confidence intervals, hypothesis tests

**4.6 Data Analysis Plan** (1 page)
- Descriptive statistics for each metric
- Hypothesis tests (paired t-tests)
- Effect sizes (Cohen's d)
- Visualization: Trade-off curves, heatmaps

---

### **Chapter 5: Implementation (6-8 pages)**
**Timeline**: Write in Month 4 (after working system complete)

**Purpose**: Demonstrate you actually built this and it's reproducible

**Structure**:

**5.1 System Architecture** (2 pages)
- Diagram: Data layer → Model layer → Drift simulation → Detection → Evaluation
- Module responsibilities
- Design decisions (why modular? reproducibility, testability)

**5.2 Data Layer** (1 page)
- Dataset loading
- Preprocessing steps (normalization, handling missing values)
- Train-test split strategy

**5.3 Model Training** (1 page)
- Random Forest configuration
- Training on each dataset
- Baseline performance (Table: Dataset | Baseline Accuracy)

**5.4 Drift Simulation** (1 page)
- How you generate drift (code pseudocode for each pattern)
- Validation: How you verified drift is correct

**5.5 Drift Detectors** (1 page)
- Implementation details of KS, PSI, Wasserstein
- API consistency (all inherit from base class)
- Computational complexity measurements

**5.6 Experiment Execution** (1 page)
- How experiments are orchestrated (config-driven)
- Logging strategy
- Resource monitoring

---

### **Chapter 6: Results (15-20 pages)**
**Timeline**: Write in Month 7 (after experiments complete)

**Structure**:

**6.1 Overview** (1 page)
- How many experiments ran?
- How long did they take?
- Did everything work? (quick troubleshooting notes)

**6.2 Baseline Performance** (2 pages)
- Detection characteristics without constraints
- KS vs PSI vs Wasserstein on unconstrained baseline
- Table: Method | Detection Delay (mean) | FPR | Memory
- Figures: Distribution of detection delays across datasets

**6.3 Impact of Monitoring Frequency** (3-4 pages)
- Research question: "How does checking less often affect detection?"
- Figure: Detection delay vs monitoring frequency (line plot, 3 methods, 4 datasets)
- Figure: False positive rate vs monitoring frequency
- Text: "When frequency decreased from every 10 to every 1000 samples, KS-Test detection delay increased by X% on dataset A, Y% on dataset B..."
- Key finding: "Gradient threshold at ~500 samples—diminishing returns beyond"

**6.4 Impact of Window Size** (3-4 pages)
- Research question: "How does smaller recent data window affect detection?"
- Figure: Detection delay vs window size
- Figure: Accuracy drop vs window size (what degrades before detection?)
- Text: Interpretation of patterns
- Key finding: "Window size less impactful than frequency; minimum 100 samples needed for stable signals"

**6.5 Impact of Memory Budget** (2-3 pages)
- Research question: "What if you can only store small reference set?"
- Figure: Detection delay vs available memory
- Show trade-off between reference set size and detection quality
- PSI should shine here (low memory, still effective)
- Key finding: "PSI achieves 90% of KS-Test accuracy with 10MB budget vs 100MB"

**6.6 Drift Type Interactions** (2-3 pages)
- Does constraint impact differ for sudden vs gradual drift?
- Figure: Detection delay by drift type (grouped bars)
- Text: "Sudden drift detectable even with constraints, gradual drift requires more frequency/larger windows"

**6.7 Method Robustness Profiles** (2 pages)
- Which method performs best under which constraint profile?
- Table: Method | Best for | Worst for | Recommendation
- Heatmap: Method × Constraint profile → Detection delay
- "KS-Test best for clean, large windows; PSI best for constrained memory; Wasserstein overkill for most scenarios"

**6.8 Accuracy Degradation Analysis** (1-2 pages)
- How much does model accuracy drop before drift is detected?
- Table: Dataset | Method | Mean accuracy drop before detection
- "Under extreme constraints, accuracy drops 10-15% before detection; acceptable with retrain frequency"

**6.9 Resource Consumption** (1-2 pages)
- Actual CPU time measurements
- Memory usage per detector per configuration
- Monitoring overhead (% of total system time)
- "Drift detection adds <2% overhead in most scenarios"

---

### **Chapter 7: Discussion (8-10 pages)**
**Timeline**: Write in Month 7 (alongside results)

**Structure**:

**7.1 Interpretation of Findings** (3 pages)
- Restate main results in context of research questions
- Explain WHY patterns exist
- Example: "Detection delay increases with frequency because fewer checks means missing subtle shifts. At 10-sample frequency we catch 98% of gradual drift, but at 1000-sample frequency we miss early warning signs."

**7.2 Practical Implications** (2 pages)
- Who benefits from this? (practitioners in resource-constrained settings)
- Decision framework: "If you have budget for X monitoring, use method Y"
- Concrete recommendations:
  - "For IoT with 50MB memory: Use PSI, check every 100 samples, minimum 100-sample window"
  - "For academic lab on single server: Use KS-Test, check every 500 samples, 500-sample window OK"
  - "For small startup: Use PSI (simple, proven), frequency less critical than window size"

**7.3 Comparison to Prior Work** (2 pages)
- How do your findings align/contradict existing literature?
- "Smith et al. claimed adaptive windows optimal, but our constrained-frequency scenario shows fixed windows often better (simpler, more predictable)"
- "Industry practice of PSI in finance validated: excellent constraint robustness"

**7.4 Limitations** (1.5 pages)
- What you DIDN'T test
- Generalizability concerns
- Honest assessment: "We only tested tabular data, not time series. Covariate drift focused, not concept drift."
- Why these limits exist: "Time constraints, scope management"

**7.5 Future Work** (1 page)
- How would someone extend this?
- Open questions: "What about adaptive frequency? What about federated learning scenarios?"

---

### **Chapter 8: Conclusion (2-3 pages)**
**Timeline**: Write in Month 8

**Structure**:

**8.1 Summary** (0.5 page)
- What was the problem?
- What did you do?
- What did you find?

**8.2 Key Contributions** (1 page)
- **Empirical**: "We quantified constraint impacts for 3 methods across 4 datasets"
- **Methodological**: "We demonstrated systematic approach to constraint evaluation"
- **Practical**: "We provide deployment guidelines for practitioners"

**8.3 Broader Impact** (0.5 page)
- Why this matters for ML systems in the real world
- Connection to sustainability (less monitoring overhead = less energy)
- Democratization (enables smaller players to do drift detection)

**8.4 Final Thought** (0.5 page)
- Restate importance of resource-constrained evaluation
- Vision: "As ML systems deploy more broadly, understanding real-world constraints becomes critical"

---

### **Chapter 9: References (2-3 pages)**
**Timeline**: Build continuously (Months 1-8)

**Format**: Use consistent citation style (IEEE or Harvard)

**Organization**:
- Alphabetical by author
- Full citation format (journal, issue, DOI)
- Verify as you go (don't leave this for the end!)

---

## Total Page Count

| Chapter | Pages | Month |
|---------|-------|-------|
| 0. Abstract | 1 | 8-9 |
| 1. Introduction | 4-5 | 2 |
| 2. Literature Review | 10-12 | 2 |
| 3. Research Problem | 6-8 | 3-4 |
| 4. Experimental Design | 8-10 | 3 |
| 5. Implementation | 6-8 | 4 |
| 6. Results | 15-20 | 7 |
| 7. Discussion | 8-10 | 7 |
| 8. Conclusion | 2-3 | 8 |
| 9. References | 2-3 | Continuous |
| **TOTAL** | **62-80** | |

---

## Writing Tips

### ✅ DO:
- Write as you go (don't save everything for Month 8)
- Use active voice ("We evaluated" not "It was evaluated")
- Include figures/tables (break up text)
- Tell a story (each section flows to next)
- Use concrete numbers (not "many" but "47")

### ❌ DON'T:
- Copy-paste from papers
- Write chapter 1 first (write lit review first, use to inform intro)
- Make figures/tables in PowerPoint (use matplotlib, save as PDF)
- Submit first draft without revision
- Assume reader knows your domain (explain clearly)

### Writing Schedule
```
Month 1: Research plan notes (not chapter, just thinking)
Month 2: Introduction draft + Lit Review draft
Month 3: Research problem + Methods draft
Month 4: Implementation details as you code
Month 5-6: Experimental running (write small analysis docs)
Month 7: Full results + discussion draft
Month 8: Revise all chapters, abstract last
Month 9: Final polish + submission
```

---

## Checklist Before Submission

- [ ] Abstract is self-contained (readable without thesis)
- [ ] Introduction motivates problem clearly
- [ ] Literature review shows comprehensive background knowledge
- [ ] Methodology is reproducible from description
- [ ] Results are presented objectively (not cherry-picked)
- [ ] Discussion honestly addresses limitations
- [ ] Conclusion revisits research questions
- [ ] All figures/tables have captions and citations
- [ ] References are complete and consistent
- [ ] No spelling/grammar errors (use Grammarly)
- [ ] Numbering of chapters/sections is consistent
- [ ] Page numbers on all pages
- [ ] Margins consistent (typically 2cm all sides)

