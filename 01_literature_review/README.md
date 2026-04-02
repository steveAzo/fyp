# Literature Review: The RIGHT Way

## What is a Proper Literature Review? (Not What You've Been Doing)

### ❌ WRONG Approach (Reading Randomly)
- Googling "drift detection"
- Reading papers in random order
- Highlighting random sentences
- No synthesis or critical evaluation
- Result: No coherent understanding

### ✅ RIGHT Approach (Systematic Review)
1. **Define clear research boundaries** → What papers are relevant?
2. **Search systematically** → Use keywords + databases
3. **Screen papers** → Which ones actually matter?
4. **Extract knowledge** → Organize findings by theme
5. **Synthesize** → Connect ideas, identify gaps, your contribution
6. **Write coherently** → Tell a story, not a list

---

## Step 1: Define Your Search Scope (MONTH 1, Week 1)

### Research Questions Driving Literature Review
Based on your proposal, answer these:

**Main**: "Under what conditions do lightweight drift detectors fail?"

**Sub-questions for literature**:
1. How is data drift defined and formalized in ML?
2. What drift detection methods exist? (advantages/limitations of each)
3. What resource constraints exist in real deployments?
4. How do existing papers handle constraints?
5. What gaps exist in empirical evaluation?

### Keyword Strategy (NOT just "drift detection")

**Concept 1: Drift Detection**
- Keywords: "data drift", "concept drift", "drift detection", "model drift"
- Synonyms: "dataset shift", "distribution shift", "covariate shift"

**Concept 2: Resource Constraints**
- Keywords: "resource-constrained", "edge computing", "lightweight", "memory-efficient"
- Synonyms: "limited resources", "IoT deployment", "on-device ML"

**Concept 3: Monitoring & Systems**
- Keywords: "model monitoring", "MLOps", "production ML", "model degradation"
- Synonyms: "model management", "model surveillance", "deployment"

**Combined Search Queries** (use these in Google Scholar / arXiv):
- "data drift detection" (Start here - foundational)
- "concept drift machine learning" (Different type of drift)
- "covariate shift" (Specific drift type you're focusing on)
- "model monitoring production" (Practical deployment context)
- "edge machine learning resource constraints" (Your constraint domain)
- "lightweight drift detection" (Directly relevant)
- "distribution shift detection" (Alternative terminology)

---

## Step 2: Systematic Search (MONTH 1, Week 2-3)

### Where to Search
1. **Google Scholar** (scholar.google.com)
   - Good for broad coverage
   - Filter by year (2015-2025 for recent work)

2. **arXiv** (arxiv.org/list/cs.LG)
   - Preprints, often more recent
   - Categories: Machine Learning, Systems, Distributed Computing

3. **Conference/Journal Proceedings**
   - NeurIPS, ICML, ICLR (general ML)
   - MLSys, SysML (systems perspective)
   - VLDB, SIGMOD (data management)

4. **University Library Access**
   - ACM Digital Library (if available through University of Ghana)
   - IEEE Xplore
   - SpringerLink

### Search Protocol

**For each search query:**

```
Search: "data drift detection"
Date range: 2015-2025
Results: ~5,000 papers

ScreeningLevel 1 (Title + Abstract):
- Read title: Does it mention drift OR distribution shift?
- Read abstract: Is it about detection/monitoring (not just problem existence)?
- Keep: ~200 papers (4% of results)

Screening Level 2 (Full Text Skim):
- Read introduction: Clear problem statement?
- Read methodology: Evaluates detection methods empirically?
- Keep: ~30-50 papers (15-25% of Level 1)
```

**Record EVERYTHING in a spreadsheet:**

```
| Title | Authors | Year | Venue | Key Contribution | Relevant to... | Notes |
|-------|---------|------|-------|------------------|--------|-------|
| "Detecting Concept Drift..." | Smith et al | 2018 | ICML | Proposes DDM algorithm | Detection Methods | Used stream classification, not reg |
| "Resource-Efficient Monitoring..." | Jones | 2021 | MLSys | Studies lightweight detectors | Constraints | Small window evaluation, good |
```

---

## Step 3: Critical Reading & Extraction (MONTH 1, Week 4 - MONTH 2, Week 1)

### For EACH paper you keep, extract:

**1. Problem Statement**
- What problem does the paper address?
- Is it relevant to YOUR gap?

**2. Technical Contribution**
- What method/insight do they propose?
- How does it work (high level)?

**3. Experimental Evaluation**
- What datasets did they test on?
- What metrics did they measure?
- Did they test under constraints? (If yes → GOLD!)

**4. Limitations**
- What do they NOT test?
- What assumptions do they make?
- What's missing?

**5. YOUR Gap Analysis**
- How does this relate to your research question?
- Does it address resource constraints? (Probably not fully)
- Will your work extend/contradict/build on this?

### Template for Each Paper

Save as `01_literature_review/notes/AUTHOR_YEAR_SUMMARY.md`:

```markdown
# Paper: "Detecting Concept Drift in Streaming Data"
**Authors**: Smith & Jones  
**Year**: 2018  
**Venue**: ICML  
**Full Citation**: Smith et al. (2018). Detecting...

## 1. Problem They Address
- Stream classification where class distribution changes over time
- Existing methods detect drift, but performance under low monitoring frequency unknown

## 2. Their Solution
- Proposes "AdaptiveWindowDrift" detector
- Uses adaptive window sizing (larger windows when stable, smaller when drift detected)
- Claims to reduce false positives vs fixed-window KS-test

## 3. Experimental Setup
- **Datasets**: 5 synthetic + 3 real streaming datasets
- **Baselines**: KS-test, DDM, ADWIN
- **Metrics**: Detection delay, false positive rate, computational time
- **Constraints tested**: NO (they assume unlimited resources)

## 4. Results Summary
- Faster detection than DDM (120 vs 200 samples on average)
- Lower FPR than fixed KS-test (0.02 vs 0.05)
- Computational overhead only 5% (good!)

## 5. Limitations
- Only synthetic data, limited real-world datasets
- No memory consumption analysis
- Assumes ability to adjust window size online (not applicable to edge devices)
- Only concept drift, not covariate drift

## 6. Relevance to MY Project
- RELEVANT: Detection methods evaluation
- RELEVANT: False positive rate metrics
- LIMITATION: No resource constraint study
- MY GAP: They don't test what happens with FIXED small windows (edge device constraint)

## 7. How I'll Use This
- Adapt their experimental metrics (detection delay, FPR) for my work
- Include AdaptiveWindowDrift as one of my baseline detectors
- Explicitly test their method under memory/window constraints (extending their work)

## 8. Key Quotes to Remember
> "The primary challenge is balancing detection timeliness with false positive rate" (p. 5)

This directly supports my motivation!
```

---

## Step 4: Organize by Theme (MONTH 2, Week 2)

**Create synthesis documents** in `01_literature_review/notes/`:

### Document 1: DRIFT FUNDAMENTALS
```markdown
# Data Drift Fundamentals

## 1. Definition & Types
- **Covariate Drift**: P(X) changes, P(Y|X) constant
  - Example: Customer age distribution shifts
  - Common in finance, marketing
  - Source: [Smith 2018], [Jones 2019]

- **Concept Drift**: P(Y|X) changes
  - Example: Fraud patterns evolve as criminals adapt
  - Harder to detect, requires retraining
  - Source: [Brown 2020]

- **Label Shift**: P(Y) changes
  - Example: Fraud rate increases from 1% to 5%
  - May not require retraining if P(X|Y) stable
  - Source: [Green 2021]

## 2. Impact on Model Performance
- Undetected drift → prediction accuracy degrades
- Detection delay determines business impact
- Retraining cost vs drift detection cost trade-off

## 3. Key Papers
- [Foundational] Quionero-Candela et al. (2009) - Dataset Shift in ML
- [Concept Drift Focus] Gama et al. (2014) - Learning from Streaming Data
- [Covariate Shift] Sugiyama & Müller (2013) - Covariate Shift Adaptation
```

### Document 2: DETECTION METHODS
```markdown
# Drift Detection Methods: A Comparative Analysis

## 1. Statistical Tests

### Kolmogorov-Smirnov Test
- **How**: Compares cumulative distributions, returns p-value
- **Pros**: Non-parametric, no assumptions, well-established
- **Cons**: Univariate only, needs large samples
- **Computational cost**: O(n log n)
- **Memory cost**: O(n) (store full reference distribution)
- **Papers**: [KS baseline] Smith 2018, Jones 2019

### Population Stability Index (PSI)
- **How**: Bins distribution, measures proportion shift
- **Pros**: Single score, works in practice (finance), low memory
- **Cons**: Requires binning choices, less sensitive to shape
- **Computational cost**: O(n)
- **Memory cost**: O(k) where k=number of bins (~10)
- **Papers**: [PSI practical use] Brown 2020, Finance industry standard

### Wasserstein Distance
- **How**: Optimal transport distance between distributions
- **Pros**: Captures distribution shape well
- **Cons**: Expensive computation, less interpretable
- **Computational cost**: O(n^3) or O(n log n) with approximations
- **Memory cost**: O(n)
- **Papers**: [Wasserstein applications] Green 2021, Black 2022

## 2. Adaptive Methods
- [Method]: [Citation]
- [Discussion of trade-offs]

## 3. Summary Table
| Method | Cost | Accuracy | Best For |
|--------|------|----------|----------|
| KS-Test | Medium | High | Clean data, covariate drift |
| PSI | Low | Medium | Constrained resources |
| Wasserstein | High | High | Complex distributions |
```

### Document 3: RESOURCE CONSTRAINTS LANDSCAPE
```markdown
# Resource Constraints in Production ML

## 1. Real-World Deployment Scenarios

### Edge Devices
- Memory: 10-100 MB available
- CPU: ARM cores, <1 GHz
- Network: Intermittent connectivity
- Examples: IoT sensors, mobile phones
- Papers: [Edge ML] Smith 2021, Jones 2022

### Small Organizations
- Cloud budget: ~$100-500/month
- Infrastructure: Single server or small cluster
- Expertise: Limited (no ML ops team)
- Papers: [Cost-aware ML] Brown 2020

### Academic Settings
- GPU: None or limited access
- CPU: Lab workstations
- Data: Smaller datasets (~100k samples)

## 2. Specific Constraints
- **Memory budget**: Affects reference data storage
- **Monitoring frequency**: How often can you check?
- **Latency**: Max time per detection check?
- **Bandwidth**: Can you send data to cloud?

## 3. Gaps in Literature
- Few papers explicitly test drift detection under constraints
- No systematic evaluation of trade-offs
- **YOUR PROJECT FILLS THIS GAP**
```

### Document 4: RESEARCH GAPS (Your Motivation!)
```markdown
# Research Gaps & Your Contribution

## Gap 1: Lack of Systematic Constraint Study
- Papers assume unlimited resources
- Few measure actual CPU/memory costs
- No trade-off analysis (frequency vs detection latency)

## Gap 2: Limited Real-World Validation
- Most work on synthetic/benchmark data
- Few test on edge devices or low-resource environments
- Industry practices (like PSI in finance) not rigorously evaluated

## Gap 3: Missing Practitioner Guidance
- When should a small org use KS vs PSI?
- What monitoring frequency is enough?
- How much memory should you budget?
- LITERATURE DOESN'T ANSWER THIS

## Why Your Project Matters
Your systematic empirical evaluation will:
1. Quantify trade-offs practitioners don't currently know
2. Provide decision framework for method selection
3. Establish minimum resource requirements
4. Enable future work on adaptive monitoring

## Expected Contribution
- Gap 1 addressed: "We systematically measure constraint impact"
- Gap 2 addressed: "We test across 4 datasets + simulate realistic constraints"
- Gap 3 addressed: "We provide practitioner guidelines"
```

---

## Step 5: Write the Literature Review Chapter (MONTH 2, Week 3-4)

**Output**: Save as `03_thesis/chapters/02_literature_review.md`

### Chapter Structure (10-12 pages, ~3000 words)

```markdown
# Chapter 2: Literature Review

## 2.1 Introduction (0.5 pages)
- Context: ML in production, need for monitoring
- Problem: Models degrade due to data drift
- Solution space: Many drift detection methods exist
- YOUR ANGLE: But what about resource constraints?

## 2.2 Data Drift: Concepts & Definitions (2 pages)
- Define drift formally (mathematical notation)
- Types: covariate, concept, label
- Real-world examples for each type
- Impact on model performance (with numbers from papers)
- **End with**: "While drift is well-understood, how to detect it under constraints is not"

## 2.3 Drift Detection Methods (3 pages)
### 2.3.1 Statistical Tests
- KS-Test (definition + advantages/disadvantages)
- PSI (definition + practical use cases)
- Other tests (mentioned briefly)

### 2.3.2 Adaptive Methods
- Concept drift methods (DDM, ADWIN)
- Why they're not focused here

### 2.3.3 Comparative Analysis
- Table: Method | Computational Cost | Memory | Interpretability
- "As shown in Table 2.1, methods vary significantly in their resource demands"

## 2.4 Resource-Constrained Environments (2 pages)
- Edge devices (definitions, examples, constraints)
- Small organizations (budget, expertise constraints)
- Academic settings
- Specific constraint types: memory, frequency, latency
- **Key insight**: Most ML literature ignores these constraints

## 2.5 Current State of Practice (1 page)
- What do practitioners actually use?
- Why? (PSI in finance because it's simple + proven)
- Industry approach vs research approach mismatch
- "In practice, practitioners use simple methods due to constraints, but rigorous evaluation of this choice is missing"

## 2.6 Research Gaps & Your Contribution (1.5 pages)
### 2.6.1 Identified Gaps
- Gap 1: No systematic study of constraint impact
- Gap 2: Trade-off analysis missing
- Gap 3: Practitioner guidelines absent

### 2.6.2 Your Project's Contribution
- "We systematically evaluate how monitoring frequency affects detection delay"
- "We measure actual resource costs (CPU, memory) for each method"
- "We provide decision framework for constraint-aware method selection"

## 2.7 Summary (0.5 pages)
- Recap: Drift is important, detection methods exist, but constraints unexplored
- Transition: "Our empirical evaluation addresses these gaps..."

---

**Cited Papers**: [1] Smith et al. 2018, [2] Jones et al. 2019, ...
```

---

## Literature Review Timeline (MONTHS 1-2)

| Week | Activity | Output | Time |
|------|----------|--------|------|
| **Month 1** | | | |
| Week 1 | Define search scope, keywords | Search strategy doc | 4-5 hours |
| Week 2-3 | Systematic search across databases | Spreadsheet of ~50 papers | 10-12 hours |
| Week 4 | Level 2 screening (full text skim) | 30-40 papers shortlisted | 8-10 hours |
| **Month 2** | | | |
| Week 1 | Deep reading & extraction (10-15 papers) | Detailed summaries | 15-20 hours |
| Week 2 | Organize by theme (synthesis docs) | 4 synthesis documents | 10 hours |
| Week 3-4 | Write literature review chapter | Chapter 2 draft | 12-15 hours |
| **TOTAL** | | Complete literature review | ~70 hours |

---

## Key Metrics to Track

### Literature Review Quality
- ✅ Papers from diverse venues (not just top conferences)
- ✅ Chronological spread (old foundational + recent work)
- ✅ International authorship (not just Western authors)
- ✅ Synthesis documents show YOU understand it (not just summaries)
- ✅ Your gaps clearly identified (motivation for your work)

### When to Stop Reading
- You're seeing the same papers cited repeatedly = exhaustive
- New papers don't add fundamentally new concepts = complete
- You can explain drift detection & constraints without notes = mastered

---

## Common Mistakes to Avoid

❌ Reading 100 papers randomly  
✅ Reading 30-40 papers systematically

❌ Copying paper abstracts into your review  
✅ Synthesizing ideas across papers

❌ "Smith did X, Jones did Y, Brown did Z"  
✅ "The field has converged on X method (Smith 2018, Jones 2019), but Y remains unexplored"

❌ Literature review chapter is 20 pages  
✅ Literature review chapter is 10-12 pages (then results section shows your work)

❌ No connection to YOUR project  
✅ Each section ends with "and this is why our study is needed"

---

## Expected Outcomes from Literature Review

### By End of Month 2, You Should Know:

1. **The drift landscape**
   - Types of drift (covariate, concept, label)
   - How it manifests in different domains
   - Why it matters (business impact numbers)

2. **Detection methods**
   - 3-4 main methods in detail (KS, PSI, Wasserstein)
   - Their strengths, weaknesses, computational costs
   - How they compare to each other

3. **Constraint impact**
   - What constraints exist in real deployments
   - Why they matter (cost trade-offs)
   - What current literature is missing

4. **Your specific contribution**
   - What GAP you're filling
   - Why it matters
   - How your study will be novel

5. **Execution knowledge**
   - What datasets to use (see dataset section)
   - What methods to implement (top 2-3)
   - What metrics to measure
   - How to design experiments

### Your Literature Review Section Should:
- Be readable by your non-ML advisor
- Build clear motivation for your work
- Establish you understand the field
- Clearly state YOUR gap and contribution

