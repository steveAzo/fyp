# Deep Extraction: Taplin & Hunt 2019 - Population Accuracy Index vs PSI

**Full Citation:** Taplin, R., & Hunt, C. (2019). The Population Accuracy Index: A New Measure of Population Stability for Model Monitoring. *Risks*, 7(2), 53. https://doi.org/10.3390/risks7020053

**Paper Type:** Methodological critique + proposal

**Relevance to Your Work:** CRITICAL - PSI is one of your 3 detectors. This paper exposes its fundamental limitations.

---

## 1. Problem & Context

### What problem are they solving?
Risk models in banking (credit scoring, default prediction) are developed on historical data but applied to current portfolios years later. **Population stability** monitoring checks whether the model remains "fit-for-purpose" when the input distribution changes.

### Why does it matter?
- Basel Accord & IFRS 9 require banks to monitor models continuously
- If current customer characteristics differ from development data, model predictions may be unreliable
- Population Stability Index (PSI) is the **industry standard** for this monitoring
- **But PSI has serious flaws** (this paper's central claim)

### Context for your work:
You're evaluating PSI under resource constraints. This paper shows PSI has **conceptual** problems even before considering constraints. Your work will test whether these problems worsen under limited monitoring frequency, window size, or memory.

---

## 2. Key Contribution

### Main Claims:
1. **PSI measures the wrong thing:** It detects ANY distribution change, not changes that make the model unfit
2. **PSI fails on interpolation:** When review data is LESS variable than development data (concentrated in the middle), PSI flags false positives even though the model is MORE reliable
3. **PSI requires bucketing:** Treating numerical variables as categorical loses information
4. **PSI is univariate only:** Cannot detect multivariate drift (extrapolation in 2D space even when each feature looks fine)

### The Killer Example (Table 1 vs Table 2):
- **Development data:** Standard normal N(0,1), divided into 10 equal-frequency buckets
- **Table 1 (extrapolation):** Review data N(0, 1.6²) - MORE spread out, tails have 21% vs expected 10%
  - PSI = 0.25 (flags as "significant change")
  - **Correct interpretation:** Model being extrapolated beyond development range - WARNING JUSTIFIED
  
- **Table 2 (interpolation):** Review data N(0, 0.674²) - LESS spread out, concentrated in middle
  - PSI = 0.25 (SAME value, flags as "significant change")
  - **Wrong interpretation:** Data is MORE concentrated where model was trained - NO WARNING NEEDED, model is actually MORE reliable

**Implication:** PSI cannot distinguish between dangerous extrapolation and safe interpolation.

### Proposed Solution: Prediction Accuracy Index (PAI)
**PAI = average variance of predicted response at review / average variance at development**

- PAI > 1: Predictions are LESS accurate on review data (extrapolation)
- PAI < 1: Predictions are MORE accurate on review data (interpolation)
- PAI ≈ 1: No change in predictive accuracy

For Table 1: PAI = 1.78 (78% worse - flags correctly)  
For Table 2: PAI = 0.73 (27% better - no flag)

---

## 3. Methods & Evaluation

### How PSI is calculated:
```
PSI = Σ(Oᵢ - Eᵢ) × ln(Oᵢ / Eᵢ)
```
where:
- Oᵢ = observed frequency in category i at review
- Eᵢ = expected frequency in category i at development
- Requires K mutually exclusive categories

**Interpretation guidelines (Siddiqi 2005):**
- PSI < 0.10: No significant change
- PSI 0.10-0.25: Small change, investigate
- PSI > 0.25: Significant change

### How PAI is calculated:

**Simple regression (1 variable):**
```
PAI = (1/2) × [1 + Σ(rⱼ - x̄)² / Σ(xᵢ - x̄)²]
```
where:
- rⱼ = review data points
- xᵢ = development data points  
- x̄ = mean of development data
- Numerator: variance of review data around development mean
- Denominator: variance of development data around its mean

**Multiple regression (k variables):**
```
PAI = [Σ rⱼᵀ V rⱼ / N] / [Σ xᵢᵀ V xᵢ / n]
```
where:
- V = variance-covariance matrix of regression coefficients
- Available from regression output

**PAI Interpretation:**
- PAI < 1.1: No significant deterioration
- PAI 1.1-1.5: Deterioration, investigate
- PAI > 1.5: Significant deterioration

### Evaluation approach:
- **Mathematical proofs:** Shows PAI = 1 for categorical variables with equal development frequencies (desirable property PSI lacks)
- **Synthetic examples:** Normal distributions with varying spread (Table 1, 2, Figure 1)
- **Multivariate example:** 2D correlated data (Figure 2) - shows univariate PSI misses multivariate extrapolation that MPAI (Multivariate PAI) detects

---

## 4. Key Results

### PSI failures demonstrated:

**Problem 1: Bucketing numerical variables**
- PSI REQUIRES categories (can't handle continuous variables directly)
- Bucketing loses information about HOW EXTREME outliers are
- Extreme values just get placed in outermost bucket, distance ignored

**Problem 2: Treats buckets as unordered**
- PSI treats categories as unordered (like "industry sector")
- When buckets represent ordered numerical ranges, this is inappropriate
- Moving from bucket 1→2 is treated same as 1→10

**Problem 3: Interpolation = false positive**
- When model declines high-risk customers (a GOOD outcome), portfolio becomes less variable
- PSI interprets this as "significant change" and triggers warnings
- Authors: "unnecessary expense re-developing a model that remains fit-for-purpose"

**Problem 4: Multivariate drift missed**
- Applied PSI to two correlated variables (Figure 2):
  - Univariate PSI for X₁ = 0.93 (acceptable)
  - Univariate PSI for X₂ = 1.02 (acceptable)
  - Multivariate PAI = 5.43 (review data in region with NO development data - extrapolation!)
- PSI checked each variable separately, missed that combination is out-of-distribution

**Problem 5: Sensitive to number of buckets**
- Too many buckets: PSI detects trivial differences
- Too few buckets: PSI misses important differences
- No principled way to choose bucket count

**Problem 6: Breaks with zero frequencies**
- PSI undefined or infinite if any bucket has 0% in development but >0% in review
- Requires ad-hoc bucket merging

### PAI advantages:
1. Works on raw numerical variables (no bucketing needed)
2. Distinguishes extrapolation (bad) from interpolation (good)
3. Can be computed for multiple variables simultaneously (MPAI)
4. Directly interpretable (1.5 = 50% worse predictive accuracy)
5. Doesn't break with sparse categories
6. Can include non-linear terms (quadratics, interactions) to test linearity assumptions

---

## 5. Limitations of This Work

### What they DON'T address:
1. **Only tested on regression models** - unclear how PAI applies to classification (though they mention logistic regression using log-odds)
2. **No real-world validation** - only synthetic examples, no banking data experiments
3. **Computational cost not discussed** - PAI requires variance-covariance matrix, potentially expensive for high-dimensional data
4. **No comparison on actual drift detection rates** - no ROC curves, no FPR/TPR analysis
5. **Assumes model form is correct** - PAI measures prediction variance, not whether model relationships still hold
6. **Still can't detect concept drift** - both PSI and PAI only look at X distribution, not P(Y|X) changes

### Questions left unanswered:
- How does PAI perform under different sample sizes? (resource constraint!)
- What happens with correlated features in MPAI? (multicollinearity issues?)
- How to set PAI threshold in practice? (1.5 is arbitrary)
- Does PAI work for non-linear models (tree-based, neural nets)?

### Tone/Positioning:
- Paper is from *Risks* journal (finance/banking focus), not ML venue
- Written for practitioners (bankers, auditors), not researchers
- Strong advocacy tone ("PSI has severe deficiencies", "PAI should be industry standard")
- No experimental comparison against other drift detectors (KS-Test, Wasserstein, MMD)

---

## 6. Gap This Creates for Your Work

### Why this matters for your FYP:

**The Central Question:**
*If PSI has conceptual flaws (interpolation false positives, bucketing losses, univariate blindness), do these flaws GET WORSE under resource constraints?*

### Specific gaps you'll address:

**Gap 1: PSI under small window sizes**
- Paper shows PSI breaks with sparse categories
- Your test: Does small window size (50, 100, 500 samples) make sparse categories more common?
- Prediction: PSI will have more undefined/infinite values at small windows

**Gap 2: PSI under infrequent monitoring**
- Paper assumes you can monitor anytime
- Your test: If you only monitor every 100 or 500 samples, does delayed detection make PSI's false positives MORE costly?
- Prediction: False positive on interpolation → unnecessary model redevelopment during the delay window

**Gap 3: Multivariate PSI workaround**
- Paper shows univariate PSI misses 2D extrapolation
- Common practice: Run PSI on model score (1D projection of k-dimensional space)
- Your test: Does score-based PSI still miss multivariate drift? How does it compare to Wasserstein (inherently multivariate)?

**Gap 4: Computational cost under memory budget**
- Paper doesn't discuss computation
- PAI requires storing full reference data + computing covariance matrix
- Your test: Under 10MB, 50MB memory limits, can you even compute PAI? Or forced to use simpler PSI?

**Gap 5: No ground truth evaluation**
- Paper uses synthetic examples, no drift detection accuracy metrics
- Your contribution: Compare PSI vs Wasserstein vs KS-Test on REAL drift (sudden, gradual, recurring) with FPR, detection delay, missed drift rates

### How to position PSI in your thesis:

**Chapter 2 (Literature Review):**
"Population Stability Index (PSI) is the industry standard for monitoring distribution shift in banking applications (Siddiqi 2005, Taplin & Hunt 2019). However, Taplin & Hunt (2019) demonstrate critical limitations: PSI flags false positives when data becomes less variable (interpolation), requires bucketing continuous variables (losing information), and cannot detect multivariate drift when monitoring features independently. These conceptual issues motivate exploring alternative detectors (KS-Test, Wasserstein) and evaluating whether resource constraints exacerbate PSI's limitations."

**Chapter 4 (Experimental Design):**
"We include PSI despite its known limitations (Taplin & Hunt 2019) because: (1) it remains the industry standard, (2) we test whether small window sizes increase sparse category problems, (3) we evaluate score-based PSI as a multivariate workaround, (4) we measure whether its false positives on interpolation occur in practice with real drift patterns."

**Chapter 7 (Discussion):**
"Our results confirm Taplin & Hunt's (2019) concerns: PSI exhibited X false positives on gradual drift scenarios where data concentrated (interpolation case). Under 50-sample windows, PSI had Y undefined values due to zero-frequency buckets. Score-based PSI failed to detect compound drift in Z% of cases, validating concerns about univariate monitoring. However, PSI's computational simplicity (O(n) with fixed buckets) made it viable under 10MB memory constraints where Wasserstein failed."

---

## 7. Key Quotes to Use

### On PSI's industry dominance:
> "The PSI is closely related to well-established entropy measures... It is also described in Siddiqi (2005)... and has become an industry standard."

### On the core problem:
> "The PSI captures any differences between the development and review data rather than focussing on those differences that suggest the model is not fit for the purpose of estimation on the review data."

### On interpolation false positives:
> "The fact that high-risk customers are less frequent in the review data compared to the development data is an expected outcome of a new model deployment and should not be interpreted, as the PSI does, as evidence that the model is no longer fit-for-purpose."

### On multivariate blindness:
> "While the review data does not represent extreme variables for either variable... there is a visual pattern whereby the review data tends to be in the lower right corner (high x₁ and low x₂) where no development data exists. Thus, in a multivariate sense, extrapolation is involved."

### On bucketing losses:
> "If the model uses a numerical variable, then assessing population stability using a categorical (bucketed) version may not capture changes in stability appropriately."

---

## 8. Connections to Other Papers You'll Read

### Papers that will support Taplin's critique:
- **Raza et al. 2015** - Uses KS-Test instead of PSI for covariate shift, validates alternative
- **Rabanser et al. 2018** - Empirical comparison mentions PSI limitations, tests MMD/KS/χ²
- **Viehmann 2021** - Shows Wasserstein respects feature geometry (addresses bucketing issue)

### Papers that use PSI (uncritically):
- Any banking/credit scoring papers pre-2019 - assume PSI is valid
- You'll note: "Prior to Taplin & Hunt (2019), PSI was accepted without critical evaluation of its interpolation failure mode"

### Papers you'll cite together:
"While Population Stability Index remains common (Siddiqi 2005), recent work has identified fundamental limitations (Taplin & Hunt 2019), motivating evaluation of statistical tests like Kolmogorov-Smirnov (Raza et al. 2015) and optimal transport methods like Wasserstein distance (Viehmann 2021, Rabanser et al. 2018) as alternatives."

---

## 9. Technical Details for Implementation

### PSI calculation (from paper):
```python
# Your implementation will need:
def calculate_psi(development_data, review_data, n_bins=10):
    # 1. Bucket development data (equal frequency or equal width)
    bins = np.percentile(development_data, np.linspace(0, 100, n_bins+1))
    
    # 2. Get expected frequencies
    expected = np.ones(n_bins) / n_bins  # Equal frequency case
    
    # 3. Assign review data to bins
    observed = np.histogram(review_data, bins=bins)[0] / len(review_data)
    
    # 4. Calculate PSI (handle zeros!)
    observed = np.clip(observed, 1e-10, None)  # Avoid log(0)
    psi = np.sum((observed - expected) * np.log(observed / expected))
    
    return psi
```

### PSI bucketing strategies (you'll need to choose):
1. **Equal-frequency:** Development data split so each bin has equal count (paper's approach)
2. **Equal-width:** Fixed intervals (e.g., 0-0.1, 0.1-0.2 for LVR ratio)
3. **Domain-driven:** Buckets chosen by business logic

For fair comparison: Use equal-frequency (10 bins) to match banking practice

### Score-based PSI (multivariate workaround):
```python
# Instead of PSI per feature, compute on model output
scores_dev = model.predict(X_dev)
scores_review = model.predict(X_review)
psi_score = calculate_psi(scores_dev, scores_review, n_bins=10)
```

This is what you'll test against Wasserstein (inherently multivariate)

---

## 10. Questions for Deep Reading Other Papers

As you read the 16 remaining Tier 1 papers, ask:

1. **Do they use PSI?** If yes, do they acknowledge Taplin's limitations?
2. **Do they bucket numerical features?** If yes, do they discuss information loss?
3. **Do they test univariate vs multivariate detection?** Relevant to PSI's univariate blindness
4. **What's their computational cost?** Compare to PSI's O(n) simplicity
5. **Do they handle interpolation?** Or assume all distribution change = bad?

---

## Summary: What to Tell Your Professor

**"I've completed deep reading of Taplin & Hunt 2019 on PSI. Key findings:**

1. **PSI is industry standard but fundamentally flawed** - flags false positives when data becomes less variable (interpolation), even though model is MORE reliable
2. **Bucketing loses information** - extreme outliers just get placed in outermost category, actual distance ignored
3. **Univariate blindness** - checking features separately misses multivariate extrapolation
4. **Implication for my work:** I need to test whether PSI's conceptual flaws worsen under resource constraints (small windows → sparse buckets, infrequent monitoring → costly false positives). I'll compare against KS-Test and Wasserstein which don't require bucketing and work on raw distributions.

**This validates my multi-detector evaluation approach - no single detector is perfect, their tradeoffs under constraints are unknown."**

---

**Paper Status:** ✅ DEEPLY READ  
**Extraction Time:** [Current date]  
**Next Action:** Read Raza 2015 (KS-Test validation) and Rabanser 2018 (empirical comparison) to build detector comparison foundation
