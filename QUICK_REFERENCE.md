# FYP QUICK REFERENCE CARD

Print this. Put it on your wall. Use it daily.

---

## THE BIG PICTURE

Your FYP is an **empirical study** asking:
> **"When and why do lightweight drift detection approaches fail or succeed under resource constraints?"**

**You're NOT** building a new algorithm  
**You ARE** systematically testing existing algorithms under real constraints

---

## YOUR TIMELINE (9 Months)

```
MONTH 1-2: 📚 LITERATURE REVIEW
           Read 30-40 papers
           Identify research gap
           Write Chapter 2

MONTH 3-4: 💻 IMPLEMENTATION
           Build drift detectors (KS, PSI, Wasserstein)
           Build drift simulators (sudden, gradual, etc.)
           Build experiment runner
           Write Chapters 1, 3, 4

MONTH 5-6: 🧪 EXPERIMENTATION
           Run 500+ configurations
           Collect 2000-5000 experiment runs
           Log all results

MONTH 7:   📊 ANALYSIS
           Statistical analysis
           Generate figures/tables
           Write Chapter 6

MONTH 8:   ✍️ WRITING
           Chapters 7-8
           Revise all
           Fix grammar

MONTH 9:   🎉 SUBMIT
           Final polish
           Submit
           Celebrate!
```

---

## YOUR 3 DATASETS (Pick Credit as First)

| Dataset | Size | Why | Status |
|---------|------|-----|--------|
| **Credit Default** | 30k | Realistic, proven, drift-prone | 🟢 START HERE |
| Churn | 7k | Smaller, faster prototyping | 🔵 Month 3 |
| Maintenance | 10k | IoT-relevant | 🔵 Month 4 |
| Bike Sharing | 17k | Natural drift patterns | 🔵 Month 4 |

---

## YOUR 3 DETECTORS

| Detector | Cost | Best For | Status |
|----------|------|----------|--------|
| **KS-Test** | Medium | Baseline, clean data | 🟢 IMPLEMENT |
| **PSI** | Low | Resource constraints | 🟢 IMPLEMENT |
| Wasserstein | High | Complex distributions | 🔵 Week 3 |

---

## YOUR 4 CONSTRAINTS TO TEST

1. **Monitoring Frequency**: Check every 10 / 50 / 100 / 500 / 1000 samples
2. **Window Size**: Use last 50 / 100 / 500 / 1000 / 5000 samples
3. **Memory Budget**: 10MB / 50MB / 100MB / unlimited
4. **Drift Types**: Sudden / Gradual / Recurring / Compound

---

## YOUR METRICS (What You Measure)

| Metric | Why It Matters | Good | Bad |
|--------|---|---|---|
| **Detection Delay** | How fast do you catch drift? | <200 samples | >1000 samples |
| **False Positive Rate** | How many false alarms? | <5% | >20% |
| **Missed Drift** | Did you completely miss it? | 0% missed | Any missed |
| **Accuracy Drop** | How much degrades before detection? | <5% drop | >15% drop |
| **CPU Time** | How expensive to check? | <10ms | >100ms |
| **Memory Used** | How much RAM needed? | <100MB | >500MB |

---

## YOUR THESIS (9 Chapters)

```
1. ABSTRACT ............ (1 page)     ← Write LAST
2. INTRODUCTION ........ (4-5 pages)   ← Your motivation
3. LITERATURE REVIEW ... (10-12 pages) ← What we know
4. RESEARCH PROBLEM .... (6-8 pages)   ← What you're studying
5. METHODOLOGY ......... (8-10 pages)  ← How you study it
6. IMPLEMENTATION ...... (6-8 pages)   ← What you built
7. RESULTS ............. (15-20 pages) ← What you found
8. DISCUSSION .......... (8-10 pages)  ← Why it matters
9. CONCLUSION .......... (2-3 pages)   ← So what?
10. REFERENCES ......... (2-3 pages)   ← Citations

TOTAL: 60-80 pages
```

---

## THIS WEEK TODO

- [ ] Download Credit dataset
- [ ] `pip install -r requirements.txt`
- [ ] Create 5 files (trainer.py, logger.py, base.py, ks_test.py, psi.py)
- [ ] Run `01_baseline.py` ← SHOULD PRINT "✓ No false positives"
- [ ] Read QUICK_START.md if stuck

**Target**: 1-2 hours total

---

## DOCUMENTS TO READ

| Document | Length | When | Why |
|----------|--------|------|-----|
| QUICK_START.md | 10 min | NOW | Get first prototype running |
| PROJECT_STRUCTURE.md | 10 min | TODAY | Understand folder layout |
| EXECUTION_PLAN.md | 20 min | THIS WEEK | Detailed timeline & code |
| 01_literature_review/README.md | 30 min | WEEK 2 | Plan your lit review |
| 03_thesis/README.md | 30 min | WEEK 2 | Understand thesis structure |

**Total reading**: ~2 hours (ESSENTIAL, not optional)

---

## REPOSITORY STRUCTURE

```
FYP/
├── 01_literature_review/  ← Read papers, take notes here
├── 02_implementation/      ← Write ALL code here
│   ├── data/              ← Load & preprocess
│   ├── models/            ← Train models
│   ├── detectors/         ← Drift detection
│   ├── simulators/        ← Generate drift
│   ├── experiments/       ← Run experiments
│   └── evaluation/        ← Analyze results
├── 03_thesis/             ← Write all chapters here
├── 04_results/            ← Experiment outputs
└── DOCs                   ← Guides & plans (you're reading them)
```

---

## MONTHLY MILESTONES

| Month | Deliverable | Status |
|-------|-------------|--------|
| 1 | Literature review search done | 📝 TODO |
| 2 | Chapter 2 (Lit Review) + Chapter 1 (Intro) | 📝 TODO |
| 3 | All detectors implemented, Chapters 3-4 | 🔧 TODO |
| 4 | Drift simulators working, experiment runner ready | 🔧 TODO |
| 5 | 1000+ experiments completed | 🧪 TODO |
| 6 | All experiments done (2000-5000 runs) | 🧪 TODO |
| 7 | Chapter 6 (Results) written | 📊 TODO |
| 8 | Chapters 7-8 (Discussion & Conclusion) + revisions | ✍️ TODO |
| 9 | SUBMITTED! | 🎉 TODO |

---

## RULES TO LIVE BY

1. **Read the documentation.** It's all here. Don't skip it.
2. **Follow the timeline.** Adjust if needed, but stay disciplined.
3. **Code modularly.** Each file has one job.
4. **Log everything.** Helps you debug later.
5. **Test as you go.** Don't wait until the end.
6. **Write while coding.** Chapters 1-2 in Month 2, not Month 8.
7. **Commit to git.** Backup your work.
8. **Ask for help early.** Advisor office hours exist for this.

---

## COMMON PITFALLS (And How to Avoid Them)

| Pitfall | Why Bad | How to Fix |
|---------|---------|-----------|
| Reading 100 random papers | Never understand the field | Read 30-40 systematically (follow the guide) |
| Adding all datasets immediately | Debugging nightmare | Get 1 dataset working first (Credit) |
| Trying to be perfect immediately | Wastes time | MVP first, then improve |
| Leaving writing until end | Rushed, poor quality | Write as you code (Chapter per month) |
| Hardcoding values | Impossible to change later | Use config.yaml for everything |
| No version control | Lost work | Commit to git daily |
| Not testing | Surprise failures | Test each module independently |

---

## WHEN YOU'RE STUCK

**Problem**: "I don't know where to start"  
**Solution**: Read QUICK_START.md, follow it step-by-step

**Problem**: "Code doesn't work"  
**Solution**: Check error message, read relevant module's README.md

**Problem**: "Don't understand the timeline"  
**Solution**: Read EXECUTION_PLAN.md, section "Immediate Next Steps"

**Problem**: "Literature review overwhelmed"  
**Solution**: Read 01_literature_review/README.md, follow week-by-week plan

**Problem**: "Thesis structure unclear"  
**Solution**: Read 03_thesis/README.md, each chapter has template

**Problem**: "Nothing works, giving up"  
**Solution**: Call your advisor. That's literally their job.

---

## SUCCESS LOOKS LIKE

**End of Week 1**: `01_baseline.py` runs successfully  
**End of Month 1**: You've read 15 papers, understand the gap  
**End of Month 2**: Chapters 1-2 drafted, 3 detectors implemented  
**End of Month 4**: Full system working, ready to run 100s of experiments  
**End of Month 7**: Results analyzed, patterns clear  
**End of Month 8**: Full thesis drafted, advisor happy  
**End of Month 9**: SUBMITTED! 🎉

---

## YOUR SECRET WEAPON

**You have everything:**
- Clear problem (from proposal.md)
- Solid methodology (from outline.md)
- Modular structure (just created)
- Detailed timeline (EXECUTION_PLAN.md)
- Code examples (in EXECUTION_PLAN.md)
- Literature guide (01_literature_review/README.md)
- Thesis guide (03_thesis/README.md)

**Most FYP students don't have this.**

**You're set up to succeed.**

---

## FINAL WORDS

This is a **solid research project.**

It's **feasible** (you can do this in 9 months).

It's **relevant** (resource-constrained ML is a real problem).

It's **novel** (nobody has empirically evaluated this trade-off).

It will **look impressive** on your resume/cv.

It will **get you a good grade** if you follow the plan.

It might **get published** if results are interesting.

---

## GET STARTED NOW

1. Download the dataset
2. Install dependencies
3. Run the first prototype
4. Read QUICK_START.md if stuck
5. Do this TODAY

---

**You've got this. Let's go.** 🚀

Questions? Re-read the relevant document. The answer is there.

Can't find the answer? Ask your advisor (that's what they're for).

Good luck! 🎓

