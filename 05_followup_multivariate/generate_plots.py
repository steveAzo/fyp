from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Validated palette (references/palette.md) -- categorical slots 1-8, chrome
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"      # slot 1
ORANGE = "#eb6834"    # slot 2
BLUE_LIGHT = "#86b6ef"   # sequential step 250 (ordinal-safe light step)
BLUE_DARK = "#184f95"    # sequential step 600
MUTED_GRAY = "#c3c2b7"   # baseline/axis -- used here as de-emphasis fill
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": MUTED_GRAY,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

OUT = "results/plots"
DETECTOR_LABELS = {"ks": "KS", "psi": "PSI", "wasserstein": "Wasserstein", "mmd": "MMD", "c2st": "C2ST", "hdddm": "HDDDM"}

marginal = pd.read_csv("results/results.csv")
correlation = pd.read_csv("results/results_correlation.csv")


def savefig(fig, name):
    path = f"{OUT}/{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Fig 1: correlation-grid miss rate by detector -- EMPHASIS (MMD vs everyone)
# ---------------------------------------------------------------------------
miss = correlation.groupby("detector")["missed_detection"].mean().sort_values(ascending=True)
labels = [DETECTOR_LABELS[d] for d in miss.index]
colors = [BLUE if d == "mmd" else MUTED_GRAY for d in miss.index]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(labels, miss.values * 100, color=colors, height=0.6)
for bar, v in zip(bars, miss.values * 100):
    ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
            va="center", ha="left", fontsize=10, color=INK_PRIMARY)
ax.set_xlabel("Miss rate on correlation-only drift (%)")
ax.set_xlim(0, 75)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_title("MMD is the only detector that reliably sees correlation-only drift",
             fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
fig.text(0.01, -0.09,
         "HDDDM's per-feature-averaged construction and C2ST's train/test-split classifier\n"
         "both leave it clustered with the univariate detectors, despite being labeled multivariate.",
         fontsize=8.5, color=INK_MUTED, ha="left")
savefig(fig, "fig4_correlation_miss_rate_emphasis")


# ---------------------------------------------------------------------------
# Fig 2: dumbbell -- miss rate per detector, marginal -> correlation
# ---------------------------------------------------------------------------
miss_m = marginal.groupby("detector")["missed_detection"].mean()
miss_c = correlation.groupby("detector")["missed_detection"].mean()
order = miss_c.sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
y = np.arange(len(order))
for i, d in enumerate(order):
    ax.plot([miss_m[d] * 100, miss_c[d] * 100], [i, i], color=MUTED_GRAY, linewidth=2, zorder=1)
ax.scatter(miss_m[order] * 100, y, color=BLUE_LIGHT, s=90, zorder=2, label="Marginal-shift grid", edgecolor=SURFACE, linewidth=1)
ax.scatter(miss_c[order] * 100, y, color=BLUE_DARK, s=90, zorder=2, label="Correlation-shift grid", edgecolor=SURFACE, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels([DETECTOR_LABELS[d] for d in order])
ax.set_xlabel("Miss rate (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("Detectors cluster near zero under marginal shift, then spread apart\nthe moment the shift only lives in the joint structure",
             fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
savefig(fig, "fig5_miss_rate_dumbbell")


# ---------------------------------------------------------------------------
# Fig 3: effect-size hierarchy -- epsilon-squared, detector vs constraints
# ---------------------------------------------------------------------------
eps2_df = pd.read_csv("results/significance/effect_size_comparison_eps2.csv")
# add the 6-way detector epsilon-squared (already computed in run_analysis.py output)
detector_eps2 = {"marginal": 0.0151, "correlation": 0.2361}

factors = ["detector (6-way)", "monitoring_frequency", "window_size"]
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(factors))
width = 0.35
marg_vals = [detector_eps2["marginal"],
             eps2_df.query("condition=='marginal' and factor=='monitoring_frequency'")["epsilon_squared"].iloc[0],
             eps2_df.query("condition=='marginal' and factor=='window_size'")["epsilon_squared"].iloc[0]]
corr_vals = [detector_eps2["correlation"],
             eps2_df.query("condition=='correlation' and factor=='monitoring_frequency'")["epsilon_squared"].iloc[0],
             eps2_df.query("condition=='correlation' and factor=='window_size'")["epsilon_squared"].iloc[0]]

b1 = ax.bar(x - width / 2, marg_vals, width, color=BLUE_LIGHT, label="Marginal-shift grid")
b2 = ax.bar(x + width / 2, corr_vals, width, color=BLUE_DARK, label="Correlation-shift grid")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", fontsize=9, color=INK_PRIMARY)
ax.set_xticks(x)
ax.set_xticklabels(["Detector identity\n(6-way)", "Monitoring\nfrequency", "Window\nsize"])
ax.set_ylabel("Effect size (ε², Kruskal-Wallis)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("Configuration dominates detector choice under marginal shift,\nand that hierarchy inverts under correlation-only shift",
             fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
savefig(fig, "fig3_effect_size_hierarchy")


# ---------------------------------------------------------------------------
# Fig 4: accuracy degradation by dataset x condition (malignancy)
# ---------------------------------------------------------------------------
deg_m = marginal.groupby("dataset")["accuracy_degradation_matched"].mean()
deg_c = correlation.groupby("dataset")["accuracy_degradation_matched"].mean()
datasets = ["credit", "telco", "maintenance"]
dataset_labels = ["Credit", "Telco", "Maintenance"]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(datasets))
width = 0.35
b1 = ax.bar(x - width / 2, [deg_m[d] * 100 for d in datasets], width, color=BLUE_LIGHT, label="Marginal-shift grid")
b2 = ax.bar(x + width / 2, [deg_c[d] * 100 for d in datasets], width, color=BLUE_DARK, label="Correlation-shift grid")
# Negative bars here are tiny (~-0.1pp) -- labeling "below the bar" lands right on
# top of the x-tick labels. Anchor negative-bar labels just above zero instead
# (interior to the plot, nowhere near the tick row) rather than below the bar.
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        label_y = h + 0.15 if h >= 0 else 0.15
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{h:.2f}", ha="center", fontsize=9, color=INK_PRIMARY, va="bottom")
ax.axhline(0, color=MUTED_GRAY, linewidth=1)
ax.set_ylim(-0.6, 5.6)
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels)
ax.set_ylabel("Mean accuracy degradation (pp)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("Correlation-only drift is easy for MMD to detect but largely benign,\nand maintenance is the outlier where accuracy genuinely suffers",
             fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
savefig(fig, "fig6_accuracy_degradation")

# ---------------------------------------------------------------------------
# Fig 5: window_size trend -- small multiples, marginal vs correlation
# ---------------------------------------------------------------------------
wm = marginal.groupby("window_size")["detection_delay_samples"].median()
wc = correlation.groupby("window_size")["detection_delay_samples"].median()

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
axes[0].plot(wm.index.astype(str), wm.values, marker="o", color=BLUE_DARK, linewidth=2, markersize=7)
axes[0].set_title("Marginal-shift grid", fontsize=11, color=INK_PRIMARY, loc="left")
axes[0].set_xlabel("Window size")
axes[0].set_ylabel("Median detection delay (samples)")

axes[1].plot(wc.index.astype(str), wc.values, marker="o", color=BLUE_DARK, linewidth=2, markersize=7)
axes[1].set_title("Correlation-shift grid", fontsize=11, color=INK_PRIMARY, loc="left")
axes[1].set_xlabel("Window size")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for x, y in zip(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata()):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=INK_PRIMARY)

fig.suptitle("Detection delay grows with window size in both conditions,\nso configuration's effect on speed replicates regardless of what's being detected",
             fontsize=12, color=INK_PRIMARY, x=0.01, ha="left", y=1.06)
fig.tight_layout()
savefig(fig, "fig1_window_size_trend")


# ---------------------------------------------------------------------------
# Fig 6: drift-type comparison -- grouped bar, marginal vs correlation
# ---------------------------------------------------------------------------
correlation["drift_type_base"] = correlation["drift_type"].str.replace("correlation_", "", regex=False)
dt_m = marginal.groupby("drift_type")["detection_delay_samples"].median()
dt_c = correlation.groupby("drift_type_base")["detection_delay_samples"].median()
drift_types = ["sudden", "gradual", "recurring"]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(drift_types))
width = 0.35
b1 = ax.bar(x - width / 2, [dt_m[d] for d in drift_types], width, color=BLUE_LIGHT, label="Marginal-shift grid")
b2 = ax.bar(x + width / 2, [dt_c[d] for d in drift_types], width, color=BLUE_DARK, label="Correlation-shift grid")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h:.0f}", ha="center", fontsize=9, color=INK_PRIMARY)
ax.set_xticks(x)
ax.set_xticklabels(["Sudden", "Gradual", "Recurring"])
ax.set_ylabel("Median detection delay (samples)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("Gradual drift is hardest to detect in both conditions,\nreplicating Paper 1's finding at the new detector-class scale",
             fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
savefig(fig, "fig2_drift_type_comparison")

print("\nAll figures generated.")
