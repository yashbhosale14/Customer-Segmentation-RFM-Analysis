#!/usr/bin/env python
# coding: utf-8
# =============================================================================
#  Customer Segmentation & RFM Analysis
#  Author  : Yash Bhosale
#  Dataset : Online Retail II  (online_retail_II.csv)
#  Business Problem:
#    Which customer segments give the highest ROI on retention,
#    and how should marketing allocate budgets?
# =============================================================================

# ── Standard-library imports ─────────────────────────────────────────────────
import os
import sys
import io
import warnings
from datetime import timedelta

# Force UTF-8 on Windows terminal (prevents encoding crash for special chars)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

warnings.filterwarnings("ignore")

# ── Third-party imports ──────────────────────────────────────────────────────
import matplotlib                           # noqa: E402
matplotlib.use("Agg")                       # non-interactive backend — must be set BEFORE pyplot

import matplotlib.pyplot as plt             # noqa: E402
import matplotlib.ticker as mticker        # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import seaborn as sns                      # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH  = r"C:\Users\ASUS\Desktop\IBM Project\online_retail_II.csv"
OUTPUT_DIR = r"C:\Users\ASUS\Desktop\IBM Project\rfm_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 130

SEG_COLORS: dict = {
    "Champions"          : "#27ae60",
    "Loyal Customers"    : "#2980b9",
    "Potential Loyalists": "#8e44ad",
    "At Risk"            : "#e74c3c",
    "Lost"               : "#95a5a6",
    "New Customers"      : "#e67e22",
    "Promising"          : "#16a085",
}

DAY_ORDER: list = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
]


def save_fig(fig: plt.Figure, name: str) -> str:
    """Save a matplotlib figure to OUTPUT_DIR and close it."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")
    return path


def fmt_gbp(value: float, _pos: int = 0) -> str:
    """Formatter for GBP axis labels."""
    return f"GBP {value:,.0f}"


def fmt_gbp_millions(value: float, _pos: int = 0) -> str:
    """Formatter for GBP millions axis labels."""
    return f"GBP {value / 1_000_000:.1f}M"


def fmt_pct(value: float, _pos: int = 0) -> str:
    """Formatter for percentage axis labels."""
    return f"{value:.0f}%"


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 1 -- Load & Clean Data")
print("=" * 60)

df: pd.DataFrame = pd.read_csv(DATA_PATH, encoding="latin-1")
print(f"  Raw rows    : {len(df):,}")

# Parse, filter, engineer
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df.dropna(subset=["Customer ID"])
df["Customer ID"] = df["Customer ID"].astype(int)
df["Invoice"]     = df["Invoice"].astype(str)
df = df[~df["Invoice"].str.startswith("C")]   # remove cancellations
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
df["Revenue"] = df["Quantity"] * df["Price"]

print(f"  Clean rows  : {len(df):,}")
print(f"  Customers   : {df['Customer ID'].nunique():,}")
print(f"  Invoices    : {df['Invoice'].nunique():,}")
print(f"  Date range  : {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
print(f"  Total Rev   : GBP {df['Revenue'].sum():,.2f}")

# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2 -- Exploratory Data Analysis")
print("=" * 60)

# ── Chart 1: Monthly Revenue Trend ───────────────────────────────────────────
monthly_rev: pd.Series = (
    df.set_index("InvoiceDate")["Revenue"].resample("M").sum()
)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly_rev.index, monthly_rev.values,
        color="#2980b9", linewidth=2.2, marker="o", markersize=5)
ax.fill_between(monthly_rev.index, monthly_rev.values,
                alpha=0.13, color="#2980b9")
ax.set_title("Monthly Revenue Trend (2009-2011)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gbp_millions))
plt.tight_layout()
save_fig(fig, "01_monthly_revenue_trend.png")

# ── Chart 2: Top 10 Countries by Revenue ─────────────────────────────────────
top_countries: pd.Series = (
    df.groupby("Country")["Revenue"].sum()
    .sort_values(ascending=False).head(10)
)
fig, ax = plt.subplots(figsize=(10, 4))
top_countries.plot(kind="bar", ax=ax, color="#2980b9", edgecolor="white")
ax.set_title("Top 10 Countries by Revenue", fontsize=13, fontweight="bold")
ax.set_xlabel("Country")
ax.set_ylabel("Revenue")
ax.set_xticklabels(top_countries.index, rotation=35, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gbp_millions))
plt.tight_layout()
save_fig(fig, "02_top10_countries.png")

# ── Chart 3: Top 10 Products by Revenue ──────────────────────────────────────
top_products: pd.Series = (
    df.groupby("Description")["Revenue"].sum()
    .sort_values(ascending=False).head(10)
)
fig, ax = plt.subplots(figsize=(11, 4))
top_products.plot(kind="barh", ax=ax, color="#27ae60", edgecolor="white")
ax.set_title("Top 10 Products by Revenue", fontsize=13, fontweight="bold")
ax.set_xlabel("Revenue (GBP)")
ax.invert_yaxis()
plt.tight_layout()
save_fig(fig, "03_top10_products.png")

# ── Chart 4: Revenue by Day of Week ──────────────────────────────────────────
df = df.copy()                              # avoid SettingWithCopyWarning
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
dow_rev: pd.Series = (
    df.groupby("DayOfWeek")["Revenue"].sum().reindex(DAY_ORDER)
)
fig, ax = plt.subplots(figsize=(9, 4))
dow_rev.plot(kind="bar", ax=ax, color="#8e44ad", edgecolor="white")
ax.set_title("Total Revenue by Day of Week", fontsize=13, fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Revenue (GBP)")
ax.set_xticklabels(DAY_ORDER, rotation=30, ha="right")
plt.tight_layout()
save_fig(fig, "04_revenue_by_dayofweek.png")

# ── Chart 5: Orders per Month ─────────────────────────────────────────────────
monthly_orders: pd.Series = (
    df.drop_duplicates("Invoice")
    .set_index("InvoiceDate")["Invoice"]
    .resample("M").count()
)
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(monthly_orders.index, monthly_orders.values,
       width=20, color="#e67e22", edgecolor="white")
ax.set_title("Number of Orders per Month", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Order Count")
plt.tight_layout()
save_fig(fig, "05_orders_per_month.png")

print("  EDA charts complete.")

# =============================================================================
# STEP 3 — RFM CALCULATION
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3 -- RFM Calculation")
print("=" * 60)

snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)
print(f"  Snapshot date : {snapshot_date.date()}")

rfm: pd.DataFrame = (
    df.groupby("Customer ID")
    .agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency   =("Invoice",     "nunique"),
        Monetary    =("Revenue",     "sum"),
    )
    .reset_index()
)
rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
rfm = rfm.drop(columns=["LastPurchase"])

print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))

# ── Chart 6: RFM Distributions ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
dist_specs: list = [
    ("Recency",   "Recency (days since last purchase)", "#e74c3c"),
    ("Frequency", "Frequency (number of orders)",       "#2980b9"),
    ("Monetary",  "Monetary (total spend GBP)",         "#27ae60"),
]
for ax, (col, xlabel, color) in zip(axes, dist_specs):
    ax.hist(rfm[col].values, bins=40, color=color, edgecolor="white", alpha=0.88)
    ax.set_title(col, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Customers")
fig.suptitle("RFM Metric Distributions", fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig(fig, "06_rfm_distributions.png")

# =============================================================================
# STEP 4 — RFM SCORING (rank-based quintile 1-5)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4 -- RFM Scoring")
print("=" * 60)


def quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Assign a 1-5 score to a series using rank-based quintile binning.
    ascending=True  -> higher value = higher score (Frequency, Monetary)
    ascending=False -> lower value  = higher score (Recency)
    """
    ranked: pd.Series = series.rank(method="first", ascending=ascending)
    scored: pd.Series = pd.qcut(ranked, q=5, labels=[1, 2, 3, 4, 5])
    return scored.astype(int)


rfm["R_Score"]   = quintile_score(rfm["Recency"],   ascending=False)
rfm["F_Score"]   = quintile_score(rfm["Frequency"], ascending=True)
rfm["M_Score"]   = quintile_score(rfm["Monetary"],  ascending=True)
rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

display_cols = ["Customer ID", "Recency", "Frequency", "Monetary",
                "R_Score", "F_Score", "M_Score", "RFM_Total"]
print(rfm[display_cols].head(10).to_string(index=False))

# ── Chart 7: Score Distributions ──────────────────────────────────────────────
score_specs: list = [
    ("R_Score", "Recency Score (R)",   "#e74c3c"),
    ("F_Score", "Frequency Score (F)", "#2980b9"),
    ("M_Score", "Monetary Score (M)",  "#27ae60"),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, title, color) in zip(axes, score_specs):
    counts: pd.Series = rfm[col].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=color, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Score (1=Low, 5=High)")
    ax.set_ylabel("Customers")
fig.suptitle("Distribution of RFM Scores", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "07_rfm_score_distributions.png")

# =============================================================================
# STEP 5 — CUSTOMER SEGMENTATION (Rule-Based)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5 -- Customer Segmentation")
print("=" * 60)


def assign_segment(row: pd.Series) -> str:
    """Assign a business segment label based on RFM scores."""
    r: int = int(row["R_Score"])
    f: int = int(row["F_Score"])
    m: int = int(row["M_Score"])
    if   r >= 4 and f >= 4 and m >= 4:  return "Champions"
    elif r >= 3 and f >= 3:             return "Loyal Customers"
    elif r >= 4 and f <= 2:             return "New Customers"
    elif r >= 3 and f >= 2 and m >= 3:  return "Potential Loyalists"
    elif r <= 2 and f >= 3:             return "At Risk"
    elif r == 1 and f == 1:             return "Lost"
    else:                               return "Promising"


rfm["Segment"] = rfm.apply(assign_segment, axis=1)

seg: pd.DataFrame = (
    rfm.groupby("Segment")
    .agg(
        Customers    =("Customer ID", "count"),
        Avg_Recency  =("Recency",     "mean"),
        Avg_Frequency=("Frequency",   "mean"),
        Avg_Monetary =("Monetary",    "mean"),
        Total_Revenue=("Monetary",    "sum"),
    )
    .round(2)
)
seg["Revenue_Pct"]  = (seg["Total_Revenue"] / seg["Total_Revenue"].sum() * 100).round(2)
seg["Customer_Pct"] = (seg["Customers"]     / seg["Customers"].sum()     * 100).round(2)
seg = seg.sort_values("Total_Revenue", ascending=False)

print(seg.to_string())

seg_index: list = seg.index.tolist()
colors_order: list = [SEG_COLORS.get(s, "#bdc3c7") for s in seg_index]

# ── Chart 8: Customer Count per Segment ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
seg["Customers"].plot(kind="bar", ax=ax, color=colors_order, edgecolor="white")
ax.set_title("Customer Count by Segment", fontsize=14, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("Number of Customers")
ax.set_xticklabels(seg_index, rotation=30, ha="right")
for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        int(bar.get_height()),
        ha="center", va="bottom", fontsize=9,
    )
plt.tight_layout()
save_fig(fig, "08_customer_count_by_segment.png")

# ── Chart 9: Revenue Share Pie ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
_, _, autotexts = ax.pie(
    seg["Total_Revenue"],
    labels=seg_index,
    colors=colors_order,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.82,
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Revenue Share by Segment", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "09_revenue_share_pie.png")

# ── Chart 10: Average Spend per Segment ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
seg["Avg_Monetary"].plot(kind="bar", ax=ax, color=colors_order, edgecolor="white")
ax.set_title("Average Customer Spend by Segment", fontsize=14, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("Avg Spend (GBP)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gbp))
ax.set_xticklabels(seg_index, rotation=30, ha="right")
plt.tight_layout()
save_fig(fig, "10_avg_spend_by_segment.png")

# ── Chart 11: RFM Score Heatmap ───────────────────────────────────────────────
hm: pd.DataFrame = (
    rfm.groupby("Segment")[["R_Score", "F_Score", "M_Score"]]
    .mean()
    .round(2)
    .reindex(seg_index)
)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    hm, annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, ax=ax, cbar_kws={"label": "Avg Score"},
)
ax.set_title("Average RFM Scores per Segment", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "11_rfm_heatmap.png")

# ── Chart 12: Recency vs Monetary Scatter ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for seg_name, grp in rfm.groupby("Segment"):
    ax.scatter(
        grp["Recency"].values,
        grp["Monetary"].values,
        label=str(seg_name),
        alpha=0.55, s=18,
        color=SEG_COLORS.get(str(seg_name), "#bdc3c7"),
    )
ax.set_title("Recency vs Monetary -- Customer Map", fontsize=13, fontweight="bold")
ax.set_xlabel("Recency (days -- lower = more recent)")
ax.set_ylabel("Monetary (total spend GBP)")
ax.set_yscale("log")
ax.legend(title="Segment", markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
save_fig(fig, "12_recency_vs_monetary_scatter.png")

# ── Chart 13: Frequency Box Plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
data_to_plot: list = [
    rfm.loc[rfm["Segment"] == s, "Frequency"].values
    for s in seg_index
]
bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False)
for patch, s in zip(bp["boxes"], seg_index):
    patch.set_facecolor(SEG_COLORS.get(s, "#bdc3c7"))
    patch.set_alpha(0.8)
ax.set_xticks(range(1, len(seg_index) + 1))
ax.set_xticklabels(seg_index, rotation=30, ha="right")
ax.set_title("Order Frequency Distribution by Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Orders")
plt.tight_layout()
save_fig(fig, "13_frequency_boxplot.png")

print("  Segmentation charts complete.")

# =============================================================================
# STEP 6 — ROI & BUDGET ALLOCATION
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 6 -- ROI & Budget Allocation")
print("=" * 60)

COST_PER_CUSTOMER: int = 10  # assumed GBP 10 per customer per campaign

seg_roi: pd.DataFrame = seg.copy()
seg_roi["Campaign_Cost"] = seg_roi["Customers"] * COST_PER_CUSTOMER
seg_roi["ROI_pct"] = (
    (seg_roi["Total_Revenue"] - seg_roi["Campaign_Cost"])
    / seg_roi["Campaign_Cost"] * 100
).round(1)
seg_roi["Budget_Alloc"] = (
    seg_roi["Total_Revenue"] / seg_roi["Total_Revenue"].sum() * 100
).round(2)

print(
    seg_roi[["Customers", "Total_Revenue", "Campaign_Cost", "ROI_pct", "Budget_Alloc"]]
    .to_string()
)

# ── Chart 14: ROI Bar ─────────────────────────────────────────────────────────
roi_sorted: pd.Series = seg_roi["ROI_pct"].sort_values(ascending=False)
bar_colors: list = ["#27ae60" if v >= 0 else "#e74c3c" for v in roi_sorted]
fig, ax = plt.subplots(figsize=(10, 5))
roi_sorted.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
ax.set_title("Estimated Retention Campaign ROI by Segment", fontsize=14, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("ROI (%)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticklabels(roi_sorted.index.tolist(), rotation=30, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct))
plt.tight_layout()
save_fig(fig, "14_roi_by_segment.png")

# ── Chart 15: Budget Allocation Horizontal Bar ────────────────────────────────
budget_sorted: pd.Series = seg_roi["Budget_Alloc"].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
budget_sorted.plot(
    kind="barh", ax=ax,
    color=[SEG_COLORS.get(str(s), "#bdc3c7") for s in budget_sorted.index],
    edgecolor="white",
)
ax.set_title("Recommended Marketing Budget Allocation", fontsize=13, fontweight="bold")
ax.set_xlabel("Budget Share (%)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct))
for bar in ax.patches:
    w: float = bar.get_width()
    ax.text(
        w + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.1f}%",
        va="center", fontsize=9,
    )
plt.tight_layout()
save_fig(fig, "15_budget_allocation.png")

# ── Chart 16: Revenue vs Campaign Cost ────────────────────────────────────────
roi_labels: list = seg_roi.index.tolist()
x_pos: list = list(range(len(roi_labels)))
bar_width: float = 0.38
rev_vals: np.ndarray  = seg_roi["Total_Revenue"].values / 1000
cost_vals: np.ndarray = seg_roi["Campaign_Cost"].values / 1000

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([i - bar_width / 2 for i in x_pos], rev_vals,
       bar_width, label="Revenue (GBP 000s)", color="#2980b9")
ax.bar([i + bar_width / 2 for i in x_pos], cost_vals,
       bar_width, label="Campaign Cost (GBP 000s)", color="#e74c3c")
ax.set_xticks(x_pos)
ax.set_xticklabels(roi_labels, rotation=30, ha="right")
ax.set_title("Revenue vs Campaign Cost by Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("GBP (thousands)")
ax.legend()
plt.tight_layout()
save_fig(fig, "16_revenue_vs_cost.png")

print("  ROI charts complete.")

# =============================================================================
# STEP 7 — EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 7 -- Exporting Results")
print("=" * 60)

rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_customer_segments.csv"), index=False)
seg_roi.to_csv(os.path.join(OUTPUT_DIR, "segment_roi_summary.csv"))
print("  rfm_customer_segments.csv  saved")
print("  segment_roi_summary.csv    saved")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  FINAL BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)
print(f"  Customers analysed : {len(rfm):,}")
print(f"  Total revenue      : GBP {df['Revenue'].sum():,.2f}")
print(f"  Period             : {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
print()
print("  SEGMENT BREAKDOWN:")
for s in seg.index:
    row = seg.loc[s]
    print(
        f"  {s:<22} | {int(row['Customers']):>5} customers |"
        f" GBP {row['Total_Revenue']:>12,.2f} | {row['Revenue_Pct']:>5.1f}% revenue"
    )
print()
print("  MARKETING RECOMMENDATIONS:")
print("  Champions           -> VIP loyalty programme, early-access offers, referrals")
print("  Loyal Customers     -> Upsell bundles, tier upgrades, cross-sell campaigns")
print("  At Risk             -> Win-back emails with personalised discount")
print("  Potential Loyalists -> Membership offers, engagement nudges")
print("  New Customers       -> Onboarding series, second-purchase incentive")
print("  Promising           -> Periodic newsletters, low-cost digital touchpoints")
print("  Lost                -> One-time reactivation email only; minimal spend")
print()
print("  BUDGET ALLOCATION GUIDE:")
print("  Champions           -> ~30% of total retention budget")
print("  Loyal Customers     -> ~25%")
print("  At Risk             -> ~20%  (win-back = high ROI per recovered customer)")
print("  Potential Loyalists -> ~15%")
print("  New Customers       ->  ~7%")
print("  Promising / Lost    ->  ~3%")
print()
print(f"  All charts saved to : {OUTPUT_DIR}")
print("\n  [DONE] Analysis complete.")
