# Customer Segmentation & RFM Analysis

**Author:** Yash Bhosale  
**Dataset:** Online Retail II — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)  
**File:** `online_retail_II.csv`

---

## Project Description

This project performs a full **Customer Segmentation and RFM (Recency, Frequency, Monetary) Analysis** on an online retail dataset covering transactions from **December 2009 to December 2011**.

### Business Problem
> *Which customer segments give the highest ROI on retention, and how should marketing allocate budgets?*

The project answers this by:
1. Cleaning and exploring the transactional data
2. Computing RFM metrics for every customer
3. Scoring customers on a 1–5 quintile scale per metric
4. Assigning each customer to a business-meaningful segment (Champions, Loyal, At Risk, etc.)
5. Estimating campaign ROI per segment and recommending budget allocation

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| pandas | Data loading, cleaning, aggregation |
| numpy | Numerical operations |
| matplotlib | Chart creation and saving |
| seaborn | Statistical visualisations (heatmap, themes) |

No machine learning libraries are used — the analysis is entirely statistical and rule-based.

---

## Project Structure

```
IBM Project/
├── online_retail_II.csv              # Raw dataset
├── YashBhosale_CustomerSegmentation.py   # Main analysis script
├── requirements.txt                  # Python dependencies
├── YashBhosale_ProjectReport.docx    # Full project report
├── README.md                         # This file
└── rfm_outputs/                      # Generated charts & CSVs
    ├── 01_monthly_revenue_trend.png
    ├── 02_top10_countries.png
    ├── 03_top10_products.png
    ├── 04_revenue_by_dayofweek.png
    ├── 05_orders_per_month.png
    ├── 06_rfm_distributions.png
    ├── 07_rfm_score_distributions.png
    ├── 08_customer_count_by_segment.png
    ├── 09_revenue_share_pie.png
    ├── 10_avg_spend_by_segment.png
    ├── 11_rfm_heatmap.png
    ├── 12_recency_vs_monetary_scatter.png
    ├── 13_frequency_boxplot.png
    ├── 14_roi_by_segment.png
    ├── 15_budget_allocation.png
    ├── 16_revenue_vs_cost.png
    ├── rfm_customer_segments.csv
    └── segment_roi_summary.csv
```

---

## Setup & Run Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Ensure `online_retail_II.csv` is located at:
```
C:\Users\ASUS\Desktop\IBM Project\online_retail_II.csv
```

### 3. Run the analysis
```bash
python YashBhosale_CustomerSegmentation.py
```

All 16 charts will be saved in the `rfm_outputs/` folder. Results CSVs are also exported there.

---

## Key Findings

| Segment | Customers | Revenue Share | Avg Spend (GBP) |
|---|---|---|---|
| Champions | 1,293 | 68.2% | 9,355 |
| Loyal Customers | 1,409 | 15.5% | 1,948 |
| At Risk | 825 | 9.2% | 1,982 |
| Promising | 1,323 | 3.5% | 462 |
| New Customers | 440 | 2.2% | 896 |
| Lost | 514 | 1.0% | 345 |
| Potential Loyalists | 74 | 0.5% | 1,139 |

### Budget Allocation Recommendation

| Segment | Recommended Budget Share |
|---|---|
| Champions | ~30% |
| Loyal Customers | ~25% |
| At Risk | ~20% |
| Potential Loyalists | ~15% |
| New Customers | ~7% |
| Promising / Lost | ~3% |

---

## RFM Scoring Logic

| Metric | Meaning | Score Direction |
|---|---|---|
| **Recency** | Days since last purchase | Lower = better (score 5) |
| **Frequency** | Number of unique orders | Higher = better (score 5) |
| **Monetary** | Total spend (GBP) | Higher = better (score 5) |

Scores are assigned using **quintile ranking** (1 = bottom 20%, 5 = top 20%) to handle skewed distributions.
