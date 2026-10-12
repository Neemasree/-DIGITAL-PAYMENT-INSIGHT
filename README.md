# Digital Payment Insight — Use Case 13

**DS Sprint Assessment 2 | U23AD491 Data Science**

This repository contains an end-to-end educational data-science workflow for the assessment case *Digital Payment Insight*: synthetic data generation, data quality repair, EDA, feature engineering, a Random Forest baseline, Excel analysis, and a business pitch.

> **Synthetic-data notice:** All records are artificially generated. `synthetic_review_label` is a simulated review-priority proxy, not confirmed fraud. Model metrics do not represent production fraud-detection performance.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/build_project.py
```

Run from the repository root. The script creates the complete local project outputs: five raw CSV tables, cleaned/enriched transactions and summaries, six charts, JSON metrics, a saved model, an Excel workbook, and an eight-slide PowerPoint pitch.

## Included source files

- `src/build_project.py` — reproducible end-to-end pipeline.
- `requirements.txt` — Python dependencies.
- `notebooks/digital_payment_eda.ipynb` — EDA walkthrough (run after generating data).
- `docs_assessment_sheet.md` — assessment-sheet answer draft.
- `reports/model_metrics.json`, `reports/key_findings.json` — current run's measured metrics/findings.
- `data/`, `reports/`, `excel/`, `presentation/`, `models/` — generated output folders after running the script.

## Assessment workflow

1. Generate synthetic transactions, customers, devices, merchants and alerts with injected data-quality issues.
2. Remove duplicate transaction IDs, standardize channel labels, impute missing categories from merchant reference data, parse timestamps and join reference tables.
3. Engineer transaction hour, night-time flag, log amount, new-device and region-mismatch features.
4. Explore amounts, hourly volume, segments, devices, merchant categories and regions.
5. Train and evaluate a Random Forest using a stratified 80/20 holdout split.
6. Export charts, an Excel workbook with analysis tables/charts and a short business pitch.

## Measured synthetic baseline

| Metric | Value |
|---|---:|
| Raw rows | 12,035 |
| Clean rows | 12,000 |
| Duplicates removed | 35 |
| Missing merchant categories filled | 300 |
| Accuracy | 0.7342 |
| Precision | 0.3855 |
| Recall | 0.3166 |
| F1-score | 0.3476 |
| ROC-AUC | 0.6169 |

The model predicts a simulated review-priority label. The results are modest and must not be described as real fraud-detection performance. The Excel output contains summary tables and charts; if your assessor requires native Excel PivotTables, create/refresh those in desktop Excel.

## Limitations and responsible use

This is an educational prototype. Synthetic labels cannot establish fraud. Do not block payments automatically based on this model. Operational use requires properly governed real data, confirmed outcomes, leakage checks, false-positive evaluation and human oversight.
