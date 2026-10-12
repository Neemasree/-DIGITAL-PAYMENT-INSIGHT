# Digital Payment Insight — Use Case 13

**DS Sprint Assessment 2 | U23AD491 Data Science**

An end-to-end educational data-science project to understand digital-payment transaction behaviour and identify patterns that may deserve operational review, following the provided Use Case 13 assessment brief.

> **Synthetic-data notice:** All records are artificially generated. `synthetic_review_label` is a simulated review-priority proxy, not confirmed fraud or a real-world outcome. Model metrics do not represent production fraud-detection performance.

## What the project produces

Running `python src/build_project.py` generates the complete working project artifacts:
- Five source CSVs in `data/raw/`: transactions, customers, devices, merchants and alerts.
- Cleaned joined transactions and summary CSVs in `data/processed/`.
- A trained Random Forest pipeline saved under `models/`.
- Metrics and EDA charts under `reports/`.
- An Excel workbook with summary analysis, cleaned data, analysis tabs and charts under `excel/`.
- An eight-slide business pitch under `presentation/`.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/build_project.py
```

The generator uses a fixed random seed for repeatability. Run from the repository root. It creates synthetic data and re-generates reports, the model, workbook and presentation.

## Assessment workflow

1. Generate synthetic source tables with intentional quality issues.
2. Clean duplicate transaction IDs, normalize channels, fill missing merchant categories using merchant reference data, parse timestamps and join the data.
3. Engineer time, amount, device and region features.
4. Explore amount distribution, hourly volume, customer segment, device, merchant category and region patterns.
5. Train and evaluate a Random Forest classifier with a stratified holdout split.
6. Export a report-ready Excel workbook, charts and a short business pitch.

## Assessment-sheet draft

See [docs_assessment_sheet.md](docs_assessment_sheet.md) for the concise answer organised by round.

## Important limitations

This is an educational prototype. Simulated review-priority labels do not prove fraud. Do not automatically block payments based on this model. Any operational use would require real, governed transaction records, confirmed review outcomes, leakage checks, threshold validation and human oversight.
