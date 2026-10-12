# Digital Payment Insight — Use Case 13

**DS Sprint Assessment 2 | U23AD491 Data Science**

A reproducible data-science prototype for identifying digital-payment transaction patterns that may deserve operational review. This follows the provided Use Case 13 brief: Digital Payment Insight.

> **Synthetic-data notice:** All records in this project are artificially generated. `synthetic_review_label` is a rule-generated review-priority proxy, not confirmed fraud or a real-world outcome. Model metrics are from this synthetic experiment only and should not be represented as production fraud-detection performance.

## Project deliverables

- `data/raw/`: five generated data tables — transactions, customers, devices, merchants, and alerts.
- `data/processed/`: cleaned/enriched transactions and EDA summaries.
- `notebooks/digital_payment_eda.ipynb`: notebook walkthrough.
- `src/build_project.py`: reproducible pipeline to regenerate datasets, preprocessing, analysis, plots, ML, Excel workbook and presentation.
- `models/synthetic_review_model.joblib`: trained scikit-learn model pipeline.
- `reports/figures/`: EDA visualizations and confusion matrix.
- `reports/model_metrics.json` and `reports/key_findings.json`: measured outputs from the generated dataset.
- `excel/digital_payment_analysis.xlsx`: analysis workbook, summary, formula examples and charts.
- `presentation/digital_payment_insight_pitch.pptx`: business insights and recommendations.
- `docs_assessment_sheet.md`: concise assessment-sheet response.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/build_project.py
```

The script uses a fixed random seed for reproducible synthetic generation. Run from the repository root. The notebook can be opened with Jupyter.

## Business question

Which transaction patterns differ from normal customer behaviour and deserve operational review?

## Workflow

1. Generate synthetic source tables with data-quality issues.
2. Clean duplicate transaction IDs, standardize channel labels, handle missing merchant categories, parse timestamps, and join context tables.
3. Engineer hour, night-time, log-amount, new-device and region-mismatch features.
4. Explore amount distributions, hourly volume, customer segments, devices, merchant categories and regions.
5. Train a Random Forest classifier against the synthetic review-priority proxy using an 80/20 stratified train/test split.
6. Export charts, the Excel workbook, measured metrics and business-pitch slides.

## Results from the current generated run

| Result | Value |
|---|---:|
| Raw transaction rows | 12,035 |
| Clean transaction rows | 12,000 |
| Duplicate rows removed | 35 |
| Missing merchant categories filled | 300 |
| Test-set rows | 2,400 |
| Accuracy | 0.7342 |
| Precision | 0.3855 |
| Recall | 0.3166 |
| F1-score | 0.3476 |
| ROC-AUC | 0.6169 |

The default classification threshold is 0.50. The results are modest and refer only to synthetic proxy labels. Real operational use would require authentic, confirmed historical outcomes, leakage checks, appropriate threshold validation and human review.

## Assessment write-up

See [docs_assessment_sheet.md](docs_assessment_sheet.md) for concise answers organised by the assessment rounds.

## Limitations and responsible use

This is an educational prototype, not a fraud-detection product. Synthetic labels reflect simulated rules and do not establish that a transaction is fraudulent. Avoid automatically blocking payments based only on this model. Validate insights on appropriately governed, real data before operational use.
