# DS Sprint Assessment — Use Case 13: Digital Payment Insight

> All figures below refer to synthetic educational data. The review label is a simulated proxy, not confirmed fraud.

## Round 1 — Data Understanding, Preparation & EDA
**Problem statement:** Analyze digital-payment transactions to identify patterns that differ from normal customer behaviour and may deserve operational review.

**Dataset details:** Synthetic data generated for this project; five CSVs: transactions, customers, devices, merchants and alerts. Expected transaction rows: 12,035 raw including 35 duplicate rows; 12,000 after deduplication.

**Preprocessing:** Removed duplicate transaction IDs; filled missing merchant categories from merchant reference data; standardized channel labels; capped extreme amounts for a modelling feature while retaining original amounts for analysis; joined customer, device, merchant and alert details.

**EDA findings:** Run `python src/build_project.py` and use the generated `reports/key_findings.json` for the exact results from your run. The analysis covers device comparison, hourly volume, merchant category and region patterns.

## Round 2 — Feature Engineering & Machine Learning
**Selected features:** log_amount, hour, is_night, new_device_flag, region_mismatch, tenure_months, customer_segment, merchant_category, channel, device_type, customer_region. **Target:** synthetic_review_label.

**Algorithm:** Random Forest Classifier.

**Justification:** Models nonlinear interactions between numeric and categorical behavioural features after preprocessing and one-hot encoding.

**Performance:** Use the actual values generated in `reports/model_metrics.json` (accuracy, precision, recall, F1-score, ROC-AUC). Metrics describe agreement with synthetic proxy labels only, not real-world fraud detection.

## Round 3 — Visualization, Intelligence Layer & Prototype
**Charts:** Amount distribution, hourly volume, new-vs-known device comparison, merchant-category review rate, regional trends and confusion matrix.

**Prototype:** Reproducible Python pipeline that generates synthetic source tables, cleans and joins records, engineers features, trains a baseline model and exports metrics, charts, an Excel workbook and pitch deck.

**Recommendations:** (1) Prioritize human review when several unusual signals coincide. (2) Validate monitoring thresholds using real, confirmed historical outcomes before operational use.

## Final Submission
**GitHub repository:** https://github.com/Neemasree/-DIGITAL-PAYMENT-INSIGHT

**Conclusion:** The project demonstrates an end-to-end transaction-monitoring analysis using synthetic data, EDA and a baseline ML model. Real-world effectiveness remains unverified and requires authentic, confirmed historical data.
