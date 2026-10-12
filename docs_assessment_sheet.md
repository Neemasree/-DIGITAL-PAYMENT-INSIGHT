# DS Sprint Assessment — Use Case 13: Digital Payment Insight

> Figures below refer to synthetic educational data. The label is a simulated review-priority proxy, not confirmed fraud.

## Round 1 — Data Understanding, Preparation & EDA
**Problem statement:** Analyze digital-payment transactions to identify patterns that differ from normal customer behaviour and may deserve operational review.

**Dataset details:** Synthetic data generated for this project; five CSVs: transactions, customers, devices, merchants and alerts. Raw transactions: 12,035; cleaned transactions: 12,000; joined cleaned table: 24 columns.

**Preprocessing:** Removed 35 duplicate transaction records; filled 300 missing merchant categories using merchant reference data; standardized channel labels; treated extreme values using a capped modelling feature while retaining original amounts for analysis; joined customer, device, merchant and alert tables.

**EDA findings:** (1) New-device transactions had a simulated review rate of approximately 39.64% versus 19.26% for known devices. (2) Peak hourly transaction volume occurred at 19:00, with 1,011 transactions.

## Round 2 — Feature Engineering & Machine Learning
**Features:** log_amount, hour, is_night, new_device_flag, region_mismatch, tenure_months, customer_segment, merchant_category, channel, device_type, customer_region. **Target:** synthetic_review_label.

**Algorithm:** Random Forest Classifier. **Justification:** Models non-linear relationships between numerical and categorical behavioural features.

**Held-out metrics:** 80/20 stratified split; 2,400 test transactions. Accuracy: 0.7342; precision: 0.3855; recall: 0.3166; F1-score: 0.3476; ROC-AUC: 0.6169. These score agreement with synthetic proxy labels only.

## Round 3 — Visualization, Intelligence Layer & Prototype
**Charts:** Amount distribution, hourly volume, new-vs-known device comparison, merchant-category review rate, regional trends, confusion matrix.

**Insights:** (1) New-device transactions had a higher simulated review rate than known-device transactions. (2) Travel had the highest observed synthetic merchant-category review rate, approximately 25.37%.

**Prototype:** Python pipeline for cleaning, joins, behavioural feature engineering, a Random Forest model, charts, Excel analysis and a business-pitch deck.

**Recommendations:** (1) Prioritize human review of combined signals such as new-device use, amount deviations, night activity and regional mismatch. (2) Validate monitoring thresholds against confirmed historical outcomes before deployment.

## Final Submission
**Repository:** https://github.com/Neemasree/-DIGITAL-PAYMENT-INSIGHT

**Conclusion:** The prototype demonstrates an end-to-end analysis workflow on synthetic payment data. The findings are educational and require validation with authentic, confirmed outcomes before operational use.
