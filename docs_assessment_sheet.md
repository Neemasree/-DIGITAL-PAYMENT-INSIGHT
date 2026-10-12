# DS Sprint Assessment Sheet — Use Case 13: Digital Payment Insight

> **Important:** Values and findings below are from the included synthetic dataset. All review labels are simulated proxies, not real fraud outcomes.

## Round 1 — Data Understanding, Preparation & EDA

**Problem statement:** Analyze digital-payment transactions to identify patterns that differ from normal customer behaviour. Prioritize unusual patterns for operational review using transaction amount, device activity, transaction timing and regional behaviour.

**Dataset details:** Digital Payment Insight synthetic dataset; generated locally for this educational assessment from five CSV files: transactions, customers, devices, merchants and alerts. Raw transactions: 12,035; cleaned unique transactions: 12,000. The joined cleaned dataset contains 24 columns.

**Preprocessing:** Removed 35 duplicate transaction records; filled 300 missing merchant categories using merchant reference data; standardized inconsistent channel labels; treated extreme amounts using a capped modelling feature while retaining original amounts for descriptive analysis; joined transaction, customer, device, merchant and alert tables.

**EDA observations:**
1. Synthetic review rate: approximately 39.64% for new-device transactions versus 19.26% for known-device transactions.
2. Highest transaction volume was at 19:00 (7 PM), with 1,011 transactions.

## Round 2 — Feature Engineering & Machine Learning

**Selected features:** log_amount, hour, is_night, new_device_flag, region_mismatch, tenure_months, customer segment, merchant category, channel, device type and customer region. Target: synthetic_review_label.

**Algorithm:** Random Forest Classifier.

**Justification:** Handles nonlinear relationships and interactions between numeric and categorical transaction features without requiring linear decision boundaries.

**Measured holdout performance:** 80/20 stratified split; 2,400 test transactions. Accuracy = 0.7342; precision = 0.3855; recall = 0.3166; F1-score = 0.3476; ROC-AUC = 0.6169.

**Caution:** Metrics measure agreement with simulated review-priority labels only; they are not real-world fraud-detection performance.

## Round 3 — Visualization, Intelligence Layer & Prototype

**Visualizations:** Transaction amount distribution, hourly volume, device comparison, merchant-category review rate, regional trends and model confusion matrix.

**Key insights:**
1. New-device transactions had a higher simulated review rate than known-device transactions.
2. Travel had the highest observed synthetic merchant-category review rate (approximately 25.37%).

**Prototype:** Reproducible Python pipeline cleans and joins synthetic transaction data, engineers behavioural features, trains a Random Forest model and exports charts, metrics, an Excel analysis workbook and a presentation.

**Recommendations:**
1. Prioritize human review where new-device activity combines with unusually high amounts, night-time activity or regional mismatch.
2. Validate monitoring thresholds against genuine confirmed historical outcomes before any operational deployment.

## Final Submission

**GitHub repository:** https://github.com/Neemasree/-DIGITAL-PAYMENT-INSIGHT

**Conclusion:** The project demonstrates an end-to-end analysis workflow using synthetic data, EDA, feature engineering and machine learning. The prototype highlights patterns that could guide operational review, but real-world usefulness must be validated with authentic historical data.
