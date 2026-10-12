from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.chart import BarChart, Reference
from pptx import Presentation
from pptx.util import Pt

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(49113)
N = 12000
for folder in ("data/raw", "data/processed", "reports/figures", "models", "excel", "presentation"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

# 1. Generate linked synthetic reference datasets.
n_customers, n_merchants, n_devices = 1800, 400, 3600
customer_ids = np.array([f"C{i:05d}" for i in range(1, n_customers + 1)])
merchant_ids = np.array([f"M{i:04d}" for i in range(1, n_merchants + 1)])
device_ids = np.array([f"D{i:05d}" for i in range(1, n_devices + 1)])
segments = RNG.choice(["Student", "Regular", "Premium", "Business"], n_customers, p=[.22, .48, .22, .08])
regions = RNG.choice(["North", "South", "East", "West", "Central"], n_customers, p=[.20, .27, .15, .23, .15])
customers = pd.DataFrame({
    "customer_id": customer_ids, "customer_segment": segments, "customer_region": regions,
    "tenure_months": np.clip(RNG.gamma(2.5, 18, n_customers).astype(int), 1, 120)
})
categories = np.array(["Grocery", "Dining", "Travel", "Electronics", "Utilities", "Fashion", "Gaming", "Healthcare", "Fuel", "Other"])
merchants = pd.DataFrame({
    "merchant_id": merchant_ids,
    "merchant_category": RNG.choice(categories, n_merchants),
    "merchant_region": RNG.choice(["North", "South", "East", "West", "Central"], n_merchants)
})
devices = pd.DataFrame({
    "device_id": device_ids,
    "device_type": RNG.choice(["Android", "iOS", "Web", "Tablet"], n_devices, p=[.46, .28, .20, .06]),
    "new_device_flag": RNG.choice([0, 1], n_devices, p=[.84, .16]),
    "device_history_complete": RNG.choice([1, 0], n_devices, p=[.94, .06])
})

# 2. Generate transaction events with evening volume peaks and injected quality issues.
hour_probs = np.array([.01,.005,.003,.002,.002,.003,.01,.025,.055,.065,.06,.06,.065,.06,.055,.05,.05,.06,.075,.085,.075,.055,.03,.015])
hour_probs /= hour_probs.sum()
hours = RNG.choice(np.arange(24), N, p=hour_probs)
times = (pd.Timestamp("2026-09-01") + pd.to_timedelta(RNG.integers(0, 30, N), unit="D")
         + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(RNG.integers(0, 3600, N), unit="s"))
ci, mi, di = RNG.integers(0, n_customers, N), RNG.integers(0, n_merchants, N), RNG.integers(0, n_devices, N)
segment, category = segments[ci], merchants.merchant_category.to_numpy()[mi]
new_device = devices.new_device_flag.to_numpy()[di]
mismatch = regions[ci] != merchants.merchant_region.to_numpy()[mi]
base = {"Student": 250, "Regular": 650, "Premium": 1500, "Business": 2200}
amount = np.array([base[x] for x in segment]) * RNG.lognormal(0, .72, N)
amount *= np.select(
    [category == "Travel", category == "Electronics", category == "Utilities", category == "Grocery", category == "Gaming"],
    [2.1, 2.6, .45, .55, .8], default=1.0
)
night = np.isin(hours, [0, 1, 2, 3, 4])
spike = RNG.random(N) < .035
amount = np.where(spike, amount * RNG.uniform(5, 12, N), amount)
amount = np.where((new_device == 1) & night, amount * RNG.uniform(1.4, 2.5, N), amount)
log_amt = np.log1p(amount)
threshold = np.select([segment == "Student", segment == "Regular", segment == "Premium", segment == "Business"], [7.9, 8.6, 9.3, 9.8])
risk = 1.15 * (log_amt > threshold) + 1.1 * (new_device == 1) + night + .7 * mismatch + 1.5 * spike + RNG.normal(0, .5, N)
prob = 1 / (1 + np.exp(-(risk - 2.25)))
labels = (RNG.random(N) < prob).astype(int)
channels = RNG.choice(["Mobile App", "Web", "UPI", "Card", "mobile app", "WEB", "upi"], N, p=[.30,.16,.22,.12,.08,.06,.06])
transactions = pd.DataFrame({
    "transaction_id": [f"T{i:07d}" for i in range(1, N + 1)],
    "customer_id": customer_ids[ci], "merchant_id": merchant_ids[mi], "device_id": device_ids[di],
    "transaction_time": times, "amount": np.round(amount, 2), "merchant_category": category, "channel": channels
})
missing_indices = RNG.choice(N, int(N * .025), replace=False)
transactions.loc[missing_indices, "merchant_category"] = np.nan
transactions = pd.concat([transactions, transactions.sample(35, random_state=13)], ignore_index=True)
label_map = dict(zip([f"T{i:07d}" for i in range(1, N + 1)], labels))
unique_transactions = transactions.drop_duplicates("transaction_id", keep="first")
transaction_labels = unique_transactions.transaction_id.map(label_map).fillna(0).astype(int)
alerts = pd.DataFrame({
    "transaction_id": unique_transactions.transaction_id,
    "review_status": np.where(transaction_labels.to_numpy() == 1, RNG.choice(["Reviewed", "Pending"], len(unique_transactions), p=[.7, .3]), "No review"),
    "reason_category": np.where(transaction_labels.to_numpy() == 1, RNG.choice(["High amount", "New device", "Unusual hour", "Region mismatch", "Multiple signals"], len(unique_transactions)), "Routine"),
    "synthetic_review_label": transaction_labels.to_numpy()
})
for name, frame in [("transactions", transactions), ("customers", customers), ("devices", devices), ("merchants", merchants), ("alerts", alerts)]:
    frame.to_csv(ROOT / "data/raw" / f"{name}.csv", index=False)

# 3. Clean and enrich.
raw_rows = len(transactions)
clean = transactions.drop_duplicates("transaction_id", keep="first").copy()
clean["transaction_time"] = pd.to_datetime(clean["transaction_time"], errors="coerce")
clean["channel"] = clean["channel"].astype(str).str.strip().str.lower().replace({
    "mobile app": "mobile_app", "web": "web", "upi": "upi", "card": "card"
})
clean["merchant_category"] = clean["merchant_category"].fillna("Unknown")
merchant_category_lookup = merchants.set_index("merchant_id")["merchant_category"]
missing_category_mask = clean["merchant_category"].eq("Unknown")
clean.loc[missing_category_mask, "merchant_category"] = clean.loc[missing_category_mask, "merchant_id"].map(merchant_category_lookup).fillna("Unknown")
clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce")
amount_cap = clean["amount"].quantile(.995)
clean["amount_capped"] = clean["amount"].clip(upper=amount_cap)
clean["hour"] = clean["transaction_time"].dt.hour
clean["day_of_week"] = clean["transaction_time"].dt.day_name()
clean["is_night"] = clean["hour"].isin([0, 1, 2, 3, 4]).astype(int)
clean = clean.merge(customers, on="customer_id", how="left")
clean = clean.merge(devices[["device_id", "device_type", "new_device_flag", "device_history_complete"]], on="device_id", how="left")
clean = clean.merge(merchants[["merchant_id", "merchant_region"]], on="merchant_id", how="left")
clean["region_mismatch"] = (clean["customer_region"] != clean["merchant_region"]).astype(int)
clean["log_amount"] = np.log1p(clean["amount_capped"].clip(lower=0))
clean["synthetic_review_label"] = clean["transaction_id"].map(label_map).fillna(0).astype(int)
clean = clean.merge(alerts[["transaction_id", "review_status", "reason_category"]], on="transaction_id", how="left")
clean.to_csv(ROOT / "data/processed/cleaned_transactions.csv", index=False)

summaries = {
    "segment_summary": clean.groupby("customer_segment").agg(transaction_count=("transaction_id","count"), median_amount=("amount","median"), p95_amount=("amount", lambda s: s.quantile(.95)), review_rate=("synthetic_review_label","mean")).reset_index(),
    "hour_summary": clean.groupby("hour").agg(transaction_count=("transaction_id","count"), review_rate=("synthetic_review_label","mean"), median_amount=("amount","median")).reset_index(),
    "merchant_category_summary": clean.groupby("merchant_category").agg(transaction_count=("transaction_id","count"), review_count=("synthetic_review_label","sum"), review_rate=("synthetic_review_label","mean"), median_amount=("amount","median")).reset_index().sort_values("review_rate", ascending=False),
    "device_summary": clean.groupby("new_device_flag").agg(transaction_count=("transaction_id","count"), review_rate=("synthetic_review_label","mean"), median_amount=("amount","median")).reset_index(),
    "region_summary": clean.groupby("customer_region").agg(transaction_count=("transaction_id","count"), review_count=("synthetic_review_label","sum"), review_rate=("synthetic_review_label","mean")).reset_index()
}
for name, frame in summaries.items():
    frame.to_csv(ROOT / "data/processed" / f"{name}.csv", index=False)

# 4. ML baseline: use a split to estimate agreement with synthetic review labels.
features = ["log_amount", "hour", "is_night", "new_device_flag", "region_mismatch", "tenure_months", "customer_segment", "merchant_category", "channel", "device_type", "customer_region"]
target = "synthetic_review_label"
X, y = clean[features], clean[target]
numeric_features = ["log_amount", "hour", "is_night", "new_device_flag", "region_mismatch", "tenure_months"]
categorical_features = ["customer_segment", "merchant_category", "channel", "device_type", "customer_region"]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])
model = Pipeline([("preprocess", preprocessor), ("classifier", RandomForestClassifier(
    n_estimators=180, min_samples_leaf=3, class_weight="balanced_subsample", random_state=49113, n_jobs=-1
))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=49113, stratify=y)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
metrics = {
    "dataset_rows_after_cleaning": int(len(clean)), "raw_transaction_rows": int(raw_rows),
    "duplicate_rows_removed": int(raw_rows - len(clean)), "missing_merchant_categories_filled": int(len(missing_indices)),
    "feature_count_before_encoding": int(len(features)), "target_positive_rate": float(y.mean()), "test_rows": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, predicted)), "precision": float(precision_score(y_test, predicted, zero_division=0)),
    "recall": float(recall_score(y_test, predicted, zero_division=0)), "f1_score": float(f1_score(y_test, predicted, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)), "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    "threshold_note": "Default classifier threshold 0.50; synthetic proxy labels only."
}
(ROOT / "reports/model_metrics.json").write_text(json.dumps(metrics, indent=2))
joblib.dump(model, ROOT / "models/synthetic_review_model.joblib")

# 5. EDA charts.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9,5)); sns.histplot(clean["amount"], bins=60, log_scale=(False, True)); plt.title("Transaction Amount Distribution"); plt.xlabel("Amount"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/amount_distribution.png", dpi=150); plt.close()
hourly = summaries["hour_summary"]
plt.figure(figsize=(10,5)); sns.barplot(data=hourly, x="hour", y="transaction_count", color="#3973ac"); plt.title("Transaction Volume by Hour"); plt.xlabel("Hour of Day"); plt.ylabel("Transactions"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/hourly_volume.png", dpi=150); plt.close()
device_summary = summaries["device_summary"]
plt.figure(figsize=(7,5)); sns.barplot(data=device_summary, x="new_device_flag", y="review_rate", color="#db8b35"); plt.xticks([0,1],["Known device","New device"]); plt.ylabel("Synthetic review rate"); plt.title("Review Rate by Device History"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/device_comparison.png", dpi=150); plt.close()
merchant_summary = summaries["merchant_category_summary"]
plt.figure(figsize=(10,5)); sns.barplot(data=merchant_summary, x="merchant_category", y="review_rate", color="#568b67"); plt.xticks(rotation=35, ha="right"); plt.ylabel("Synthetic review rate"); plt.title("Review Rate by Merchant Category"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/merchant_category_review_rate.png", dpi=150); plt.close()
region_summary = summaries["region_summary"]
plt.figure(figsize=(8,5)); sns.barplot(data=region_summary, x="customer_region", y="review_rate", color="#6c6fb0"); plt.ylabel("Synthetic review rate"); plt.title("Review Rate by Customer Region"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/regional_trends.png", dpi=150); plt.close()
plt.figure(figsize=(5,4)); sns.heatmap(confusion_matrix(y_test, predicted), annot=True, fmt="d", cmap="Blues", cbar=False); plt.xlabel("Predicted"); plt.ylabel("Actual synthetic label"); plt.title("Holdout Confusion Matrix"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/confusion_matrix.png", dpi=150); plt.close()

peak = hourly.loc[hourly.transaction_count.idxmax()]
device_new_rate = float(device_summary.loc[device_summary.new_device_flag == 1, "review_rate"].iloc[0])
device_known_rate = float(device_summary.loc[device_summary.new_device_flag == 0, "review_rate"].iloc[0])
top_category = merchant_summary.iloc[0]
findings = {
    "peak_volume_hour": int(peak.hour), "peak_volume_count": int(peak.transaction_count),
    "new_device_review_rate": device_new_rate, "known_device_review_rate": device_known_rate,
    "highest_review_rate_category": str(top_category.merchant_category), "highest_category_review_rate": float(top_category.review_rate),
    "note": "Synthetic data and simulated review-priority labels only; not confirmed fraud."
}
(ROOT / "reports/key_findings.json").write_text(json.dumps(findings, indent=2))

# 6. Excel workbook with readable tables and chart tabs.
workbook = Workbook()
summary_sheet = workbook.active
summary_sheet.title = "Executive Summary"
navy = "17365D"
summary_sheet["A1"] = "DIGITAL PAYMENT INSIGHT — SYNTHETIC DATA"
summary_sheet["A1"].font = Font(bold=True, size=16, color="FFFFFF")
summary_sheet["A1"].fill = PatternFill("solid", fgColor=navy)
summary_sheet.merge_cells("A1:D1")
workbook_rows = [
    ("Metric","Value"), ("Raw transactions",raw_rows), ("Clean transactions",len(clean),
    ), ("Duplicate records removed",raw_rows-len(clean)), ("Missing merchant categories filled",len(missing_indices)),
    ("Accuracy",metrics["accuracy"]), ("Precision",metrics["precision"]), ("Recall",metrics["recall"]),
    ("F1-score",metrics["f1_score"]), ("ROC-AUC",metrics["roc_auc"]),
    ("Peak hour",f"{int(peak.hour):02d}:00"), ("Transactions at peak hour",int(peak.transaction_count)),
    ("New-device synthetic review rate",device_new_rate), ("Known-device synthetic review rate",device_known_rate)
]
for row in workbook_rows: summary_sheet.append(row)
for cell in summary_sheet[2]:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=navy)
for row in summary_sheet.iter_rows(min_row=3):
    row[0].font = Font(bold=True)
    if "rate" in str(row[0].value).lower():
        row[1].number_format = "0.00%"
summary_sheet["A17"] = "All rows and labels are synthetic; scores do not measure real fraud detection."
summary_sheet.merge_cells("A17:D17")
for sheet_name, frame in [("Cleaned Data",clean), ("By Segment",summaries["segment_summary"]), ("By Hour",hourly), ("By Merchant",merchant_summary), ("By Device",device_summary), ("By Region",region_summary)]:
    sheet = workbook.create_sheet(sheet_name)
    sheet.append(list(frame.columns))
    for row in frame.astype(object).where(pd.notna(frame), "").itertuples(index=False, name=None):
        sheet.append(list(row))
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=navy)
    for column_idx, column_name in enumerate(frame.columns, 1):
        vals = [str(column_name)] + frame[column_name].astype(str).head(100).tolist()
        sheet.column_dimensions[sheet.cell(1, column_idx).column_letter].width = min(max(len(v) for v in vals) + 2, 28)
        if "rate" in str(column_name).lower():
            for row_idx in range(2, sheet.max_row + 1):
                sheet.cell(row_idx, column_idx).number_format = "0.00%"
for title, source, value_column in [("Hourly Volume Chart","By Hour",2), ("Merchant Rate Chart","By Merchant",4), ("Region Rate Chart","By Region",4)]:
    sheet = workbook.create_sheet(title)
    sheet["A1"] = title
    source_sheet = workbook[source]
    chart = BarChart()
    chart.title = title
    chart.add_data(Reference(source_sheet, min_col=value_column, min_row=1, max_row=source_sheet.max_row), titles_from_data=True)
    chart.set_categories(Reference(source_sheet, min_col=1, min_row=2, max_row=source_sheet.max_row))
    chart.height = 8; chart.width = 16
    sheet.add_chart(chart, "A3")
workbook.save(ROOT / "excel/digital_payment_analysis.xlsx")

# 7. Short business pitch deck.
presentation = Presentation()
def add_slide(title, bullets):
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = title
    frame = slide.placeholders[1].text_frame
    frame.clear()
    for index, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        paragraph.font.size = Pt(21)
add_slide("Digital Payment Insight", ["Use Case 13 | DS Sprint Assessment 2", "Synthetic educational data for transaction monitoring", "Goal: identify patterns that deserve operational review"])
add_slide("Business Question & Data", [f"{raw_rows:,} raw rows → {len(clean):,} cleaned transactions", "Five linked tables: transactions, customers, devices, merchants and alerts", "Injected issues include duplicate IDs, missing categories and inconsistent channel labels"])
add_slide("Cleaning & Feature Engineering", ["Removed duplicate transaction IDs and filled missing categories from merchant reference data", "Standardized channels, parsed times and joined reference tables", "Features: log amount, hour, night-time, new device and region mismatch"])
add_slide("EDA Findings", [f"Peak transaction hour: {int(peak.hour):02d}:00 ({int(peak.transaction_count):,} rows)", f"New-device synthetic review rate: {device_new_rate:.1%}; known device: {device_known_rate:.1%}", f"Highest category synthetic review rate: {top_category.merchant_category} ({top_category.review_rate:.1%})"])
add_slide("Machine Learning", ["Algorithm: Random Forest Classifier", f"Accuracy {metrics['accuracy']:.3f} | Precision {metrics['precision']:.3f} | Recall {metrics['recall']:.3f}", f"F1-score {metrics['f1_score']:.3f} | ROC-AUC {metrics['roc_auc']:.3f}", "Metrics evaluate synthetic proxy labels only"])
add_slide("Monitoring Priorities", ["Review combinations of new-device use, unusual amounts, night-time activity and region mismatch", "Use hourly, category and region trends to guide human review", "Do not automatically block payments with this prototype"])
add_slide("Recommendations", ["Validate thresholds against confirmed historical review outcomes", "Measure false positives and false negatives with domain experts", "Keep human review and governance in the decision loop"])
add_slide("Conclusion & Limitations", ["End-to-end synthetic data, EDA, ML, Excel and reporting workflow", "Educational prototype only—not validated fraud detection", "Replace synthetic data and labels with approved real data before operational deployment"])
presentation.save(ROOT / "presentation/digital_payment_insight_pitch.pptx")
print(json.dumps({"output_directory": str(ROOT), "raw_rows": raw_rows, "clean_rows": len(clean), "metrics": metrics, "findings": findings}, indent=2))
