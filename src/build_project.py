"""
Digital Payment Insight — reproducible synthetic-data project.
Run from repository root: python src/build_project.py
All records and review labels are simulated for educational use, not real fraud data.
"""
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

# Reference tables.
n_customers, n_merchants, n_devices = 1800, 400, 3600
customer_ids = np.array([f"C{i:05d}" for i in range(1, n_customers + 1)])
merchant_ids = np.array([f"M{i:04d}" for i in range(1, n_merchants + 1)])
device_ids = np.array([f"D{i:05d}" for i in range(1, n_devices + 1)])
segments = RNG.choice(["Student", "Regular", "Premium", "Business"], n_customers, p=[.22, .48, .22, .08])
regions = RNG.choice(["North", "South", "East", "West", "Central"], n_customers, p=[.20, .27, .15, .23, .15])
customers = pd.DataFrame({"customer_id":customer_ids, "customer_segment":segments, "customer_region":regions,
                          "tenure_months":np.clip(RNG.gamma(2.5,18,n_customers).astype(int),1,120)})
categories = np.array(["Grocery","Dining","Travel","Electronics","Utilities","Fashion","Gaming","Healthcare","Fuel","Other"])
merchants = pd.DataFrame({"merchant_id":merchant_ids, "merchant_category":RNG.choice(categories,n_merchants),
                          "merchant_region":RNG.choice(["North","South","East","West","Central"],n_merchants)})
devices = pd.DataFrame({"device_id":device_ids,
                        "device_type":RNG.choice(["Android","iOS","Web","Tablet"],n_devices,p=[.46,.28,.20,.06]),
                        "new_device_flag":RNG.choice([0,1],n_devices,p=[.84,.16]),
                        "device_history_complete":RNG.choice([1,0],n_devices,p=[.94,.06])})

# Transactions distributed across 30 days, with evening concentration and injected extreme activity.
hour_probs=np.array([.01,.005,.003,.002,.002,.003,.01,.025,.055,.065,.06,.06,.065,.06,.055,.05,.05,.06,.075,.085,.075,.055,.03,.015])
hour_probs=hour_probs/hour_probs.sum()
hours=RNG.choice(np.arange(24),N,p=hour_probs)
times=(pd.Timestamp("2026-09-01")+pd.to_timedelta(RNG.integers(0,30,N),unit="D")
       +pd.to_timedelta(hours,unit="h")+pd.to_timedelta(RNG.integers(0,3600,N),unit="s"))
ci,mi,di=RNG.integers(0,n_customers,N),RNG.integers(0,n_merchants,N),RNG.integers(0,n_devices,N)
segment,category=segments[ci],merchants.merchant_category.to_numpy()[mi]
new_device=devices.new_device_flag.to_numpy()[di]
mismatch=(regions[ci]!=merchants.merchant_region.to_numpy()[mi])
base={"Student":250,"Regular":650,"Premium":1500,"Business":2200}
amount=np.array([base[x] for x in segment])*RNG.lognormal(0,.72,N)
amount*=np.select([category=="Travel",category=="Electronics",category=="Utilities",category=="Grocery",category=="Gaming"],
                  [2.1,2.6,.45,.55,.8],default=1.0)
night=np.isin(hours,[0,1,2,3,4])
spike=RNG.random(N)<.035
amount=np.where(spike,amount*RNG.uniform(5,12,N),amount)
amount=np.where((new_device==1)&night,amount*RNG.uniform(1.4,2.5,N),amount)
log_amt=np.log1p(amount)
threshold=np.select([segment=="Student",segment=="Regular",segment=="Premium",segment=="Business"],[7.9,8.6,9.3,9.8])
risk=1.15*(log_amt>threshold)+1.1*(new_device==1)+night+.7*mismatch+1.5*spike+RNG.normal(0,.5,N)
prob=1/(1+np.exp(-(risk-2.25)))
labels=(RNG.random(N)<prob).astype(int)
channels=RNG.choice(["Mobile App","Web","UPI","Card","mobile app","WEB","upi"],N,p=[.30,.16,.22,.12,.08,.06,.06])
transactions=pd.DataFrame({"transaction_id":[f"T{i:07d}" for i in range(1,N+1)],"customer_id":customer_ids[ci],
 "merchant_id":merchant_ids[mi],"device_id":device_ids[di],"transaction_time":times,
 "amount":np.round(amount,2),"merchant_category":category,"channel":channels})
missing_idx=RNG.choice(N,int(N*.025),replace=False)
transactions.loc[missing_idx,"merchant_category"]=np.nan
transactions=pd.concat([transactions,transactions.sample(35,random_state=13)],ignore_index=True)
label_map=dict(zip([f"T{i:07d}" for i in range(1,N+1)],labels))
unique_tx=transactions.drop_duplicates("transaction_id",keep="first")
label_series=unique_tx.transaction_id.map(label_map).fillna(0).astype(int)
alerts=pd.DataFrame({"transaction_id":unique_tx.transaction_id,
 "review_status":np.where(label_series.to_numpy()==1,RNG.choice(["Reviewed","Pending"],len(unique_tx),p=[.7,.3]),"No review"),
 "reason_category":np.where(label_series.to_numpy()==1,RNG.choice(["High amount","New device","Unusual hour","Region mismatch","Multiple signals"],len(unique_tx)),"Routine"),
 "synthetic_review_label":label_series.to_numpy()})
for name,frame in [("transactions",transactions),("customers",customers),("devices",devices),("merchants",merchants),("alerts",alerts)]:
    frame.to_csv(ROOT/"data/raw"/f"{name}.csv",index=False)

# Clean and join reference data.
clean=transactions.drop_duplicates("transaction_id",keep="first").copy()
raw_rows=len(transactions)
clean["transaction_time"]=pd.to_datetime(clean["transaction_time"],errors="coerce")
clean["channel"]=clean["channel"].astype(str).str.strip().str.lower().replace({"mobile app":"mobile_app","web":"web","upi":"upi","card":"card"})
clean["merchant_category"]=clean["merchant_category"].fillna("Unknown")
cat_map=merchants.set_index("merchant_id")["merchant_category"]
unknown=clean["merchant_category"].eq("Unknown")
clean.loc[unknown,"merchant_category"]=clean.loc[unknown,"merchant_id"].map(cat_map).fillna("Unknown")
clean["amount"]=pd.to_numeric(clean["amount"],errors="coerce")
amount_cap=clean["amount"].quantile(.995)
clean["amount_capped"]=clean["amount"].clip(upper=amount_cap)
clean["hour"]=clean["transaction_time"].dt.hour
clean["day_of_week"]=clean["transaction_time"].dt.day_name()
clean["is_night"]=clean["hour"].isin([0,1,2,3,4]).astype(int)
clean=clean.merge(customers,on="customer_id",how="left")
clean=clean.merge(devices[["device_id","device_type","new_device_flag","device_history_complete"]],on="device_id",how="left")
clean=clean.merge(merchants[["merchant_id","merchant_region"]],on="merchant_id",how="left")
clean["region_mismatch"]=(clean["customer_region"]!=clean["merchant_region"]).astype(int)
clean["log_amount"]=np.log1p(clean["amount_capped"].clip(lower=0))
clean["synthetic_review_label"]=clean["transaction_id"].map(label_map).fillna(0).astype(int)
clean=clean.merge(alerts[["transaction_id","review_status","reason_category"]],on="transaction_id",how="left")
clean.to_csv(ROOT/"data/processed/cleaned_transactions.csv",index=False)

summaries={
 "segment_summary":clean.groupby("customer_segment").agg(transaction_count=("transaction_id","count"),median_amount=("amount","median"),p95_amount=("amount",lambda x:x.quantile(.95)),review_rate=("synthetic_review_label","mean")).reset_index(),
 "hour_summary":clean.groupby("hour").agg(transaction_count=("transaction_id","count"),review_rate=("synthetic_review_label","mean"),median_amount=("amount","median")).reset_index(),
 "merchant_category_summary":clean.groupby("merchant_category").agg(transaction_count=("transaction_id","count"),review_count=("synthetic_review_label","sum"),review_rate=("synthetic_review_label","mean"),median_amount=("amount","median")).reset_index().sort_values("review_rate",ascending=False),
 "device_summary":clean.groupby("new_device_flag").agg(transaction_count=("transaction_id","count"),review_rate=("synthetic_review_label","mean"),median_amount=("amount","median")).reset_index(),
 "region_summary":clean.groupby("customer_region").agg(transaction_count=("transaction_id","count"),review_count=("synthetic_review_label","sum"),review_rate=("synthetic_review_label","mean")).reset_index()
}
for name,frame in summaries.items():
    frame.to_csv(ROOT/"data/processed"/f"{name}.csv",index=False)

# Random Forest baseline on an 80/20 stratified holdout.
features=["log_amount","hour","is_night","new_device_flag","region_mismatch","tenure_months","customer_segment","merchant_category","channel","device_type","customer_region"]
target="synthetic_review_label"
X,y=clean[features],clean[target]
numeric=["log_amount","hour","is_night","new_device_flag","region_mismatch","tenure_months"]
categorical=["customer_segment","merchant_category","channel","device_type","customer_region"]
preprocess=ColumnTransformer([
 ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
 ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
])
model=Pipeline([("preprocess",preprocess),("classifier",RandomForestClassifier(n_estimators=180,min_samples_leaf=3,class_weight="balanced_subsample",random_state=49113,n_jobs=-1))])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=49113,stratify=y)
model.fit(X_train,y_train)
pred=model.predict(X_test); score=model.predict_proba(X_test)[:,1]
metrics={"dataset_rows_after_cleaning":int(len(clean)),"raw_transaction_rows":int(raw_rows),
 "duplicate_rows_removed":int(raw_rows-len(clean)),"missing_merchant_categories_filled":int(len(missing_idx)),
 "test_rows":int(len(X_test)),"target_positive_rate":float(y.mean()),
 "accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred,zero_division=0)),
 "recall":float(recall_score(y_test,pred,zero_division=0)),"f1_score":float(f1_score(y_test,pred,zero_division=0)),
 "roc_auc":float(roc_auc_score(y_test,score)),"confusion_matrix":confusion_matrix(y_test,pred).tolist(),
 "threshold_note":"Default classifier threshold 0.50; synthetic proxy labels only."}
(ROOT/"reports/model_metrics.json").write_text(json.dumps(metrics,indent=2))
joblib.dump(model,ROOT/"models/synthetic_review_model.joblib")

# Exploratory visuals.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9,5)); sns.histplot(clean["amount"],bins=60,log_scale=(False,True)); plt.title("Transaction Amount Distribution"); plt.xlabel("Amount"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/amount_distribution.png",dpi=150); plt.close()
hour_df=summaries["hour_summary"]
plt.figure(figsize=(10,5)); sns.barplot(data=hour_df,x="hour",y="transaction_count",color="#3973ac"); plt.title("Transaction Volume by Hour"); plt.xlabel("Hour of Day"); plt.ylabel("Transactions"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/hourly_volume.png",dpi=150); plt.close()
device_df=summaries["device_summary"]
plt.figure(figsize=(7,5)); sns.barplot(data=device_df,x="new_device_flag",y="review_rate",color="#db8b35"); plt.xticks([0,1],["Known device","New device"]); plt.ylabel("Synthetic review rate"); plt.title("Review Rate by Device History"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/device_comparison.png",dpi=150); plt.close()
cat_df=summaries["merchant_category_summary"]
plt.figure(figsize=(10,5)); sns.barplot(data=cat_df,x="merchant_category",y="review_rate",color="#568b67"); plt.xticks(rotation=35,ha="right"); plt.ylabel("Synthetic review rate"); plt.title("Review Rate by Merchant Category"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/merchant_category_review_rate.png",dpi=150); plt.close()
region_df=summaries["region_summary"]
plt.figure(figsize=(8,5)); sns.barplot(data=region_df,x="customer_region",y="review_rate",color="#6c6fb0"); plt.ylabel("Synthetic review rate"); plt.title("Review Rate by Customer Region"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/regional_trends.png",dpi=150); plt.close()
plt.figure(figsize=(5,4)); sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt="d",cmap="Blues",cbar=False); plt.xlabel("Predicted"); plt.ylabel("Actual synthetic label"); plt.title("Holdout Confusion Matrix"); plt.tight_layout(); plt.savefig(ROOT/"reports/figures/confusion_matrix.png",dpi=150); plt.close()

peak=hour_df.loc[hour_df.transaction_count.idxmax()]
new_rate=float(device_df.loc[device_df.new_device_flag==1,"review_rate"].iloc[0])
known_rate=float(device_df.loc[device_df.new_device_flag==0,"review_rate"].iloc[0])
topcat=cat_df.iloc[0]
findings={"peak_hour":int(peak.hour),"peak_hour_transactions":int(peak.transaction_count),
 "new_device_review_rate":new_rate,"known_device_review_rate":known_rate,
 "highest_category_by_synthetic_review_rate":str(topcat.merchant_category),
 "highest_category_review_rate":float(topcat.review_rate),
 "note":"Synthetic dataset and simulated review-priority labels only; not confirmed fraud."}
(ROOT/"reports/key_findings.json").write_text(json.dumps(findings,indent=2))

# Excel workbook: executive summary, cleaned data, analysis tables and chart tabs.
wb=Workbook(); ws=wb.active; ws.title="Executive Summary"
navy="17365D"
ws["A1"]="DIGITAL PAYMENT INSIGHT — SYNTHETIC DATA"; ws["A1"].font=Font(bold=True,size=16,color="FFFFFF"); ws["A1"].fill=PatternFill("solid",fgColor=navy); ws.merge_cells("A1:D1")
rows=[("Metric","Value"),("Raw transactions",raw_rows),("Clean transactions",len(clean)),("Duplicate records removed",raw_rows-len(clean)),("Missing merchant categories filled",len(missing_idx)),("Accuracy",metrics["accuracy"]),("Precision",metrics["precision"]),("Recall",metrics["recall"]),("F1-score",metrics["f1_score"]),("ROC-AUC",metrics["roc_auc"]),("Peak hour",f"{int(peak.hour):02d}:00"),("Transactions at peak hour",int(peak.transaction_count)),("New-device synthetic review rate",new_rate),("Known-device synthetic review rate",known_rate)]
for row in rows: ws.append(row)
for cell in ws[2]: cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor=navy)
for row in ws.iter_rows(min_row=3):
    row[0].font=Font(bold=True)
    if row[0].value and "rate" in str(row[0].value).lower(): row[1].number_format="0.00%"
ws["A17"]="All rows and labels are synthetic. Scores do not measure real fraud detection."; ws.merge_cells("A17:D17")
for name,frame in [("Cleaned Data",clean),("By Segment",summaries["segment_summary"]),("By Hour",hour_df),("By Merchant",cat_df),("By Device",device_df),("By Region",region_df)]:
    sh=wb.create_sheet(name)
    sh.append(list(frame.columns))
    for row in frame.astype(object).where(pd.notna(frame),"").itertuples(index=False,name=None): sh.append(list(row))
    sh.freeze_panes="A2"; sh.auto_filter.ref=sh.dimensions
    for cell in sh[1]: cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor=navy)
    for idx,col in enumerate(frame.columns,1):
        sh.column_dimensions[sh.cell(1,idx).column_letter].width=min(max(len(str(x)) for x in [col]+frame[col].astype(str).head(100).tolist())+2,28)
        if "rate" in str(col).lower():
            for rr in range(2,sh.max_row+1): sh.cell(rr,idx).number_format="0.00%"
for title,source,ycol in [("Hourly Volume Chart","By Hour",2),("Merchant Rate Chart","By Merchant",4),("Region Rate Chart","By Region",4)]:
    sh=wb.create_sheet(title); sh["A1"]=title; src=wb[source]; chart=BarChart(); chart.title=title
    chart.add_data(Reference(src,min_col=ycol,min_row=1,max_row=src.max_row),titles_from_data=True)
    chart.set_categories(Reference(src,min_col=1,min_row=2,max_row=src.max_row)); chart.height=8; chart.width=16; sh.add_chart(chart,"A3")
wb.save(ROOT/"excel/digital_payment_analysis.xlsx")

# Short business pitch deck.
prs=Presentation()
def add_slide(title,bullets):
    slide=prs.slides.add_slide(prs.slide_layouts[1]); slide.shapes.title.text=title
    tf=slide.placeholders[1].text_frame; tf.clear()
    for i,text in enumerate(bullets):
        para=tf.paragraphs[0] if i==0 else tf.add_paragraph(); para.text=text; para.level=0; para.font.size=Pt(21)
add_slide("Digital Payment Insight",["Use Case 13 | DS Sprint Assessment 2","Synthetic educational data for transaction-monitoring analysis","Goal: identify unusual patterns that deserve operational review"])
add_slide("Business Question & Data",[f"{raw_rows:,} raw records → {len(clean):,} cleaned transactions","Five linked tables: transactions, customers, devices, merchants and alerts","Injected data issues include duplicates, missing categories and inconsistent labels"])
add_slide("Cleaning & Features",["Removed duplicate transaction IDs and imputed missing categories from merchant reference data","Standardized channels, parsed timestamps and joined context tables","Engineered log amount, hour, night-time, new-device and region-mismatch features"])
add_slide("EDA Findings",[f"Peak transaction hour: {int(peak.hour):02d}:00 ({int(peak.transaction_count):,} records)",f"New-device synthetic review rate: {new_rate:.1%}; known-device: {known_rate:.1%}",f"Highest-category synthetic review rate: {topcat.merchant_category} ({topcat.review_rate:.1%})"])
add_slide("Machine Learning",["Algorithm: Random Forest Classifier",f"Accuracy {metrics['accuracy']:.3f} | Precision {metrics['precision']:.3f} | Recall {metrics['recall']:.3f}",f"F1-score {metrics['f1_score']:.3f} | ROC-AUC {metrics['roc_auc']:.3f}","Metrics reflect simulated labels only; model performance is modest"])
add_slide("Monitoring Priorities",["Review combinations of new-device use, amount deviations, night-time activity and region mismatch","Use hourly, regional and category patterns to guide analyst attention","Do not automatically block transactions with this prototype"])
add_slide("Recommendations",["Validate thresholds using confirmed historical review outcomes","Measure false positives and false negatives with domain experts","Retain human review, governance and monitoring"])
add_slide("Conclusion & Limitations",["End-to-end synthetic data, EDA, ML, Excel and reporting workflow","Educational prototype only—not validated fraud detection","Replace synthetic data and labels with approved real data before deployment"])
prs.save(ROOT/"presentation/digital_payment_insight_pitch.pptx")
print(json.dumps({"output_directory":str(ROOT),"raw_rows":raw_rows,"clean_rows":len(clean),"metrics":metrics,"findings":findings},indent=2))
