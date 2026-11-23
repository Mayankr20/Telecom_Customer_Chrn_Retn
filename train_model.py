import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay)

RANDOM_STATE = 42
BASE_DIR = r"F:\Projects"
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 60)
print("TRAINING CUSTOMER CHURN RETENTION MODEL PIPELINE")
print("=" * 60)

# 1. Load Data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# TotalCharges has blank spaces for tenure=0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# 2. Feature Engineering (Exact features from customer retention analysis)
print("Performing feature engineering...")
data = df.copy()
cust_ids = data["customerID"]
data = data.drop(columns=["customerID"])

service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
data["NumServices"]       = sum((data[c] == "Yes").astype(int) for c in service_cols)
data["NoSupportServices"] = ((data["OnlineSecurity"] != "Yes") & (data["TechSupport"] != "Yes")).astype(int)
data["AvgMonthlySpend"]   = data["TotalCharges"] / data["tenure"].replace(0, 1)
data["BillingSpike"]      = data["MonthlyCharges"] - data["AvgMonthlySpend"]
data["IsMonthToMonth"]    = (data["Contract"] == "Month-to-month").astype(int)
data["AutoPay"]           = data["PaymentMethod"].str.contains("automatic").astype(int)
data["NewCustomer"]       = (data["tenure"] <= 6).astype(int)

X = data.drop(columns="Churn")
y = data["Churn"]

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgMonthlySpend", "BillingSpike"]
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
bin_cols = [c for c in X.columns if c not in num_cols + cat_cols]

print(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical, {len(bin_cols)} binary")

# 3. Train/Test Split
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, cust_ids, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 4. Preprocessor
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat_cols),
    ("bin", "passthrough", bin_cols),
])

Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)
feature_names = [n.split("__")[1] if "__" in n else n for n in pre.get_feature_names_out()]

print(f"Transformed shapes: Train {Xtr.shape}, Test {Xte.shape}")

# 5. Define Models
print("Configuring models...")
lr = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
gb = GradientBoostingClassifier(n_estimators=180, learning_rate=0.05, max_depth=3,
                                subsample=0.8, random_state=RANDOM_STATE)

stack = StackingClassifier(
    estimators=[("lr", lr), ("dt", dt), ("rf", rf), ("gb", gb)],
    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    stack_method="predict_proba",
    n_jobs=-1
)

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Stacked Ensemble": stack
}

# 6. Fit Models & Evaluate
print("Fitting models...")
rows = []
for name, m in models.items():
    t_start = time.time()
    m.fit(Xtr, y_train)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    dur = time.time() - t_start
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, p)
    rows.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "ROC-AUC": round(auc, 3),
        "TrainTimeSec": round(dur, 1)
    })
    print(f"  -> {name:20s} | ROC-AUC: {auc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} ({dur:.1f}s)")

metrics_df = pd.DataFrame(rows)
print("\nModel Benchmark Summary:")
print(metrics_df.to_string(index=False))

# 7. Threshold Tuning for Stacking Classifier
print("\nOptimizing decision threshold via Cross-Validation...")
oof = cross_val_predict(
    stack, Xtr, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=1),
    method="predict_proba"
)[:, 1]
ths = np.arange(0.2, 0.8, 0.01)
f1s = [f1_score(y_train, oof >= t) for t in ths]
best_t = float(ths[int(np.argmax(f1s))])
print(f"Optimal Decision Threshold: {best_t:.2f} (Max CV F1: {max(f1s):.3f})")

p_stack = stack.predict_proba(Xte)[:, 1]
pred_opt = (p_stack >= best_t).astype(int)
print(f"Test Set Performance at Threshold {best_t:.2f}:")
print(f"  Recall:    {recall_score(y_test, pred_opt):.3f}")
print(f"  Precision: {precision_score(y_test, pred_opt):.3f}")
print(f"  F1-Score:  {f1_score(y_test, pred_opt):.3f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, p_stack):.3f}")

# 8. Feature Importances
rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
gb_importances = pd.Series(gb.feature_importances_, index=feature_names).sort_values(ascending=False)
avg_importances = ((rf_importances + gb_importances) / 2.0).sort_values(ascending=False)

# 9. Visualizations
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# ROC Curves
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626"]
for (name, m), c in zip(models.items(), colors):
    p = m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    auc_val = roc_auc_score(y_test, p)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (0.50)")
ax.set_title("Model Comparison - ROC Curves", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontweight="bold")
ax.set_ylabel("True Positive Rate", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "roc_curves.png"), dpi=150)
plt.close(fig)

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, pred_opt)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Stay (0)", "Churn (1)"], yticklabels=["Stay (0)", "Churn (1)"], ax=ax)
ax.set_xlabel("Predicted Label", fontweight="bold")
ax.set_ylabel("Actual Label", fontweight="bold")
ax.set_title(f"Confusion Matrix (Threshold = {best_t:.2f})", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(9, 6))
avg_importances.head(15).iloc[::-1].plot(kind="barh", color="#4f46e5", ax=ax)
ax.set_title("Top 15 Churn Predictors (Ensemble Importance)", fontsize=13, fontweight="bold")
ax.set_xlabel("Relative Importance Score", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)

# 10. Generate Comprehensive Retention Dashboard CSV
def get_risk_tier(p):
    if p >= 0.70:
        return "HIGH"
    elif p >= best_t:
        return "MEDIUM"
    return "LOW"

def assign_retention_action(row, drivers):
    contract = str(row.get("Contract", ""))
    monthly = float(row.get("MonthlyCharges", 0))
    spike = float(row.get("BillingSpike", 0))
    tenure = float(row.get("tenure", 0))
    tech = str(row.get("TechSupport", ""))
    
    if spike > 10 or "Billing" in " ".join(drivers):
        return "Bill review & loyalty discount offer"
    elif contract == "Month-to-month" or "Month" in " ".join(drivers):
        return "Offer 1-year contract lock with 15% discount"
    elif tech == "No" or "Support" in " ".join(drivers):
        return "Complimentary Premium Tech Support setup"
    elif tenure <= 6:
        return "Dedicated Onboarding Specialist check-in"
    else:
        return "Executive VIP Account Review call"

print("\nGenerating customer drivers and retention interventions...")
test_records = []
for i in range(len(X_test)):
    cid = id_test.iloc[i]
    prob = float(p_stack[i])
    orig_row = X_test.iloc[i].to_dict()
    x_vec = Xte[i]
    
    # Fast local driver approximation: feature value scaled by ensemble importance
    contributions = []
    for f_idx, f_name in enumerate(feature_names):
        val = x_vec[f_idx]
        imp = avg_importances.get(f_name, 0.01)
        # Features with positive standard value that contribute toward risk
        if val > 0:
            score = val * imp
            contributions.append((f_name, score))
    
    contributions.sort(key=lambda x: -x[1])
    top3 = contributions[:3]
    while len(top3) < 3:
        top3.append(("tenure", 0.01))
    
    drv_strings = [f"{name} (+{round(sc, 2)})" for name, sc in top3]
    action = assign_retention_action(orig_row, [t[0] for t in top3])
    
    rec = {
        "customerID": cid,
        "ChurnProb": round(prob, 3),
        "Contract": orig_row["Contract"],
        "MonthlyCharges": orig_row["MonthlyCharges"],
        "Tenure": orig_row["tenure"],
        "Risk": get_risk_tier(prob),
        "Driver_1": drv_strings[0],
        "Driver_2": drv_strings[1],
        "Driver_3": drv_strings[2],
        "SuggestedAction": action,
        "InternetService": orig_row["InternetService"],
        "PaymentMethod": orig_row["PaymentMethod"],
        "TechSupport": orig_row["TechSupport"],
        "OnlineSecurity": orig_row["OnlineSecurity"],
        "TotalCharges": orig_row["TotalCharges"],
        "BillingSpike": round(orig_row["BillingSpike"], 2),
        "NumServices": orig_row["NumServices"]
    }
    test_records.append(rec)

scored_df = pd.DataFrame(test_records)
# Save prioritized at-risk customers for dashboard
retention_csv_path = os.path.join(BASE_DIR, "retention_dashboard.csv")
scored_df.sort_values("ChurnProb", ascending=False).to_csv(retention_csv_path, index=False)
print(f"Saved {len(scored_df)} scored customers to: {retention_csv_path}")

# 11. Save Complete Model Artifacts
artifacts = {
    "stack": stack,
    "models": models,
    "pre": pre,
    "feature_names": feature_names,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "bin_cols": bin_cols,
    "service_cols": service_cols,
    "best_t": best_t,
    "metrics_df": metrics_df,
    "avg_importances": avg_importances,
    "scored_df": scored_df
}

artifacts_path = os.path.join(BASE_DIR, "model_artifacts.joblib")
joblib.dump(artifacts, artifacts_path)
print(f"Model artifacts successfully dumped to: {artifacts_path}")
print("=" * 60)
print("TRAINING & ARTIFACT EXPORT COMPLETE!")
print("=" * 60)
