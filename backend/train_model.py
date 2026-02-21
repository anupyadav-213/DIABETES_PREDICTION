# ==========================================
# DIABETES PREDICTION - ENHANCED ML PIPELINE
# ==========================================

# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# 2. Load Dataset
# ==========================================
df = pd.read_csv("diabetes.csv")
print("Dataset Shape:", df.shape)
print(df.head())
print("\nClass Distribution:\n", df['Outcome'].value_counts())

# ==========================================
# 3. Feature Engineering
# ==========================================
# Fix missing values (0 is not physiologically valid for these)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Add meaningful interaction features
df['Glucose_BMI']          = df['Glucose'] * df['BMI']
df['Age_BMI']              = df['Age'] * df['BMI']
df['Glucose_Insulin_ratio'] = df['Glucose'] / (df['Insulin'] + 1)
df['BMI_category']         = pd.cut(df['BMI'],
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=[0, 1, 2, 3]).astype(int)
df['Age_group']            = pd.cut(df['Age'],
                                    bins=[0, 30, 45, 60, 100],
                                    labels=[0, 1, 2, 3]).astype(int)
df['High_glucose']         = (df['Glucose'] > 140).astype(int)

print("\nNew Feature Set:", df.columns.tolist())

# ==========================================
# 4. Split Features and Target
# ==========================================
X = df.drop('Outcome', axis=1)
print("Number of features:", X.shape[1])
y = df['Outcome']

# ==========================================
# 5. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 6. Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==========================================
# 7. Handle Class Imbalance (SMOTE)
# ==========================================
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("\nAfter SMOTE:", np.bincount(y_train_res))

# ==========================================
# 8. Model Dictionary (Tuned Hyperparams)
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(
        C=0.5, max_iter=1000, class_weight='balanced', solver='lbfgs'
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    "SVM": CalibratedClassifierCV(
        SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        eval_metric='logloss', use_label_encoder=False, random_state=42
    ),
}

# ==========================================
# 9. Train, Evaluate & Collect Metrics
# ==========================================
results    = {}
roc_data   = {}
cv_scores  = {}
skf        = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n" + "="*60)
for name, model in models.items():
    # Fit
    model.fit(X_train_res, y_train_res)
    y_pred  = model.predict(X_test_scaled)

    # Probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_test_scaled)

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name]  = {"Accuracy": acc, "ROC-AUC": roc_auc}
    roc_data[name] = (fpr, tpr, roc_auc)

    # Cross-validation on original (unSMOTEd) data
    cv = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_scores[name] = cv

    print(f"\nModel: {name}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print(f"  CV ROC-AUC    : {cv.mean():.4f} ± {cv.std():.4f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("  Classification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# 10. Visualizations
# ==========================================
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11})

# --- Figure 1: EDA ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight='bold', y=1.01)

raw = pd.read_csv("diabetes.csv")
eda_features = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'DiabetesPedigreeFunction']
colors = ['#4C72B0', '#DD8452']

for ax, feat in zip(axes.flatten(), eda_features):
    for outcome, color in zip([0, 1], colors):
        subset = raw[raw['Outcome'] == outcome][feat]
        ax.hist(subset, bins=25, alpha=0.6, color=color,
                label=f"{'No Diabetes' if outcome==0 else 'Diabetes'}")
    ax.set_title(feat)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("eda_distributions.png", bbox_inches='tight')
# plt.show()
print("Saved: eda_distributions.png")

# --- Figure 2: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches='tight')
# plt.show()
print("Saved: correlation_heatmap.png")

# --- Figure 3: Model Comparison (Accuracy & ROC-AUC) ---
model_names = list(results.keys())
accuracies  = [results[m]["Accuracy"] for m in model_names]
aucs        = [results[m]["ROC-AUC"]  for m in model_names]

x   = np.arange(len(model_names))
w   = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - w/2, accuracies, w, label='Accuracy', color='#4C72B0', alpha=0.85)
bars2 = ax.bar(x + w/2, aucs,       w, label='ROC-AUC',  color='#DD8452', alpha=0.85)

ax.set_ylabel('Score')
ax.set_title('Model Comparison: Accuracy vs ROC-AUC', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylim(0.6, 1.0)
ax.legend()

for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{bar.get_height():.3f}', ha='center', fontsize=9)
for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{bar.get_height():.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches='tight')
# plt.show()
print("Saved: model_comparison.png")

# --- Figure 4: ROC Curves ---
fig, ax = plt.subplots(figsize=(9, 7))
palette = sns.color_palette("tab10", len(models))
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance')

for (name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), palette):
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC={roc_auc:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – All Models", fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("roc_curves.png", bbox_inches='tight')
# plt.show()
print("Saved: roc_curves.png")

# --- Figure 5: Cross-Validation Box Plots ---
fig, ax = plt.subplots(figsize=(12, 6))
cv_df = pd.DataFrame(cv_scores)
cv_df.boxplot(ax=ax, patch_artist=True,
              boxprops=dict(facecolor='#4C72B0', alpha=0.6),
              medianprops=dict(color='orange', lw=2))
ax.set_title('10-Fold Cross-Validation ROC-AUC Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('ROC-AUC Score')
ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.axhline(0.80, color='red', linestyle='--', lw=1.2, label='0.80 baseline')
ax.legend()
plt.tight_layout()
plt.savefig("cv_boxplots.png", bbox_inches='tight')
# plt.show()
print("Saved: cv_boxplots.png")

# --- Figure 6: Confusion Matrices Grid ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold')

for ax, (name, model) in zip(axes.flatten(), models.items()):
    y_pred = model.predict(X_test_scaled)
    cm     = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig("confusion_matrices.png", bbox_inches='tight')
# plt.show()
print("Saved: confusion_matrices.png")

# --- Figure 7: Feature Importance (XGBoost) ---
xgb_model = models["XGBoost"]
importances = xgb_model.feature_importances_
feat_names  = X.columns.tolist()
feat_df     = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_df     = feat_df.sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
colors_bar = ['#4C72B0' if i < len(feat_df)-3 else '#DD8452'
              for i in range(len(feat_df))]
ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors_bar, alpha=0.85)
ax.set_title("XGBoost Feature Importances", fontsize=14, fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches='tight')
# plt.show()
print("Saved: feature_importance.png")

# ==========================================
# 11. Final Summary Table
# ==========================================
print("\n" + "="*55)
print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
print("="*55)
for name in model_names:
    acc = results[name]["Accuracy"]
    ra  = results[name]["ROC-AUC"]
    print(f"{name:<25} {acc:>10.4f} {ra:>10.4f}")
print("="*55)

best_model = max(results, key=lambda m: results[m]["ROC-AUC"])
print(f"\n🏆 Best Model: {best_model}")
print(f"   Accuracy : {results[best_model]['Accuracy']:.4f}")
print(f"   ROC-AUC  : {results[best_model]['ROC-AUC']:.4f}")

import joblib

joblib.dump(scaler, "scaler.pkl")
joblib.dump(models[best_model], "model.pkl")

print("Model saved successfully!")

# import pickle

# # Save model
# pickle.dump(model, open("model.pkl", "wb"))

# # Save scaler
# pickle.dump(scaler, open("scaler.pkl", "wb"))

# print("Model and scaler saved successfully!")