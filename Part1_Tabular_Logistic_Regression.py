# =============================================================================
# PART 1 : LOGISTIC REGRESSION ON TABULAR DATA
# Dataset : Breast Cancer Wisconsin (Diagnostic)
# Kaggle  : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 : LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PART 1 : LOGISTIC REGRESSION — TABULAR DATA")
print("  Dataset : Breast Cancer Wisconsin (Diagnostic)")
print("  Kaggle  : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")
print("=" * 65)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

print(f"\n[INFO] Dataset Shape   : {df.shape}")
print(f"[INFO] Number of Features : {len(data.feature_names)}")
print(f"[INFO] Classes         : {list(data.target_names)}")
print(f"[INFO] Missing Values  : {df.isnull().sum().sum()}")
print(f"\n[Class Distribution]")
print(df['diagnosis'].value_counts().to_string())
print(f"\n[First 5 Rows]")
print(df[list(data.feature_names[:5]) + ['diagnosis']].head())
print(f"\n[Statistical Summary (first 5 features)]")
print(df[list(data.feature_names[:5])].describe().round(4))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 : EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Generating EDA plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Breast Cancer Dataset — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1 : Class Distribution
counts = df['diagnosis'].value_counts()
bars = axes[0, 0].bar(counts.index, counts.values, color=['#e74c3c', '#2ecc71'], edgecolor='black', width=0.5)
axes[0, 0].set_title('Class Distribution', fontsize=13)
axes[0, 0].set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, val + 2, str(val),
                    ha='center', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0, max(counts.values) + 40)

# Plot 2 : Feature Correlation Heatmap (top 8 features)
top_features = list(data.feature_names[:8]) + ['target']
corr = df[top_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[0, 1], annot_kws={'size': 7}, linewidths=0.5)
axes[0, 1].set_title('Feature Correlation Heatmap (Top 8 Features)', fontsize=13)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=7)
axes[0, 1].tick_params(axis='y', rotation=0, labelsize=7)

# Plot 3 : Mean Radius Distribution by Class
for label, color in zip(['Malignant', 'Benign'], ['#e74c3c', '#2ecc71']):
    subset = df[df['diagnosis'] == label]['mean radius']
    axes[1, 0].hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor='black')
axes[1, 0].set_title('Distribution of Mean Radius by Class', fontsize=13)
axes[1, 0].set_xlabel('Mean Radius')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Plot 4 : Box Plot — Mean Area by Diagnosis
df.boxplot(column='mean area', by='diagnosis', ax=axes[1, 1],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue', color='navy'),
           medianprops=dict(color='red', linewidth=2))
axes[1, 1].set_title('Mean Area by Diagnosis', fontsize=13)
axes[1, 1].set_xlabel('Diagnosis')
axes[1, 1].set_ylabel('Mean Area')
plt.sca(axes[1, 1])
plt.title('Mean Area by Diagnosis')

plt.tight_layout()
plt.savefig('d:\\ML Assignment 1\\part1_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part1_eda.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 : PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
X = df[list(data.feature_names)]
y = df['target']

# Train / Test Split (80% / 20%), stratified to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling — StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)          # transform test with same params

print(f"\n[Preprocessing Complete]")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 : TRAIN LOGISTIC REGRESSION MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Training Logistic Regression model...")

model = LogisticRegression(
    solver='lbfgs',    # Good default solver for small/medium datasets
    max_iter=10000,    # Enough iterations to converge
    C=1.0,             # Regularization strength (inverse) — default
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  CV Scores per Fold : {[round(s, 4) for s in cv_scores]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 : EVALUATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*45}")
print(f"  TEST RESULTS")
print(f"{'='*45}")
print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 : EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Generating evaluation plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Logistic Regression — Tabular Data Evaluation', fontsize=15, fontweight='bold')

# — Confusion Matrix —
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'Confusion Matrix\n(Accuracy: {acc * 100:.2f}%)', fontsize=12)

# — ROC Curve —
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='darkorange')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=11)
axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].set_title('ROC Curve', fontsize=12)
axes[1].legend(loc='lower right')

# — Learning Curve —
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)
t_mean, t_std = train_scores.mean(axis=1), train_scores.std(axis=1)
v_mean, v_std = val_scores.mean(axis=1), val_scores.std(axis=1)

axes[2].plot(train_sizes, t_mean, 'o-', color='#3498db', label='Training Accuracy', lw=2)
axes[2].plot(train_sizes, v_mean, 'o-', color='#27ae60', label='Validation Accuracy', lw=2)
axes[2].fill_between(train_sizes, t_mean - t_std, t_mean + t_std, alpha=0.15, color='#3498db')
axes[2].fill_between(train_sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color='#27ae60')
axes[2].set_title('Learning Curve', fontsize=12)
axes[2].set_xlabel('Training Set Size', fontsize=11)
axes[2].set_ylabel('Accuracy', fontsize=11)
axes[2].legend()
axes[2].set_ylim([0.88, 1.02])

plt.tight_layout()
plt.savefig('d:\\ML Assignment 1\\part1_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part1_evaluation.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 : FEATURE IMPORTANCE (Model Coefficients)
# ─────────────────────────────────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': model.coef_[0]
}).reindex(pd.Series(model.coef_[0]).abs().sort_values(ascending=False).index)
coef_df = coef_df.head(15)

plt.figure(figsize=(11, 6))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coefficient']]
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Top 15 Feature Importances — Logistic Regression Coefficients\n'
          '(Green = increases Benign probability | Red = increases Malignant probability)',
          fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
plt.tight_layout()
plt.savefig('d:\\ML Assignment 1\\part1_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part1_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PART 1 SUMMARY")
print("=" * 65)
print(f"  Dataset          : Breast Cancer Wisconsin (569 samples, 30 features)")
print(f"  Train/Test Split : 80% / 20%  →  {X_train.shape[0]} / {X_test.shape[0]} samples")
print(f"  Scaler           : StandardScaler (zero mean, unit variance)")
print(f"  Model            : LogisticRegression (solver=lbfgs, C=1.0)")
print(f"  5-Fold CV Acc    : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
print(f"  Test Accuracy    : {acc * 100:.2f}%")
print(f"  ROC-AUC Score    : {auc:.4f}")
print("=" * 65)
print("\n[PART 1 COMPLETE]\n")
