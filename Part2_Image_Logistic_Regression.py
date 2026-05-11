# =============================================================================
# PART 2 : LOGISTIC REGRESSION ON IMAGE DATA
# Dataset : Fashion-MNIST (Zalando Clothing Images)
# Kaggle  : https://www.kaggle.com/datasets/zalando-research/fashionmnist
#
# WHAT IS MNIST?
# MNIST = Modified National Institute of Standards and Technology
# A dataset of 70,000 handwritten digit images (0-9), created in 1998
# by Yann LeCun. It became the "Hello World" of machine learning.
#
# WHAT IS FASHION-MNIST?
# Created by Zalando Research (2017) as a harder, more realistic
# drop-in replacement for MNIST. Same format: 70,000 images, 28x28
# grayscale pixels, 10 classes — but clothing items, not digits.
#
# Classes:
#   0=T-shirt/Top  1=Trouser    2=Pullover  3=Dress   4=Coat
#   5=Sandal       6=Shirt      7=Sneaker   8=Bag     9=Ankle Boot
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]

# ---------------------------------------------------------------------------
# STEP 1 : LOAD DATASET
# ---------------------------------------------------------------------------
print("=" * 65)
print("  PART 2 : LOGISTIC REGRESSION - IMAGE DATA")
print("  Dataset : Fashion-MNIST (Zalando Clothing Images)")
print("  Kaggle  : https://www.kaggle.com/datasets/zalando-research/fashionmnist")
print("=" * 65)
print("""
  [MNIST vs Fashion-MNIST]
  MNIST (1998, Yann LeCun): 70,000 handwritten digit images (0-9).
  Considered too easy for modern models (>99% accuracy achievable).

  Fashion-MNIST (2017, Zalando): Same 28x28 grayscale format, same
  size, but 10 clothing categories. Much harder and more practical.
  We use this as our image classification challenge.
""")

print("[INFO] Loading Fashion-MNIST from OpenML cache...")
fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X_full = fmnist.data
y_full = fmnist.target.astype(int)

print(f"[INFO] Dataset loaded : {X_full.shape[0]} images, {X_full.shape[1]} features each")
print(f"[INFO] Image size     : 28x28 pixels flattened to 784 features")
print(f"[INFO] Pixel range    : {int(X_full.min())} to {int(X_full.max())}")

# ---------------------------------------------------------------------------
# STEP 2 : SAMPLE 12,000 IMAGES (subset for speed)
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 12000
rng = np.random.RandomState(42)
indices = rng.choice(len(X_full), SAMPLE_SIZE, replace=False)
X = X_full[indices]
y = y_full[indices]

print(f"\n[INFO] Using {SAMPLE_SIZE} samples from {len(X_full):,} total")
print("[INFO] Class Distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    bar = '#' * (count // 20)
    print(f"  {CLASS_NAMES[cls]:15s} (class {cls}): {count}  {bar}")

# ---------------------------------------------------------------------------
# STEP 3 : VISUALIZE SAMPLE IMAGES
# ---------------------------------------------------------------------------
print("\n[INFO] Generating sample image grid...")
fig, axes = plt.subplots(5, 10, figsize=(15, 8))
fig.suptitle('Fashion-MNIST — 5 Sample Images per Clothing Class',
             fontsize=14, fontweight='bold', y=1.01)

for cls in range(10):
    cls_indices = np.where(y == cls)[0]
    for row in range(5):
        ax = axes[row, cls]
        ax.imshow(X[cls_indices[row]].reshape(28, 28), cmap='gray_r', interpolation='nearest')
        if row == 0:
            ax.set_title(CLASS_NAMES[cls], fontsize=7.5, fontweight='bold', pad=3)
        ax.axis('off')

plt.tight_layout()
plt.savefig('d:/ML Assignment 1/part2_sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part2_sample_images.png")

# ---------------------------------------------------------------------------
# STEP 4 : PREPROCESSING
# ---------------------------------------------------------------------------
# Normalize pixel values: 0-255 -> 0.0-1.0
X_normalized = X / 255.0

# Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[Preprocessing Complete]")
print(f"  Normalization  : pixels / 255 -> [0.0, 1.0]")
print(f"  Train samples  : {X_train.shape[0]}")
print(f"  Test  samples  : {X_test.shape[0]}")
print(f"  Feature dims   : {X_train.shape[1]}  (28x28 flattened)")

# ---------------------------------------------------------------------------
# STEP 5 : TRAIN LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------------------------
print("\n[INFO] Training Logistic Regression...")
print("  Solver   : saga   (efficient for large/high-dimensional data)")
print("  Strategy : One-vs-Rest (OvR) — trains 10 binary classifiers")
print("  C        : 0.5    (moderate regularization for 784 features)")
print("  This may take 1-2 minutes...")

model = LogisticRegression(
    solver='saga',
    max_iter=300,
    C=0.5,
    multi_class='auto',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("[INFO] Training complete!")

# ---------------------------------------------------------------------------
# STEP 6 : EVALUATE
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("  TEST RESULTS")
print("=" * 55)
print(f"  Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
print("\n[Classification Report — Per Clothing Class]")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ---------------------------------------------------------------------------
# STEP 7 : CONFUSION MATRIX + PER-CLASS ACCURACY
# ---------------------------------------------------------------------------
print("[INFO] Generating evaluation plots...")
cm = confusion_matrix(y_test, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Logistic Regression — Fashion-MNIST Evaluation', fontsize=15, fontweight='bold')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(ax=axes[0], cmap='Blues', colorbar=True, xticks_rotation=45)
axes[0].set_title(f'Confusion Matrix — 10 Classes\n(Accuracy: {acc * 100:.2f}%)', fontsize=12)
short = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'A.Boot']
axes[0].set_xticklabels(short, rotation=40, ha='right', fontsize=8)
axes[0].set_yticklabels(short, fontsize=8)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
bars = axes[1].bar(range(10), per_class_acc, color=colors, edgecolor='black')
axes[1].set_xticks(range(10))
axes[1].set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Per-Class Recognition Accuracy', fontsize=12)
axes[1].set_ylim([0, 1.2])
avg_label = f'Overall Avg ({acc * 100:.1f}%)'
axes[1].axhline(y=acc, color='red', linestyle='--', lw=2, label=avg_label)
axes[1].legend(fontsize=10)
for bar, v in zip(bars, per_class_acc):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f'{v:.2f}', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('d:/ML Assignment 1/part2_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part2_evaluation.png")

# ---------------------------------------------------------------------------
# STEP 8 : CORRECT vs INCORRECT PREDICTIONS
# ---------------------------------------------------------------------------
correct_idx = np.where(y_pred == y_test)[0]
wrong_idx   = np.where(y_pred != y_test)[0]

fig, axes = plt.subplots(2, 10, figsize=(16, 4))
fig.suptitle('Sample Predictions  |  Row 1: Correct (Green)    Row 2: Incorrect (Red)',
             fontsize=11, fontweight='bold')

for i in range(10):
    # Row 1 — correct
    idx = correct_idx[i]
    ax = axes[0, i]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray_r')
    ax.set_title(CLASS_NAMES[y_pred[idx]][:7], fontsize=6.5, color='#27ae60', fontweight='bold')
    for s in ax.spines.values():
        s.set_edgecolor('#27ae60')
        s.set_linewidth(2.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 2 — incorrect
    if i < len(wrong_idx):
        idx = wrong_idx[i]
        ax = axes[1, i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='OrRd')
        pred_name = CLASS_NAMES[y_pred[idx]][:5]
        true_name = CLASS_NAMES[y_test[idx]][:5]
        ax.set_title(f'P:{pred_name}\nA:{true_name}', fontsize=5.8,
                     color='#e74c3c', fontweight='bold')
        for s in ax.spines.values():
            s.set_edgecolor('#e74c3c')
            s.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('d:/ML Assignment 1/part2_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part2_predictions.png")

# ---------------------------------------------------------------------------
# STEP 9 : LEARNED WEIGHT TEMPLATES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Learned Model Weights — What the Model Sees for Each Clothing Class\n'
             '(Red = strong activation for this class  |  Blue = suppression)',
             fontsize=11, fontweight='bold')

for cls in range(10):
    ax = axes[cls // 5, cls % 5]
    w = model.coef_[cls].reshape(28, 28)
    vmax = np.abs(w).max()
    im = ax.imshow(w, cmap='RdBu_r', interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax.set_title(CLASS_NAMES[cls], fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, format='%.1f')

plt.tight_layout()
plt.savefig('d:/ML Assignment 1/part2_learned_weights.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] part2_learned_weights.png")

# ---------------------------------------------------------------------------
# STEP 10 : MOST CONFUSED PAIRS
# ---------------------------------------------------------------------------
cm2 = cm.copy()
np.fill_diagonal(cm2, 0)
print("\n[Top 5 Most Confused Clothing Pairs]")
top5 = np.argsort(cm2.ravel())[::-1][:5]
for rank, fi in enumerate(top5, 1):
    a, p = fi // 10, fi % 10
    print(f"  {rank}. '{CLASS_NAMES[a]}' mistaken as '{CLASS_NAMES[p]}'  ({cm2[a, p]} times)")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  PART 2 SUMMARY")
print("=" * 65)
print(f"  Dataset    : Fashion-MNIST (Zalando)")
print(f"  Kaggle     : https://www.kaggle.com/datasets/zalando-research/fashionmnist")
print(f"  Samples    : {SAMPLE_SIZE} (from 70,000 total)")
print(f"  Images     : 28x28 grayscale -> 784 flattened features")
print(f"  Normalize  : pixels / 255 -> [0.0, 1.0]")
print(f"  Split      : 80/20 -> {X_train.shape[0]} train / {X_test.shape[0]} test")
print(f"  Model      : LogisticRegression (saga, C=0.5, One-vs-Rest)")
print(f"  Classes    : 10 clothing categories")
print(f"  Accuracy   : {acc * 100:.2f}%")
print("=" * 65)
print("\n[PART 2 COMPLETE]")
