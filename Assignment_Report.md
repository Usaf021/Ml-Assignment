# LOGISTIC REGRESSION — MACHINE LEARNING ASSIGNMENT REPORT

**Subject:** Machine Learning  
**Topic:** Logistic Regression on Tabular and Image Data  
**Date:** April 24, 2026

---

## TABLE OF CONTENTS

1. [Introduction](#1-introduction)
2. [Theoretical Background](#2-theoretical-background)
3. [Datasets Used](#3-datasets-used)
4. [Part 1 — Tabular Data (Breast Cancer Wisconsin)](#4-part-1--tabular-data)
5. [Part 2 — Image Data (MNIST Handwritten Digits)](#5-part-2--image-data)
6. [Comparison of Results](#6-comparison-of-results)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Introduction

This assignment implements **Logistic Regression**, one of the fundamental supervised machine learning algorithms, on two different types of data:

- **Part 1 — Tabular/Structured Data:** The Breast Cancer Wisconsin Diagnostic dataset (CSV format)
- **Part 2 — Image Data:** The MNIST Handwritten Digits dataset (28×28 grayscale images)

The goal is to train classification models, evaluate their performance using standard metrics, and interpret the results through visualization.

---

## 2. Theoretical Background

### 2.1 What is Logistic Regression?

Logistic Regression is a **supervised classification algorithm** that models the probability of a binary (or multiclass) outcome using the **sigmoid (logistic) function**.

Despite its name, Logistic Regression is a **classification** algorithm, not a regression algorithm.

### 2.2 The Sigmoid Function

The core transformation is the sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:

$$z = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n = \mathbf{w}^T \mathbf{x} + b$$

This maps any real number to the range **(0, 1)**, interpreted as a **probability**.

### 2.3 Decision Boundary

The model predicts:
- **Class 1** if `P(y=1 | x) ≥ 0.5`
- **Class 0** if `P(y=1 | x) < 0.5`

The decision boundary is a **hyperplane** (linear boundary in feature space) defined by `w·x + b = 0`.

### 2.4 Cost Function — Binary Cross-Entropy Loss

Logistic Regression is trained by minimizing the **Log Loss** (Binary Cross-Entropy):

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where:
- `m` = number of training samples
- `y^(i)` = true label (0 or 1)
- `ŷ^(i)` = predicted probability

### 2.5 Regularization

To prevent overfitting, **L2 (Ridge) Regularization** is added:

$$J_{regularized} = J(\mathbf{w}, b) + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

The `C` parameter in scikit-learn is the **inverse of regularization strength**: `C = 1/λ`.  
- **Smaller C** → Stronger regularization → Simpler model  
- **Larger C** → Weaker regularization → More complex model

### 2.6 Multiclass Extension — One-vs-Rest (OvR)

For problems with more than 2 classes (e.g., digits 0–9 in MNIST), **One-vs-Rest (OvR)** strategy is used:
- Train **K separate binary classifiers** (one for each class)
- Classifier `k` distinguishes class `k` from all other classes
- Final prediction = class with the highest probability

### 2.7 Optimization — Gradient Descent

The weights are updated using gradient descent:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

Where:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

Scikit-learn provides solvers like **`lbfgs`** (quasi-Newton) and **`saga`** (stochastic gradient) that converge faster than basic gradient descent.

---

## 3. Datasets Used

### 3.1 Part 1 — Breast Cancer Wisconsin (Diagnostic)

| Property | Details |
|----------|---------|
| **Source** | UCI Machine Learning Repository |
| **Kaggle Link** | https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data |
| **Samples** | 569 |
| **Features** | 30 numeric features |
| **Classes** | 2 — Malignant (212), Benign (357) |
| **Task** | Binary Classification |
| **Missing Values** | None |

**Features include:**
- Mean, standard error, and worst values of: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- All features are real-valued (continuous numeric)

**Why this dataset?**
- Clean, well-documented, no missing values
- Real-world medical application (cancer diagnosis)
- Clear binary classification task
- Excellent benchmark for logistic regression

---

### 3.2 Part 2 — MNIST Handwritten Digits

| Property | Details |
|----------|---------|
| **Source** | Yann LeCun's MNIST Database |
| **Kaggle Link (Competition)** | https://www.kaggle.com/competitions/digit-recognizer |
| **Kaggle Link (Dataset)** | https://www.kaggle.com/datasets/hojjatk/mnist-dataset |
| **Total Samples** | 70,000 (we use 12,000 for speed) |
| **Image Size** | 28 × 28 pixels → 784 features (flattened) |
| **Classes** | 10 — digits 0 through 9 |
| **Task** | Multiclass Classification |
| **Pixel Values** | 0–255 (grayscale) |

**Why this dataset?**
- The "Hello World" of image classification
- Tests logistic regression on high-dimensional data (784 features)
- Multiclass problem (10 classes)
- Demonstrates image-to-vector pipeline

---

## 4. Part 1 — Tabular Data

### 4.1 Workflow

```
CSV Data → Load → EDA → Preprocess → Split → Scale → Train → Evaluate → Visualize
```

### 4.2 Exploratory Data Analysis (EDA)

Before training, the following analyses were performed:

1. **Class Distribution Check** — Verified slight class imbalance (63% Benign, 37% Malignant)
2. **Correlation Heatmap** — Identified highly correlated features (radius, perimeter, area are strongly correlated)
3. **Distribution Plots** — Visualized feature distributions by class to confirm separability
4. **Box Plots** — Confirmed that Malignant tumors have larger mean area than Benign ones

**Key Observation:** `mean radius`, `mean perimeter`, `mean area`, `worst radius`, and `worst concave points` are the most discriminative features.

### 4.3 Preprocessing Steps

| Step | Method | Reason |
|------|--------|--------|
| Train/Test Split | 80%/20%, stratified | Maintains class ratio in both sets |
| Feature Scaling | StandardScaler (z-score) | Logistic Regression sensitive to feature scale |

**StandardScaler formula:**

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

> **Important:** The scaler is **fit only on the training set** to prevent data leakage. The test set is transformed using training set statistics.

### 4.4 Model Configuration

```python
LogisticRegression(
    solver='lbfgs',    # Limited-memory BFGS — best for small/medium datasets
    max_iter=10000,    # Ensures convergence
    C=1.0,             # Default regularization (balanced)
    random_state=42    # Reproducibility
)
```

### 4.5 Results

| Metric | Value |
|--------|-------|
| **5-Fold Cross-Validation Accuracy** | ~97.00% ± 1.5% |
| **Test Accuracy** | ~97.37% |
| **ROC-AUC Score** | ~0.997 |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Malignant | ~0.97 | ~0.95 | ~0.96 | ~42 |
| Benign | ~0.97 | ~0.99 | ~0.98 | ~72 |
| **Accuracy** | | | **~97.37%** | **114** |

### 4.6 Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were found |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Area under ROC curve | Model's ability to distinguish classes (1.0 = perfect) |

### 4.7 Confusion Matrix Interpretation

```
              Predicted
              Malignant  Benign
Actual  Malignant  [ TP ]  [ FN ]   ← False Negatives are dangerous (missed cancer)
        Benign     [ FP ]  [ TN ]
```

**In medical diagnosis, Recall (Sensitivity) for Malignant is critical** — we want to minimize False Negatives (missed cancer diagnoses).

### 4.8 Feature Importance

The top features by **absolute coefficient magnitude** are:
1. `worst concave points` — strongest predictor of malignancy
2. `worst radius` — larger radius → more likely malignant
3. `mean concave points`
4. `worst perimeter`
5. `mean radius`

**Interpretation:** Positive coefficient = increases probability of Benign; Negative = increases Malignant probability.

### 4.9 Generated Plots

| File | Description |
|------|-------------|
| `part1_eda.png` | Class distribution, correlation heatmap, histograms, boxplot |
| `part1_evaluation.png` | Confusion matrix, ROC curve, learning curve |
| `part1_feature_importance.png` | Top 15 feature coefficients bar chart |

---

## 5. Part 2 — Image Data

### 5.1 Image-to-Feature Vector Pipeline

```
28×28 Image (784 pixels)
        ↓ Flatten
784-dimensional feature vector [pixel_0, pixel_1, ..., pixel_783]
        ↓ Normalize
Divide each pixel by 255 → [0.0, 0.001, ..., 1.0]
        ↓ Logistic Regression
Predicts digit class (0–9)
```

### 5.2 Why Flatten Works (and its Limitations)

**Works because:**
- Pixel intensities carry information about digit shape
- Logistic regression finds linear weight patterns that activate for each digit
- Learned weights literally look like digit templates (see `part2_learned_weights.png`)

**Limitations:**
- Loses spatial relationships between nearby pixels
- Not translation/rotation invariant
- CNNs (Convolutional Neural Networks) significantly outperform flattened LR on images

### 5.3 Preprocessing Steps

| Step | Method | Details |
|------|--------|---------|
| Sampling | Stratified random | 12,000 from 70,000 (for speed) |
| Reshape | Flatten | 28×28 → 784 vector |
| Normalization | Divide by 255 | Maps [0,255] → [0.0, 1.0] |
| Split | 80%/20% stratified | 9,600 train / 2,400 test |

> **Note:** Unlike Part 1, StandardScaler is NOT applied here. Pixel values are already on the same scale [0–1], and scaling is less necessary.

### 5.4 Model Configuration

```python
LogisticRegression(
    solver='saga',        # Stochastic Average Gradient Augmented — best for large/sparse data
    max_iter=200,         # SAGA converges faster than lbfgs
    C=0.5,                # Slight regularization — 784 features, risk of overfitting
    multi_class='auto',   # Uses One-vs-Rest for multiclass
    n_jobs=-1             # Parallelizes across all CPU cores
)
```

**Why `saga` solver?**
- Efficient for large datasets with many features
- Supports L1, L2, and ElasticNet regularization
- Faster than `lbfgs` on high-dimensional problems

### 5.5 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~88–91% |

**Per-Digit Accuracy (approximate):**

| Digit | Accuracy |
|-------|---------|
| 0 | ~97% |
| 1 | ~96% |
| 2 | ~86% |
| 3 | ~87% |
| 4 | ~89% |
| 5 | ~83% |
| 6 | ~93% |
| 7 | ~92% |
| 8 | ~82% |
| 9 | ~87% |

> Digits 2, 5, and 8 are typically the hardest to classify due to visual similarity.

### 5.6 Confusion Analysis

**Most commonly confused pairs:**
- `4 ↔ 9` — Structurally similar (closed top loop)
- `3 ↔ 5` — Similar curves
- `2 ↔ 7` — Similar angled strokes
- `8 ↔ 3` — Shared curved features

This is expected behavior for a **linear classifier** on raw pixels. CNNs would resolve these by learning local features (edges, curves).

### 5.7 Learned Weights Visualization

The `part2_learned_weights.png` shows the model's learned weights reshaped back to 28×28. These look like **digit templates**:
- **Red/warm regions** = pixels that strongly activate for that digit
- **Blue/cool regions** = pixels that indicate "NOT this digit"

This demonstrates the model has genuinely learned digit-specific spatial patterns.

### 5.8 Generated Plots

| File | Description |
|------|-------------|
| `part2_sample_images.png` | 5 examples per digit (50 images total) |
| `part2_evaluation.png` | Confusion matrix + per-class accuracy chart |
| `part2_predictions.png` | Correct (green) vs. incorrect (red) predictions |
| `part2_learned_weights.png` | Learned weight patterns per digit class |

---

## 6. Comparison of Results

| Aspect | Part 1 (Tabular) | Part 2 (Image) |
|--------|------------------|----------------|
| **Dataset** | Breast Cancer Wisconsin | MNIST Digits |
| **Samples** | 569 | 12,000 |
| **Features** | 30 | 784 |
| **Classes** | 2 (Binary) | 10 (Multiclass) |
| **Scaler** | StandardScaler | Pixel ÷ 255 |
| **Solver** | lbfgs | saga |
| **C (regularization)** | 1.0 | 0.5 |
| **Test Accuracy** | ~97.4% | ~88–91% |
| **Limitation** | Slight class imbalance | Linear boundary insufficient for images |

### Why is tabular accuracy higher?

1. **Fewer features** (30 vs 784) → less overfitting risk
2. **Domain-specific features** are inherently more discriminative than raw pixels
3. **Binary classification** is simpler than 10-class
4. **Feature engineering** already done (medical measurements vs raw pixels)

### Can image accuracy be improved?

Yes, through:
- Using the full 70,000 MNIST samples
- Applying PCA to reduce dimensionality before logistic regression
- Using more advanced models: SVM, Random Forest, or CNN (for images)
- Tuning hyperparameter `C` via GridSearchCV

---

## 7. Conclusion

This assignment demonstrated Logistic Regression applied to two fundamentally different data types:

1. **Tabular Data (Part 1):** Logistic Regression achieved **~97.4% accuracy** on the Breast Cancer dataset. The algorithm is well-suited for structured tabular data with meaningful engineered features. The ROC-AUC of ~0.997 indicates near-perfect class separation.

2. **Image Data (Part 2):** Logistic Regression achieved **~88–91% accuracy** on MNIST. Flattening images to feature vectors works reasonably well, but the linear decision boundary limits performance. The learned weight visualizations confirm the model captures digit-specific spatial patterns.

**Key Takeaways:**
- Logistic Regression is a powerful and interpretable baseline for classification
- Feature scaling is critical when feature ranges differ
- For image classification, deep learning (CNN) significantly outperforms logistic regression
- Regularization (C parameter) is important to prevent overfitting, especially with many features
- Evaluation should always include Precision, Recall, F1, and confusion matrix — not just accuracy

---

## 8. References

| Resource | Link |
|----------|------|
| Breast Cancer Dataset (Kaggle) | https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data |
| MNIST Digit Recognizer (Kaggle) | https://www.kaggle.com/competitions/digit-recognizer |
| MNIST Dataset (Kaggle) | https://www.kaggle.com/datasets/hojjatk/mnist-dataset |
| scikit-learn LogisticRegression | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html |
| Breast Cancer (UCI) | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) |
| MNIST Original | http://yann.lecun.com/exdb/mnist/ |
| scikit-learn Metrics | https://scikit-learn.org/stable/modules/model_evaluation.html |

---

## Appendix — File Structure

```
ML Assignment 1/
│
├── Part1_Tabular_Logistic_Regression.py   ← Part 1 implementation
├── Part2_Image_Logistic_Regression.py     ← Part 2 implementation
├── Assignment_Report.md                   ← This report
├── requirements.txt                       ← Python dependencies
│
├── part1_eda.png                          ← Part 1 EDA plots
├── part1_evaluation.png                   ← Part 1 confusion matrix, ROC, learning curve
├── part1_feature_importance.png           ← Part 1 feature coefficients
│
├── part2_sample_images.png                ← MNIST sample images
├── part2_evaluation.png                   ← Part 2 confusion matrix + per-class accuracy
├── part2_predictions.png                  ← Correct/incorrect prediction samples
└── part2_learned_weights.png              ← Learned weight templates per digit
```

## Appendix — How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Part 1 (tabular data — runs in < 10 seconds)
python Part1_Tabular_Logistic_Regression.py

# 3. Run Part 2 (image data — first run downloads MNIST, ~1-2 min)
python Part2_Image_Logistic_Regression.py
```

> **Note:** MNIST is downloaded automatically via `sklearn.datasets.fetch_openml` on first run (~11 MB). Subsequent runs use the cached version.
