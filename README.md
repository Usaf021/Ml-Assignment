# Machine Learning Assignment 1 — Logistic Regression

**Topic:** Logistic Regression on Tabular Data and Image Data  
**Subject:** Machine Learning  
**Date:** April 24, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Setup & Installation](#3-setup--installation)
4. [How to Run](#4-how-to-run)
5. [What the Code Does — Step by Step](#5-what-the-code-does--step-by-step)
   - [Part 1 — Breast Cancer (Tabular Data)](#part-1--breast-cancer-tabular-data)
   - [Part 2 — Fashion-MNIST (Image Data)](#part-2--fashion-mnist-image-data)
6. [Understanding the Math](#6-understanding-the-math)
7. [Output Files](#7-output-files)
8. [Comparative Analysis of Models](#8-comparative-analysis-of-models)
9. [How to Present This Assignment](#9-how-to-present-this-assignment)

---

## 1. Project Overview

This assignment applies **Logistic Regression** — a fundamental supervised machine learning algorithm — to two completely different types of data:

| | Part 1 | Part 2 |
|---|---|---|
| **Data Type** | Tabular / Structured | Images |
| **Dataset** | Breast Cancer Wisconsin | Fashion-MNIST |
| **Task** | Binary Classification | 10-Class Classification |
| **Features** | 30 medical measurements | 784 pixels (28×28 image) |
| **Goal** | Detect Malignant vs Benign tumors | Identify clothing category |

The purpose is to understand **how the same algorithm behaves on different data**, what preprocessing is needed for each, and how to evaluate a classifier properly.

---

## 2. Project Structure

```
ML Assignment 1/
│
├── README.md                              ← You are here
├── Assignment_Report.md                   ← Detailed written report
├── requirements.txt                       ← Python package dependencies
│
├── Part1_Tabular_Logistic_Regression.py   ← Part 1 code (Breast Cancer)
├── Part2_Image_Logistic_Regression.py     ← Part 2 code (Fashion-MNIST)
│
│── (Generated after running Part 1)
├── part1_eda.png                          ← EDA plots
├── part1_evaluation.png                   ← Confusion matrix, ROC curve, learning curve
├── part1_feature_importance.png           ← Top feature coefficients
│
└── (Generated after running Part 2)
├── part2_sample_images.png               ← Sample clothing images per class
├── part2_evaluation.png                  ← Confusion matrix + per-class accuracy
├── part2_predictions.png                 ← Correct (green) vs incorrect (red) predictions
└── part2_learned_weights.png             ← What the model "learned" per class
```

---

## 3. Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1 — Create a virtual environment (recommended)

```powershell
# In the project folder
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations, arrays |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical plots (heatmaps, etc.) |
| `scikit-learn` | Machine learning models and metrics |
| `scipy` | Scientific computing utilities |

---

## 4. How to Run

Run each part independently from the project folder:

```powershell
# Part 1 — Tabular Data (runs in ~10 seconds)
python Part1_Tabular_Logistic_Regression.py

# Part 2 — Image Data (runs in 2–5 minutes due to large dataset)
python Part2_Image_Logistic_Regression.py
```

Each script:
- Prints detailed progress and results to the terminal
- Saves plots as `.png` files in the same folder
- Requires no manual downloads — data is fetched automatically via scikit-learn / OpenML

> **Note for Part 2:** The first run downloads Fashion-MNIST (~30 MB) and caches it locally. Subsequent runs are faster.

---

## 5. What the Code Does — Step by Step

### Part 1 — Breast Cancer (Tabular Data)

The complete pipeline is:

```
Load Data → EDA → Preprocess → Train/Test Split → Scale Features → Train Model → Evaluate → Visualize
```

#### Step 1: Load Dataset
- Uses `sklearn.datasets.load_breast_cancer()` — no manual download needed
- 569 samples, 30 features, 2 classes (Malignant / Benign)
- Features include medical measurements: radius, texture, perimeter, area, smoothness, etc.

#### Step 2: Exploratory Data Analysis (EDA)
Before training, the data is examined to understand its properties:
- **Class distribution** — confirms slight imbalance (63% Benign, 37% Malignant)
- **Correlation heatmap** — identifies highly correlated features (radius, perimeter, area move together)
- **Feature distributions** — histograms showing class separation per feature
- **Box plots** — confirms Malignant tumors have larger `mean area`

> Key finding: `worst concave points`, `worst radius`, and `mean concave points` are the most discriminative features.

#### Step 3: Preprocessing
| Step | What happens | Why |
|------|-------------|-----|
| Train/Test Split | 80% train, 20% test, stratified | Keeps class ratio in both sets |
| StandardScaler | Subtracts mean, divides by std dev | LR is sensitive to feature scale |

> **Critical rule:** The scaler is **fit ONLY on training data**, then applied to both train and test. Fitting on test data would be "data leakage" (cheating).

#### Step 4: Train the Model
```python
LogisticRegression(solver='lbfgs', max_iter=10000, C=1.0, random_state=42)
```
- `lbfgs` solver: efficient quasi-Newton optimizer, best for small/medium datasets
- `C=1.0`: default regularization (inverse of regularization strength)
- `max_iter=10000`: ensures the optimizer converges fully

#### Step 5: Evaluate
- **5-Fold Cross-Validation** — tests generalization on 5 different splits
- **Confusion Matrix** — shows TP, TN, FP, FN breakdown
- **ROC-AUC Curve** — measures ranking quality across all thresholds
- **Precision-Recall Curve** — important for imbalanced classes
- **Learning Curve** — shows whether the model needs more data

#### Step 6: Feature Importance
- Logistic Regression coefficients are extracted and plotted
- Larger absolute value = more important feature
- Sign tells you which class the feature pushes toward

---

### Part 2 — Fashion-MNIST (Image Data)

The pipeline is different because the input is raw pixels:

```
Load Images → Sample Subset → Flatten (28×28 → 784) → Normalize → Split → Train → Evaluate → Visualize
```

#### Step 1: Load Dataset
- Uses `fetch_openml('Fashion-MNIST')` — downloads and caches automatically
- 70,000 images, 28×28 pixels each, 10 clothing classes
- Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

#### Step 2: Sample a Subset
- Only 12,000 of the 70,000 images are used (for reasonable training time)
- Sampled randomly with `RandomState(42)` for reproducibility

#### Step 3: Image-to-Feature Vector (Flattening)
```
28×28 pixel grid  →  [pixel_0, pixel_1, pixel_2, ..., pixel_783]  →  784 numbers
```
This converts each image into a row of 784 numbers. The model treats each pixel as an independent feature.

#### Step 4: Normalization
```python
X = X / 255.0   # Maps pixel values from [0, 255] to [0.0, 1.0]
```
> Unlike Part 1, `StandardScaler` is NOT used here. All features (pixels) are already on the same scale, so simple division by 255 is sufficient.

#### Step 5: Train the Model
```python
LogisticRegression(solver='saga', max_iter=200, C=0.5, multi_class='auto', n_jobs=-1)
```
- `saga` solver: stochastic gradient, best for large/high-dimensional datasets
- `C=0.5`: slightly stronger regularization than Part 1 (784 features = higher overfitting risk)
- `n_jobs=-1`: uses all CPU cores in parallel
- Internally uses **One-vs-Rest (OvR)**: trains 10 separate binary classifiers

#### Step 6: Evaluate
- **Confusion matrix** — 10×10 grid showing which classes get confused with which
- **Per-class accuracy** — bar chart showing accuracy for each clothing type
- **Prediction samples** — shows correctly classified (green border) vs wrong (red border)

#### Step 7: Visualize Learned Weights
- Reshapes each class's 784 weight values back to 28×28
- Displays them as heatmaps — they literally look like templates of each clothing item
- Red = "this pixel is important for this class", Blue = "this pixel argues against this class"

---

## 6. Understanding the Math

### The Sigmoid Function
Logistic Regression maps a linear combination of features to a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$

- Output is always between 0 and 1 → interpreted as a probability
- If output ≥ 0.5 → predict Class 1; if < 0.5 → predict Class 0

### Loss Function (Binary Cross-Entropy)
The model is trained by minimizing this loss:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### Regularization (L2 / Ridge)
Prevents overfitting by penalizing large weights:

$$J_{reg} = J + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

The `C` parameter in scikit-learn = $1 / \lambda$. Smaller C = stronger penalty = simpler model.

### Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Accuracy | (TP + TN) / Total | Overall correctness |
| Precision | TP / (TP + FP) | How reliable positive predictions are |
| Recall | TP / (TP + FN) | How many actual positives were caught |
| F1-Score | 2 × P × R / (P + R) | Balance of Precision and Recall |
| ROC-AUC | Area under ROC curve | Separability across all thresholds |

> In medical diagnosis (Part 1), **Recall for Malignant is most important** — a missed cancer (False Negative) is far more dangerous than a false alarm.

---

## 7. Output Files

### Part 1 Outputs

| File | What to look for |
|------|-----------------|
| `part1_eda.png` | Class imbalance, correlated features, class separability |
| `part1_evaluation.png` | Confusion matrix (few FN = good), steep ROC curve (AUC ~0.997) |
| `part1_feature_importance.png` | Top features with largest coefficients |

### Part 2 Outputs

| File | What to look for |
|------|-----------------|
| `part2_sample_images.png` | Visual examples of each clothing class |
| `part2_evaluation.png` | Which clothing types get confused most |
| `part2_predictions.png` | Green = correct, Red = wrong (see what the model struggles with) |
| `part2_learned_weights.png` | Template-like patterns the model learned per class |

---

## 8. Comparative Analysis of Models

This section provides a structured, metric-by-metric comparison of the two Logistic Regression models trained in this assignment. Both models use the same algorithm but differ in data type, dimensionality, number of classes, and preprocessing — making the comparison meaningful for understanding how context shapes model behavior.

---

### 8.1 Model Configuration Comparison

| Property | Part 1 — Breast Cancer | Part 2 — Fashion-MNIST |
|----------|----------------------|----------------------|
| **Data type** | Tabular / Structured | Image (flattened pixels) |
| **Dataset size** | 569 samples | 12,000 samples (subset) |
| **Features** | 30 (engineered medical values) | 784 (raw pixel intensities) |
| **Classes** | 2 — Malignant, Benign | 10 — clothing categories |
| **Classification type** | Binary | Multiclass (One-vs-Rest) |
| **Solver** | `lbfgs` | `saga` |
| **Regularization C** | 1.0 | 0.5 |
| **Feature scaling** | StandardScaler (z-score) | Divide by 255 |
| **Train / Test split** | 80% / 20% stratified | 80% / 20% stratified |

---

### 8.2 Overall Performance Metrics

| Metric | Part 1 (Breast Cancer) | Part 2 (Fashion-MNIST) |
|--------|----------------------|----------------------|
| **Test Accuracy** | **~97.4%** | **~88–91%** |
| **Cross-Val Accuracy** | ~97.0% ± 1.5% | — |
| **ROC-AUC** | **~0.997** | — (multiclass, not directly comparable) |
| **Macro Avg Precision** | ~0.97 | ~0.88–0.91 |
| **Macro Avg Recall** | ~0.97 | ~0.88–0.91 |
| **Macro Avg F1-Score** | ~0.97 | ~0.88–0.90 |
| **Weighted Avg F1-Score** | ~0.97 | ~0.88–0.90 |

> **Macro average** treats all classes equally regardless of size.  
> **Weighted average** accounts for class imbalance by weighting each class by its number of samples.

---

### 8.3 Per-Class Metric Breakdown

#### Part 1 — Breast Cancer (Binary)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Malignant (0)** | ~0.97 | ~0.95 | ~0.96 | ~42 |
| **Benign (1)** | ~0.97 | ~0.99 | ~0.98 | ~72 |
| **Macro Avg** | ~0.97 | ~0.97 | ~0.97 | 114 |
| **Weighted Avg** | ~0.97 | ~0.97 | ~0.97 | 114 |

**Observations:**
- Precision and Recall are nearly balanced for both classes — the model generalizes well
- Recall for Malignant (~0.95) is slightly lower than for Benign (~0.99) — a small number of malignant cases are missed (False Negatives)
- In a clinical setting, maximizing Malignant Recall is the priority; the current value is strong

#### Part 2 — Fashion-MNIST (Multiclass)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| T-shirt/Top | ~0.82 | ~0.84 | ~0.83 |
| Trouser | ~0.97 | ~0.97 | ~0.97 |
| Pullover | ~0.82 | ~0.78 | ~0.80 |
| Dress | ~0.87 | ~0.90 | ~0.88 |
| Coat | ~0.84 | ~0.83 | ~0.83 |
| Sandal | ~0.96 | ~0.96 | ~0.96 |
| Shirt | ~0.70 | ~0.68 | ~0.69 |
| Sneaker | ~0.94 | ~0.96 | ~0.95 |
| Bag | ~0.97 | ~0.97 | ~0.97 |
| Ankle Boot | ~0.95 | ~0.96 | ~0.95 |

**Observations:**
- **Best performing classes:** Trouser, Sandal, Bag, Sneaker, Ankle Boot (distinct visual shapes → easy to separate with a linear boundary)
- **Worst performing class:** Shirt (~0.69 F1) — visually similar to T-shirt and Coat; a linear classifier cannot separate them reliably
- **Pullover and Coat** also underperform — overlapping pixel distributions make them hard to distinguish

---

### 8.4 Confusion Analysis

#### Part 1 — Confusion Matrix Structure

```
              Predicted
              Malignant   Benign
Actual  Malignant  [ TP ]    [ FN ]   ← minimizing FN is critical
        Benign     [ FP ]    [ TN ]
```

- Very few off-diagonal errors — the classes are well separated in feature space
- Any False Negatives (missed malignant tumors) are the most dangerous type of error here

#### Part 2 — Most Confused Class Pairs

| Predicted as → | Often confused with |
|---------------|--------------------|
| Shirt | T-shirt/Top, Coat |
| Pullover | Coat, Shirt |
| Coat | Pullover, Shirt |
| T-shirt/Top | Shirt |

- Confusion clusters around **similar-looking garments** — a fundamental limitation of linear models on raw pixels
- Structurally distinct items (Trouser, Bag, Sandal) are rarely confused

---

### 8.5 Metric-by-Metric Comparison

#### Accuracy

| Model | Accuracy | Verdict |
|-------|----------|---------|
| Part 1 — Breast Cancer | ~97.4% | Excellent — near human-level for this task |
| Part 2 — Fashion-MNIST | ~88–91% | Good for a linear model on raw pixels |

- The ~8–9% gap is primarily explained by problem complexity, not model failure
- Accuracy alone is insufficient: Part 1 has class imbalance (63% Benign), so a naive model predicting only Benign would score 63% — making 97.4% genuinely impressive

#### Precision

- **Part 1:** ~0.97 for both classes — when the model predicts a class, it is almost always correct
- **Part 2:** Ranges from ~0.70 (Shirt) to ~0.97 (Trouser, Bag) — high variance across classes
- **Conclusion:** Tabular features provide much stronger class separation than raw pixels

#### Recall

- **Part 1:** ~0.95 (Malignant), ~0.99 (Benign) — very high; the model catches almost all true cases
- **Part 2:** Ranges from ~0.68 (Shirt) to ~0.97 (Trouser) — Shirt has a notably high miss rate
- **Conclusion:** Low recall in Part 2 for visually ambiguous classes reflects the ceiling of linear classification on images

#### F1-Score

- **Part 1 macro F1:** ~0.97 — consistent performance across both classes
- **Part 2 macro F1:** ~0.88 — pulled down by poor Shirt performance
- F1 is the most informative single metric when class sizes differ, because it balances both precision and recall

#### ROC-AUC (Part 1 only)

- AUC ~0.997 is near-perfect — the model can rank a randomly chosen malignant case above a randomly chosen benign case 99.7% of the time
- ROC-AUC is not directly applicable to multiclass problems in the same way; in Part 2 it would require a one-vs-rest AUC computed per class

---

### 8.6 Root Cause Analysis — Why the Models Differ

| Factor | Favors Part 1 | Explanation |
|--------|--------------|-------------|
| Feature quality | ✅ Part 1 | Medical measurements encode domain knowledge; pixels are raw signal |
| Dimensionality | ✅ Part 1 | 30 features vs 784 — fewer features = less risk of overfitting |
| Class count | ✅ Part 1 | Binary (2 classes) is always simpler than 10-class OvR |
| Class separability | ✅ Part 1 | Malignant and Benign are well-separated in feature space |
| Model suitability | ✅ Part 1 | Logistic Regression is near-optimal for clean tabular data |
| Image structure | ❌ Part 1 | Flattening destroys spatial pixel relationships — CNNs handle this better |
| Dataset size | ❌ Part 1 | 569 samples is small; Part 2 uses 12,000 |

---

### 8.7 Strengths and Weaknesses of Each Model

#### Part 1 — Breast Cancer Model

| Strengths | Weaknesses |
|-----------|------------|
| Very high accuracy and AUC | Small dataset (569 samples) |
| Balanced precision and recall | Slight class imbalance (63/37) |
| Interpretable feature coefficients | Only 2 classes — limited generalization test |
| Fast training (<1 second) | Assumes linear decision boundary |

#### Part 2 — Fashion-MNIST Model

| Strengths | Weaknesses |
|-----------|------------|
| Handles high-dimensional input (784 features) | Raw pixel features lose spatial context |
| Scales to 10-class OvR automatically | Shirt/Pullover/Coat confusion unresolvable by linear model |
| Learned weights are visually interpretable | Requires more data and longer training |
| Good baseline (88–91%) for a linear model | CNNs easily achieve 93–95% on same data |

---

### 8.8 Recommendations for Improvement

| Improvement | Applicable to | Expected Gain |
|-------------|--------------|---------------|
| Use full 70,000 MNIST samples | Part 2 | +1–2% accuracy |
| Apply PCA before LR (e.g., 100 components) | Part 2 | +1–3% accuracy, faster training |
| Tune `C` via GridSearchCV | Both | Small but consistent gain |
| Use SVM with RBF kernel | Both | Better non-linear boundary |
| Use Random Forest / Gradient Boosting | Part 1 | Handles feature interactions better |
| Use CNN (Convolutional Neural Network) | Part 2 | +5–8% accuracy; resolves spatial confusion |
| SMOTE oversampling for class imbalance | Part 1 | Improves Malignant recall further |

---

## 9. How to Present This Assignment

### Recommended Presentation Flow (5–10 minutes)

**1. Introduction (1 min)**
> "This assignment implements Logistic Regression on two types of data — structured tabular data and raw image data — to understand how the same algorithm behaves differently across domains."

**2. Theory — What is Logistic Regression? (1–2 min)**
- Show the sigmoid function formula
- Explain: linear combination of features → probability → class decision
- Mention: trained by minimizing Binary Cross-Entropy loss
- Mention: L2 regularization via `C` parameter

**3. Part 1 — Breast Cancer (2–3 min)**
- Show `part1_eda.png` → explain what you found in EDA (class imbalance, correlated features)
- Explain the preprocessing: train/test split → StandardScaler
- Show `part1_evaluation.png` → walk through the confusion matrix (few FN = good for medical use)
- Show the ROC curve → AUC ~0.997 means near-perfect separation
- Show `part1_feature_importance.png` → explain which features matter most and why

**4. Part 2 — Fashion-MNIST (2–3 min)**
- Show `part2_sample_images.png` → explain the 10 clothing classes
- Explain the flattening pipeline: 28×28 image → 784 features → normalize by /255
- Explain why `saga` solver and stronger regularization (C=0.5) were chosen
- Show `part2_evaluation.png` → point out which classes get confused and why
- Show `part2_learned_weights.png` → this is visually striking; explain that weights form clothing templates

**5. Comparative Analysis (2 min)**
- Walk through Section 8 of the README — metric-by-metric comparison
- Highlight the per-class F1 table for Part 2: point out Shirt (~0.69) vs Trouser (~0.97) and explain why
- Show the confusion pairs table: Shirt ↔ T-shirt, Pullover ↔ Coat — the linear model's ceiling
- State the root cause: feature quality and class count, not a flaw in the algorithm
- Close with the improvement table: mention CNN for images, PCA + LR, full dataset

---

### Key Points to Emphasize

- **Data leakage prevention:** Scaler is fit only on training data — always mention this
- **Stratified splitting:** Preserves class ratio in both splits — important for imbalanced data
- **Solver choice matters:** `lbfgs` for small data, `saga` for large/high-dimensional data
- **Evaluation beyond accuracy:** Use precision, recall, F1, and confusion matrix — especially for imbalanced classes
- **Interpretability:** Logistic Regression's coefficients give direct feature importance — a major advantage over black-box models

### Common Questions You May Be Asked

| Question | Answer |
|----------|--------|
| Why use logistic regression for classification, not linear regression? | Linear regression can output values outside [0,1] and doesn't model probabilities. Logistic uses the sigmoid to constrain output to (0,1). |
| What is data leakage? | Fitting the scaler on test data would let training indirectly "see" test statistics, inflating performance metrics artificially. |
| Why is recall more important than precision in Part 1? | In cancer diagnosis, a False Negative (missed cancer) is far more dangerous than a False Positive (extra biopsy). |
| Why does flattening images work? | Pixel intensities still carry shape information. The model learns which pixels are bright/dark for each class. But it loses spatial relationships. |
| How would you improve Part 2? | Use a CNN (learns local spatial features via convolution), or apply PCA to reduce 784 features before logistic regression. |
| What does the `C` parameter do? | C = 1/λ. Smaller C = stronger L2 regularization = smaller weights = simpler model. Use smaller C when you have many features. |

---

## References

| Resource | Link |
|----------|------|
| Breast Cancer Dataset | https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data |
| Fashion-MNIST Dataset | https://www.kaggle.com/datasets/zalando-research/fashionmnist |
| scikit-learn LogisticRegression | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html |
| scikit-learn Model Evaluation | https://scikit-learn.org/stable/modules/model_evaluation.html |
| UCI Breast Cancer (original) | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) |
