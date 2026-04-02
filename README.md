# 🚗 CO2 Emissions Prediction — Linear & Logistic Regression from Scratch

A machine learning assignment that uses the **CO2 Emissions dataset** to build and evaluate both a **linear regression model implemented from scratch** (gradient descent) and a **logistic regression model** (SGD classifier) for emission class classification — with full EDA, preprocessing, and evaluation.

---

## 🗂️ Project Structure

```
├── group_9_20236076_20237003_20237015.py   # Full pipeline: EDA → preprocessing → models
└── Group_9_20236076_20237003_20237015.pdf  # Project report / results documentation
```

---

## 📊 Dataset

- **File:** `co2_emissions_data.csv`
- **Targets:**
  - `CO2 Emissions(g/km)` — continuous (regression target)
  - `Emission Class` — ordinal categorical (classification target): `VERY LOW`, `LOW`, `MODERATE`, `HIGH`
- **Features:** Vehicle attributes including engine specs, fuel type, make/model, and other numeric/categorical features

---

## 🔍 Exploratory Data Analysis

### a) Data Loading & Inspection
- Loaded dataset and inspected shape, types, and sample rows

### b) Feature Analysis
- **Missing values** — checked via `isnull().sum()`
- **Scale check** — described numeric features; plotted histograms and boxplots to assess spread and scale differences
- **Pairplot** — visualized pairwise relationships with `Emission Class` as hue, histogram diagonals
- **Correlation heatmap** — Seaborn heatmap of numeric features + CO2 emissions target

---

## ⚙️ Preprocessing

| Step | Details |
|------|---------|
| Target separation | `CO2 Emissions(g/km)` and `Emission Class` separated from feature matrix |
| Feature removal | `Model` dropped due to excessive cardinality (2,053 unique values) |
| Categorical encoding | `Emission Class` → Ordinal encoding (`VERY LOW=0` … `HIGH=3`); nominal features → dummy encoding with `drop_first=True` to avoid multicollinearity |
| Train/test split | Shuffled before splitting to prevent ordering bias |
| Feature scaling | `StandardScaler` applied to numeric features |

---

## 🤖 Models

### d) Linear Regression — Gradient Descent (From Scratch)

Implemented entirely using NumPy without any sklearn regression estimator:

1. **Theta initialization** — weights initialized to zeros
2. **Cost function** — Mean Squared Error (MSE)
3. **Gradient Descent loop** — iteratively updates weights using the gradient of the cost
4. **Cost history plot** — verifies convergence over iterations
5. **Evaluation** — R² score on test set

### e) Logistic Regression — SGD Classifier

Built using `sklearn.linear_model.SGDClassifier` for multi-class emission category prediction:

1. **Model fitting** — trained on preprocessed training set
2. **Accuracy evaluation** — `accuracy_score` on test set
3. **Manual verification** — manual implementation cross-checked against sklearn output

---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## ▶️ Running the Project

```bash
# Ensure co2_emissions_data.csv is in the working directory
python group_9_20236076_20237003_20237015.py
```

The script runs sequentially through all sections: data loading → EDA visualizations → preprocessing → linear regression from scratch → logistic regression classifier.

---

## 👥 Team Members

| Student ID |
|------------|
| Laila      |
| Jumanah    |
| Malak      |
