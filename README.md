# ❤️ Cardiovascular Disease Prediction

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
A comprehensive machine learning project for predicting cardiovascular disease using patient medical data. This project implements and compares multiple ML algorithms to identify the most effective model for CVD prediction.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Workflow](#-project-workflow)
- [Feature Engineering](#-feature-engineering)
- [Models & Algorithms](#-models--algorithms)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Data Preprocessing](#-data-preprocessing)
- [Usage Examples](#-usage-examples)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Cardiovascular diseases (CVD) are the leading cause of death globally. Early detection and prediction can save lives. This project uses machine learning to predict the presence or absence of CVD based on patient examination results and medical history.

### Project Highlights

- 📊 **70,000 patient records** analyzed from [Kaggle Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)
- 🔬 **15 engineered features** for better predictions
- 🤖 **6 ML algorithms** implemented and compared
- 🎯 **82.27% accuracy** achieved - **Significantly outperforming the 73% benchmark** most solutions achieve
- 📈 **99.95% precision** on cardiovascular disease detection
---

## 📊 Dataset Description

<details>
<summary><b>Objective Features</b> (Factual Information)</summary>

| Feature | Description | Type | Unit |
|---------|-------------|------|------|
| `age` | Patient age | Integer | Days |
| `gender` | Patient gender | Categorical | 1 or 2 |
| `height` | Patient height | Integer | cm |
| `weight` | Patient weight | Float | kg |

</details>

<details>
<summary><b>Examination Features</b> (Medical Tests)</summary>

| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| `ap_hi` | Systolic blood pressure | Integer | mmHg |
| `ap_lo` | Diastolic blood pressure | Integer | mmHg |
| `cholesterol` | Cholesterol level | Categorical | 1: normal, 2: above normal, 3: well above normal |
| `gluc` | Glucose level | Categorical | 1: normal, 2: above normal, 3: well above normal |

</details>

<details>
<summary><b>Subjective Features</b> (Patient Lifestyle)</summary>

| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| `smoke` | Smoking status | Binary | 0: No, 1: Yes |
| `alco` | Alcohol intake | Binary | 0: No, 1: Yes |
| `active` | Physical activity | Binary | 0: No, 1: Yes |

</details>

### 🎯 Target Variable

- **`cardio`**: Presence (1) or absence (0) of cardiovascular disease

---

## ✨ Key Features

- ✅ **Automated Model Caching**: Pre-trained models are saved and loaded automatically
- ✅ **Feature Alignment**: Ensures consistent feature order during prediction
- ✅ **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- ✅ **Advanced Feature Engineering**: 6 new features created from existing data
- ✅ **Statistical Feature Selection**: ANOVA F-test for feature importance
- ✅ **Robust Outlier Handling**: IQR method applied per target category
- ✅ **Multiple Model Comparison**: 6 different algorithms evaluated
- ✅ **Complete Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

---


###  Download Dataset

Download the dataset from Kaggle:
- **Dataset Link**: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)
- Place `cardio_train.csv` in the project root directory

**Note**: You'll need a Kaggle account to download the dataset.

---

## 🚀 Quick Start

### Run the Jupyter Notebook

```bash
jupyter notebook "final notebook.ipynb"
```
## 🔄 Project Workflow

```
1. Data Loading & Exploration
   ↓
2. Data Cleaning & Preprocessing
   ↓
3. Exploratory Data Analysis (EDA)
   ↓
4. Outlier Detection & Handling
   ↓
5. Feature Engineering
   ↓
6. Feature Selection (ANOVA F-test)
   ↓
7. Train-Test Split (75-25)
   ↓
8. Model Training (6 algorithms)
   ↓
9. Model Evaluation & Comparison
   ↓
10. Results Export & Visualization
```

---

## 🔧 Feature Engineering

We engineered **6 new features** to improve model performance:

| Feature | Formula | Description |
|---------|---------|-------------|
| **`age_years`** | `age / 365` | Age converted from days to years |
| **`bmi`** | `weight / (height²)` | Body Mass Index |
| **`pulse_pressure`** | `ap_hi - ap_lo` | Difference between systolic and diastolic BP |
| **`health_index`** | `active - 0.5×smoke - 0.5×alco` | Combined lifestyle health score |
| **`cholesterol_gluc_interaction`** | `cholesterol × gluc` | Interaction between cholesterol and glucose |
| **`MAP`** | `(2×ap_lo + ap_hi) / 3` | Mean Arterial Pressure |

### Why These Features?

- **BMI**: Strong indicator of cardiovascular risk
- **Pulse Pressure**: Reflects arterial stiffness
- **Health Index**: Combines lifestyle factors into single metric
- **MAP**: Better indicator of perfusion than systolic/diastolic alone
- **Interactions**: Captures combined effects of related variables

---

## 🤖 Models & Algorithms

### Implemented Models

| # | Model | Hyperparameters | Use Case |
|---|-------|----------------|----------|
| 1 | **Logistic Regression** | Default | Baseline linear model |
| 2 | **Decision Tree** | `max_depth=7` | **Best performer** |
| 3 | **Decision Tree** | `max_depth=10, min_samples_split=7` | Deeper tree comparison |
| 4 | **Random Forest** | `n_estimators=3` | Ensemble method (small) |
| 5 | **Random Forest** | `n_estimators=5` | Ensemble method (medium) |
| 6 | **Gradient Boosting** | `n_estimators=3` | Boosting approach |
---

## 📈 Results & Performance

### 🏆 Champion Model: Decision Tree (depth=7)

**Why it's the best:**
- ✅ Excellent generalization (minimal overfitting)
- ✅ Near-perfect precision (99.95%)
- ✅ Strong F1 Score (0.7845)
- ✅ Balanced performance across all metrics
- ✅ Interpretable and explainable
- 🚀 **Beats the 73% accuracy benchmark** that most Kaggle solutions achieve

### 💪 Competitive Advantage

Most solutions on Kaggle struggle to exceed **73% accuracy**. Our approach achieves:
- ✨ **82.27% accuracy** - a **+9.27% improvement** over typical solutions
- 🎯 **Advanced feature engineering** that captures hidden patterns
- 🔬 **Proper outlier handling** using IQR method per target category
- 📊 **Statistical feature selection** ensuring only meaningful features are used

**What makes this different?**
1. Smart feature engineering (BMI, Pulse Pressure, Health Index, MAP)
2. Category-aware outlier treatment
3. Optimal model depth selection (avoiding overfitting)
4. Comprehensive evaluation across multiple metrics

### Training Set Performance

| Model | Accuracy ⬆️ | Recall | Precision ⬆️ | F1 Score ⬆️ | ROC-AUC |
|-------|------------|--------|--------------|------------|---------|
| Logistic Regression | 60.58% | 62.14% | 60.23% | 61.17% | 60.58% |
| **Decision Tree (depth=7)** | **82.28%** | 64.61% | **99.91%** | 78.47% | 82.27% |
| Decision Tree (depth=10) | 82.76% | 69.49% | 94.55% | 80.11% | 82.75% |
| Random Forest (trees=3) | 94.20% | 92.63% | 95.62% | 94.10% | 94.20% |
| Random Forest (trees=5) | **95.48%** | **94.01%** | 96.85% | **95.41%** | **95.48%** |
| Gradient Boosting | 81.26% | 62.50% | **100.00%** | 76.92% | 81.25% |

### Test Set Performance (Most Important!)

| Model | Accuracy ⬆️ | Recall | Precision ⬆️ | F1 Score ⬆️ | ROC-AUC ⬆️ |
|-------|------------|--------|--------------|------------|-----------|
| Logistic Regression | 60.14% | 62.08% | 59.74% | 60.89% | 60.14% |
| **Decision Tree (depth=7)** ⭐ | **82.27%** | 64.56% | **99.95%** | **78.45%** | **82.26%** |
| Decision Tree (depth=10) | 82.51% | 69.26% | 94.20% | 79.83% | 82.50% |
| Random Forest (trees=3) | 79.37% | 76.58% | 81.09% | 78.77% | 79.37% |
| Random Forest (trees=5) | 80.00% | 76.31% | 82.37% | 79.22% | 80.00% |
| Gradient Boosting | 81.24% | 62.46% | **100.00%** | 76.89% | 81.23% |

### 🔍 Key Insights & Analysis

#### ✅ Strengths
1. **Decision Tree (depth=7)** achieves the best balance with **zero overfitting**
2. **Near-perfect precision** (99.95%) means very few false positives
3. Consistent performance between train and test sets
4. Interpretable model - can explain predictions to medical staff

#### ⚠️ Important Observations
1. **Random Forest shows overfitting**: 95.48% train vs 80% test accuracy
2. **Gradient Boosting** has perfect precision but lower recall (misses some cases)
3. **Logistic Regression** performs poorly - data has non-linear relationships
4. **Recall vs Precision tradeoff**: Decision Tree balances both well

#### 🏥 Medical Context
- **High Precision** is critical: Reduces false alarms and unnecessary treatments
- **Good Recall** is important: Catches most at-risk patients
- **F1 Score** provides balanced view for medical decision-making

---

## 🗂️ Project Structure

```
cardiovascular-disease-prediction/
│
├── 📊 Data/
│   └── cardio_train.csv              # Main dataset (70K records)
│
├── 📓 Notebooks/
│   └── final notebook.ipynb          # Main analysis notebook
│
├── 💾 models_cache/                  # Saved trained models
│   ├── logistic_regression.pickle
│   ├── decision_tree_max_depth_7.pickle ⭐
│   ├── decision_tree_max_depth_10.pickle
│   ├── random_forest_trees_3.pickle
│   ├── random_forest_trees_5.pickle
│   └── gradient_boosting_estimator_3.pickle
│
├── 📁 models_predictions/            # Model predictions
│   ├── Logistic Regression.csv
│   ├── Decision Tree max-depth=7.csv
│   └── ... (other models)
│
├── 📈 Results/
│   └── eval_dataset.csv              # Complete evaluation metrics
│
└── 📄 README.md                      # This file
```

---

## 🧹 Data Preprocessing

### 1. Data Quality Checks
- ✅ **No missing values** in dataset
- ✅ **No duplicate records** found
- ✅ **Balanced target classes** (no bias)
- ✅ **70,000 complete records**

### 2. Outlier Handling

We use the **IQR (Interquartile Range)** method:

**Applied to:** `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`  
**Excluded:** Categorical features (`cholesterol`, `gluc`, `smoke`, `alco`, `active`)

**Why IQR?**
- Preserves more data than hard filtering
- Applied per target category for better accuracy
- Reduces extreme values without removing records

### 3. Feature Selection (ANOVA F-test)

Statistical significance testing using F-statistic:

```
Rule: Keep features with p-value < 0.05
```

**All features passed** the significance test, indicating strong predictive power.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

1. 🐛 **Report Bugs**: Open an issue with details
2. 💡 **Suggest Features**: Share your ideas
3. 📝 **Improve Documentation**: Fix typos, add examples
4. 🔬 **Add Models**: Implement new algorithms
5. 📊 **Improve Visualizations**: Better charts and graphs
---

## 👥 Authors & Acknowledgments

### Project Team

**Abdelrahman Mohamed**  
**Ahmed Tamer**
**Ahmed Hani**
### Acknowledgments

- 📊 **Dataset**: [Cardiovascular Disease Dataset on Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data) by Svetlana Ulianova
- 🎯 **Achievement**: Significantly outperformed the typical 73% accuracy benchmark with 82.27% accuracy
- 🙏 **Community**: Python data science community and Kaggle contributors
---

## 📚 Additional Resources

### Learn More About CVD

- [WHO - Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)
- [CDC - Heart Disease Facts](https://www.cdc.gov/heartdisease/)
---
### Stay Updated

- ⭐ **Star** this repository
- 👁️ **Watch** for updates
- 🍴 **Fork** to contribute

---
**Made with ❤️ for Medical AI Research**

If this project helped you, please consider giving it a ⭐!

[⬆ Back to Top](#-cardiovascular-disease-prediction)

</div>
