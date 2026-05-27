# Air Quality Prediction using Machine Learning

## Overview

This project focuses on predicting the **Air Quality Index (AQI)** levels using weather conditions and pollution sensor data.
The task is formulated as a **multi-class classification problem**, where the target variable `AirQuality` ranges from **1 to 5**.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Handling class imbalance using SMOTE
* Model training and evaluation
* Hyperparameter tuning
* Final prediction generation for submission

---

# Dataset

The dataset contains:

* Weather-related features
* Pollution sensor measurements
* Time-based features
* Target variable: `AirQuality`

## Files

* `train.csv` → Training dataset
* `test.csv` → Testing dataset
* `code_.ipynb` → Complete notebook implementation
* `requirements.txt` → Required Python libraries

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost

---

# Project Workflow

## 1. Data Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("train.csv")
```

---

## 2. Data Preprocessing

### Null Value Check

The dataset was checked for missing values.

```python
df.isnull().sum()
```

No missing values were found.

---

## 3. Exploratory Data Analysis (EDA)

### Target Distribution

The target variable `AirQuality` was found to be highly imbalanced.

### Correlation Analysis

Correlation analysis was performed to identify important features.

Key observations:

* `LevelPM2` and `LevelPM10` showed the highest positive correlation with AirQuality.
* Humidity showed negative correlation.
* Time-based cyclic features had very weak correlations.

---

## 4. Feature Selection

The following columns were removed:

* Time-related features
* Features with very low correlation

Dropped columns:

```python
['LocalDate', 'LocalHour', 'SineDay', 'CosineDay',
 'SineWeek', 'CosineWeek', 'SineMonth', 'CosineMonth',
 'SineYear', 'CosineYear', 'WindSine', 'WindCosine', 'dt']
```

---

## 5. Encoding Categorical Features

The categorical feature `WeatherType` was encoded using Label Encoding.

Encoding:

* Clear → 0
* Clouds → 1
* Rain → 2
* Snow → 3

---

## 6. Handling Class Imbalance

The dataset was highly imbalanced, so **SMOTE (Synthetic Minority Oversampling Technique)** was applied only on the training data.

```python
from imblearn.over_sampling import SMOTE
```

This balanced all classes equally.

---

## 7. Feature Scaling

Standardization was applied using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler
```

---

# Models Used

## 1. Logistic Regression

Used as a baseline model.

### Performance

* Test Accuracy: ~79-80%

---

## 2. Random Forest Classifier

Random Forest significantly improved prediction performance.

### Performance

* Test Accuracy: ~96-97%
* Strong generalization performance

---

## 3. Hyperparameter Tuned Random Forest

### Best Parameters

```python
RandomForestClassifier(n_estimators=200)
```

### Performance

* Test Accuracy: ~97.15%
* Cross-validation Accuracy: ~98.75%

This was the best-performing model.

---

## 4. XGBoost Classifier

### Performance

* Test Accuracy: ~96%
* Strong alternative to Random Forest

---

# Final Conclusion

| Model               | Performance |
| ------------------- | ----------- |
| Logistic Regression | Moderate    |
| Random Forest       | Excellent   |
| Tuned Random Forest | Best        |
| XGBoost             | Very Good   |

## Best Model

The **Hyperparameter-Tuned Random Forest Classifier** achieved the best balance of:

* Accuracy
* Precision
* Recall
* F1 Score

and was selected as the final model.

---

# Submission Generation

Predictions were generated for:

* Random Forest
* XGBoost

Submission files:

* `submission_rf.csv`
* `final_submission.csv`

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Raj9818/Air-Quality.git
```

Move into the project directory:

```bash
cd Air-Quality
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Project

Open the notebook:

```bash
jupyter notebook
```

Run:

```bash
code_.ipynb
```

---

# Future Improvements

* Deep Learning models
* Advanced feature engineering
* Better hyperparameter tuning
* Real-time AQI prediction deployment
* Web application integration

---

# Author

**Raj Singh**

GitHub: [Raj9818 GitHub Profile](https://github.com/Raj9818?utm_source=chatgpt.com)
