# Housing Price Prediction using Machine Learning

## Overview

This project implements an **end-to-end machine learning pipeline** to predict housing prices using the California Housing dataset.

The project demonstrates the complete workflow used in practical machine learning systems:

* Data preprocessing
* Stratified sampling
* Feature transformation pipelines
* Model training
* Cross-validation evaluation
* Model comparison

The goal is to build a reliable regression system and compare multiple machine learning models.

---

## Project Structure

```
housing-price-prediction-ml
│
├── data
│   └── housing.csv
│
├── notebooks
│   │── analyzing-data.ipynb
│   └── analyzing-data.ipynb
│
├── src
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Machine Learning Pipeline

The project implements the following workflow:

1. **Data Loading**

   * Housing dataset loaded using Pandas.

2. **Stratified Sampling**

   * Dataset split using income categories to preserve distribution.

3. **Feature Engineering**

   * Numerical features
   * Categorical feature: `ocean_proximity`

4. **Data Preprocessing**

   * Missing value imputation
   * Standardization of numerical features
   * One-hot encoding of categorical features

5. **Pipeline Construction**

   * `Pipeline`
   * `ColumnTransformer`

6. **Model Training**

   * Linear Regression
   * Decision Tree Regressor
   * Random Forest Regressor

7. **Model Evaluation**

   * Root Mean Squared Error (RMSE)
   * Cross-validation

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/housing-price-prediction-ml.git
```

Navigate into the project directory:

```
cd housing-price-prediction-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Run the main script:

```
python src/main.py
```

The program trains multiple models and prints cross-validation RMSE statistics.

---

## Models Implemented

| Model             | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| Linear Regression | Baseline regression model                       |
| Decision Tree     | Non-linear model for capturing complex patterns |
| Random Forest     | Ensemble model for improved prediction accuracy |

---

## Future Improvements

Planned enhancements:

* Hyperparameter tuning using GridSearchCV
* Feature importance analysis
* Custom feature engineering
* Model persistence using `joblib`
* REST API for predictions
* Web interface for interactive predictions
* Implementation of regression algorithms from scratch

---

## Learning Objectives

This project focuses on understanding:

* Machine learning pipelines
* Data preprocessing techniques
* Model evaluation strategies
* Regression model comparison

---

## Author

Created as part of a machine learning learning journey.
