# Predicting Adult Income Using Machine Learning!
![adult_income](https://github.com/user-attachments/assets/a981cd94-5b17-4bf2-bd96-b050c8349614)

## Overview

This repository contains a comprehensive analysis of predicting adult income using various machine learning algorithms, focusing primarily on logistic regression. The goal is to classify whether an individual's income exceeds $50K per year based on demographic and financial attributes. The analysis involves data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation using multiple algorithms.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Dataset](#dataset)
5. [Analysis Steps](#analysis-steps)
   - [1. Data Loading and Exploration](#1-data-loading-and-exploration)
   - [2. Data Cleaning and Preprocessing](#2-data-cleaning-and-preprocessing)
   - [3. Feature Engineering](#3-feature-engineering)
   - [4. Model Building and Evaluation](#4-model-building-and-evaluation)
   - [5. Model Tuning](#5-model-tuning)
   - [6. Summary of Results](#6-summary-of-results)
6. [How to Use](#how-to-use)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Project Structure

The repository is organized as follows:

```
predicting-adult-income/
│
├── data/
│   └── adult.csv                             # Raw dataset used in the analysis
│
├── notebooks/
│   └── Logistic_Regression_for_Predicting_Adult_Income.ipynb  # Jupyter Notebook with full analysis
│
├── README.md                                  # Project documentation
├── LICENSE                                    # License for the project
└── .gitignore                                 # Git ignore file
```

## Prerequisites

Before running the notebook, ensure you have the following installed:

- **Python 3.x**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **mlxtend**

You can install the required Python packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost mlxtend
```

## Dataset

The dataset used in this analysis, `adult.csv`, contains demographic information about individuals, including attributes like age, work class, education, marital status, occupation, race, gender, and more. The target variable is `income`, which is binary, indicating whether an individual's income is `<=50K` or `>50K`.

## Analysis Steps

### 1. Data Loading and Exploration
The dataset is loaded using Pandas, followed by initial exploration to understand its structure, including dimensions, summary statistics, and distribution of key variables.

### 2. Data Cleaning and Preprocessing
The data is cleaned by handling missing values, particularly in columns such as `workclass`, `occupation`, and `native-country`. Categorical variables are encoded using techniques like One-Hot Encoding.

### 3. Feature Engineering
New features are created by transforming existing attributes to better represent the data's underlying patterns. The dataset is also standardized to optimize model performance.

### 4. Model Building and Evaluation
Multiple machine learning models are built and evaluated, including:
- **Logistic Regression**
- **Support Vector Classification (SVC)**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**
- **Decision Tree Classifier**
- **K-Neighbors Classifier**
- **Gaussian Naive Bayes**

Each model is evaluated using cross-validation techniques and metrics such as accuracy scores.

### 5. Model Tuning
Hyperparameter tuning is performed using GridSearchCV and RandomizedSearchCV for each model to identify the best parameters that improve model performance.

### 6. Summary of Results
The Random Forest and Gradient Boosting models achieved the highest accuracy scores (86.8%) in predicting whether an individual's income is greater than $50K, demonstrating the effectiveness of ensemble methods in classification tasks.

## How to Use

1. **Clone this repository**:
    ```bash
    git clone https://github.com/yourusername/predicting-adult-income.git
    cd predicting-adult-income
    ```

2. **Prepare your data**:
    - Ensure that `adult.csv` is placed in the `data/` directory.

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook notebooks/Logistic_Regression_for_Predicting_Adult_Income.ipynb
    ```

4. **Follow the steps in the notebook** to perform the analysis and understand the model-building process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this analysis was derived from the UCI Machine Learning Repository.
- Special thanks to the developers of open-source libraries like Pandas, NumPy, Scikit-learn, and XGBoost that made this analysis possible.

---

This README provides a comprehensive guide to understanding and running the income prediction analysis using logistic regression and other machine learning algorithms. Feel free to customize it further based on your specific need
![image](https://github.com/user-attachments/assets/fd0cf93f-10e1-4f00-a8ac-c148a0f62c9d)
