#  Cookie Cats Player Retention Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Project Overview

This project focuses on predicting **7-day player retention** for the popular mobile puzzle game **Cookie Cats** using Machine Learning.

Player retention is one of the most important metrics in the gaming industry. It helps game companies understand whether players continue playing the game after installation. Improving player retention directly increases user engagement and company revenue.

In this project, I performed a complete end-to-end Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and model evaluation using multiple Scikit-learn algorithms.

---

# Business Problem

Cookie Cats is a popular mobile puzzle game.

The company conducted an A/B test by placing the first gate at two different levels:

- Gate 30
- Gate 40

The goal is to determine whether changing the gate location affects player retention after seven days.

The objective of this project is to build a Machine Learning model that predicts whether a player will return after seven days based on their gameplay behavior.

---

#  Dataset Information

**Dataset Name**

Cookie Cats A/B Testing Dataset

**Source**

Kaggle

Dataset Link

https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats

---

# Dataset Features

| Feature | Description |
|----------|-------------|
| userid | Unique player ID |
| version | A/B Test version (Gate 30 or Gate 40) |
| sum_gamerounds | Total game rounds played |
| retention_1 | Returned after 1 day |
| retention_7 | Returned after 7 days (Target Variable) |

---

#  Target Variable

retention_7

The Machine Learning model predicts whether a player will return after 7 days.

---

#  Project Workflow

The project follows an end-to-end Machine Learning pipeline.

## Module 1 — Data Loading

Performed:

- Loaded dataset using Pandas
- Displayed first records
- Checked dataset shape
- Displayed column names
- Checked data types
- Generated statistical summary

---

## Module 2 — Data Cleaning

Performed:

- Missing value analysis
- Duplicate value detection
- Unique value inspection
- Dataset validation

Result:

- No missing values
- No duplicate records
- Clean dataset

---

## Module 3 — Exploratory Data Analysis (EDA)

Performed multiple visualizations including:

- Version Count Plot
- Day 1 Retention Plot
- Day 7 Retention Plot
- Distribution Plot
- Histogram
- Box Plot
- Violin Plot
- Correlation Heatmap
- Pair Plot
- KDE Plot
- Feature Comparison
- Outlier Analysis
- Player Distribution Graphs

Business insights were extracted from every visualization.

---

## Module 4 — Feature Engineering

Performed:

- Data preparation
- Feature selection
- Label Encoding
- Input-output separation

---

## Module 5 — Train-Test Split

Split the dataset into:

- Training Data
- Testing Data

Used Scikit-learn Train_Test_Split.

---

## Module 6 — Machine Learning Models

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

---

## Module 7 — Hyperparameter Tuning

Performed:

- GridSearchCV
- Cross Validation
- Random Forest Optimization

Selected the best-performing model.

---

## Module 8 — Model Saving

Saved the final optimized model using Joblib.

Output:

best_model.joblib

---

# Final Results

Best Model

Random Forest Classifier

Performance

Accuracy

80.33%

ROC-AUC Score

88.56%

The tuned Random Forest model achieved the best overall performance among all evaluated models.

---

#  Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- SciPy

Development Environment

- Visual Studio Code

Version Control

- Git
- GitHub

---

#  Project Structure

```
cookies_cats_project/

│

├── data/

│ └── cookie_cats.csv

│

├── models/

│ └── best_model.joblib

│

├── outputs/

│ ├── plots/

│ └── reports/

│

├── src/

│ ├── data_preprocessing.py

│ ├── data_cleaning.py

│ ├── eda.py

│ ├── feature_engineering.py

│ ├── train_test_split.py

│ ├── train_model.py

│ ├── hyperparameter_tuning.py

│ ├── save_model.py

│ └── evaluate_model.py

│

├── README.md

├── requirements.txt

└── .gitignore
```

---

#  Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cookies-cats-player-retention.git
```

Move into project directory

```bash
cd cookies-cats-player-retention
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python src/train_model.py
```

---

#  Future Improvements

Possible future enhancements include:

- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- Streamlit Deployment
- Docker Container
- Cloud Deployment

---

#  Learning Outcomes

This project helped me understand:

- End-to-End Machine Learning Workflow
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Classification Algorithms
- Hyperparameter Tuning
- Model Evaluation
- Model Saving
- GitHub Project Management

---

# Author

**Muhammad Awais**

Machine Learning Enthusiast

Python | Scikit-Learn | Data Science | Artificial Intelligence

---

# ⭐ If you found this project useful, please consider giving it a Star.
