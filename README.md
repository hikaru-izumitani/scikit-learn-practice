# scikit-learn Practice

A practice repository for learning the fundamentals of machine learning with scikit-learn.

## Purpose

This repository is a hands-on learning log rather than a production-ready machine learning project.

The goal is to understand the basic machine learning workflow by implementing small examples from scratch and visualizing their behavior.

## Topics Covered

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forests
* Data Preprocessing
* Cross-Validation
* Hyperparameter Tuning
* Pipelines
* Model Evaluation

## Machine Learning Workflow

The exercises gradually build the following workflow:

```text
Data
 ↓
Train / Test Split
 ↓
Preprocessing
 ↓
Model Training
 ↓
Cross-Validation
 ↓
Hyperparameter Tuning
 ↓
Pipeline
 ↓
Model Evaluation
```

## Key Concepts Learned

### Supervised Learning

Understanding the difference between regression and classification, including:

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forests

### Data Preprocessing

Using techniques such as `StandardScaler` to transform features before model training.

An important concept is avoiding data leakage by fitting preprocessing steps only on the training data.

### Model Validation

Using cross-validation to evaluate model performance across multiple data splits instead of relying on a single train/test split.

### Hyperparameter Tuning

Using `GridSearchCV` to systematically compare different hyperparameter combinations.

### Pipelines

Using scikit-learn `Pipeline` to combine preprocessing and model training into a single workflow.

This also helps prevent preprocessing leakage during cross-validation.

### Model Evaluation

Evaluating classification models using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* False Positives (FP)
* False Negatives (FN)

The exercises helped me understand that model performance is not only about accuracy. The type of error a model makes can also have important implications for a real-world business problem.

## Repository Structure

```text
scikit-learn-practice/
├── 01_linear_regression/
├── 02_logistic_regression/
├── 03_decision_tree/
├── 04_random_forest/
├── 05_preprocessing/
├── 06_cross_validation/
├── 07_hyperparameter_tuning/
├── 08_pipeline/
└── 09_model_evaluation/
```

## Note

The datasets used in these exercises are intentionally small and simplified.

They are designed to make the underlying machine learning concepts easy to understand and visualize, rather than to demonstrate production-level model performance.

The next step is to apply these concepts to more realistic datasets and domain-specific problems, particularly in finance and business analytics.
