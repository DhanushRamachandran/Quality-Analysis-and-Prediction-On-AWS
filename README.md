### MLFlow experiment tracker
Wine Quality Analysis & Prediction using Elastic Net Regression

# End-to-End ML Pipeline with MLflow Experiment Tracking on AWS EC2
## Project Overview

This repository presents a complete machine learning workflow for predicting wine quality based on physicochemical properties using the Elastic Net Regression model.

The project emphasizes:

 Systematic ML experimentation

 Feature engineering & data preprocessing

 Model evaluation & hyperparameter tuning

 Production-style experiment tracking using MLflow

 Remote MLflow tracking server hosted on AWS EC2 instance

 Reproducible end-to-end pipelines

This project demonstrates how real-world ML systems are built — where model experimentation, tracking, and reproducibility are critical.

# Problem Statement

The goal is to build a regression model that predicts wine quality score (0-10) using chemical characteristics such as acidity, sugar content, sulphates, alcohol, etc.

Dataset used: Wine Quality Dataset (UCI Repository)

Tasks:

Explore data through EDA

Engineer and scale features

Build Elastic Net regression model

Log experiments to MLflow

Evaluate and compare runs


#  Why Elastic Net?

Elastic Net Regression combines L1 + L2 regularization, making it ideal when:

Challenge	Elastic Net Benefit
High multicollinearity	Removes & shrinks correlated features
Need balanced sparsity & stability	Combines Lasso + Ridge strength
Generalization required	Prevents overfitting

Loss Function:
Loss = 𝑀𝑆𝐸 + 𝛼( 𝜆1∣∣𝑤∣∣1 + 𝜆2∣∣𝑤∣∣22 )
Where:
α = regularization strength

l1_ratio = balance between L1 & L2
| Category            | Tools                                            |
| ------------------- | ------------------------------------------------ |
| Language            | Python                                           |
| Model               | Elastic Net (scikit-learn)                       |
| Experiment Tracking | MLflow                                           |
| Deployment Platform | AWS EC2 (Ubuntu)                                 |
| Libraries           | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| Logging             | MLflow & logging module                          |

## What gets tracked?

Model parameters (alpha, l1_ratio)

Metrics (MAE, RMSE, R²)

Artifacts (model pickle, plots)

Run timestamps & tags

# Evaluation mwtrics

| Metric   | Interpretation                    |
| -------- | --------------------------------- |
| MAE      | Average absolute prediction error |
| RMSE     | Penalizes large errors            |
| R² Score | Variance explained by model       |


# AWS storage and usage
| Component                                   | Storage          | Purpose                |
| ------------------------------------------- | ---------------- | ---------------------- |
| Experiment Metadata (runs, params, metrics) | SQLite / AWS RDS | Persistent tracking DB |
| Artifacts (model files, plots, conda.yaml)  | **Amazon S3**    | Scalable storage       |
| MLflow App                                  | AWS EC2          | Remote tracking server |

Screen shots:


