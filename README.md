### MLFlow experiment tracker
Wine Quality Analysis & Prediction using Elastic Net Regression

# End-to-End ML Pipeline with MLflow Experiment Tracking on AWS EC2
## Project Overview

This repository presents a complete machine learning workflow for predicting wine quality based on physicochemical properties using the Elastic Net Regression model.

This project places strong emphasis on building a production-grade machine learning workflow that aligns with real-world industry standards. It incorporates systematic ML experimentation, where different model configurations, regularization strengths, and preprocessing strategies are rigorously tested and compared. The pipeline includes comprehensive feature engineering and data preprocessing, ensuring that input variables are clean, scaled, and meaningful for modeling. It implements thorough model evaluation and hyperparameter tuning to achieve optimal predictive performance and robustness. A key highlight is the production-style experiment tracking framework powered by MLflow, enabling structured logging of metrics, parameters, and model artifacts. The MLflow tracking server is deployed remotely on an AWS EC2 instance, with artifacts stored in Amazon S3, closely mirroring real-world MLOps deployment patterns used in enterprise environments. Overall, this project demonstrates how modern machine learning systems are developed, with a focus on traceability, scalability, reproducibility, and operational readiness—critical elements for delivering reliable and maintainable ML solutions in production.

# Problem Statement

The objective of this project is to develop a regression model capable of predicting the wine quality score (ranging from 0–10) based on a set of physicochemical attributes such as acidity, residual sugar, sulphates, alcohol percentage, and more. The widely-used Wine Quality Dataset from the UCI Machine Learning Repository is utilized for this study, providing a robust benchmark for evaluating model performance and experimentation. The workflow follows a structured machine learning approach, from exploratory data understanding to model training, tuning, and experiment management using MLflow.

Key tasks include:

1. Conducting exploratory data analysis (EDA) to understand distribution patterns, correlations, and feature relationships

2. Performing feature engineering and scaling to ensure high-quality model inputs

3. Training an Elastic Net regression model, leveraging combined L1 + L2 regularization

4. Logging parameters, metrics, models, and artifacts using MLflow

5. Evaluating and comparing ML experiments to select the best-performing configuration

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
| Experiment Metadata (runs, params, metrics) | SQLite           | Persistent tracking DB |
| Artifacts (model files, plots, conda.yaml)  | Amazon S3        | Scalable storage       |
| MLflow App                                  | AWS EC2          | Remote Ubuntu Instance |

Screen shots:
<img width="1212" height="103" alt="image" src="https://github.com/user-attachments/assets/8078e197-a78d-47d5-8d3f-b4bfed17d6db" />
<img width="1352" height="409" alt="image" src="https://github.com/user-attachments/assets/061d17c3-a8ac-47d0-bb7d-9baad973fb5f" />
<img width="1841" height="669" alt="image" src="https://github.com/user-attachments/assets/436ee676-4d93-4efd-af74-898419d43824" />
<img width="1838" height="412" alt="image" src="https://github.com/user-attachments/assets/feef3165-aba2-4c44-a2bd-8bd5e5831939" />
<img width="1867" height="687" alt="image" src="https://github.com/user-attachments/assets/4ec1a65a-93aa-4404-a72c-65eadca8a45c" />


