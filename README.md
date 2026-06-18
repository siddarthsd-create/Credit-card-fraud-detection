📌 Overview

Credit card fraud is a significant challenge in the financial industry, resulting in substantial financial losses each year. Since fraudulent transactions represent only a small fraction of total transactions, traditional machine learning models often struggle to identify them effectively.
This project builds an end-to-end fraud detection pipeline that:
Handles class imbalance using SMOTE
Trains and evaluates machine learning models
Compares model performance using key classification metrics
Visualizes results through confusion matrices

🎯 Problem Statement

Fraud detection is a highly imbalanced classification problem where:
Genuine transactions vastly outnumber fraudulent ones
False negatives can lead to financial losses
False positives can impact customer experience
Fraud patterns are often complex and non-linear
The goal is to develop a robust model capable of accurately identifying fraudulent transactions while minimizing classification errors.

📊 Dataset

This project uses the popular Credit Card Fraud Detection Dataset available on Kaggle.
Features
Feature
Description
Time
Seconds elapsed between transactions
Amount
Transaction amount
V1-V28
PCA-transformed anonymized features
Class
Target variable (0 = Genuine, 1 = Fraudulent)
Dataset Characteristics
Highly imbalanced dataset
Binary classification problem
Real-world financial transaction data

🛠️ Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn (SMOTE)

📦 Installation

Clone the repository:
Bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection-Using-Machine-Learning.git

cd Credit-Card-Fraud-Detection-Using-Machine-Learning
Install dependencies:
Bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

🚀 Project Workflow

1. Data Loading
Load the credit card transaction dataset.
Python
df = pd.read_csv("creditcard.csv")
2. Data Exploration
Check dataset structure
Analyze class distribution
Identify imbalance between fraud and non-fraud transactions
3. Train-Test Split
Python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
4. Apply SMOTE
Balance the minority fraud class using Synthetic Minority Oversampling Technique.
Python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
5. Model Training
Implemented models:
Logistic Regression
Python
LogisticRegression(max_iter=1000)
Random Forest
Python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
6. Model Evaluation
Metrics used:
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix

📈 Results

The models were evaluated on unseen test data.
Performance Metrics
Precision
Recall
F1 Score
ROC-AUC
Best Performing Model

🏆 Random Forest

Random Forest generally outperformed Logistic Regression by capturing complex, non-linear fraud patterns more effectively.

🔍 Key Findings

The dataset is highly imbalanced.
SMOTE successfully balances fraud and non-fraud classes.
Accuracy alone is not a reliable metric for fraud detection.
Precision and Recall provide better insights into model performance.
Random Forest achieved superior fraud detection capability.

📉 Visualization

The project generates:
Class distribution analysis
Confusion matrix heatmaps
Model performance comparisons
Example:
Python
cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

📂 Project Structure

Plain text
Credit-Card-Fraud-Detection-Using-Machine-Learning/
│
├── creditcard.csv
├── fraud_detection.py
├── requirements.txt
├── README.md
└── results/
    ├── confusion_matrix.png
    └── evaluation_metrics.csv

🔮 Future Improvements

Hyperparameter tuning using GridSearchCV
XGBoost implementation
Real-time fraud detection pipeline
Deep Learning models (ANN, LSTM)
Deployment using Flask/FastAPI
Model monitoring and drift detection

👨‍💻 Author
Siddarth 
