#  Heart Disease Risk Prediction System

An end-to-end **machine learning application** that predicts the probability of heart disease based on patient clinical features. The project demonstrates a full **ML engineering workflow**, including data preprocessing, model training, evaluation, and deployment through an interactive web interface.

The system enables users to input patient health metrics and obtain **real-time risk predictions** using a trained machine learning model.

This project was built as part of a portfolio to demonstrate practical experience in **machine learning model development, evaluation, and deployment**.

---

#  Live Application

## Streamlit Web App

[Heart Disease Risk Predictor](https://cardio-risk.streamlit.app)

The application allows users to:

- Estimate heart disease risk from clinical parameters
- Run batch predictions using uploaded CSV files
- View model performance metrics

---

# Machine Learning Pipeline

The project implements a structured ML pipeline following best practices used in real-world ML systems.

## 1. Data Processing

The dataset contains clinical indicators commonly used in cardiovascular risk analysis.

### Features

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise induced angina
- ST depression (oldpeak)

### Preprocessing Steps

- Feature selection
- Handling categorical variables
- Numerical feature scaling
- Train/test split

---

## 2. Model Training

Multiple classification models were trained and evaluated.

### Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting

The final model was selected based on performance across multiple evaluation metrics.

---

## 3. Model Evaluation

Model performance was assessed using several classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Using multiple metrics ensures a reliable and balanced evaluation of model performance.

---

## 4. Model Serialization

The trained model is exported using **joblib**:
models/heart_model.pkl

This allows the model to be loaded efficiently for inference within the web application.

---

#  Application Architecture

The project follows a **modular machine learning system design**, separating model training from the user interface.
heart-disease-predictor
│
├── app
│ ├── streamlit_app.py
│ ├── utils.py
│ └── pages
│ ├── 1_Single_Prediction.py
│ ├── 2_Batch_Prediction.py
│ ├── 3_Model_Performance.py
│ └── 4_About.py
│
├── src
│ ├── train.py
│ ├── predict.py
│
├── models
│ ├── heart_model.pkl
│ └── metrics.json
│
├── requirements.txt
└── README.md

---

# 🌐 Web Application

The user interface is implemented using **Streamlit**, enabling interactive access to the trained model.

## Key Features

### Single Patient Prediction

Users can input patient clinical features through a web form to receive:

- Predicted risk classification
- Probability of heart disease
- Risk category

### Batch Prediction

The application allows uploading CSV datasets to perform predictions for multiple patients simultaneously.

Outputs include:

- Predicted labels
- Probability scores
- Downloadable results

### Model Performance Dashboard

Users can explore model evaluation results including:

- Model comparison metrics
- Best model selection
- Overall performance statistics

---

# Technology Stack

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- NumPy
- Pandas

## Data Visualization

- Matplotlib
- Seaborn

## Web Application

- Streamlit

## Model Serialization

- Joblib

## Version Control

- Git
- GitHub

---

#  Installation

## 1. Clone the repository

```bash
git clone https://github.com/indiradu/heart-disease-predictor.git
cd heart-disease-predictor
```
## 2. Install dependencies
```bash
pip install -r requirements.txt
```
## 3. Run the application
```bash
streamlit run app/streamlit_app.py
```
## 4. Open the application
http://localhost:8501

#  Example Prediction Output
## Example output returned by the model:
- Prediction: Low Risk
- Probability: 0.39
- Risk Band: Low

## ⚠️ Disclaimer
This application is intended for educational and research purposes only and should not be used for medical diagnosis or clinical decision-making.

## Author
Indira Duisembayeva

## GitHub:
https://github.com/indiradu