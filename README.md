# Heart Health AI Predictor 🫀

A machine learning-powered web application that predicts the likelihood of heart disease based on clinical parameters. This project demonstrates an end-to-end data science pipeline, from data preprocessing and model training to deployment using Streamlit.



## 🚀 Overview
The Heart Health AI Predictor uses a trained classification model to analyze patient data (such as age, cholesterol levels, and blood pressure) to provide an immediate risk assessment. The goal is to assist medical professionals with a data-driven second opinion.

## 🛠️ Tech Stack
* **Language:** Python 3.14
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn, Random Forest, Logistic Regression
* **Data Handling:** Pandas, NumPy
* **Data Processing:** Principal Component Analysis (PCA), Standard Scaling, Category Encoding
* **Model Persistence:** Joblib

## 📊 Project Workflow
1.  **Exploratory Data Analysis (EDA):** Analyzing the Heart Disease dataset to identify key features.
2.  **Preprocessing:** Handling categorical variables using Label Encoding and One-Hot Encoding. Scaling numerical features for optimal model performance.
3.  **Feature Engineering:** Applying PCA to reduce dimensionality while preserving 95% of the variance.
4.  **Modeling:** Training and evaluating multiple classifiers to select the best-performing model.
5.  **Deployment:** Creating an interactive dashboard for real-time predictions.



## 📂 Repository Structure
```text
├── app.py                     # Streamlit web application
├── main.py                    # Model training and preprocessing script
├── HeartDiseaseTrain-Test.csv # Project dataset
├── final_model.joblib         # Trained machine learning model
├── preprocessor.joblib        # Saved preprocessing objects (Scalers/Encoders)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
