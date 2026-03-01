# 🎓 Student Exam Score Prediction using XGBoost

Deployment Link :- https://student-marks-prediction-model.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Model Accuracy](https://img.shields.io/badge/Accuracy-73%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

This project predicts **student exam scores** based on academic, behavioral, and environmental factors using an optimized **XGBoost Regression model**.

The goal of this project is to analyze how various features such as study habits, attendance, sleep quality, and exam difficulty influence student performance.



## 🧠 Problem Statement

Educational institutions often struggle to identify key factors affecting student performance.

This project aims to:

- Predict student exam scores
- Identify influential features
- Build a robust regression model using boosting techniques
- Apply hyperparameter tuning for performance optimization



## 📊 Dataset Features

The model is trained on the following input features:

- age  
- gender  
- course  
- study_hours  
- class_attendance  
- internet_access  
- sleep_hours  
- sleep_quality  
- study_method  
- facility_rating  
- exam_difficulty  

🎯 **Target Variable:**
- `exam_score`



## ⚙️ Machine Learning Workflow

### 1️⃣ Data Preprocessing
- Handled categorical features using **Label Encoding**
- Feature-target separation
- Train-test split
- Data consistency validation

### 2️⃣ Model Selection
- Algorithm: **XGBoost Regressor**
- Reason: Handles non-linearity, interactions, and complex patterns effectively

### 3️⃣ Hyperparameter Tuning
Optimized parameters such as:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

Used tuning techniques to improve model generalization and reduce overfitting.



## 📈 Model Performance

- 📊 R² Score: ~0.73
- Model Accuracy: ~73%
- Good generalization performance on unseen data

The model captures significant patterns between study habits and exam scores while maintaining reasonable bias-variance tradeoff.


## 📂 Project Structure

Student-Marks-Prediction/

├── model.pkl

├── encoder.pkl

├── app.py 

├── requirements.txt

└── README.md



## 🚀 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn (for visualization)
- Pickle (Model Serialization)

## 🔍 Key Learnings

- Understanding boosting algorithms (XGBoost)
- Importance of hyperparameter tuning
- Handling categorical variables using encoding
- Evaluating regression models using R²
- Feature influence on student performance



## 📌 Future Improvements

- Add Feature Importance visualization
- Try advanced encoding (OneHot / Target Encoding)
- Perform cross-validation
- Deploy interactive Streamlit dashboard
- Compare with other ensemble models (Random Forest, Gradient Boosting)

# 🚀 Installation & Usage

## 🔹 1️⃣ Clone the Repository

git clone https://github.com/akshitgajera1013/Student-Marks-Prediction.git

cd Student-Marks-Prediction

pip install -r requirements.txt

streamlit run app.py

## 👨‍💻 Author

**Akshit Gajera**  
Aspiring Data Scientist | Machine Learning Enthusiast  

GitHub: https://github.com/akshitgajera1013  





