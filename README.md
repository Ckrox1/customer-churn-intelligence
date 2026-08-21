# Customer Churn Intelligence

An end-to-end machine learning application for **customer churn prediction and customer segmentation**, built with Python and Streamlit.

This project combines supervised and unsupervised machine learning to identify customers who are likely to churn and discover meaningful customer segments based on behavior, spending, tenure, and service adoption.

---

## 🚀 Project Overview

Customer churn is a major business problem for subscription-based companies.

This project addresses the problem from two complementary perspectives:

- **Churn Prediction:** Predict whether a customer is likely to leave.
- **Customer Segmentation:** Group customers with similar behavioral and spending characteristics.

The final application provides an interactive Streamlit dashboard where users can enter customer information, receive churn predictions, and identify customer segments.

---

## ✨ Features

### Churn Prediction

- Data preprocessing and feature engineering
- Multiple classification algorithms
- Model comparison
- Cross-validation
- Hyperparameter tuning
- XGBoost optimization
- Classification threshold optimization
- Churn probability estimation
- Interactive prediction interface

### Customer Segmentation

- Feature engineering for customer behavior
- Feature scaling
- K-Means clustering
- Elbow method for cluster selection
- Silhouette score analysis
- DBSCAN clustering
- Cluster profiling
- Business interpretation of customer segments

### Application

- Interactive Streamlit dashboard
- Churn prediction interface
- Customer segmentation interface
- Customer risk interpretation
- Cluster-level customer insights

---

## 🧠 Machine Learning Techniques

### Supervised Learning

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

### Unsupervised Learning

- K-Means Clustering
- DBSCAN

### Model Optimization

- Train/Test Split
- Cross-Validation
- Hyperparameter Tuning
- Randomized Search
- Classification Threshold Optimization

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.8020 | 0.6568 | 0.5321 | 0.5879 | **0.8453** |
| Logistic Regression | **0.8027** | 0.6611 | 0.5267 | 0.5863 | 0.8428 |
| Gradient Boosting | 0.8013 | **0.6667** | 0.5027 | 0.5732 | 0.8427 |
| Random Forest | 0.7871 | 0.6217 | 0.5053 | 0.5575 | 0.8229 |
| SVM | 0.7857 | 0.6429 | 0.4332 | 0.5176 | 0.7965 |
| Decision Tree | 0.7495 | 0.5283 | 0.5241 | 0.5262 | 0.6770 |

XGBoost achieved the highest ROC-AUC among the evaluated models and was selected as the primary churn model.

---

## 🎯 Classification Threshold Optimization

The default classification threshold of `0.50` was analyzed rather than being used blindly.

Because missing a potentially churning customer can be costly, the threshold was optimized based on F1-score.

The selected threshold was **0.30**.

| Metric | Default Threshold | Optimized Threshold |
|---|---:|---:|
| Recall | 0.521 | **0.770** |
| F1 Score | 0.583 | **0.621** |
| ROC-AUC | 0.845 | 0.845 |

The optimized threshold substantially increased recall, allowing the system to identify a larger proportion of potentially churning customers.

---

## 👥 Customer Segmentation

K-Means clustering was used to identify customer groups based on:

- Tenure
- Monthly charges
- Total charges
- Number of services
- Internet usage
- Technical support
- Online security

The final K-Means model produced **8 customer segments**.

### Example Customer Segments

#### Cluster 2 — Premium High-Engagement Customers

- Average tenure: 62.7 months
- Average monthly charges: 91.29
- Average total charges: 5716.71
- Average services: 6.44
- 100% have technical support
- 76% have online security

**Business interpretation:** High-value customers with long relationships and strong service adoption. They are strong candidates for retention and loyalty programs.

#### Cluster 0 — High-Value Established Customers

- Average tenure: 56.2 months
- Average monthly charges: 94.61
- Average total charges: 5314.02
- Average services: 5.58

**Business interpretation:** Long-term customers with high spending and broad service adoption. Retention and premium-service strategies may be valuable.

#### Cluster 6 — Newer Moderate-Spending Customers

- Average tenure: 17.4 months
- Average monthly charges: 74.53
- Average services: 2.83
- Low technical-support adoption
- Low online-security adoption

**Business interpretation:** Potential targets for onboarding and cross-selling additional services.

#### Cluster 3 — Low-Value Basic Customers

- Average tenure: 30.6 months
- Average monthly charges: 21.08
- Average total charges: 665.22
- Average services: 1.22
- No internet service

**Business interpretation:** Low-spending customers with limited service adoption. Marketing strategies should be cost-efficient and targeted.

---

## 🔬 Data Science Workflow

### Churn Prediction

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Feature Encoding
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Model Selection
     ↓
Threshold Optimization
     ↓
Model Serialization
     ↓
Streamlit Application
```

### Customer Segmentation

```text
Customer Data
     ↓
Feature Selection
     ↓
Feature Scaling
     ↓
K-Means
     ↓
Elbow Method
     ↓
Silhouette Analysis
     ↓
Cluster Profiling
     ↓
Business Interpretation
```

---

## 🛠️ Tech Stack

### Programming
- Python

### Data Science
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Visualization
- Matplotlib
- Seaborn

### Application
- Streamlit

### Model Persistence
- Joblib

### Development
- Jupyter Notebook
- Visual Studio Code
- Git & GitHub

---

## 📁 Project Structure

```text
customer-churn-intelligence/
│
├── app.py
├── test_model.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── churn_threshold.pkl
│   ├── kmeans_model.pkl
│   ├── cluster_scaler.pkl
│   ├── cluster_features.pkl
│   ├── cluster_profile.csv
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
└── assets/
    ├── dashboard.png
    ├── churn_prediction.png
    └── customer_segmentation.png
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-intelligence.git
cd customer-churn-intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📌 Key Insights

### Churn Prediction

XGBoost produced the strongest ROC-AUC performance among the evaluated models, reaching approximately **0.845**.

Threshold optimization increased churn recall from approximately **52% to 77%**, demonstrating how model probabilities can be adapted to business objectives.

### Customer Segmentation

K-Means identified **8 distinct customer groups** with different combinations of customer lifetime, spending, service adoption, internet usage, technical support, and online security.

These segments can support targeted retention, cross-selling, and customer engagement strategies.

---

## 📈 Future Improvements

- Address class imbalance using techniques such as SMOTE
- Add explainable AI using SHAP
- Add feature-importance visualizations
- Automate model retraining
- Add customer-level churn explanations
- Build a real-time prediction API
- Add database integration
- Deploy to the cloud
- Monitor model performance over time

---

## 👨‍💻 Author

**Chirag Goel**

B.Tech — Geoinformatics  
NSUT Delhi

---

## ⭐ Project Highlights

This project demonstrates practical experience with:

- End-to-end machine learning
- Supervised learning
- Unsupervised learning
- Feature engineering
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Ensemble learning
- XGBoost
- Clustering
- Streamlit application development
- Model deployment workflow
