import joblib
import pandas as pd

# Load model
model = joblib.load("models/churn_model.pkl")

# Load threshold
threshold = joblib.load("models/churn_threshold.pkl")

print("Model loaded successfully!")
print("Threshold:", threshold)


# Test customer
customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 1026.00,
    "NumberOfServices": 5,
    "HasInternet": 1,
    "HasTechSupport": 0,
    "HasOnlineSecurity": 0,
    "AvgMonthlySpend": 85.50,
    "TenureGroup": "1-2 Years"
}])

# Predict probability
probability = model.predict_proba(customer)[0, 1]

# Apply selected threshold
prediction = int(probability >= threshold)

print("\nPrediction Results")
print("------------------")
print(f"Churn Probability: {probability:.2%}")
print(f"Threshold: {threshold:.2f}")

if prediction == 1:
    print("Prediction: CHURN")
else:
    print("Prediction: NO CHURN")