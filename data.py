import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dSet = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

def sleep_quality_label(x):
    if x <= 4:
        return "Poor"
    elif x <= 6:
        return "Average"
    else:
        return "Good"

dSet["SleepQualityLabel"] = dSet["Quality of Sleep"].apply(
    lambda x: "Good Sleep Quality:" if x >= 7 else "Bad Sleep Quality:"
)

#print(df["SleepQualityLabel"].value_counts()) prints number amount of people sleep quality

features = [
    "Sleep Duration",
    "Stress Level",
    "Physical Activity Level",
    "Daily Steps",
    "Heart Rate",
    "BMI Category"
]

X = dSet[features]
y = dSet["SleepQualityLabel"]

X = pd.get_dummies(X, columns=["BMI Category"], drop_first=True)
print(X.head())

print(X.shape)
print(y.value_counts())

#Train the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

#Evaluate the model
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Feature importance
feature_names = X.columns
coefficients = model.coef_[0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print(importance)