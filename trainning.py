import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data\synthetic_health_data_with_risk.csv")

# Fill missing values in 'Pre-existing Conditions' with 'None'
data["Pre-existing Conditions"].fillna("None", inplace=True)

# Ensure all expected categories are included
lifestyle_values = ["Active", "Moderate", "Dormant"]
condition_values = ["None", "Diabetes", "Blood Pressure", "Diabetes & Blood Pressure"]
risk_values = ["Low", "Moderate", "High"]

# Encode categorical variables
label_encoders = {
    "Lifestyle": LabelEncoder(),
    "Pre-existing Conditions": LabelEncoder(),
    "Health Risk": LabelEncoder()
}

label_encoders["Lifestyle"].fit(lifestyle_values)
label_encoders["Pre-existing Conditions"].fit(condition_values)
label_encoders["Health Risk"].fit(risk_values)

# Apply encoding
data["Lifestyle"] = label_encoders["Lifestyle"].transform(data["Lifestyle"])
data["Pre-existing Conditions"] = label_encoders["Pre-existing Conditions"].transform(data["Pre-existing Conditions"])
data["Health Risk"] = label_encoders["Health Risk"].transform(data["Health Risk"])

# Split dataset
X = data[["Age", "Lifestyle", "Pre-existing Conditions"]]
y = data["Health Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and encoders
joblib.dump(model, "health_risk_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Model and encoders saved successfully!")
