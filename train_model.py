import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Folder containing CSV files
dataset_path = "dataset"

X = []
y = []

# Read all CSV files
for file in os.listdir(dataset_path):

    if file.endswith(".csv"):

        label = file.replace(".csv", "")

        data = pd.read_csv(dataset_path + "/" + file, header=None)

        for row in data.values:
            X.append(row)
            y.append(label)

print(f"Total Samples: {len(X)}")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/hand_sign_model.pkl")

print("✅ Model trained successfully!")
print("✅ Model saved in models/hand_sign_model.pkl")