import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib   # for saving the model

# 1. Load dataset safely
csv_path = "all_gestures.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f" CSV file not found at path: {csv_path}")

df = pd.read_csv(csv_path)

# Check if 'label' column exists
if "label" not in df.columns:
    raise ValueError(" 'label' column not found in the CSV file.")

# 2. Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

# 4. Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# 5. Accuracy
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

# 6. SAVE THE MODEL
model_path = "gesture_model.pkl"
joblib.dump(model, model_path)

print(f"\n Model saved successfully as {model_path}")
