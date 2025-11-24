import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib   # <-- required

# 1. Load dataset
df = pd.read_csv("all_gestures.csv")

X = df.drop("label", axis=1)
y = df["label"]

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Train model
model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

# 4. Show accuracy
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))

# 5. SAVE THE MODEL (write here)
joblib.dump(model, "gesture_model.pkl")

print("\nModel saved as gesture_model.pkl")

