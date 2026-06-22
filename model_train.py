import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# Load dataset
df = pd.read_csv("cardio_train.csv", sep=";")

# 🔎 Debug: check actual column names
print("Columns in dataset before cleaning:", df.columns.tolist())

# Clean column names (remove spaces, hidden BOM characters)
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)

print("Columns after cleaning:", df.columns.tolist())

# Ensure required columns exist
if "cardio" not in df.columns or "id" not in df.columns:
    raise ValueError("❌ 'cardio' or 'id' column not found. Please check dataset headers.")

# Split features and target
X = df.drop(columns=["cardio", "id"])
y = df["cardio"]

# Convert age (days → years)
if "age" in X.columns:
    X["age"] = X["age"] / 365

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Train
model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)

# Save model + scaler
model.save("cardio_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")
