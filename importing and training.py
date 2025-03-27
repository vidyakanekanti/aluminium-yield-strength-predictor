import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Load dataset
file_path = "features_al_data.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# Debug: Show available columns
print("Columns in dataset:", df.columns)

# Encode categorical features
label_encoder = LabelEncoder()
df["Processing"] = label_encoder.fit_transform(df["Processing"])

# Define input (X) and target (y)
X = df[["Processing", "Tensile Strength (MPa)"]]
y = df["Yield Strength (MPa)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Save model and encoder
with open("regression_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model and Label Encoder saved successfully!")
