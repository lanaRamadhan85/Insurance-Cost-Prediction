"""
Script untuk melatih model Linear Regression dan menyimpannya sebagai .pkl file
Jalankan script ini terlebih dahulu sebelum menjalankan Flask app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv('data/insurance.csv')

# Membuat copy untuk preprocessing
df_processed = df.copy()

# Encoding variabel kategorikal menggunakan LabelEncoder
print("Preprocessing data...")
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df_processed['sex_encoded'] = le_sex.fit_transform(df_processed['sex'])
df_processed['smoker_encoded'] = le_smoker.fit_transform(df_processed['smoker'])
df_processed['region_encoded'] = le_region.fit_transform(df_processed['region'])

# Menyimpan encoders untuk digunakan di Flask app
encoders = {
    'sex': le_sex,
    'smoker': le_smoker,
    'region': le_region
}

# Memisahkan features dan target
X = df_processed[['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']]
y = df_processed['charges']

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  MAE: ${train_mae:.2f}")

print(f"\nTest Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  MAE: ${test_mae:.2f}")

# Menyimpan model dan encoders
print("\nSaving model and encoders...")
os.makedirs('model', exist_ok=True)

# Simpan model
with open('model/insurance_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Simpan encoders
with open('model/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✓ Model saved to: model/insurance_model.pkl")
print("✓ Encoders saved to: model/encoders.pkl")
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run the Flask app with: python app.py")

