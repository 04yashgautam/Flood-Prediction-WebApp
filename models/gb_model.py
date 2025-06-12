# Gradient Boosting Regressor for Flood Probability Prediction
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('../data/processed/X_train.npy')
y_train = np.load('../data/processed/y_train.npy')
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Create metrics directory if it doesn't exist
os.makedirs('../models/metrics', exist_ok=True)

# Save metrics
with open('../models/metrics/gb_metrics.txt', 'w') as f:
    f.write(f"Gradient Boosting Regressor MSE: {mse:.4f}\n")
    f.write(f"R^2 Score: {r2:.4f}\n")

# Plot training deviance (loss curve)
plt.figure(figsize=(10, 5))
plt.plot(model.train_score_, 'b-')
plt.title('Training Deviance')
plt.xlabel('Boosting Stages')
plt.ylabel('Loss (Deviance)')
plt.tight_layout()
plt.savefig('../models/metrics/gb_training_deviance.png')
plt.show()

# Save model
joblib.dump(model, '../models/gb_model.pkl')
print("Gradient Boosting model saved to ../models/gb_model.pkl")