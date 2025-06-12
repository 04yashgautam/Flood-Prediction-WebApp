# Random Forest Classifier for Flood Risk Classification
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('../data/processed/X_train.npy')
y_train = np.load('../data/processed/y_train.npy')
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

# Load feature names from original dataset
RAW_DATA_PATH = "../data/raw/flood_data.csv"
df = pd.read_csv(RAW_DATA_PATH)
feature_names = df.drop(columns=['FloodProbability']).columns.tolist()

# Discretize target for classification
def discretize(y):
    return np.digitize(y, bins=[0.33, 0.66])  # 0: low, 1: medium, 2: high

y_train_disc = discretize(y_train)
y_test_disc = discretize(y_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_disc)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_disc, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Create metrics directory
os.makedirs('../models/metrics', exist_ok=True)

# Save metrics
with open('../models/metrics/rf_metrics.txt', 'w') as f:
    f.write(f"Random Forest Accuracy: {accuracy:.2%}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test_disc, y_pred)) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_disc, y_pred))

# Plot feature importances
plt.figure(figsize=(14, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('../models/metrics/rf_feature_importance.png')
plt.show()

# Save model if it's the best
current_best = 0.0  # Update based on comparison with other models
if accuracy > current_best:
    joblib.dump(model, '../best_model.pkl')
    print("Random Forest model saved as the best model to ../best_model.pkl")