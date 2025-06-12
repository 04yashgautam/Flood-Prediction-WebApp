import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('../data/processed/X_train.npy')
y_train = np.load('../data/processed/y_train.npy')
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

# Load original column names (features only)
RAW_DATA_PATH = "../data/raw/flood_data.csv"
df = pd.read_csv(RAW_DATA_PATH)
feature_names = df.drop(columns=['FloodProbability']).columns.tolist()

# Discretize flood probabilities into 3 classes
def discretize(y):
    return np.digitize(y, bins=[0.33, 0.66])  # 0: low, 1: medium, 2: high

y_train_disc = discretize(y_train)
y_test_disc = discretize(y_test)

# Train model
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train_disc)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_disc, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Create metrics directory
os.makedirs('../models/metrics', exist_ok=True)

# Save metrics
with open('../models/metrics/lr_metrics.txt', 'w') as f:
    f.write(f"Logistic Regression Accuracy: {accuracy:.2%}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test_disc, y_pred)) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_disc, y_pred))

# Plot coefficients with feature names
plt.figure(figsize=(14, 6))
for i, class_label in enumerate(['Low', 'Medium', 'High']):
    plt.plot(model.coef_[i], label=f'Class {class_label}')
plt.xticks(ticks=range(len(feature_names)), labels=feature_names, rotation=90)
plt.title('Logistic Regression Coefficients by Class')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../models/metrics/lr_coefficients.png')
plt.show()

# Save model
joblib.dump(model, '../models/lr_model.pkl')
print("Logistic Regression model saved to ../models/lr_model.pkl")