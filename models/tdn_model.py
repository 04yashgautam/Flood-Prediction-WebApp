import os
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Ensure directories exist before saving
def ensure_dir(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Load preprocessed data
X_train = np.load('../data/processed/X_train.npy')
y_train = np.load('../data/processed/y_train.npy')
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

# Ensure sequences are created properly
def create_sequences(X, y, window=3):
    if len(X) <= window:
        raise ValueError("Window size is too large for dataset length.")

    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])

    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)  # Ensure correct shape

window_size = 3
X_train_seq, y_train_seq = create_sequences(X_train, y_train, window=window_size)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, window=window_size)

print("X_train_seq shape:", X_train_seq.shape)

# Build TDNN Model
model = Sequential([
    InputLayer(input_shape=(window_size, X_train_seq.shape[-1])),
    SimpleRNN(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=2
)





# Compute accuracy
y_test_seq_binary = (y_test_seq > 0.5).astype(int).flatten()
y_pred_prob = model.predict(X_test_seq)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_test_seq_binary, y_pred)
print(f"TDNN Accuracy: {accuracy:.2%}")


# Save metrics safely
metrics_path = '../models/metrics/tdnn_metrics.txt'
ensure_dir(metrics_path)
with open(metrics_path, 'w') as f:
    f.write(f"TDNN Accuracy: {accuracy:.2%}\n")

# Plot training history
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plot_path = '../models/metrics/tdnn_training_history.png'
ensure_dir(plot_path)
plt.savefig(plot_path)
plt.show()

# Save Model
model_path = '../best_model_tdnn.h5'
ensure_dir(model_path)
model.save(model_path)
print(f"Model saved to {model_path}")