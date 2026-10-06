import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# LOAD V4 DATA
# ============================================

X_train = np.load("Data/processed/X_train_v4.npy")
y_train = np.load("Data/processed/y_train_v4.npy")

X_dev = np.load("Data/processed/X_dev_v4.npy")
y_dev = np.load("Data/processed/y_dev_v4.npy")

X_test = np.load("Data/processed/X_test_v4.npy")
y_test = np.load("Data/processed/y_test_v4.npy")

print("Training data:", X_train.shape)
print("Development data:", X_dev.shape)
print("Test data:", X_test.shape)

# ============================================
# GENRE NAMES
# ============================================

genres = [
    "classical",
    "jazz",
    "metal",
    "pop"
]

print("\nGenre classes:")
print(genres)

# ============================================
# BASELINE MODEL
# ============================================

model = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation="tanh",
    solver="adam",
    learning_rate_init=0.001,
    batch_size="auto",
    max_iter=100,
    random_state=42,
    verbose=True
)

print("\n============================================")
print("BASELINE MODEL")
print("============================================")

print("Architecture: 500 -> 128 -> 4")
print("Activation: tanh")
print("Optimizer: Adam")
print("Learning rate: 0.001")
print("Maximum iterations: 100")

# ============================================
# TRAIN
# ============================================

print("\nStarting training...")

model.fit(
    X_train,
    y_train
)

# ============================================
# PREDICTIONS
# ============================================

train_predictions = model.predict(X_train)

dev_predictions = model.predict(X_dev)

test_predictions = model.predict(X_test)

# ============================================
# ACCURACY
# ============================================

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

dev_accuracy = accuracy_score(
    y_dev,
    dev_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\n============================================")
print("ACCURACY RESULTS")
print("============================================")

print(
    f"Training accuracy:    {train_accuracy * 100:.2f}%"
)

print(
    f"Development accuracy: {dev_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy:        {test_accuracy * 100:.2f}%"
)

# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=genres,
        digits=4
    )
)

# ============================================
# CONFUSION MATRIX
# ============================================

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)

# ============================================
# FINAL SUMMARY
# ============================================

print("\n============================================")
print("FINAL BASELINE SUMMARY")
print("============================================")

print(f"Training Accuracy:    {train_accuracy * 100:.2f}%")
print(f"Development Accuracy: {dev_accuracy * 100:.2f}%")
print(f"Test Accuracy:        {test_accuracy * 100:.2f}%")