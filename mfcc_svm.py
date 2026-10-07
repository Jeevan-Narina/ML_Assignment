import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATA
# ============================================================

X_train = np.load("Data/processed/mfcc_train.npy")
X_dev = np.load("Data/processed/mfcc_dev.npy")
X_test = np.load("Data/processed/mfcc_test.npy")

y_train = np.load("Data/processed/y_train.npy")
y_dev = np.load("Data/processed/y_dev.npy")
y_test = np.load("Data/processed/y_test.npy")

print("MFCC RBF-SVM")
print("=" * 60)

print("Train:", X_train.shape)
print("Dev:  ", X_dev.shape)
print("Test: ", X_test.shape)


# ============================================================
# 2. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

# IMPORTANT:
# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler for dev and test
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 3. HYPERPARAMETER SEARCH
# ============================================================

C_values = [0.1, 1, 10, 100]
gamma_values = ["scale", 0.001, 0.01, 0.1]

best_dev_accuracy = 0
best_C = None
best_gamma = None
best_model = None


print("\nSearching for best RBF-SVM...")
print("-" * 60)

for C in C_values:

    for gamma in gamma_values:

        model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        dev_predictions = model.predict(X_dev_scaled)

        dev_accuracy = accuracy_score(
            y_dev,
            dev_predictions
        )

        print(
            f"C={C:<5} "
            f"gamma={str(gamma):<8} "
            f"Dev Accuracy={dev_accuracy * 100:.2f}%"
        )

        if dev_accuracy > best_dev_accuracy:

            best_dev_accuracy = dev_accuracy
            best_C = C
            best_gamma = gamma
            best_model = model


# ============================================================
# 4. BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST MFCC RBF-SVM")
print("=" * 60)

print("Best C:", best_C)
print("Best gamma:", best_gamma)
print(f"Best Dev Accuracy: {best_dev_accuracy * 100:.2f}%")


# ============================================================
# 5. TRAINING ACCURACY
# ============================================================

train_predictions = best_model.predict(X_train_scaled)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")


# ============================================================
# 6. FINAL TEST EVALUATION
# ============================================================

test_predictions = best_model.predict(X_test_scaled)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

genres = [
    "classical",
    "jazz",
    "metal",
    "pop"
]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=genres,
        digits=4
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


print("\nDone.")