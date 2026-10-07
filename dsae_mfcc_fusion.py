import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DSAE + MFCC FUSION")
print("=" * 60)
print("Using device:", device)


# ============================================================
# 2. LOAD ORIGINAL DSAE DATA
# ============================================================

X_train = np.load("Data/processed/X_train.npy")
X_dev = np.load("Data/processed/X_dev.npy")
X_test = np.load("Data/processed/X_test.npy")

y_train = np.load("Data/processed/y_train.npy")
y_dev = np.load("Data/processed/y_dev.npy")
y_test = np.load("Data/processed/y_test.npy")


# ============================================================
# 3. LOAD MFCC FEATURES
# ============================================================

mfcc_train = np.load(
    "Data/processed/mfcc_train.npy"
)

mfcc_dev = np.load(
    "Data/processed/mfcc_dev.npy"
)

mfcc_test = np.load(
    "Data/processed/mfcc_test.npy"
)

print("\nMFCC shapes:")
print("Train:", mfcc_train.shape)
print("Dev:  ", mfcc_dev.shape)
print("Test: ", mfcc_test.shape)


# ============================================================
# 4. DEFINE DSAE ENCODER
# ============================================================

class DSAEEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_fc1 = nn.Linear(500, 256)
        self.encoder_fc2 = nn.Linear(256, 192)
        self.encoder_fc3 = nn.Linear(192, 128)
        self.encoder_fc4 = nn.Linear(128, 64)

    def forward(self, x):

        x = torch.sigmoid(
            self.encoder_fc1(x)
        )

        x = torch.tanh(
            self.encoder_fc2(x)
        )

        x = torch.relu(
            self.encoder_fc3(x)
        )

        x = torch.relu(
            self.encoder_fc4(x)
        )

        return x


# ============================================================
# 5. LOAD TRAINED DSAE ENCODER
# ============================================================

encoder = DSAEEncoder().to(device)

checkpoint = torch.load(
    "dsae_best.pth",
    map_location=device
)

encoder.load_state_dict(
    {
        "encoder_fc1.weight":
            checkpoint["encoder_fc1.weight"],

        "encoder_fc1.bias":
            checkpoint["encoder_fc1.bias"],

        "encoder_fc2.weight":
            checkpoint["encoder_fc2.weight"],

        "encoder_fc2.bias":
            checkpoint["encoder_fc2.bias"],

        "encoder_fc3.weight":
            checkpoint["encoder_fc3.weight"],

        "encoder_fc3.bias":
            checkpoint["encoder_fc3.bias"],

        "encoder_fc4.weight":
            checkpoint["encoder_fc4.weight"],

        "encoder_fc4.bias":
            checkpoint["encoder_fc4.bias"]
    }
)

encoder.eval()

print("\nDSAE encoder loaded successfully.")


# ============================================================
# 6. EXTRACT DSAE LATENT FEATURES
# ============================================================

def extract_latent_features(X):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        latent = encoder(X_tensor)

    return latent.cpu().numpy()


latent_train = extract_latent_features(
    X_train
)

latent_dev = extract_latent_features(
    X_dev
)

latent_test = extract_latent_features(
    X_test
)


print("\nDSAE latent shapes:")
print("Train:", latent_train.shape)
print("Dev:  ", latent_dev.shape)
print("Test: ", latent_test.shape)


# ============================================================
# 7. FUSE DSAE + MFCC
# ============================================================

X_train_fused = np.concatenate(
    [
        latent_train,
        mfcc_train
    ],
    axis=1
)

X_dev_fused = np.concatenate(
    [
        latent_dev,
        mfcc_dev
    ],
    axis=1
)

X_test_fused = np.concatenate(
    [
        latent_test,
        mfcc_test
    ],
    axis=1
)


print("\nFused feature shapes:")
print("Train:", X_train_fused.shape)
print("Dev:  ", X_dev_fused.shape)
print("Test: ", X_test_fused.shape)


# ============================================================
# 8. STANDARDIZE
# ============================================================

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(
    X_train_fused
)

# Apply the same scaler
X_dev_scaled = scaler.transform(
    X_dev_fused
)

X_test_scaled = scaler.transform(
    X_test_fused
)


# ============================================================
# 9. RBF-SVM HYPERPARAMETER SEARCH
# ============================================================

C_values = [
    0.1,
    1,
    10,
    100
]

gamma_values = [
    "scale",
    0.001,
    0.01,
    0.1
]


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

        model.fit(
            X_train_scaled,
            y_train
        )

        dev_predictions = model.predict(
            X_dev_scaled
        )

        dev_accuracy = accuracy_score(
            y_dev,
            dev_predictions
        )

        print(
            f"C={C:<5} "
            f"gamma={str(gamma):<8} "
            f"Dev Accuracy="
            f"{dev_accuracy * 100:.2f}%"
        )

        if dev_accuracy > best_dev_accuracy:

            best_dev_accuracy = dev_accuracy
            best_C = C
            best_gamma = gamma
            best_model = model


# ============================================================
# 10. BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST DSAE + MFCC FUSION MODEL")
print("=" * 60)

print("Best C:", best_C)
print("Best gamma:", best_gamma)

print(
    f"Best Dev Accuracy: "
    f"{best_dev_accuracy * 100:.2f}%"
)


# ============================================================
# 11. TRAIN ACCURACY
# ============================================================

train_predictions = best_model.predict(
    X_train_scaled
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(
    f"Train Accuracy: "
    f"{train_accuracy * 100:.2f}%"
)


# ============================================================
# 12. TEST ACCURACY
# ============================================================

test_predictions = best_model.predict(
    X_test_scaled
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 13. CLASSIFICATION REPORT
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
# 14. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


print("\nDone.")