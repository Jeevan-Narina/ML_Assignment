import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD V4 DATA
# ============================================================

X = np.load("Data/processed/X_v4.npy")
y = np.load("Data/processed/y_v4.npy")

print("Original data:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# FIRST SPLIT
# 75% TRAIN
# 25% TEMPORARY
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# SECOND SPLIT
# TEMPORARY -> 50% DEV + 50% TEST
#
# Final:
# 75% TRAIN
# 12.5% DEV
# 12.5% TEST
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# SAVE SPLITS
# ============================================================

np.save("Data/processed/X_train_v4.npy", X_train)
np.save("Data/processed/y_train_v4.npy", y_train)

np.save("Data/processed/X_dev_v4.npy", X_dev)
np.save("Data/processed/y_dev_v4.npy", y_dev)

np.save("Data/processed/X_test_v4.npy", X_test)
np.save("Data/processed/y_test_v4.npy", y_test)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("V4 DATA SPLIT COMPLETE")
print("========================================")

print("\nTrain:")
print("X:", X_train.shape)
print("y:", y_train.shape)

print("\nDev:")
print("X:", X_dev.shape)
print("y:", y_dev.shape)

print("\nTest:")
print("X:", X_test.shape)
print("y:", y_test.shape)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

genres = ["classical", "jazz", "metal", "pop"]

print("\nTrain class distribution:")
for i, genre in enumerate(genres):
    print(f"{genre}: {np.sum(y_train == i)}")

print("\nDev class distribution:")
for i, genre in enumerate(genres):
    print(f"{genre}: {np.sum(y_dev == i)}")

print("\nTest class distribution:")
for i, genre in enumerate(genres):
    print(f"{genre}: {np.sum(y_test == i)}")