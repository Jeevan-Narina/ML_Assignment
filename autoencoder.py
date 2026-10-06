import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# CONFIGURATION
# ============================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30

INPUT_SIZE = 500
LATENT_SIZE = 64

# ============================================
# LOAD DATA
# ============================================

X_train = np.load("Data/processed/X_train.npy")
X_dev = np.load("Data/processed/X_dev.npy")
X_test = np.load("Data/processed/X_test.npy")

print("Training data:", X_train.shape)
print("Development data:", X_dev.shape)
print("Test data:", X_test.shape)

# ============================================
# CONVERT TO PYTORCH TENSORS
# ============================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ============================================
# DATA LOADERS
# ============================================

train_dataset = TensorDataset(X_train_tensor)
dev_dataset = TensorDataset(X_dev_tensor)
test_dataset = TensorDataset(X_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================
# VANILLA AUTOENCODER
# ============================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder: 500 -> 256 -> 192 -> 128 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(500, 256),
            nn.Tanh(),

            nn.Linear(256, 192),
            nn.Tanh(),

            nn.Linear(192, 128),
            nn.Tanh(),

            nn.Linear(128, 64)
        )

        # Decoder: 64 -> 128 -> 192 -> 256 -> 500
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),

            nn.Linear(128, 192),
            nn.Tanh(),

            nn.Linear(192, 256),
            nn.Tanh(),

            nn.Linear(256, 500)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed


# ============================================
# MODEL
# ============================================

model = Autoencoder()

print("\n============================================")
print("VANILLA AUTOENCODER")
print("============================================")

print(model)

# ============================================
# LOSS AND OPTIMIZER
# ============================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================
# TRAINING
# ============================================

print("\n============================================")
print("TRAINING")
print("============================================")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for (batch,) in train_loader:

        optimizer.zero_grad()

        reconstructed = model(batch)

        loss = criterion(reconstructed, batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * batch.size(0)

    train_loss /= len(train_loader.dataset)

    # ----------------------------------------
    # DEVELOPMENT LOSS
    # ----------------------------------------

    model.eval()

    dev_loss = 0.0

    with torch.no_grad():

        for (batch,) in dev_loader:

            reconstructed = model(batch)

            loss = criterion(reconstructed, batch)

            dev_loss += loss.item() * batch.size(0)

    dev_loss /= len(dev_loader.dataset)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.6f} "
        f"| Dev Loss: {dev_loss:.6f}"
    )

# ============================================
# TEST RECONSTRUCTION LOSS
# ============================================

model.eval()

test_loss = 0.0

with torch.no_grad():

    for (batch,) in test_loader:

        reconstructed = model(batch)

        loss = criterion(reconstructed, batch)

        test_loss += loss.item() * batch.size(0)

test_loss /= len(test_loader.dataset)

print("\n============================================")
print("FINAL RESULTS")
print("============================================")

print(f"Test reconstruction loss: {test_loss:.6f}")

# ============================================
# EXTRACT 64-D LATENT REPRESENTATIONS
# ============================================

with torch.no_grad():

    X_train_encoded = model.encoder(X_train_tensor).numpy()
    X_dev_encoded = model.encoder(X_dev_tensor).numpy()
    X_test_encoded = model.encoder(X_test_tensor).numpy()

# ============================================
# SAVE LATENT REPRESENTATIONS
# ============================================

np.save(
    "Data/processed/X_train_encoded.npy",
    X_train_encoded
)

np.save(
    "Data/processed/X_dev_encoded.npy",
    X_dev_encoded
)

np.save(
    "Data/processed/X_test_encoded.npy",
    X_test_encoded
)

print("\nLatent representations saved.")

print("Train latent:", X_train_encoded.shape)
print("Dev latent  :", X_dev_encoded.shape)
print("Test latent :", X_test_encoded.shape)