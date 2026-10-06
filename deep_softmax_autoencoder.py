import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 500
NUM_CLASSES = 4

LATENT_DIM = 64

BATCH_SIZE = 512
LEARNING_RATE = 0.0001
EPOCHS = 200

CLASSIFICATION_WEIGHT = 0.1

DROPOUT_RATE = 0.2

SEED = 42

GENRES = ["classical", "jazz", "metal", "pop"]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load("Data/processed/X_train_v4.npy")
y_train = np.load("Data/processed/y_train_v4.npy")

X_dev = np.load("Data/processed/X_dev_v4.npy")
y_dev = np.load("Data/processed/y_dev_v4.npy")

X_test = np.load("Data/processed/X_test_v4.npy")
y_test = np.load("Data/processed/y_test_v4.npy")


print("\nDataset shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Dev:  ", X_dev.shape, y_dev.shape)
print("Test: ", X_test.shape, y_test.shape)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
y_dev_tensor = torch.tensor(y_dev, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

dev_dataset = TensorDataset(
    X_dev_tensor,
    y_dev_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


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


# ============================================================
# DSAE MODEL
# ============================================================

class DeepSoftmaxAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # 500 -> 256 -> 192 -> 128 -> 64
        # ----------------------------------------------------

        self.encoder_fc1 = nn.Linear(500, 256)
        self.encoder_fc2 = nn.Linear(256, 192)
        self.encoder_fc3 = nn.Linear(192, 128)
        self.encoder_fc4 = nn.Linear(128, 64)

        # Original dropout keep probability = 0.8
        # Therefore dropout rate = 0.2
        self.encoder_dropout = nn.Dropout(DROPOUT_RATE)


        # ----------------------------------------------------
        # DECODER
        # 64 -> 128 -> 192 -> 256 -> 500
        # ----------------------------------------------------

        self.decoder_fc1 = nn.Linear(64, 128)
        self.decoder_fc2 = nn.Linear(128, 192)
        self.decoder_fc3 = nn.Linear(192, 256)
        self.decoder_fc4 = nn.Linear(256, 500)

        self.decoder_dropout = nn.Dropout(DROPOUT_RATE)


        # ----------------------------------------------------
        # CLASSIFIER
        # 64 -> 32 -> 16 -> 4
        # ----------------------------------------------------

        self.classifier_fc1 = nn.Linear(64, 32)
        self.classifier_fc2 = nn.Linear(32, 16)
        self.classifier_fc3 = nn.Linear(16, NUM_CLASSES)


        # ----------------------------------------------------
        # WEIGHT INITIALIZATION
        # ----------------------------------------------------

        self.apply(self.initialize_weights)


    @staticmethod
    def initialize_weights(module):

        if isinstance(module, nn.Linear):

            # Xavier / Glorot initialization
            nn.init.xavier_uniform_(module.weight)

            # Zero bias
            nn.init.zeros_(module.bias)


    # ========================================================
    # ENCODER
    # ========================================================

    def encode(self, x):

        # Layer 1: Sigmoid
        x = torch.sigmoid(
            self.encoder_fc1(x)
        )

        # Layer 2: Tanh
        x = torch.tanh(
            self.encoder_fc2(x)
        )

        # Dropout after layer 2
        x = self.encoder_dropout(x)

        # Layer 3: ReLU
        x = torch.relu(
            self.encoder_fc3(x)
        )

        # Layer 4: ReLU
        x = torch.relu(
            self.encoder_fc4(x)
        )

        return x


    # ========================================================
    # DECODER
    # ========================================================

    def decode(self, z):

        # Layer 1: Sigmoid
        x = torch.sigmoid(
            self.decoder_fc1(z)
        )

        # Layer 2: Sigmoid
        x = torch.sigmoid(
            self.decoder_fc2(x)
        )

        # Dropout after layer 2
        x = self.decoder_dropout(x)

        # Layer 3: ReLU
        x = torch.relu(
            self.decoder_fc3(x)
        )

        # Layer 4: ReLU
        x = torch.relu(
            self.decoder_fc4(x)
        )

        return x


    # ========================================================
    # CLASSIFIER
    # ========================================================

    def classify(self, z):

        # Layer 1: Tanh
        x = torch.tanh(
            self.classifier_fc1(z)
        )

        # Layer 2: Tanh
        x = torch.tanh(
            self.classifier_fc2(x)
        )

        # Final linear layer
        logits = self.classifier_fc3(x)

        return logits


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decode(z)

        logits = self.classify(z)

        return reconstruction, logits, z


# ============================================================
# CREATE MODEL
# ============================================================

model = DeepSoftmaxAutoencoder().to(device)

print("\nModel:")
print(model)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

reconstruction_loss_fn = nn.MSELoss()

classification_loss_fn = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    total_reconstruction_loss = 0.0
    total_classification_loss = 0.0

    predictions = []
    targets = []

    total_samples = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            reconstruction, logits, _ = model(inputs)

            reconstruction_loss = reconstruction_loss_fn(
                reconstruction,
                inputs
            )

            classification_loss = classification_loss_fn(
                logits,
                labels
            )

            batch_size = inputs.size(0)

            total_reconstruction_loss += (
                reconstruction_loss.item() * batch_size
            )

            total_classification_loss += (
                classification_loss.item() * batch_size
            )

            predicted = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

            total_samples += batch_size


    reconstruction_loss = (
        total_reconstruction_loss / total_samples
    )

    classification_loss = (
        total_classification_loss / total_samples
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    return (
        reconstruction_loss,
        classification_loss,
        accuracy
    )


# ============================================================
# TRAINING
# ============================================================

best_dev_accuracy = 0.0
best_epoch = 0

best_model_state = None


print("\n========================================")
print("STARTING DSAE V4 TRAINING")
print("========================================")


for epoch in range(1, EPOCHS + 1):

    model.train()

    total_reconstruction_loss = 0.0
    total_classification_loss = 0.0

    total_samples = 0


    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        reconstruction, logits, _ = model(inputs)


        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        reconstruction_loss = reconstruction_loss_fn(
            reconstruction,
            inputs
        )

        classification_loss = classification_loss_fn(
            logits,
            labels
        )


        # ----------------------------------------------------
        # Original DSAE loss
        #
        # Loss = reconstruction
        #      + 0.1 * classification
        # ----------------------------------------------------

        loss = (
            reconstruction_loss
            + CLASSIFICATION_WEIGHT * classification_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Track losses
        # ----------------------------------------------------

        batch_size = inputs.size(0)

        total_reconstruction_loss += (
            reconstruction_loss.item() * batch_size
        )

        total_classification_loss += (
            classification_loss.item() * batch_size
        )

        total_samples += batch_size


    # ========================================================
    # TRAINING AVERAGES
    # ========================================================

    train_reconstruction_loss = (
        total_reconstruction_loss / total_samples
    )

    train_classification_loss = (
        total_classification_loss / total_samples
    )


    # ========================================================
    # DEV EVALUATION
    # ========================================================

    (
        dev_reconstruction_loss,
        dev_classification_loss,
        dev_accuracy
    ) = evaluate(dev_loader)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if dev_accuracy > best_dev_accuracy:

        best_dev_accuracy = dev_accuracy

        best_epoch = epoch

        best_model_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Recon: {train_reconstruction_loss:.6f} | "
            f"Train CE: {train_classification_loss:.6f} | "
            f"Dev Recon: {dev_reconstruction_loss:.6f} | "
            f"Dev CE: {dev_classification_loss:.6f} | "
            f"Dev Acc: {dev_accuracy * 100:.2f}%"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    f"Best Dev Accuracy: "
    f"{best_dev_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)


model.load_state_dict(best_model_state)
model.to(device)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

model.eval()

test_predictions = []
test_targets = []

test_reconstruction_loss = 0.0
test_classification_loss = 0.0

total_samples = 0


with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        reconstruction, logits, _ = model(inputs)


        reconstruction_loss = reconstruction_loss_fn(
            reconstruction,
            inputs
        )

        classification_loss = classification_loss_fn(
            logits,
            labels
        )


        batch_size = inputs.size(0)

        test_reconstruction_loss += (
            reconstruction_loss.item() * batch_size
        )

        test_classification_loss += (
            classification_loss.item() * batch_size
        )

        total_samples += batch_size


        predicted = torch.argmax(
            logits,
            dim=1
        )

        test_predictions.extend(
            predicted.cpu().numpy()
        )

        test_targets.extend(
            labels.cpu().numpy()
        )


test_reconstruction_loss /= total_samples
test_classification_loss /= total_samples


test_accuracy = accuracy_score(
    test_targets,
    test_predictions
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========================================")
print("DSAE V4 FINAL TEST RESULTS")
print("========================================")

print(
    f"Test Reconstruction Loss: "
    f"{test_reconstruction_loss:.6f}"
)

print(
    f"Test Classification Loss: "
    f"{test_classification_loss:.6f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        test_targets,
        test_predictions,
        target_names=GENRES,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:\n")

print(
    confusion_matrix(
        test_targets,
        test_predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "dsae_v4_best.pth"
)

print("\nBest model saved as:")
print("dsae_v4_best.pth")