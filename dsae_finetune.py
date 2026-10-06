import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 512
LEARNING_RATE = 0.00001
EPOCHS = 50

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD V4 DATA
# ============================================================

X_train = np.load("Data/processed/X_train.npy")
y_train = np.load("Data/processed/y_train.npy")

X_dev = np.load("Data/processed/X_dev.npy")
y_dev = np.load("Data/processed/y_dev.npy")

X_test = np.load("Data/processed/X_test.npy")
y_test = np.load("Data/processed/y_test.npy")

# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

dev_dataset = TensorDataset(
    torch.tensor(X_dev, dtype=torch.float32),
    torch.tensor(y_dev, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
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
# SAME V4 ARCHITECTURE
# ============================================================

class DeepSoftmaxAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder_fc1 = nn.Linear(500, 256)
        self.encoder_fc2 = nn.Linear(256, 192)
        self.encoder_fc3 = nn.Linear(192, 128)
        self.encoder_fc4 = nn.Linear(128, 64)

        self.encoder_dropout = nn.Dropout(0.2)

        # Decoder
        self.decoder_fc1 = nn.Linear(64, 128)
        self.decoder_fc2 = nn.Linear(128, 192)
        self.decoder_fc3 = nn.Linear(192, 256)
        self.decoder_fc4 = nn.Linear(256, 500)

        self.decoder_dropout = nn.Dropout(0.2)

        # Classifier
        self.classifier_fc1 = nn.Linear(64, 32)
        self.classifier_fc2 = nn.Linear(32, 16)
        self.classifier_fc3 = nn.Linear(16, 4)


    def encode(self, x):

        x = torch.sigmoid(self.encoder_fc1(x))

        x = torch.tanh(self.encoder_fc2(x))

        x = self.encoder_dropout(x)

        x = torch.relu(self.encoder_fc3(x))

        x = torch.relu(self.encoder_fc4(x))

        return x


    def decode(self, z):

        x = torch.sigmoid(self.decoder_fc1(z))

        x = torch.sigmoid(self.decoder_fc2(x))

        x = self.decoder_dropout(x)

        x = torch.relu(self.decoder_fc3(x))

        x = torch.relu(self.decoder_fc4(x))

        return x


    def classify(self, z):

        x = torch.tanh(self.classifier_fc1(z))

        x = torch.tanh(self.classifier_fc2(x))

        return self.classifier_fc3(x)


    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decode(z)

        logits = self.classify(z)

        return reconstruction, logits, z


# ============================================================
# LOAD BEST V4 MODEL
# ============================================================

model = DeepSoftmaxAutoencoder().to(device)

model.load_state_dict(
    torch.load(
        "dsae_best.pth",
        map_location=device
    )
)

print("\nLoaded dsae_best.pth")


# ============================================================
# CLASSIFICATION-ONLY FINE-TUNING
# ============================================================

classification_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            _, logits, _ = model(inputs)

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

    return accuracy_score(
        targets,
        predictions
    )


# ============================================================
# FINE-TUNING
# ============================================================

best_dev_accuracy = 0.0
best_epoch = 0
best_model_state = None


print("\n========================================")
print("STARTING CLASSIFICATION FINE-TUNING")
print("========================================")


for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_samples = 0


    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Only classification output is used
        _, logits, _ = model(inputs)

        loss = classification_loss_fn(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        batch_size = inputs.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size


    train_loss = (
        total_loss / total_samples
    )


    # Dev accuracy
    dev_accuracy = evaluate(dev_loader)


    # Save best model
    if dev_accuracy > best_dev_accuracy:

        best_dev_accuracy = dev_accuracy

        best_epoch = epoch

        best_model_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }


    if epoch == 1 or epoch % 5 == 0:

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train CE: {train_loss:.6f} | "
            f"Dev Acc: {dev_accuracy * 100:.2f}%"
        )


# ============================================================
# LOAD BEST FINE-TUNED MODEL
# ============================================================

model.load_state_dict(
    best_model_state
)

model.to(device)


print("\n========================================")
print("BEST FINE-TUNED MODEL")
print("========================================")

print(
    f"Best Dev Accuracy: "
    f"{best_dev_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)


# ============================================================
# FINAL TEST
# ============================================================

model.eval()

predictions = []
targets = []


with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        _, logits, _ = model(inputs)

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


test_accuracy = accuracy_score(
    targets,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FINE-TUNED TEST RESULTS")
print("========================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")

print(
    classification_report(
        targets,
        predictions,
        target_names=GENRES,
        digits=4
    )
)


print("Confusion Matrix:\n")

print(
    confusion_matrix(
        targets,
        predictions
    )
)


# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    "dsae_finetuned.pth"
)

print(
    "\nSaved as: dsae_finetuned.pth"
)