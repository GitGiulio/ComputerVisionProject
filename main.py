import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
DATA_DIR = "CIFAKE/"   # <- change this to your dataset path
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters (matching paper search space)
CONV_FILTERS = 32          # Paper uses {16,32,64,128}
CONV_LAYERS = 2            # Paper uses {1,2,3}
DENSE_NEURONS = 64         # Paper uses {32..4096}
DENSE_LAYERS = 1           # Paper uses {1,2,3}


# ---------------------------------------------------------
# 2. DATA PIPELINE
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Match CIFAR-10 resolution
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------
# 3. MODEL DEFINITION
# ---------------------------------------------------------
class CIFAKE_CNN(nn.Module):
    def __init__(self, conv_filters, conv_layers, dense_neurons, dense_layers):
        super(CIFAKE_CNN, self).__init__()

        conv_blocks = []
        in_channels = 3

        for _ in range(conv_layers):
            conv_blocks.append(nn.Conv2d(in_channels, conv_filters, kernel_size=3, stride=1, padding=1))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = conv_filters

        self.conv = nn.Sequential(*conv_blocks)

        # After downsampling: input 32x32 → depends on number of conv layers
        reduction = 2 ** conv_layers
        flattened_size = (32 // reduction) * (32 // reduction) * conv_filters

        dense_blocks = []
        in_features = flattened_size

        for _ in range(dense_layers):
            dense_blocks.append(nn.Linear(in_features, dense_neurons))
            dense_blocks.append(nn.ReLU())
            in_features = dense_neurons

        dense_blocks.append(nn.Linear(in_features, 1))
        dense_blocks.append(nn.Sigmoid())

        self.fc = nn.Sequential(*dense_blocks)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = CIFAKE_CNN(CONV_FILTERS, CONV_LAYERS, DENSE_NEURONS, DENSE_LAYERS).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ---------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")


# ---------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------
def evaluate():
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            preds.extend(predictions.flatten())
            trues.extend(labels.numpy().flatten())

    accuracy = np.mean(np.array(preds) == np.array(trues))
    precision = precision_score(trues, preds)
    recall = recall_score(trues, preds)
    f1 = f1_score(trues, preds)

    print("\n--- TEST RESULTS ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ---------------------------------------------------------
# 6. MAIN LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    print(model)
    for epoch in range(EPOCHS):
        train_one_epoch(epoch)
    evaluate()