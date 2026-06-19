"""
Minimal CNN for CIFAR-10 — grayscale input (1 channel)
Architecture:
    Input 32x32x1  (grayscale)
    Conv 3x3,  8 filters, ReLU  -> 32x32x8
    MaxPool 2x2                  -> 16x16x8
    Conv 3x3, 16 filters, ReLU  -> 16x16x16
    MaxPool 2x2                  ->  8x8x16
    Conv 3x3, 32 filters, ReLU  ->  8x8x32
    MaxPool 2x2                  ->  4x4x32
    Flatten                      ->  512
    FC 64, ReLU
    FC 10, Softmax

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "cnn_cifar10.pth"

print(f"Using device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────────────────────────
# Convert to grayscale (1 channel) before normalizing.
# Grayscale mean and std computed over CIFAR-10.
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4808,), (0.2393,)),
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4808,), (0.2393,)),
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                              download=True, transform=transform_train)
test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                              download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ── Architecture ───────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 input channel (grayscale)
            nn.Conv2d(1,  8,  kernel_size=3, padding=1),  # 32x32x8
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16x16x8

            # Block 2
            nn.Conv2d(8,  16, kernel_size=3, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 8x8x16

            # Block 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 8x8x32
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 4x4x32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN().to(DEVICE)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print(f"Epoch {epoch:>2} | train loss: {running_loss/len(train_loader):.4f} | train acc: {acc:.1f}%")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print(f"           test acc: {acc:.1f}%")
    return acc

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    train_epoch(epoch)
    acc = evaluate()
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"           -> saved (best acc: {best_acc:.1f}%)")

print(f"\nTraining done. Best test acc: {best_acc:.1f}%")
print(f"Weights saved to: {WEIGHTS_PATH}")

# ── Export float weights ───────────────────────────────────────────────────────
import json, numpy as np

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

weights_export = {}
for name, param in model.named_parameters():
    weights_export[name] = param.detach().numpy().tolist()

with open("cnn_weights_float.json", "w") as f:
    json.dump(weights_export, f)

print("Float weights exported to: cnn_weights_float.json")