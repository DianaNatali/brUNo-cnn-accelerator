"""
Minimal CNN for CIFAR-10
Architecture:
    Input 32x32x3
    Conv 3x3, 8 filters, ReLU  -> 32x32x8
    MaxPool 2x2                 -> 16x16x8
    Conv 3x3, 16 filters, ReLU -> 16x16x16
    MaxPool 2x2                 -> 8x8x16
    Conv 3x3, 32 filters, ReLU -> 8x8x32
    MaxPool 2x2                 -> 4x4x32
    Flatten                     -> 512
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
# Training transform: random flip and crop for data augmentation,
# then normalize using CIFAR-10 channel mean and std.
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# Test transform: only normalize, no augmentation.
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
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
            # Block 1: learn low-level features (edges, colors)
            nn.Conv2d(3,  8,  kernel_size=3, padding=1),  # 32x32x8
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16x16x8

            # Block 2: learn mid-level features (textures, shapes)
            nn.Conv2d(8,  16, kernel_size=3, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 8x8x16

            # Block 3: learn high-level features (object parts)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 8x8x32
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 4x4x32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # 4x4x32 = 512 values
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10),     # 10 output logits, one per class
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits; CrossEntropyLoss applies softmax internally


model = SmallCNN().to(DEVICE)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# Cosine annealing: smoothly reduces LR from LR to ~0 over EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()          # clear gradients from previous step
        outputs = model(inputs)        # forward pass
        loss = criterion(outputs, targets)
        loss.backward()                # compute gradients
        optimizer.step()               # update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch:>2} | train loss: {avg_loss:.4f} | train acc: {acc:.1f}%")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():              # no gradients needed for inference
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
        torch.save(model.state_dict(), WEIGHTS_PATH)   # save best checkpoint
        print(f"           -> saved (best acc: {best_acc:.1f}%)")

print(f"\nTraining done. Best test acc: {best_acc:.1f}%")
print(f"Weights saved to: {WEIGHTS_PATH}")

# ── Export weights as plain float (input for the fixed-point golden model) ─────
import json
import numpy as np

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

weights_export = {}
for name, param in model.named_parameters():
    weights_export[name] = param.detach().numpy().tolist()

with open("cnn_weights_float.json", "w") as f:
    json.dump(weights_export, f)

print("Float weights exported to: cnn_weights_float.json")