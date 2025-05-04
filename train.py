import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loader, get_test_loader, get_full_train_set
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def train_and_eval(model_class, name, device, epochs=5, dropout=0.5, lr=0.001, use_aug=False, use_scheduler=False):
    print(f"\nTraining {name}")
    train_loader = get_loader(augmented=use_aug)
    test_loader = get_test_loader()
    model = model_class(dropout).to(device) if hasattr(model_class, '__init__') and 'dropout' in model_class.__init__.__code__.co_varnames else model_class().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) if use_scheduler else None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        print(f"{name} Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"{name} Test Accuracy: {acc:.2f}%")
    return acc

def save_model(model, exp_name, fold_idx, save_dir='saved_models'):
    # Organize by experiment subdirectory
    exp_dir = os.path.join(save_dir, exp_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace(',', '').replace('=', ''))
    os.makedirs(exp_dir, exist_ok=True)
    fname = f"fold{fold_idx+1}.pt"
    path = os.path.join(exp_dir, fname)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path

def load_model(model_class, exp_name, fold_idx, device, dropout=0.5, save_dir='saved_models'):
