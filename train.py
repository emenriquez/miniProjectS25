import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loader, get_test_loader, get_full_train_set, get_full_emnist_train_set
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import os
from utils import get_num_classes

def instantiate_model(model_class, dropout=0.5, device=None):
    """
    Instantiate a model, passing dropout if supported.
    """
    model_init_args = {}
    init_params = model_class.__init__.__code__.co_varnames
    if 'dropout' in init_params:
        model_init_args['dropout'] = dropout
    model = model_class(**model_init_args)
    if device is not None:
        model = model.to(device)
    return model

def train_and_eval(model_class, name, device, epochs=5, dropout=0.5, lr=0.001, use_aug=False, use_scheduler=False):
    """
    Train a model and evaluate on the test set.
    """
    train_loader = get_loader(augmented=use_aug)
    test_loader = get_test_loader()
    model = instantiate_model(model_class, dropout=dropout, device=device)
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
    """
    Save a model's state_dict to disk.
    """
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    fname = f"{exp_name}_{fold_idx+1}.pt"
    path = os.path.join(exp_dir, fname)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path

def load_model(model_class, exp_name, fold_idx, device, dropout=0.5, save_dir='saved_models', load_weights=True):
    """
    Load a model's state_dict from disk.
    """
    exp_dir = os.path.join(save_dir, exp_name)
    fname = f"{exp_name}_{fold_idx+1}.pt"
    path = os.path.join(exp_dir, fname)
    model = instantiate_model(model_class, dropout=dropout, device=device)
    if load_weights and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def cross_validate(model_class, exp_name, device, k=5, epochs=5, dropout=0.5, use_aug=False, DATASET='mnist', EMNIST_SPLIT='balanced'):
    """
    Perform k-fold cross-validation and save each fold's model.
    """
    print(f"\nCross-validating {exp_name} with {k}-fold CV")
    if DATASET == 'emnist':
        dataset = get_full_emnist_train_set(split=EMNIST_SPLIT, augmented=use_aug)
    else:
        dataset = get_full_train_set(augmented=use_aug)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1000, shuffle=False)
        model = instantiate_model(model_class, dropout=dropout, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        save_model(model, exp_name, fold, save_dir="saved_models")
