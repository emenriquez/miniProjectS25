import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loader, get_test_loader, get_full_train_set, get_emnist_loader, get_emnist_test_loader, get_full_emnist_train_set
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from utils import get_num_classes

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
    # Save all model folds in a single experiment directory
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    fname = f"{exp_name}_{fold_idx+1}.pt"
    path = os.path.join(exp_dir, fname)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path

def load_model(model_class, exp_name, fold_idx, device, dropout=0.5, save_dir='saved_models'):
    exp_dir = os.path.join(save_dir, exp_name)
    fname = f"{exp_name}_{fold_idx+1}.pt"
    path = os.path.join(exp_dir, fname)
    model = model_class(dropout).to(device) if hasattr(model_class, '__init__') and 'dropout' in model_class.__init__.__code__.co_varnames else model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def cross_validate(model_class, name, device, epochs=5, dropout=0.5, lr=0.001, use_aug=False, use_scheduler=False, k=5, batch_size=64, return_fold_accs=False, return_conf_matrices=False, return_probs=False, DATASET='mnist', EMNIST_SPLIT='balanced', **kwargs):
    print(f"\nCross-validating {name} with {k}-fold CV")
    if DATASET == 'emnist':
        dataset = get_full_emnist_train_set(split=EMNIST_SPLIT, augmented=use_aug)
    else:
        dataset = get_full_train_set(augmented=use_aug)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs = []
    conf_matrices = []
    all_labels_all_folds = []
    all_preds_all_folds = []
    all_probs_all_folds = []
    num_classes = get_num_classes(DATASET, EMNIST_SPLIT)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        if DATASET == 'emnist':
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=1000, shuffle=False)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=1000, shuffle=False)
        model_args = {k: v for k, v in kwargs.items() if k in model_class.__init__.__code__.co_varnames}
        model = model_class(**model_args).to(device)
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
        save_model(model, name, fold)
        model.eval()
        correct, total = 0, 0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(max_probs.cpu().numpy())
        acc = 100 * correct / total
        print(f"Fold {fold+1} Accuracy: {acc:.2f}%")
        accs.append(acc)
        if return_conf_matrices:
            cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
            conf_matrices.append(cm)
        if return_probs:
            all_labels_all_folds.extend(all_labels)
            all_preds_all_folds.extend(all_preds)
            all_probs_all_folds.extend(all_probs)
    avg_acc = sum(accs) / len(accs)
    print(f"{name} Average CV Accuracy: {avg_acc:.2f}%")
    if return_fold_accs and return_conf_matrices and return_probs:
        return avg_acc, accs, conf_matrices, (np.array(all_labels_all_folds), np.array(all_preds_all_folds), np.array(all_probs_all_folds))
    if return_fold_accs and return_conf_matrices:
        return avg_acc, accs, conf_matrices
    if return_fold_accs:
        return avg_acc, accs
    if return_conf_matrices:
        return avg_acc, conf_matrices
    if return_probs:
        return avg_acc, (np.array(all_labels_all_folds), np.array(all_preds_all_folds), np.array(all_probs_all_folds))
    return avg_acc
