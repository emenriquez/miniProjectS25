from models import SimpleCNN, ImprovedCNN
from train import cross_validate, load_model
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set this flag to True to load models instead of training
LOAD_MODELS = False  # Set to True to skip training and only load models for evaluation/analysis

# ======== RUN EXPERIMENTS =========

results = {}
per_fold_results = {}
per_fold_conf_matrices = {}

# Helper to run and store per-fold results
def run_cv(model_class, name, **kwargs):
    avg_acc, fold_accs, conf_matrices = cross_validate(model_class, name, device, return_fold_accs=True, return_conf_matrices=True, **kwargs)
    results[name] = avg_acc
    per_fold_results[name] = fold_accs
    per_fold_conf_matrices[name] = conf_matrices

if not LOAD_MODELS:
    # Run cross-validation and save models as before
    run_cv(SimpleCNN, "SimpleCNN (5 epochs)", epochs=5)
    run_cv(SimpleCNN, "SimpleCNN (10 epochs)", epochs=10)
    run_cv(SimpleCNN, "SimpleCNN (15 epochs)", epochs=15)
    run_cv(ImprovedCNN, "ImprovedCNN (15 epochs)", epochs=15)
    run_cv(ImprovedCNN, "ImprovedCNN (15 epochs, Dropout=0.3)", epochs=15, dropout=0.3)
    run_cv(ImprovedCNN, "ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug", epochs=15, dropout=0.3, use_aug=True)
    run_cv(ImprovedCNN, "ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler", epochs=15, dropout=0.3, use_scheduler=True)
else:
    # Load models for each experiment and fold, and evaluate/analyze as needed
    for exp_name, model_class in [
        ("SimpleCNN (5 epochs)", SimpleCNN),
        ("SimpleCNN (10 epochs)", SimpleCNN),
        ("SimpleCNN (15 epochs)", SimpleCNN),
        ("ImprovedCNN (15 epochs)", ImprovedCNN),
        ("ImprovedCNN (15 epochs, Dropout=0.3)", ImprovedCNN),
        ("ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug", ImprovedCNN),
        ("ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler", ImprovedCNN),
    ]:
        per_fold_results[exp_name] = []
        per_fold_conf_matrices[exp_name] = []
        for fold in range(5):
            # Set dropout value for ImprovedCNN variants
            dropout = 0.3 if 'Dropout=0.3' in exp_name else 0.5
            model = load_model(model_class, exp_name, fold, device, dropout=dropout)
            # You can add evaluation code here, e.g., run on validation/test set and collect results
            # For now, just append None as a placeholder
            per_fold_results[exp_name].append(None)
            per_fold_conf_matrices[exp_name].append(None)

# ======== FINAL RESULTS =========

print("\n========== FINAL COMPARISON ==========")
for name, acc in results.items():
    print(f"{name:35s}: {acc:.2f}%")

# ======== PLOTTING =========

os.makedirs('plots', exist_ok=True)
labels = list(results.keys())
means = [results[k] for k in labels]
stds = [np.std(per_fold_results[k]) for k in labels]

for exp_name, conf_matrices in per_fold_conf_matrices.items():
    # Create subdirectory for this experiment's plots
    exp_plot_dir = os.path.join('plots', exp_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace(',', '').replace('=', ''))
    os.makedirs(exp_plot_dir, exist_ok=True)
    # Save summary bar plot in this directory as well
    if exp_name == labels[0]:  # Only once for all experiments
        plt.figure(figsize=(10, 6))
        plt.barh(labels, means, xerr=stds, color='skyblue', capsize=8)
        plt.xlabel('Average Cross-Validation Accuracy (%)')
        plt.title('MNIST Model Experiment Results (5-fold CV)')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_plot_dir, 'experiment_results.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(exp_plot_dir, 'experiment_results.png')}")
    for fold_idx, cm in enumerate(conf_matrices):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f'{exp_name} - Fold {fold_idx+1} Confusion Matrix')
        plt.tight_layout()
        fname = os.path.join(exp_plot_dir, f"fold{fold_idx+1}_cm.png")
        plt.savefig(fname)
        plt.close()
        print(f"Confusion matrix saved to {fname}")
