from models import SimpleCNN, ImprovedCNN, MLPBaseline
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

# Set this flag to True for quick debug runs
DEBUG = True  # Set to True for quick debug runs

# Experiment name for grouping results
EXPERIMENT_NAME = "DEBUG_RUN"  # Set this to your experiment name for grouping results

# ======== RUN EXPERIMENTS =========

results = {}
per_fold_results = {}
per_fold_conf_matrices = {}

# Helper to run and store per-fold results
def run_cv(model_class, name, **kwargs):
    exp_name = f"{EXPERIMENT_NAME}_{name}"
    avg_acc, fold_accs, conf_matrices = cross_validate(model_class, exp_name, device, return_fold_accs=True, return_conf_matrices=True, **kwargs)
    results[exp_name] = avg_acc
    per_fold_results[exp_name] = fold_accs
    per_fold_conf_matrices[exp_name] = conf_matrices

if not LOAD_MODELS:
    if DEBUG:
        run_cv(SimpleCNN, "SimpleCNN (5 epochs)", epochs=5)
        run_cv(MLPBaseline, "MLPBaseline (5 epochs)", epochs=5)
    else:
        run_cv(SimpleCNN, "SimpleCNN (5 epochs)", epochs=5)
        run_cv(SimpleCNN, "SimpleCNN (10 epochs)", epochs=10)
        run_cv(SimpleCNN, "SimpleCNN (15 epochs)", epochs=15)
        run_cv(ImprovedCNN, "ImprovedCNN (15 epochs)", epochs=15)
        run_cv(ImprovedCNN, "ImprovedCNN (15 epochs, Dropout=0.3)", epochs=15, dropout=0.3)
        run_cv(ImprovedCNN, "ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug", epochs=15, dropout=0.3, use_aug=True)
        run_cv(ImprovedCNN, "ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler", epochs=15, dropout=0.3, use_scheduler=True)
        run_cv(MLPBaseline, "MLPBaseline (5 epochs)", epochs=5)
        run_cv(MLPBaseline, "MLPBaseline (10 epochs)", epochs=10)
        run_cv(MLPBaseline, "MLPBaseline (15 epochs)", epochs=15)
else:
    if DEBUG:
        for name, model_class in [
            ("SimpleCNN (5 epochs)", SimpleCNN),
            ("MLPBaseline (5 epochs)", MLPBaseline),
        ]:
            exp_name = f"{EXPERIMENT_NAME}_{name}"
            per_fold_results[exp_name] = []
            per_fold_conf_matrices[exp_name] = []
            for fold in range(5):
                dropout = 0.5
                model = load_model(model_class, exp_name, fold, device, dropout=dropout)
                per_fold_results[exp_name].append(None)
                per_fold_conf_matrices[exp_name].append(None)
    else:
        for name, model_class in [
            ("SimpleCNN (5 epochs)", SimpleCNN),
            ("SimpleCNN (10 epochs)", SimpleCNN),
            ("SimpleCNN (15 epochs)", SimpleCNN),
            ("ImprovedCNN (15 epochs)", ImprovedCNN),
            ("ImprovedCNN (15 epochs, Dropout=0.3)", ImprovedCNN),
            ("ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug", ImprovedCNN),
            ("ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler", ImprovedCNN),
            ("MLPBaseline (5 epochs)", MLPBaseline),
            ("MLPBaseline (10 epochs)", MLPBaseline),
            ("MLPBaseline (15 epochs)", MLPBaseline),
        ]:
            exp_name = f"{EXPERIMENT_NAME}_{name}"
            per_fold_results[exp_name] = []
            per_fold_conf_matrices[exp_name] = []
            for fold in range(5):
                dropout = 0.3 if 'Dropout=0.3' in name else 0.5
                model = load_model(model_class, exp_name, fold, device, dropout=dropout)
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
    exp_plot_dir = os.path.join('plots', EXPERIMENT_NAME)
    os.makedirs(exp_plot_dir, exist_ok=True)
    # Save summary bar plot in this directory as well
    if exp_name == labels[0]:  # Only once for all experiments
        plt.figure(figsize=(10, 6))
        bars = plt.barh(labels, means, xerr=stds, color='skyblue', capsize=8)
        plt.xlabel('Average Cross-Validation Accuracy (%)')
        plt.title(f'{EXPERIMENT_NAME}: MNIST Model Experiment Results (5-fold CV)')
        plt.tight_layout()
        # Annotate bars with value and std
        for bar, mean, std in zip(bars, means, stds):
            plt.text(
                mean + std + 0.5,  # Offset to the right of the bar
                bar.get_y() + bar.get_height() / 2,
                f'{mean:.2f} +/- {std:.2f}',
                va='center',
                fontsize=9
            )
        plt.savefig(os.path.join(exp_plot_dir, 'experiment_results.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(exp_plot_dir, 'experiment_results.png')}")
    # Averaged confusion matrix only
    if all(cm is not None for cm in conf_matrices):
        avg_cm = np.sum(conf_matrices, axis=0) / len(conf_matrices)
        row_sums = avg_cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(avg_cm, row_sums, where=row_sums!=0) * 100  # Convert to percentage
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=norm_cm, display_labels=np.arange(10))
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.1f')
        plt.title(f'{exp_name} - Averaged Confusion Matrix (5 folds, % normalized)')
        plt.tight_layout()
        fname = os.path.join(exp_plot_dir, f"{exp_name}_averaged_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Averaged confusion matrix saved to {fname}")
