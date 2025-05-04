import torch
import os
from models import SimpleCNN, ImprovedCNN, MLPBaseline, TemperatureScaledModel
from train import cross_validate, load_model
from evaluation import evaluate_saved_models, plot_all_results, plot_reliability_diagram, compute_ece
import numpy as np
import pandas as pd

def get_experiment_configs(EXPERIMENT_NAME, DEBUG):
    if DEBUG:
        return [
            (f"{EXPERIMENT_NAME}_SimpleCNN (5 epochs)", SimpleCNN, {"epochs": 5}),
            (f"{EXPERIMENT_NAME}_MLPBaseline (5 epochs)", MLPBaseline, {"epochs": 5}),
        ]
    else:
        return [
            (f"{EXPERIMENT_NAME}_SimpleCNN (5 epochs)", SimpleCNN, {"epochs": 5}),
            (f"{EXPERIMENT_NAME}_SimpleCNN (10 epochs)", SimpleCNN, {"epochs": 10}),
            (f"{EXPERIMENT_NAME}_SimpleCNN (15 epochs)", SimpleCNN, {"epochs": 15}),
            (f"{EXPERIMENT_NAME}_ImprovedCNN (15 epochs)", ImprovedCNN, {"epochs": 15}),
            (f"{EXPERIMENT_NAME}_ImprovedCNN (15 epochs, Dropout=0.3)", ImprovedCNN, {"epochs": 15, "dropout": 0.3}),
            (f"{EXPERIMENT_NAME}_ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug", ImprovedCNN, {"epochs": 15, "dropout": 0.3, "use_aug": True}),
            (f"{EXPERIMENT_NAME}_ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler", ImprovedCNN, {"epochs": 15, "dropout": 0.3, "use_scheduler": True}),
            (f"{EXPERIMENT_NAME}_MLPBaseline (5 epochs)", MLPBaseline, {"epochs": 5}),
            (f"{EXPERIMENT_NAME}_MLPBaseline (10 epochs)", MLPBaseline, {"epochs": 10}),
            (f"{EXPERIMENT_NAME}_MLPBaseline (15 epochs)", MLPBaseline, {"epochs": 15}),
            (f"{EXPERIMENT_NAME}_ImprovedCNN (15 epochs, Temp Scaling)", TemperatureScaledModel, {"base_model_class": ImprovedCNN, "epochs": 15, "temp_scaling": True}),
        ]

def run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device):
    results = {}
    per_fold_results = {}
    per_fold_conf_matrices = {}
    reliability_data = {}
    configs = get_experiment_configs(EXPERIMENT_NAME, DEBUG)
    if not LOAD_MODELS:
        for exp_name, model_class, kwargs in configs:
            if kwargs.get("temp_scaling", False):
                # Train base model first
                base_model = kwargs["base_model_class"](dropout=kwargs.get("dropout", 0.5)).to(device)
                avg_acc, fold_accs, conf_matrices = cross_validate(
                    lambda: base_model, exp_name+"_pretemp", device, return_fold_accs=True, return_conf_matrices=True, epochs=kwargs["epochs"]
                )
                # Fit temperature on validation set (reuse last fold's val set for simplicity)
                from data import get_full_train_set
                from torch.utils.data import Subset, DataLoader
                from sklearn.model_selection import KFold
                dataset = get_full_train_set()
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                _, val_idx = list(kf.split(dataset))[-1]
                val_loader = DataLoader(Subset(dataset, val_idx), batch_size=128, shuffle=False)
                temp_model = TemperatureScaledModel(base_model).to(device)
                temp_model.set_temperature(val_loader, device)
                # Evaluate with temperature scaling
                avg_acc, fold_accs, conf_matrices, (all_labels, all_preds, all_probs) = evaluate_saved_models(
                    lambda: temp_model, exp_name, device
                )
                results[exp_name] = avg_acc
                per_fold_results[exp_name] = fold_accs
                per_fold_conf_matrices[exp_name] = conf_matrices
                reliability_data[exp_name] = (all_labels, all_preds, all_probs)
                continue
            avg_acc, fold_accs, conf_matrices, (all_labels, all_preds, all_probs) = cross_validate(
                model_class, exp_name, device, return_fold_accs=True, return_conf_matrices=True, return_probs=True, **kwargs
            )
            results[exp_name] = avg_acc
            per_fold_results[exp_name] = fold_accs
            per_fold_conf_matrices[exp_name] = conf_matrices
            reliability_data[exp_name] = (all_labels, all_preds, all_probs)
    else:
        for exp_name, model_class, kwargs in configs:
            dropout = kwargs.get('dropout', 0.5)
            use_aug = kwargs.get('use_aug', False)
            avg_acc, fold_accs, conf_matrices, (all_labels, all_preds, all_probs) = evaluate_saved_models(model_class, exp_name, device, dropout=dropout, use_aug=use_aug)
            results[exp_name] = avg_acc
            per_fold_results[exp_name] = fold_accs
            per_fold_conf_matrices[exp_name] = conf_matrices
            reliability_data[exp_name] = (all_labels, all_preds, all_probs)
    return results, per_fold_results, per_fold_conf_matrices, reliability_data

def main(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device):
    results, per_fold_results, per_fold_conf_matrices, reliability_data = run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device)
    print("\n========== FINAL COMPARISON ==========")
    for name, acc in results.items():
        print(f"{name:35s}: {acc:.2f}%")
    plot_all_results(EXPERIMENT_NAME, results, per_fold_results, per_fold_conf_matrices)
    # Plot reliability diagrams and compute ECE
    exp_plot_dir = os.path.join('plots', EXPERIMENT_NAME)
    ece_table = []
    for exp_name, (all_labels, all_preds, all_probs) in reliability_data.items():
        plot_reliability_diagram(all_labels, all_preds, all_probs, exp_plot_dir, exp_name)
        # Compute ECE for each fold (simulate by splitting into 5 chunks)
        n = len(all_labels)
        fold_size = n // 5
        eces = []
        for i in range(5):
            start = i * fold_size
            end = (i + 1) * fold_size if i < 4 else n
            ece, _, _, _ = compute_ece(all_labels[start:end], all_preds[start:end], all_probs[start:end])
            eces.append(ece)
        mean_ece = np.mean(eces)
        std_ece = np.std(eces)
        ece_table.append({"Model": exp_name, "ECE (mean ± std)": f"{mean_ece:.4f} ± {std_ece:.4f}"})
    # Save ECE table as CSV and markdown
    ece_df = pd.DataFrame(ece_table)
    ece_df.to_csv(os.path.join(exp_plot_dir, "ece_table.csv"), index=False)
    with open(os.path.join(exp_plot_dir, "ece_table.md"), "w") as f:
        f.write(ece_df.to_markdown(index=False))
    print(f"ECE table saved to {os.path.join(exp_plot_dir, 'ece_table.csv')} and .md")
