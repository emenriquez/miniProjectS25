import torch
from models import SimpleCNN, ImprovedCNN, MLPBaseline
from train import cross_validate, load_model
from evaluation import evaluate_saved_models, plot_all_results

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
        ]

def run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device):
    results = {}
    per_fold_results = {}
    per_fold_conf_matrices = {}
    configs = get_experiment_configs(EXPERIMENT_NAME, DEBUG)
    if not LOAD_MODELS:
        for exp_name, model_class, kwargs in configs:
            avg_acc, fold_accs, conf_matrices = cross_validate(model_class, exp_name, device, return_fold_accs=True, return_conf_matrices=True, **kwargs)
            results[exp_name] = avg_acc
            per_fold_results[exp_name] = fold_accs
            per_fold_conf_matrices[exp_name] = conf_matrices
    else:
        for exp_name, model_class, kwargs in configs:
            dropout = kwargs.get('dropout', 0.5)
            use_aug = kwargs.get('use_aug', False)
            avg_acc, fold_accs, conf_matrices = evaluate_saved_models(model_class, exp_name, device, dropout=dropout, use_aug=use_aug)
            results[exp_name] = avg_acc
            per_fold_results[exp_name] = fold_accs
            per_fold_conf_matrices[exp_name] = conf_matrices
    return results, per_fold_results, per_fold_conf_matrices

def main(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device):
    results, per_fold_results, per_fold_conf_matrices = run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device)
    print("\n========== FINAL COMPARISON ==========")
    for name, acc in results.items():
        print(f"{name:35s}: {acc:.2f}%")
    plot_all_results(EXPERIMENT_NAME, results, per_fold_results, per_fold_conf_matrices)
