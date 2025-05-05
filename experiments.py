import torch
import os
from models import SimpleCNN, ImprovedCNN, MLPBaseline, TemperatureScaledModel
from train import cross_validate, load_model
from evaluation import evaluate_saved_models, plot_all_results, plot_reliability_diagram, compute_ece, plot_most_confident_misclassifications, plot_umap_embeddings, plot_tsne_embeddings, plot_confidence_histogram
import numpy as np
import pandas as pd

def get_num_classes(dataset, emnist_split):
    if dataset == 'mnist':
        return 10
    emnist_splits = {
        'byclass': 62,
        'bymerge': 47,
        'balanced': 47,
        'letters': 26,
        'digits': 10,
        'mnist': 10
    }
    return emnist_splits.get(emnist_split, 47)

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
            (f"{EXPERIMENT_NAME}_MLPBaseline (15 epochs)", MLPBaseline, {"epochs": 15}),
            (f"{EXPERIMENT_NAME}_ImprovedCNN (15 epochs, Temp Scaling)", TemperatureScaledModel, {"base_model_class": ImprovedCNN, "epochs": 15, "temp_scaling": True}),
        ]

def run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device, DATASET='mnist', EMNIST_SPLIT='balanced', NUM_WORKERS=4, BATCH_SIZE=64):
    results = {}
    per_fold_results = {}
    per_fold_conf_matrices = {}
    reliability_data = {}
    miscls_data = {}
    umap_data = {}
    configs = get_experiment_configs(EXPERIMENT_NAME, DEBUG)
    num_classes = get_num_classes(DATASET, EMNIST_SPLIT)
    k_folds = 5
    for exp_name, model_class, kwargs in configs:
        if model_class in [SimpleCNN, ImprovedCNN, MLPBaseline]:
            kwargs['num_classes'] = num_classes
        if model_class is TemperatureScaledModel and 'base_model_class' in kwargs:
            base_model_class = kwargs['base_model_class']
            base_model = base_model_class(num_classes=num_classes, dropout=kwargs.get('dropout', 0.5)).to(device)
            kwargs['base_model'] = base_model

        # Prepare extra_kwargs for models that need them (e.g., TemperatureScaledModel)
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in ['epochs', 'use_aug', 'use_scheduler', 'temp_scaling', 'base_model_class', 'dropout']}
        if model_class is TemperatureScaledModel and 'base_model' in kwargs:
            extra_kwargs['base_model'] = kwargs['base_model']

        # RESUME LOGIC: Check if all folds are already trained
        exp_dir = os.path.join("saved_models", exp_name)
        all_folds_exist = True
        for fold in range(k_folds):
            fname = f"{exp_name}_{fold+1}.pt"
            path = os.path.join(exp_dir, fname)
            if not os.path.exists(path):
                all_folds_exist = False
                break

        # TRAINING PHASE
        if not LOAD_MODELS and not all_folds_exist:
            from train import cross_validate
            cross_validate(
                model_class, exp_name, device,
                k=k_folds, epochs=kwargs.get('epochs', 5), dropout=kwargs.get('dropout', 0.5),
                use_aug=kwargs.get('use_aug', False), DATASET=DATASET, EMNIST_SPLIT=EMNIST_SPLIT,
                num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, **{k: v for k, v in kwargs.items() if k not in ['epochs', 'use_aug', 'use_scheduler', 'temp_scaling', 'base_model_class']}
            )
        elif not LOAD_MODELS and all_folds_exist:
            print(f"Skipping training for {exp_name} (all folds already exist)")

        # EVALUATION PHASE (always load weights)
        avg_acc, fold_accs, conf_matrices, (all_labels, all_preds, all_probs, all_images, all_embeddings) = evaluate_saved_models(
            model_class, exp_name, device, dropout=kwargs.get('dropout', 0.5), use_aug=kwargs.get('use_aug', False),
            DATASET=DATASET, EMNIST_SPLIT=EMNIST_SPLIT, return_images=True, return_embeddings=True, load_weights=True,
            num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, **extra_kwargs
        )
        results[exp_name] = avg_acc
        per_fold_results[exp_name] = fold_accs
        per_fold_conf_matrices[exp_name] = conf_matrices
        reliability_data[exp_name] = (all_labels, all_preds, all_probs)
        miscls_data[exp_name] = (all_images, all_preds, all_labels, all_probs)
        umap_data[exp_name] = (all_embeddings, all_preds, all_labels)
    return results, per_fold_results, per_fold_conf_matrices, reliability_data, miscls_data, umap_data

def main(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device, DATASET='mnist', EMNIST_SPLIT='balanced', NUM_WORKERS=4, BATCH_SIZE=64):
    results, per_fold_results, per_fold_conf_matrices, reliability_data, miscls_data, umap_data = run_experiments(
        EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device, DATASET, EMNIST_SPLIT, NUM_WORKERS, BATCH_SIZE)
    print("\n========== FINAL COMPARISON ==========")
    for name, acc in results.items():
        print(f"{name:35s}: {acc:.2f}%")
    plot_all_results(EXPERIMENT_NAME, results, per_fold_results, per_fold_conf_matrices)
    # Plot reliability diagrams and compute ECE
    exp_plot_dir = os.path.join('plots', EXPERIMENT_NAME)
    ece_table = []
    acc_table = []
    for exp_name, (all_labels, all_preds, all_probs) in reliability_data.items():
        plot_reliability_diagram(all_labels, all_preds, all_probs, exp_plot_dir, exp_name)
        plot_confidence_histogram(all_labels, all_preds, all_probs, exp_plot_dir, exp_name)
        # Compute ECE for each fold (simulate by splitting into 5 chunks)
        n = len(all_labels)
        fold_size = n // 5
        eces = []
        accs = []
        for i in range(5):
            start = i * fold_size
            end = (i + 1) * fold_size if i < 4 else n
            ece, _, _, _ = compute_ece(all_labels[start:end], all_preds[start:end], all_probs[start:end])
            acc = np.mean(all_labels[start:end] == all_preds[start:end])
            eces.append(ece)
            accs.append(acc)
        mean_ece = np.mean(eces)
        std_ece = np.std(eces)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        ece_table.append({"Model": exp_name, "ECE (mean ± std)": f"{mean_ece:.4f} ± {std_ece:.4f}"})
        acc_table.append({"Model": exp_name, "Accuracy (mean ± std)": f"{mean_acc*100:.2f} ± {std_acc*100:.2f}"})
    # Save ECE table as CSV and markdown
    ece_df = pd.DataFrame(ece_table)
    ece_df.to_csv(os.path.join(exp_plot_dir, "ece_table.csv"), index=False)
    with open(os.path.join(exp_plot_dir, "ece_table.md"), "w") as f:
        f.write(ece_df.to_markdown(index=False))
    print(f"ECE table saved to {os.path.join(exp_plot_dir, 'ece_table.csv')} and .md")
    # Save accuracy table as CSV and markdown
    acc_df = pd.DataFrame(acc_table)
    acc_df.to_csv(os.path.join(exp_plot_dir, "accuracy_table.csv"), index=False)
    with open(os.path.join(exp_plot_dir, "accuracy_table.md"), "w") as f:
        f.write(acc_df.to_markdown(index=False))
    print(f"Accuracy table saved to {os.path.join(exp_plot_dir, 'accuracy_table.csv')} and .md")
    # Plot most confident misclassifications for each model
    for exp_name, (all_images, all_preds, all_labels, all_probs) in miscls_data.items():
        plot_most_confident_misclassifications(
            all_images, all_preds, all_labels, all_probs,
            exp_plot_dir, exp_name, top_n=16
        )
    # UMAP embedding plots
    for exp_name, (all_embeddings, all_preds, all_labels) in umap_data.items():
        plot_umap_embeddings(
            all_embeddings, all_preds, all_labels,
            exp_plot_dir, exp_name
        )
        # t-SNE embedding plots
        plot_tsne_embeddings(
            all_embeddings, all_preds, all_labels,
            exp_plot_dir, exp_name
        )
