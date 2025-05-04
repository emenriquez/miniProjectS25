import numpy as np
import os
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

def evaluate_saved_models(model_class, exp_name, device, k=5, batch_size=1000, dropout=0.5, use_aug=False):
    from data import get_full_train_set
    from torch.utils.data import Subset, DataLoader
    from train import load_model
    from sklearn.model_selection import KFold
    dataset = get_full_train_set(augmented=use_aug)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs = []
    conf_matrices = []
    all_labels_all_folds = []
    all_preds_all_folds = []
    all_probs_all_folds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        model = load_model(model_class, exp_name, fold, device, dropout=dropout)
        model.eval()
        correct, total = 0, 0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(max_probs.cpu().numpy())
        acc = 100 * correct / total
        accs.append(acc)
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(10))
        conf_matrices.append(cm)
        all_labels_all_folds.extend(all_labels)
        all_preds_all_folds.extend(all_preds)
        all_probs_all_folds.extend(all_probs)
    avg_acc = sum(accs) / len(accs)
    return avg_acc, accs, conf_matrices, (np.array(all_labels_all_folds), np.array(all_preds_all_folds), np.array(all_probs_all_folds))

from sklearn.calibration import calibration_curve

def compute_ece(all_labels, all_preds, all_probs, n_bins=10):
    # ECE: Expected Calibration Error
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(all_probs, bins) - 1
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            acc = np.mean(all_labels[mask] == all_preds[mask])
            conf = np.mean(all_probs[mask])
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(np.sum(mask))
            ece += np.abs(acc - conf) * np.sum(mask)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    ece = ece / len(all_labels)
    return ece, bin_accs, bin_confs, bin_counts

def plot_reliability_diagram(all_labels, all_preds, all_probs, exp_plot_dir, exp_name):
    correct = (all_labels == all_preds)
    prob_true, prob_pred = calibration_curve(correct, all_probs, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram\n{exp_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = os.path.join(exp_plot_dir, f"{exp_name}_reliability_diagram.png")
    plt.savefig(fname)
    plt.close()
    print(f"Reliability diagram saved to {fname}")

def prettify_label(label, experiment_name):
    if label.startswith(experiment_name):
        label = label[len(experiment_name):]
    label = label.lstrip('_').strip()
    label = label.replace('MLPBaseline', 'MLP Baseline')
    label = label.replace('SimpleCNN', 'Simple CNN')
    label = label.replace('ImprovedCNN', 'Improved CNN')
    label = label.replace('plus', '+')
    return '\n'.join(textwrap.wrap(label, width=28))

def plot_all_results(EXPERIMENT_NAME, results, per_fold_results, per_fold_conf_matrices):
    os.makedirs('plots', exist_ok=True)
    labels = list(results.keys())
    means = [results[k] for k in labels]
    stds = [np.std(per_fold_results[k]) for k in labels]
    sorted_indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
    labels_sorted = [labels[i] for i in sorted_indices]
    means_sorted = [means[i] for i in sorted_indices]
    stds_sorted = [stds[i] for i in sorted_indices]
    pretty_labels = [prettify_label(l, EXPERIMENT_NAME) for l in labels_sorted]
    exp_plot_dir = os.path.join('plots', EXPERIMENT_NAME)
    os.makedirs(exp_plot_dir, exist_ok=True)
    # Bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.barh(pretty_labels, means_sorted, xerr=stds_sorted, color='skyblue', capsize=8)
    plt.xlabel('Average Cross-Validation Accuracy (%)', fontsize=16)
    plt.title(f'{EXPERIMENT_NAME}: MNIST Model Experiment Results (5-fold CV)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 1])
    for bar, mean, std in zip(bars, means_sorted, stds_sorted):
        plt.text(
            mean + std + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{mean:.2f} ± {std:.2f}',
            va='center',
            fontsize=14,
            clip_on=False
        )
    plt.savefig(os.path.join(exp_plot_dir, 'experiment_results.png'), bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(exp_plot_dir, 'experiment_results.png')}")
    # Confusion matrices
    for exp_name, conf_matrices in per_fold_conf_matrices.items():
        if all(cm is not None for cm in conf_matrices):
            avg_cm = np.sum(conf_matrices, axis=0) / len(conf_matrices)
            row_sums = avg_cm.sum(axis=1, keepdims=True)
            norm_cm = np.divide(avg_cm, row_sums, where=row_sums!=0) * 100
            fig, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(norm_cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=3)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('% of True Class', fontsize=14)
            plt.title(f'{exp_name} - Averaged Confusion Matrix (5 folds, % normalized)', fontsize=14, pad=20)
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            tick_marks = np.arange(10)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(tick_marks, fontsize=12)
            ax.set_yticklabels(tick_marks, fontsize=12)
            for i in range(norm_cm.shape[0]):
                for j in range(norm_cm.shape[1]):
                    val = norm_cm[i, j]
                    color = 'white' if val > 1.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 1])
            fname = os.path.join(exp_plot_dir, f"{exp_name}_averaged_confusion_matrix.png")
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            print(f"Averaged confusion matrix saved to {fname}")
