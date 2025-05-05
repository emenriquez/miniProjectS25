import numpy as np
import os
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from data import get_full_train_set, get_full_emnist_train_set
from torch.utils.data import Subset, DataLoader
from train import load_model
from sklearn.model_selection import KFold

def evaluate_saved_models(
    model_class, exp_name, device, k=5, batch_size=1000, dropout=0.5, use_aug=False,
    DATASET='mnist', EMNIST_SPLIT='balanced', return_images=False, return_embeddings=False, load_weights=True,
    num_workers=4, **extra_kwargs
):
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
    all_images_all_folds = [] if return_images else None
    all_embeddings_all_folds = [] if return_embeddings else None
    pin_memory = torch.cuda.is_available()
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        model = load_model(model_class, exp_name, fold, device, dropout=dropout, load_weights=load_weights, **extra_kwargs)
        model.eval()
        correct, total = 0, 0
        all_labels = []
        all_preds = []
        all_probs = []
        all_images = [] if return_images else None
        all_embeddings = [] if return_embeddings else None
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                # Get embeddings if requested
                if return_embeddings and hasattr(model, "forward_features"):
                    feats = model.forward_features(images)
                    all_embeddings.extend(feats.cpu().numpy())
                    outputs = model(images)
                elif return_embeddings and hasattr(model, "features"):
                    feats = model.features(images)
                    if isinstance(feats, tuple):  # e.g., (features, logits)
                        feats = feats[0]
                    all_embeddings.extend(feats.cpu().numpy())
                    outputs = model(images)
                else:
                    outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(max_probs.cpu().numpy())
                if return_images:
                    all_images.extend(images.cpu())
        acc = 100 * correct / total
        accs.append(acc)
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(10 if DATASET == 'mnist' else len(set(all_labels))))
        conf_matrices.append(cm)
        all_labels_all_folds.extend(all_labels)
        all_preds_all_folds.extend(all_preds)
        all_probs_all_folds.extend(all_probs)
        if return_images:
            all_images_all_folds.extend(all_images)
        if return_embeddings:
            all_embeddings_all_folds.extend(all_embeddings)
    avg_acc = sum(accs) / len(accs)
    results = (np.array(all_labels_all_folds), np.array(all_preds_all_folds), np.array(all_probs_all_folds))
    if return_images and return_embeddings:
        return avg_acc, accs, conf_matrices, (*results, all_images_all_folds, np.array(all_embeddings_all_folds))
    if return_images:
        return avg_acc, accs, conf_matrices, (*results, all_images_all_folds)
    if return_embeddings:
        return avg_acc, accs, conf_matrices, (*results, np.array(all_embeddings_all_folds))
    return avg_acc, accs, conf_matrices, results

from sklearn.calibration import calibration_curve

def compute_ece(all_labels, all_preds, all_probs, n_bins=10):
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

def plot_most_confident_misclassifications(
    images, preds, labels, confidences, exp_plot_dir, exp_name, top_n=16, class_names=None
):
    # Manual mapping for EMNIST "balanced" split (index -> ASCII code)
    emnist_balanced_ascii = [
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,  # 0-9
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,  # A-Z
        97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116  # a, b, d, e, f, g, h, n, q, r, t
    ]
    emnist_balanced_labels = [chr(a) for a in emnist_balanced_ascii]

    preds = np.array(preds)
    labels = np.array(labels)
    confidences = np.array(confidences)
    mis_idx = np.where(preds != labels)[0]
    if len(mis_idx) == 0:
        print("No misclassifications found.")
        return
    sorted_idx = mis_idx[np.argsort(-confidences[mis_idx])]
    top_idx = sorted_idx[:top_n]
    ncols = 4
    nrows = int(np.ceil(top_n / ncols))
    plt.figure(figsize=(3*ncols, 3*nrows))
    for i, idx in enumerate(top_idx):
        plt.subplot(nrows, ncols, i+1)
        img = images[idx]
        if isinstance(img, torch.Tensor):
            img_disp = img.squeeze().cpu().numpy()
        else:
            img_disp = img.squeeze()
        # Fix orientation for EMNIST: transpose image if shape is (28,28)
        if img_disp.ndim == 2 and img_disp.shape == (28, 28):
            img_disp = img_disp.T
        pred_idx = int(preds[idx])
        true_idx = int(labels[idx])
        # Use manual mapping for EMNIST balanced, else fallback to class_names or index
        if class_names is not None and len(class_names) == 47:
            pred_label = emnist_balanced_labels[pred_idx] if pred_idx < len(emnist_balanced_labels) else pred_idx
            true_label = emnist_balanced_labels[true_idx] if true_idx < len(emnist_balanced_labels) else true_idx
        else:
            pred_label = class_names[pred_idx] if (class_names is not None and pred_idx < len(class_names)) else pred_idx
            true_label = class_names[true_idx] if (class_names is not None and true_idx < len(class_names)) else true_idx
        plt.imshow(img_disp, cmap='gray')
        plt.axis('off')
        plt.title(f"P:{pred_label}\nT:{true_label}\nC:{confidences[idx]:.2f}")
    plt.tight_layout()
    os.makedirs(exp_plot_dir, exist_ok=True)
    fname = os.path.join(exp_plot_dir, f"{exp_name}_most_confident_misclassifications.png")
    plt.savefig(fname)
    plt.close()
    print(f"Most confident misclassifications saved to {fname}")

def plot_umap_embeddings(
    embeddings, preds, labels, exp_plot_dir, exp_name, confidences=None,
    sample_size=2000, n_neighbors=50, min_dist=0.05, random_state=42
):
    """
    Project embeddings to 2D using UMAP and plot:
    - Color by true label
    - Misclassified points are highly visible: larger, thicker, and on top
    - Optionally, color intensity by confidence if confidences is provided
    - sample_size: number of points to plot (randomly sampled for clarity)
    - n_neighbors, min_dist: UMAP parameters to control cluster tightness/islands
    - Adds class label colors to the legend
    - Normalizes embeddings before UMAP for better quality
    """
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    preds = np.array(preds)
    labels = np.array(labels)
    correct = preds == labels
    embeddings = np.array(embeddings)
    N = len(labels)

    # Handle case where embeddings is empty or 1D (avoid crash)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        print(f"Warning: No embeddings to plot for {exp_name}. Skipping UMAP plot.")
        return

    # Normalize embeddings (L2 norm)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms

    # Subsample for clarity if needed
    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        preds = preds[idx]
        labels = labels[idx]
        correct = correct[idx]
        if confidences is not None:
            confidences = np.array(confidences)[idx]
    else:
        idx = np.arange(N)
        if confidences is not None:
            confidences = np.array(confidences)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric='cosine')
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')

    # Plot correct predictions (drawn first, smaller, more transparent)
    plt.scatter(
        emb_2d[correct, 0], emb_2d[correct, 1],
        c=labels[correct], cmap=cmap, s=12, marker='o', alpha=0.25, label='Correct', linewidths=0
    )
    # Plot misclassifications (drawn on top, larger, less transparent, thick edge)
    if np.any(~correct):
        plt.scatter(
            emb_2d[~correct, 0], emb_2d[~correct, 1],
            c=labels[~correct], cmap=cmap, s=60, marker='o', alpha=0.95,
            edgecolor='black', linewidths=1.5, label='Incorrect'
        )

    # Optionally overlay confidence as color intensity (drawn behind)
    if confidences is not None:
        plt.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=confidences, cmap='viridis', s=8, alpha=0.15, label='Confidence', zorder=0
        )
        cbar = plt.colorbar()
        cbar.set_label('Confidence (max softmax)', fontsize=12)

    # Build legend for class colors
    handles = []
    if num_classes <= 20:
        class_handles = []
        for i, class_label in enumerate(unique_labels):
            color = cmap(i % cmap.N)
            class_handles.append(Line2D([0], [0], marker='o', color='w', label=f'Class {class_label}',
                                        markerfacecolor=color, markersize=8, alpha=0.8))
        handles = class_handles
    # Always add handles for correct/incorrect
    handles += [
        Line2D([0], [0], marker='o', color='w', label='Correct', markerfacecolor='gray', markersize=8, alpha=0.25),
        Line2D([0], [0], marker='o', color='black', label='Incorrect', markerfacecolor='gray', markersize=12, markeredgewidth=2)
    ]
    plt.legend(handles=handles, loc='best', fontsize=10, frameon=True)

    plt.title(f"UMAP Embeddings\nColored by True Label, Misclassified Highlighted\n{exp_name}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    os.makedirs(exp_plot_dir, exist_ok=True)
    fname = os.path.join(exp_plot_dir, f"{exp_name}_umap_embeddings.png")
    plt.savefig(fname)
    plt.close()
    print(f"UMAP embedding plot saved to {fname}")

def plot_tsne_embeddings(
    embeddings, preds, labels, exp_plot_dir, exp_name, confidences=None,
    sample_size=2000, perplexity=30, random_state=42
):
    """
    Project embeddings to 2D using t-SNE and plot:
    - Color by true label
    - Misclassified points are highly visible: larger, thicker, and on top
    - Optionally, color intensity by confidence if confidences is provided
    - sample_size: number of points to plot (randomly sampled for clarity)
    - Adds class label colors to the legend
    - Normalizes embeddings before t-SNE for better quality
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    preds = np.array(preds)
    labels = np.array(labels)
    correct = preds == labels
    embeddings = np.array(embeddings)
    N = len(labels)

    # Normalize embeddings (L2 norm)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Subsample for clarity if needed
    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        preds = preds[idx]
        labels = labels[idx]
        correct = correct[idx]
        if confidences is not None:
            confidences = np.array(confidences)[idx]
    else:
        idx = np.arange(N)
        if confidences is not None:
            confidences = np.array(confidences)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca')
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')

    # Plot correct predictions (drawn first, smaller, more transparent)
    plt.scatter(
        emb_2d[correct, 0], emb_2d[correct, 1],
        c=labels[correct], cmap=cmap, s=12, marker='o', alpha=0.25, label='Correct', linewidths=0
    )
    # Plot misclassifications (drawn on top, larger, less transparent, thick edge)
    if np.any(~correct):
        plt.scatter(
            emb_2d[~correct, 0], emb_2d[~correct, 1],
            c=labels[~correct], cmap=cmap, s=60, marker='o', alpha=0.95,
            edgecolor='black', linewidths=1.5, label='Incorrect'
        )

    # Optionally overlay confidence as color intensity (drawn behind)
    if confidences is not None:
        plt.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=confidences, cmap='viridis', s=8, alpha=0.15, label='Confidence', zorder=0
        )
        cbar = plt.colorbar()
        cbar.set_label('Confidence (max softmax)', fontsize=12)

    # Build legend for class colors
    handles = []
    if num_classes <= 20:
        class_handles = []
        for i, class_label in enumerate(unique_labels):
            color = cmap(i % cmap.N)
            class_handles.append(Line2D([0], [0], marker='o', color='w', label=f'Class {class_label}',
                                        markerfacecolor=color, markersize=8, alpha=0.8))
        handles = class_handles
    # Always add handles for correct/incorrect
    handles += [
        Line2D([0], [0], marker='o', color='w', label='Correct', markerfacecolor='gray', markersize=8, alpha=0.25),
        Line2D([0], [0], marker='o', color='black', label='Incorrect', markerfacecolor='gray', markersize=12, markeredgewidth=2)
    ]
    plt.legend(handles=handles, loc='best', fontsize=10, frameon=True)

    plt.title(f"t-SNE Embeddings\nColored by True Label, Misclassified Highlighted\n{exp_name}")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    os.makedirs(exp_plot_dir, exist_ok=True)
    fname = os.path.join(exp_plot_dir, f"{exp_name}_tsne_embeddings.png")
    plt.savefig(fname)
    plt.close()
    print(f"t-SNE embedding plot saved to {fname}")

def plot_confidence_histogram(all_labels, all_preds, all_probs, exp_plot_dir, exp_name, bins=20):
    """
    Plot histogram of predicted probabilities (confidence) for correct and incorrect predictions.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    correct_mask = (all_labels == all_preds)
    incorrect_mask = ~correct_mask
    plt.figure(figsize=(8, 5))
    plt.hist(all_probs[correct_mask], bins=bins, alpha=0.7, label='Correct', color='green', density=True)
    plt.hist(all_probs[incorrect_mask], bins=bins, alpha=0.7, label='Incorrect', color='red', density=True)
    plt.xlabel('Predicted Probability (Confidence)')
    plt.ylabel('Density')
    plt.title(f'Confidence Histogram\n{exp_name}')
    plt.legend()
    plt.tight_layout()
    os.makedirs(exp_plot_dir, exist_ok=True)
    fname = os.path.join(exp_plot_dir, f"{exp_name}_confidence_histogram.png")
    plt.savefig(fname)
    plt.close()
    print(f"Confidence histogram saved to {fname}")

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
    for exp_name, conf_matrices in per_fold_conf_matrices.items():
        if all(cm is not None for cm in conf_matrices):
            avg_cm = np.sum(conf_matrices, axis=0) / len(conf_matrices)
            row_sums = avg_cm.sum(axis=1, keepdims=True)
            norm_cm = np.divide(avg_cm, row_sums, where=row_sums!=0) * 100
            num_classes = norm_cm.shape[0]
            fig, ax = plt.subplots(figsize=(min(1.5 + 0.3*num_classes, 20), min(1.5 + 0.3*num_classes, 20)))
            im = ax.imshow(norm_cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=2)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('% of True Class', fontsize=14)
            plt.title(f'{exp_name} - Averaged Confusion Matrix (5 folds, % normalized)', fontsize=14, pad=20)
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            tick_marks = np.arange(num_classes)
            # Show all ticks for <=20 classes, else every 5th
            if num_classes <= 20:
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(tick_marks, fontsize=10, rotation=45)
                ax.set_yticklabels(tick_marks, fontsize=10)
            else:
                step = max(1, num_classes // 20)
                shown_ticks = tick_marks[::step]
                ax.set_xticks(shown_ticks)
                ax.set_yticks(shown_ticks)
                ax.set_xticklabels(shown_ticks, fontsize=8, rotation=90)
                ax.set_yticklabels(shown_ticks, fontsize=8)
            # Only annotate cells for small confusion matrices
            if num_classes <= 20:
                for i in range(norm_cm.shape[0]):
                    for j in range(norm_cm.shape[1]):
                        val = norm_cm[i, j]
                        color = 'white' if val > 1.5 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)
            plt.tight_layout(rect=[0, 0, 1, 1])
            fname = os.path.join(exp_plot_dir, f"{exp_name}_averaged_confusion_matrix.png")
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            print(f"Averaged confusion matrix saved to {fname}")
