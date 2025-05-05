# MNIST & EMNIST CNN Experiments

This project provides a modular framework for running and analyzing convolutional neural network (CNN) experiments on the MNIST and EMNIST datasets. It supports rigorous evaluation with cross-validation, model saving/loading, confidence calibration, and detailed visualizations.

## Features
- Modular code for models, data, training, experiments, and evaluation
- 5-fold cross-validation for robust evaluation
- Model saving/loading for efficient iteration
- Automatic plotting of results, confusion matrices, and reliability diagrams
- Confidence calibration metrics (ECE, reliability diagrams, temperature scaling)
- Confidence histogram plots for correct/incorrect predictions
- UMAP and t-SNE embedding plots for visualizing learned representations
- Plots of most confident misclassifications for each model
- Command-line interface for experiment configuration, including batch size and number of workers
- Supports both MNIST and all EMNIST splits (byclass, bymerge, balanced, letters, digits, mnist)
- Easily extensible for new models, datasets, or analysis

## Datasets
- **Default:** [MNIST](http://yann.lecun.com/exdb/mnist/) (downloaded automatically via torchvision)
- **Expanded:** [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) (all splits supported via torchvision)

### How to Add or Use New Datasets
- Add new dataset loading logic to `data.py`.
- Place new data in the `data/` directory if needed.
- Update experiment configs in `experiments.py` to use the new dataset.
- The project is structured to make it easy to swap or extend datasets.

## Project Structure
```
models.py        # CNN and MLP model definitions
data.py          # Data loading and augmentation
train.py         # Training, cross-validation, model save/load
experiments.py   # Experiment orchestration and workflow
evaluation.py    # Evaluation, calibration, and plotting functions
utils.py         # Utility functions (e.g., get_num_classes)
main.py          # Entry point (CLI, config, calls experiments)
requirements.txt # Python dependencies
plots/           # Output plots (auto-generated)
saved_models/    # Saved models (auto-generated)
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone git@github.com:emenriquez/miniProjectS25.git
   cd miniProjectS25
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage
- **Run experiments and generate plots:**
  ```bash
  python main.py --experiment MY_EXPERIMENT
  ```
  This will train models with 5-fold cross-validation, save results, and generate plots in the `plots/` directory.

- **Use EMNIST (with split selection):**
  ```bash
  python main.py --experiment MY_EMNIST_EXPERIMENT --dataset emnist --emnist-split byclass
  ```
  Available EMNIST splits: byclass, bymerge, balanced, letters, digits, mnist

- **Set number of DataLoader workers and batch size:**
  ```bash
  python main.py --num-workers 8 --batch-size 256
  ```

- **Debug mode (quick run):**
  ```bash
  python main.py --debug
  ```
  Only runs MLP Baseline and Simple CNN (5 epochs) for fast prototyping.

- **Skip training and analyze saved models:**
  ```bash
  python main.py --load-models --experiment MY_EXPERIMENT
  ```
  Loads previously saved models for further analysis or plotting.

- **Specify device:**
  ```bash
  python main.py --device cuda
  ```
  (or `cpu`)

## Results
- Plots, confusion matrices, reliability diagrams, **confidence histograms**, and ECE tables are saved in `plots/<experiment_name>/`.
- Each model produces a file named `<model_name>_confidence_histogram.png` showing the distribution of predicted confidences for correct and incorrect predictions.
- **UMAP and t-SNE embedding plots** (`<model_name>_umap_embeddings.png`, `<model_name>_tsne_embeddings.png`) visualize learned features.
- **Most confident misclassification plots** (`<model_name>_most_confident_misclassifications.png`) highlight errors with high confidence.
- Trained models are saved in `saved_models/<experiment_name>/`.
- Each experiment run is grouped under its own directory for easy comparison.

## Extending the Project
- Add new models to `models.py`.
- Add new datasets to `data.py` and update experiment configs in `experiments.py`.
- Add new experiments or analysis scripts as needed in `experiments.py` or `evaluation.py`.

---

*Created for research and educational purposes.*
