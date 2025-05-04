# MNIST CNN Experiments

This project provides a modular framework for running and analyzing convolutional neural network (CNN) experiments on the MNIST dataset. It supports basic evaluation with cross-validation, model saving/loading, and detailed visualizations.

## Features
- Modular code for models, data, training, experiments, and evaluation
- 5-fold cross-validation for robust evaluation
- Model saving/loading for efficient iteration
- Automatic plotting of results and confusion matrices
- Command-line interface for experiment configuration
- Easily extensible for new models or analysis

## Project Structure
```
models.py        # CNN and MLP model definitions
data.py          # Data loading and augmentation
train.py         # Training, cross-validation, model save/load
experiments.py   # Experiment orchestration and workflow
evaluation.py    # Evaluation and plotting functions
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
- Plots and confusion matrices are saved in `plots/<experiment_name>/`.
- Trained models are saved in `saved_models/<experiment_name>/`.
- Each experiment run is grouped under its own directory for easy comparison.

## Extending the Project
- Add new models to `models.py`.
- Add new experiments or analysis scripts as needed in `experiments.py` or `evaluation.py`.

---

*Created for research and educational purposes.*
