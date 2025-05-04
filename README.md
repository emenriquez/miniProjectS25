# MNIST CNN Experiments

This project provides a modular framework for running and analyzing convolutional neural network (CNN) experiments on the MNIST dataset. It supports basic evaluation with cross-validation, model saving/loading, and detailed visualizations.

## Features
- Modular code for models, data, training, and experiments
- 5-fold cross-validation for robust evaluation
- Model saving/loading for efficient iteration
- Automatic plotting of results and confusion matrices
- Easily extensible for new models or analysis

## Project Structure
```
models.py        # CNN model definitions
train.py         # Training, cross-validation, model save/load
main.py          # Entry point for running experiments and analysis
data.py          # Data loading and augmentation
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
  python main.py
  ```
  This will train models with 5-fold cross-validation, save results, and generate plots in the `plots/` directory.

- **Skip training and analyze saved models:**
  Set `LOAD_MODELS = True` in `main.py` to load previously saved models for further analysis or plotting.

## Results
- Plots and confusion matrices are saved in `plots/<experiment_name>/`.
- Trained models are saved in `saved_models/<experiment_name>/`.

## Extending the Project
- Add new models to `models.py`.
- Add new experiments or analysis scripts as needed.

---

*Created for research and educational purposes.*
