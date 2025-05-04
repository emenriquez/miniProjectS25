import argparse
from experiments import main as run_experiments
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST experiments.")
    parser.add_argument('--experiment', type=str, default="DEBUG_RUN", help='Experiment name for grouping results')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (quick, minimal experiments)')
    parser.add_argument('--load-models', action='store_true', help='Load models instead of training')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu, default: auto)')
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment
    DEBUG = args.debug
    LOAD_MODELS = args.load_models
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device)
