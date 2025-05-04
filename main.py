import argparse
from experiments import main as run_experiments
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST/EMNIST experiments.")
    parser.add_argument('--experiment', type=str, default="DEBUG_RUN", help='Experiment name for grouping results')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (quick, minimal experiments)')
    parser.add_argument('--load-models', action='store_true', help='Load models instead of training')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'], help='Dataset to use: mnist or emnist')
    parser.add_argument('--emnist-split', type=str, default='balanced',
        help='EMNIST split to use (if dataset=emnist). Options: byclass, bymerge, balanced, letters, digits, mnist')
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment
    DEBUG = args.debug
    LOAD_MODELS = args.load_models
    DATASET = args.dataset
    EMNIST_SPLIT = args.emnist_split
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_experiments(EXPERIMENT_NAME, DEBUG, LOAD_MODELS, device, DATASET, EMNIST_SPLIT)
