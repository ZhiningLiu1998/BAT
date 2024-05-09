import argparse
import yaml

IMB_TYPES = ["step", "natural"]
GNN_ARCHS = ["GCN", "GAT", "SAGE"]
BAT_MODES = ["dummy", "bat0", "bat1", "all"]


def parse_args(config_path="config.yaml"):
    """
    Parses command-line arguments and returns an argparse.Namespace object containing the parsed arguments.

    Args:
    - config_path: str
        Path to the config.yaml file.

    Returns:
    - args: argparse.Namespace
        Parsed command-line arguments.
    """
    # Load default configuration from config.yaml
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    parser = argparse.ArgumentParser(description="Argument parser for your script")

    # General
    parser.add_argument(
        "--gpu_id", type=int, default=config["general"]["gpu_id"], help="GPU ID"
    )
    parser.add_argument(
        "--seed", type=int, default=config["general"]["seed"], help="Random seed"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=config["general"]["n_runs"],
        help="The number of independent runs",
    )
    parser.add_argument(
        "--debug", type=bool, default=config["general"]["debug"], help="Debug mode"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default=config["dataset"]["dataset"], help="Dataset Name"
    )
    parser.add_argument(
        "--imb_type",
        type=str,
        default=config["dataset"]["imb_type"],
        choices=IMB_TYPES,
        help="Imbalance type",
    )
    parser.add_argument(
        "--imb_ratio",
        type=int,
        default=config["dataset"]["imb_ratio"],
        help="Imbalance Ratio",
    )

    # Architecture
    parser.add_argument(
        "--gnn_arch",
        type=str,
        default=config["architecture"]["gnn_arch"],
        choices=GNN_ARCHS,
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=config["architecture"]["n_layer"],
        help="The number of layers",
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=config["architecture"]["hid_dim"],
        help="Hidden dimension",
    )

    # Training
    parser.add_argument(
        "--lr",
        type=float,
        default=config["training"]["lr"],
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config["training"]["weight_decay"],
        help="Weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"]["epochs"],
        help="The number of training epochs",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=config["training"]["early_stop"],
        help="Early stop patience",
    )
    parser.add_argument(
        "--tqdm",
        type=bool,
        default=config["training"]["tqdm"],
        help="Enable tqdm progress bar",
    )

    # ToBE parameters
    parser.add_argument(
        "--bat_mode",
        type=str,
        default=config["bat_parameters"]["bat_mode"],
        choices=BAT_MODES,
        help="Mode of BAT, can be 'dummy', 'bat0', 'bat1', or 'all'",
    )

    # Parse args
    args = parser.parse_args()

    return args
