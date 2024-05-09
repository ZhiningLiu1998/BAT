import os
import random

import numpy as np
import torch


def seed_everything(seed):
    """
    Set random seeds for reproducibility.

    Parameters:
    - seed: int or None
        Seed value for random number generators.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(
    gnn_arch: str,
    feat_dim: int,
    hid_dim: int,
    out_dim: int,
    n_layer: int,
    use_ens_net: bool = False,
    dropout: float = 0.5,
    device: str = "cpu",
):
    """
    Create a GNN model on the specified device.

    Parameters:
    - gnn_arch: str
        GNN architecture (options: "GCN", "GAT", "SAGE").
    - feat_dim: int
        Dimension of input features.
    - hid_dim: int
        Dimension of hidden layers.
    - out_dim: int
        Dimension of output classes.
    - n_layer: int
        Number of GNN layers.
    - use_ens_net: bool, optional
        Whether to use modified GNNs for GraphENS.
    - dropout: float, optional
        Dropout rate for the model.
    - device: str, optional
        Device to use for model (e.g., "cpu" or "cuda").

    Returns:
    - model: torch.nn.Module
        Created GNN model.
    """
    # Import appropriate modules based on imbalance handling strategy
    if use_ens_net:
        from nets.gens_networks import create_gat, create_gcn, create_sage
    else:
        from nets import create_gat, create_gcn, create_sage

    init_kwargs = {
        "nlayer": n_layer,
        "nfeat": feat_dim,
        "nhid": hid_dim,
        "nclass": out_dim,
        "dropout": dropout,
    }

    gnn_arch = gnn_arch.upper()
    if gnn_arch == "GCN":
        model = create_gcn(**init_kwargs)
    elif gnn_arch == "GAT":
        model = create_gat(**init_kwargs)
    elif gnn_arch == "SAGE":
        model = create_sage(**init_kwargs)
    else:
        raise ValueError(
            f"gnn_arch must be one of ['GCN', 'GAT', 'SAGE'], got {gnn_arch}."
        )

    model = model.to(device)
    return model


def get_device(gpu_id: int = -1):
    """
    Get the specified device for computation.

    Parameters:
    - gpu_id: int, optional
        Index of the GPU to use (-1 for CPU).

    Returns:
    - device: str
        Device for computation ("cpu" or "cuda").
    """
    if gpu_id == -1:
        device = "cpu"
    else:
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(gpu_id)
            device_id = torch.cuda.current_device()
            print(
                f"Now using GPU #{device_id}: {torch.cuda.get_device_name(device_id)}"
            )
        else:
            raise ValueError("cuda is not available, specify gpu_id=-1 to use cpu.")

    return device


def print_centered(s: str, width: int, fillchar: str = "=", prefix: str = ""):
    """
    Print a string centered within a specified width.

    Parameters:
    - s: str
        String to be centered.
    - width: int
        Total width of the output.
    - fillchar: str, optional
        Character used for filling (default: "=").
    - prefix: str, optional
        Prefix to be added before the centered string.
    """
    if len(s) == 0:
        print(f"{fillchar * width}")
        return

    fill_len = (width - len(s) - 2) // 2
    print(prefix, end="")
    print(f"{fillchar * fill_len} {s} {fillchar * fill_len}")

    return
