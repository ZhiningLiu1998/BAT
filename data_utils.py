import os
import random

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.utils import index_to_mask, mask_to_index


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


def load_data(
    dataset_name,
    to_device: str,
    path="./data",
    split_type="public",
    reorder_label=False,
    verbose: bool = False,
):
    """
    Loads and preprocesses a dataset for node classification.

    Parameters:
    - dataset_name: str
        Name of the dataset.
    - to_device: str
        Device to move the data to (e.g., 'cuda' or 'cpu').
    - path: str, optional
        Path to the dataset directory.
    - split_type: str, optional
        Type of split to use for the dataset.
    - reorder_label: bool, optional
        Whether to reorder labels based on their frequency.
    - verbose: bool, optional
        Whether to print verbose loading information.

    Returns:
    - data: pyg.data.Data
        Preprocessed data for node classification.
    """
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Coauthor, Planetoid

    if verbose:
        print(f"Loading dataset {dataset_name.title()}... ", end="")

    name = dataset_name.lower()
    if name in ["cora", "citeseer", "pubmed"]:
        # Planetoid datasets
        dataset = Planetoid(
            path + "/Planetoid", name, transform=T.NormalizeFeatures(), split=split_type
        )
    elif name in ["cs", "physics"]:
        # Coauthor datasets
        dataset = Coauthor(path + "/Coauthor", name, transform=T.NormalizeFeatures())
    elif name in ["reddit"]:
        from torch_geometric.datasets import Reddit
        
        dataset = Reddit(path + '/Reddit', transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError("Not Implemented Dataset!")
    if reorder_label:
        dataset.data.y, _ = reorder_label_by_count(dataset.data.y)
    # Load data to device
    try:
        data = dataset[0].to(to_device)
    except:
        raise ValueError(f"Failed loading data to device: {to_device}")

    if verbose:
        print(f"Done!")

    return data


def get_step_imbalanced_split_data(
    data: pyg.data.Data,
    use_public_split="auto",
    imbratio=20,
    n_tail_class=None,
    head_class_size=20,
    test_ratio=0.5,  # ignored if data has pub split and use_public_split=True
    random_seed=None,
    verbose=False,
):
    """
    Generates a naturally imbalanced split for the dataset.

    Parameters:
    - data: pyg.data.Data
        Preprocessed data for node classification.
    - use_public_split: bool, optional
        Whether to use the public split.
    - imbratio: int
        Imbalance ratio for the entire dataset.
    - n_tail_class: int, optional
        Number of tail classes (ignored).
    - head_class_size: int, optional
        Size of the head classes (ignored).
    - test_ratio: float, optional
        Ratio of nodes to use for testing.
    - random_seed: int, optional
        Seed for random number generator.
    - verbose: bool, optional
        Whether to print verbose information.

    Returns:
    - data: pyg.data.Data
        Data with naturally imbalanced split.
    """
    assert isinstance(data, pyg.data.Data), "data must be a pyg.data.Data object"
    n_classes = len(data.y.unique())
    assert use_public_split in [
        "auto",
        True,
        False,
    ], f"Invalid use_public_split: {use_public_split}, must be one of ['auto', True, False]"
    assert imbratio >= 1, f"imbratio must >= 1, got {imbratio}"
    assert (
        isinstance(n_tail_class, int) or n_tail_class is None
    ), f"n_tail_class must be an int or None, got {n_tail_class}"
    assert (
        n_tail_class < n_classes if isinstance(n_tail_class, int) else True
    ), f"n_tail_class (got {n_tail_class}) must be less than n_classes ({n_classes})"
    assert (
        head_class_size >= imbratio
    ), f"head_class_size (got {head_class_size}) must >= imbratio (got {imbratio})"
    assert test_ratio >= 0 and test_ratio <= 1, "test_ratio must be in [0, 1]"
    assert (
        isinstance(random_seed, int) or random_seed is None
    ), "random_seed must be an int or None"
    assert isinstance(verbose, bool), "verbose must be a bool"
    if random_seed is not None:
        seed_everything(random_seed)

    # detect if data has public split
    if (
        hasattr(data, "train_mask")
        and hasattr(data, "val_mask")
        and hasattr(data, "test_mask")
    ):
        has_public_split = True
    else:
        has_public_split = False
        if use_public_split == True:
            raise ValueError(
                f"Dataset does not have public split, "
                f"set use_public_split=False or 'auto'."
            )
    if use_public_split == False or (
        use_public_split == "auto" and has_public_split == False
    ):
        (
            data.train_mask,
            data.val_mask,
            data.test_mask,
        ) = get_random_semi_train_val_test_split(
            data,
            n_train_num_per_class=head_class_size,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

    train_mask = data.train_mask
    classes, train_class_counts = data.y[train_mask].unique(return_counts=True)
    n_class = len(classes)

    if n_tail_class is None:
        n_tail_class = n_class // 2
    else:
        assert (
            n_tail_class < n_class
        ), f"n_tail_class ({n_tail_class}) must be less than n_class ({n_class})"

    max_class_size = train_class_counts.max().item()
    tail_class_size = max_class_size // imbratio
    tail_classes = classes[n_class - n_tail_class :]
    train_class_indices = get_train_class_indices(data.y, n_class, train_mask)

    # make imbalanced train mask
    _train_class_indices = []
    _train_mask = torch.zeros_like(train_mask).bool()
    _train_class_counts = []
    for i in range(n_class):
        # get class train indices
        class_indices = train_class_indices[i]
        if i in tail_classes and len(class_indices) > tail_class_size:
            # shuffle and select tail_class_size
            perm_idx = torch.randperm(len(class_indices))
            class_indices = class_indices[perm_idx][:tail_class_size]
        _train_class_indices.append(class_indices)
        _train_class_counts.append(len(class_indices))
        _train_mask[class_indices] = True

    if verbose:
        _, val_class_counts = data.y[data.val_mask].unique(return_counts=True)
        _, test_class_counts = data.y[data.test_mask].unique(return_counts=True)
        print(
            f"Generating step imbalanced task...\n"
            f"-----------------------------------\n"
            f"imbratio:                      {imbratio}\n"
            f"random_seed:                   {random_seed}\n"
            f"Head/Tail class number:        {n_class-n_tail_class}/{n_tail_class}\n"
            f"Head/Tail class size:          {max_class_size}/{tail_class_size}\n"
            f"Classes:                       {classes.cpu().tolist()}\n"
            f"Tail classes:                  {tail_classes.cpu().tolist()}\n"
            f"Original train class counts:   {train_class_counts.cpu().tolist()}\n"
            f"Train class counts:            {_train_class_counts}\n"
            # f"Val class counts:              {val_class_counts.cpu().tolist()}\n"
            # f"Test class counts:             {test_class_counts.cpu().tolist()}\n"
        )

    data.train_mask = _train_mask
    return data


def get_natural_imbalanced_split_data(
    data: pyg.data.Data,
    use_public_split=False,  # ignored
    imbratio=100,
    n_tail_class=None,  # ignored
    head_class_size=None,  # ignored
    test_ratio=0.5,
    random_seed=None,
    verbose=False,
):
    assert isinstance(data, pyg.data.Data), "data must be a pyg.data.Data object"
    n_classes = len(data.y.unique())
    assert (
        use_public_split == False
    ), "natural imbalance will use random splits, set use_public_split=False"
    assert imbratio >= 1, f"imbratio must >= 1, got {imbratio}"
    assert n_tail_class is None, "natural imbalance does not support n_tail_class"
    assert head_class_size is None, "natural imbalance does not support head_class_size"
    assert test_ratio >= 0 and test_ratio <= 1, "test_ratio must be in [0, 1]"
    assert (
        isinstance(random_seed, int) or random_seed is None
    ), "random_seed must be an int or None"
    assert isinstance(verbose, bool), "verbose must be a bool"

    if random_seed is not None:
        seed_everything(random_seed)

    n_nodes = data.y.shape[0]
    classes, counts = data.y.unique(return_counts=True)
    n_cls = len(classes)
    n_cls_1 = n_cls - 1
    # sort classes by counts
    head_to_tail_classes = sorted(
        classes.cpu().numpy(), key=lambda x: counts[x], reverse=True
    )

    train_index, val_index, test_index = [], [], []
    _train_class_counts = []
    # for each class
    for i, label in enumerate(head_to_tail_classes):
        cls_index = mask_to_index(data.y == label)
        # get class train/val size
        n_cls_nodes = len(cls_index)
        n_train_nodes = int(imbratio ** ((n_cls_1 - i) / n_cls_1))
        n_val_nodes = int((n_cls_nodes - n_train_nodes) * (1 - test_ratio))
        # permute the indices
        perm_index = torch.randperm(len(cls_index))
        # get class train/val/test indices
        cls_train_index = cls_index[perm_index[:n_train_nodes]]
        cls_val_index = cls_index[
            perm_index[n_train_nodes : n_train_nodes + n_val_nodes]
        ]
        cls_test_index = cls_index[perm_index[n_train_nodes + n_val_nodes :]]
        train_index.append(cls_train_index)
        val_index.append(cls_val_index)
        test_index.append(cls_test_index)
        _train_class_counts.append(len(cls_train_index))

    # transform indices to masks
    device = data.y.device
    data.train_mask = index_to_mask(torch.concat(train_index), size=n_nodes).to(device)
    data.val_mask = index_to_mask(torch.concat(val_index), size=n_nodes).to(device)
    data.test_mask = index_to_mask(torch.concat(test_index), size=n_nodes).to(device)

    if verbose:
        _, val_class_counts = data.y[data.val_mask].unique(return_counts=True)
        _, test_class_counts = data.y[data.test_mask].unique(return_counts=True)
        print(
            f"Generating natural imbalanced task...\n"
            f"-----------------------------------\n"
            f"imbratio:                      {imbratio}\n"
            f"random_seed:                   {random_seed}\n"
            f"Head/Tail class size:          {head_class_size}/{_train_class_counts[-1]}\n"
            f"Classes:                       {head_to_tail_classes}\n"
            f"Train class counts:            {_train_class_counts}\n"
            # f"Val class counts:              {val_class_counts.cpu().tolist()}\n"
            # f"Test class counts:             {test_class_counts.cpu().tolist()}\n"
        )

    return data


def reorder_label_by_count(labels, debug=False):
    """
    Reorders labels based on the count of occurrences.

    Parameters:
    - labels: torch.Tensor
        The original labels.
    - debug: bool, optional
        Whether to print debug information.

    Returns:
    - new_labels: torch.Tensor
        The reordered labels.
    - idx_map: dict
        A dictionary mapping old label indices to new label indices.
    """
    if debug:
        print("Refine label order, Many to Few")
    num_labels = labels.max() + 1
    num_labels_each_class = np.array(
        [(labels == i).sum().item() for i in range(num_labels)]
    )
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]: i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.cpu().numpy())

    return labels.new(new_labels), idx_map


def get_train_class_indices(y, n_class, train_mask):
    """
    Retrieves indices of training nodes for each class.

    Parameters:
    - y: torch.Tensor
        Node labels.
    - n_class: int
        Number of classes.
    - train_mask: torch.Tensor
        Mask indicating training nodes.

    Returns:
    - train_class_indices: list
        List of lists containing indices of training nodes for each class.
    """
    indices = torch.arange(len(y))
    train_class_indices = []
    for i in range(n_class):
        cls_indices = indices[((y == i) & train_mask)]
        train_class_indices.append(cls_indices)
    return train_class_indices


def get_random_semi_train_val_test_split(
    data,
    n_train_num_per_class=20,
    test_ratio=0.5,
    random_seed=None,
):
    """
    Generates random semi-supervised train-val-test split.

    Parameters:
    - data: pyg.data.Data
        Preprocessed data for node classification.
    - n_train_num_per_class: int, optional
        Number of training nodes per class.
    - test_ratio: float, optional
        Ratio of nodes to use for testing.
    - random_seed: int, optional
        Seed for random number generator.

    Returns:
    - train_mask: torch.Tensor
        Mask indicating training nodes.
    - val_mask: torch.Tensor
        Mask indicating validation nodes.
    - test_mask: torch.Tensor
        Mask indicating testing nodes.
    """
    from torch_geometric.utils import index_to_mask, mask_to_index

    if random_seed is not None:
        seed_everything(random_seed)

    n_nodes = data.y.shape[0]
    train_index, val_index, test_index = [], [], []
    for label in data.y.unique():
        cls_index = mask_to_index(data.y == label)
        n_class_nodes = len(cls_index)
        n_train_nodes = n_train_num_per_class
        n_val_nodes = int((n_class_nodes - n_train_nodes) * (1 - test_ratio))
        n_test_nodes = int((n_class_nodes - n_train_nodes) * test_ratio)
        perm_index = torch.randperm(len(cls_index))
        cls_train_index = cls_index[perm_index[:n_train_nodes]]
        cls_val_index = cls_index[
            perm_index[n_train_nodes : n_train_nodes + n_val_nodes]
        ]
        cls_test_index = cls_index[perm_index[n_train_nodes + n_val_nodes :]]
        train_index.append(cls_train_index)
        val_index.append(cls_val_index)
        test_index.append(cls_test_index)

    device = data.y.device
    train_mask = index_to_mask(torch.concat(train_index), size=n_nodes).to(device)
    val_mask = index_to_mask(torch.concat(val_index), size=n_nodes).to(device)
    test_mask = index_to_mask(torch.concat(test_index), size=n_nodes).to(device)
    return train_mask, val_mask, test_mask


def get_data_stat(data, store_in_data: bool = True, verbose: bool = True):
    """
    Extracts and displays data statistics from the data object.

    Parameters:
    - data: pyg.data.Data
        Preprocessed data for node classification.
    - store_in_data: bool, optional
        Whether to store statistics in the data object.
    - verbose: bool, optional
        Whether to print verbose information.

    Returns:
    - data: pyg.data.Data
        Data object with statistics if store_in_data is True.
    """

    classes, train_class_counts = data.y[data.train_mask].unique(return_counts=True)
    _, val_class_counts = data.y[data.val_mask].unique(return_counts=True)
    _, test_class_counts = data.y[data.test_mask].unique(return_counts=True)
    n_class = len(classes)
    n_node = data.x.shape[0]
    n_feat = data.x.shape[1]
    n_edge = data.edge_index.shape[1]
    train_class_indices = get_train_class_indices(data.y, n_class, data.train_mask)

    if torch.all(train_class_counts == train_class_counts.min()):
        # training dataset is balanced
        tail_classes = None
    else:
        # training dataset is imbalanced
        tail_classes = torch.where(train_class_counts == train_class_counts.min())[0]

    if store_in_data:
        data.n_node = n_node
        data.n_feat = n_feat
        data.n_edge = n_edge
        data.n_class = n_class
        data.classes = classes
        data.tail_classes = tail_classes
        data.train_class_counts = train_class_counts
        data.val_class_counts = val_class_counts
        data.test_class_counts = test_class_counts
        data.train_class_indices = train_class_indices

    if verbose:
        print(
            f"Data statistics:\n"
            f"----------------\n"
            f"Imabalanced?:      {tail_classes is not None}\n"
            f"n_node:            {n_node}\n"
            f"n_feat:            {n_feat}\n"
            f"n_edge:            {n_edge}\n"
            f"n_class:           {n_class}\n"
            f"classes:           {classes.cpu().tolist()}\n"
            f"train_class_distr: {train_class_counts.cpu().tolist()}\n"
            f"valid_class_distr: {val_class_counts.cpu().tolist()}\n"
            f"test_class_distr:  {test_class_counts.cpu().tolist()}\n"
            f"tail_classes:      {tail_classes.cpu().tolist() if tail_classes is not None else None}\n"
        )

    return data
