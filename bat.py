from abc import ABC
from abc import abstractmethod
import time
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import to_undirected
from utils import seed_everything


class BaseGraphAugmenter(ABC):
    """
    Abstract base class for graph data augmentation strategies.

    Methods:
    - init_with_data(self, data)
        Initialize the augmenter with graph data.

    - augment(self, model, x, edge_index)
        Perform graph augmentation.

    - adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor)
        Adapt labels and training mask after augmentation.
    """

    @abstractmethod
    def init_with_data(self, data: pyg.data.Data):
        """
        Initialize the augmenter with graph data.

        Parameters:
        - data: pyg.data.Data
            Graph data used for initialization.
        """
        pass

    @abstractmethod
    def augment(
        self, model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor
    ):
        """
        Perform graph augmentation.

        Parameters:
        - model: torch.nn.Module
            Graph neural network model.
        - x: torch.Tensor
            Input features of the graph nodes.
        - edge_index: torch.Tensor
            Edge indices of the graph.

        Returns:
        - augmented_x: torch.Tensor
            Augmented node features.
        - augmented_edge_index: torch.Tensor
            Augmented edge indices.
        - runtime_info: dict
            Additional runtime information from the augmentation process.
        """
        pass

    @abstractmethod
    def adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor):
        """
        Adapt labels and training mask after augmentation.

        Parameters:
        - y: torch.Tensor
            Original node labels.
        - train_mask: torch.Tensor
            Original training mask.

        Returns:
        - adapted_y: torch.Tensor
            Adapted node labels.
        - adapted_train_mask: torch.Tensor
            Adapted training mask.
        """
        pass


class DummyAugmenter(BaseGraphAugmenter):
    """
    A dummy graph augmenter for demonstration purposes.

    Methods:
    - __init__(self)
        Initializes the DummyAugmenter instance.

    - init_with_data(self, data)
        Initializes the augmenter with graph data.

    - augment(self, model, x, edge_index)
        Performs dummy graph augmentation.

    - adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor)
        Adapts labels and training mask after dummy augmentation.
    """

    def __init__(self) -> None:
        """
        Initializes the DummyAugmenter instance.
        """
        super().__init__()

    def init_with_data(self, data: pyg.data.Data):
        """
        Initializes the augmenter with graph data.

        Parameters:
        - data: pyg.data.Data
            Graph data used for initialization.

        Returns:
        - self: DummyAugmenter
        """
        return self

    def augment(
        self, model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor
    ):
        """
        Performs dummy graph augmentation.

        Parameters:
        - model: torch.nn.Module
            Graph neural network model.
        - x: torch.Tensor
            Input features of the graph nodes.
        - edge_index: torch.Tensor
            Edge indices of the graph.

        Returns:
        - augmented_x: torch.Tensor
            Augmented node features.
        - augmented_edge_index: torch.Tensor
            Augmented edge indices.
        - runtime_info: dict
            Additional runtime information from the dummy augmentation process.
        """
        return (
            x,
            edge_index,
            {
                "time_aug(ms)": 0.0,
                "node_ratio(%)": 100.0,
                "edge_ratio(%)": 100.0,
            },
        )

    def adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor):
        """
        Adapts labels and training mask after dummy augmentation.

        Parameters:
        - y: torch.Tensor
            Original node labels.
        - train_mask: torch.Tensor
            Original training mask.

        Returns:
        - adapted_y: torch.Tensor
            Adapted node labels.
        - adapted_train_mask: torch.Tensor
            Adapted training mask.
        """
        return y, train_mask


class BatAugmenter(BaseGraphAugmenter):
    """
    Topological Balanced augmEntation (BAT) for graph data.

    Parameters:
    - mode: str, optional (default: "bat0")
        The augmentation mode. Must be one of ["dummy", "bat0", "bat1"].
    - random_state: int or None, optional (default: None)
        Random seed for reproducibility.

    Methods:
    - __init__(self, mode: str = "bat0", random_state: int = None)
        Initializes the BatAugmenter instance.

    - init_with_data(self, data: pyg.data.Data)
        Initializes the augmenter with graph data.

    - augment(self, model, x, edge_index)
        Performs topology-aware graph augmentation.

    - adapt_labels_and_train_mask(self, y, train_mask)
        Adapts labels and training mask after augmentation.

    - info(self)
        Prints information about the augmenter.

    - predict_proba(model, x, edge_index, return_numpy=False)
        Computes predicted class probabilities using the model.

    - edge_sampling(edge_index, edge_sampling_proba, random_state=None)
        Performs edge sampling based on probability.

    - get_group_mean(values, labels, classes)
        Computes the mean of values within each class.

    - get_virtual_node_features(x, y_pred, classes)
        Computes virtual node features based on predicted labels.

    - get_connectivity_distribution_sparse(y_pred, edge_index, n_class, n_node, n_edge)
        Computes the distribution of neighbor labels for each node.

    - get_node_risk(self, y_pred_proba, y_pred)
        Computes node risk based on predicted class probabilities.

    - estimate_node_posterior_likelihood(self, y_pred_proba, y_neighbor_distr)
        Computes posterior likelihood for each node and class.

    - get_virual_link_proba(self, node_posterior, y_pred)
        Computes virtual link probabilities based on node posterior likelihood.
    """

    MODE_SPACE = ["dummy", "bat0", "bat1"]

    def __init__(
        self,
        mode: str = "bat0",
        random_state: int = None,
    ):
        """
        Initializes the BatAugmenter instance.

        Parameters:
        - mode: str, optional (default: "bat0")
            The augmentation mode. Must be one of ["dummy", "bat0", "bat1"].
        - random_state: int or None, optional (default: None)
            Random seed for reproducibility.
        """
        super().__init__()
        # parameter check
        assert mode in self.MODE_SPACE, f"mode must be one of {self.MODE_SPACE}"
        assert (
            isinstance(random_state, int) or random_state is None
        ), "random_state must be an integer or None"
        self.mode = mode
        self.random_state = random_state
        self.init_flag = False

    def init_with_data(self, data: pyg.data.Data):
        """
        Initializes the augmenter with graph data.

        Parameters:
        - data: pyg.data.Data
            The graph data.

        Raises:
        - AssertionError: If data is not a pyg.data.Data object or lacks required attributes.

        Returns:
        - self: BatAugmenter
        """
        assert isinstance(data, pyg.data.Data), "data must be a pyg.data.Data object"
        assert hasattr(data, "train_mask"), "data must have 'train_mask' attribute"
        assert hasattr(data, "val_mask"), "data must have 'val_mask' attribute"
        assert hasattr(data, "test_mask"), "data must have 'test_mask' attribute"

        # initialization
        x, edge_index, train_mask, y_train, device = (
            data.x,
            data.edge_index,
            data.train_mask,
            data.y[data.train_mask],
            data.x.device,
        )
        classes, train_class_counts = y_train.unique(return_counts=True)
        self.classes = classes
        self.train_class_counts = train_class_counts
        # basic stats
        self.n_node = x.shape[0]
        self.n_edge = edge_index.shape[1]
        self.n_class = len(classes)
        self.y_virtual = classes
        self.y_train = y_train
        self.train_mask = train_mask
        self.train_class_weights = train_class_counts / train_class_counts.max()
        self.empty_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        self.dummy_runtime_info = {
            "time_aug(ms)": 0.0,
            "time_unc(ms)": 0.0,
            "time_risk(ms)": 0.0,
            "time_neighbor_distr(ms)": 0.0,
            "time_gen(ms)": 0.0,
            "time_sim(ms)": 0.0,
            "node_ratio(%)": 100.0,
            "edge_ratio(%)": 100.0,
        }
        self.device = device
        self.init_flag = True

        return self

    def augment(
        self, model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor
    ):
        """
        Performs topology-aware graph augmentation.

        Parameters:
        - model: torch.nn.Module
            The model used for prediction.
        - x: torch.Tensor
            Node features.
        - edge_index: torch.Tensor
            Edge indices.

        Returns:
        - x_aug: torch.Tensor
            Augmented node features.
        - edge_index_aug: torch.Tensor
            Augmented edge indices.
        - info: dict
            Augmentation information.
        """
        assert self.init_flag, "init_with_data() must be called before augment()"
        # for reproducibility (constant seed will led to non-diverse sampling results)
        if self.random_state is not None:
            self.random_state += 1
        train_mask = self.train_mask
        # do nothing if mode is 'dummy'
        if self.mode == "dummy":
            return (x, edge_index, self.dummy_runtime_info)

        # initialization
        start_time = time.time()
        y_pred_proba = self.predict_proba(model, x, edge_index)
        y_pred = y_pred_proba.argmax(axis=1)
        y_pred[train_mask] = self.y_train

        # compute node_risk and virtual link probability
        node_risk = self.get_node_risk(y_pred_proba, y_pred)

        start_time_sim = time.time()
        if self.mode == "bat0":
            y_neighbor_distr = None
            self.time_neighbor_distr = 0.0
        else:
            y_neighbor_distr = self.get_connectivity_distribution_sparse(
                y_pred, edge_index, self.n_class, self.n_node, self.n_edge
            )

        node_posterior = self.estimate_node_posterior_likelihood(
            y_pred_proba, y_neighbor_distr
        )
        virtual_link_proba = self.get_virual_link_proba(node_posterior, y_pred)
        time_cost_sim = time.time() - start_time_sim

        start_time_gen = time.time()

        # assign link probability w.r.t node risk
        virtual_link_proba *= node_risk.reshape(-1, 1)

        # sample virtual edge_index w.r.t given probability
        virtual_adj = virtual_link_proba.T.to_sparse().coalesce()
        edge_index_candidates, edge_sampling_proba = (
            virtual_adj.indices(),
            virtual_adj.values(),
        )

        virtual_edge_index = self.edge_sampling(
            edge_index_candidates, edge_sampling_proba, self.random_state
        )
        virtual_edge_index[
            0
        ] += self.n_node  # adjust index to match original node index
        virtual_edge_index = to_undirected(virtual_edge_index)

        # compute virtual node features
        x_virtual = self.get_virtual_node_features(x, y_pred, self.classes)
        time_cost_gen = time.time() - start_time_gen

        # concatenate results
        time_cost = time.time() - start_time
        x_aug = torch.concat([x, x_virtual])
        edge_index_aug = torch.concat([edge_index, virtual_edge_index], axis=1)
        info = {
            "time_aug(ms)": time_cost * 1000,
            "time_unc(ms)": self.time_unc_comp * 1000,
            "time_risk(ms)": self.time_risk_comp * 1000,
            "time_sim(ms)": time_cost_sim * 1000,
            "time_neighbor_distr(ms)": self.time_neighbor_distr * 1000,
            "time_gen(ms)": time_cost_gen * 1000,
            "node_ratio(%)": x_aug.shape[0] / x.shape[0] * 100,
            "edge_ratio(%)": edge_index_aug.shape[1] / edge_index.shape[1] * 100,
        }
        return x_aug, edge_index_aug, info

    def adapt_labels_and_train_mask(self, y: torch.Tensor, train_mask: torch.Tensor):
        """
        Adapts labels and training mask after augmentation.

        Parameters:
        - y: torch.Tensor
            Original labels.
        - train_mask: torch.Tensor
            Original training mask.

        Returns:
        - new_y: torch.Tensor
            Adapted labels.
        - new_train_mask: torch.Tensor
            Adapted training mask.
        """
        if self.mode == "dummy":
            return y, train_mask
        new_y = torch.concat([y, self.y_virtual])
        new_train_mask = torch.concat(
            [train_mask, torch.ones_like(self.y_virtual).bool()]
        )
        return new_y, new_train_mask

    def info(self):
        """
        Prints information about the augmenter.
        """
        print(
            f"BatAugmenter(\n"
            f"    mode={self.mode},\n"
            f"    n_node={self.n_node},\n"
            f"    n_edge={self.n_edge},\n"
            f"    n_class={self.n_class},\n"
            f"    classes={self.classes.cpu()},\n"
            f"    train_class_counts={self.train_class_counts.cpu()},\n"
            f"    train_class_weights={self.train_class_weights.cpu()},\n"
            f"    device={self.device},\n"
            f")"
        )

    @staticmethod
    def predict_proba(
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_numpy: bool = False,
    ):
        """
        Computes predicted class probabilities using the model.

        Parameters:
        - model: torch.nn.Module
            The model used for prediction.
        - x: torch.Tensor
            Node features.
        - edge_index: torch.Tensor
            Edge indices.
        - return_numpy: bool, optional (default: False)
            Whether to return the probabilities as a numpy array.

        Returns:
        - pred_proba: torch.Tensor or numpy.ndarray
            Predicted class probabilities.
        """
        model.eval()
        with torch.no_grad():
            logits = model.forward(x, edge_index)
        pred_proba = torch.softmax(logits, dim=1).detach()
        if return_numpy:
            pred_proba = pred_proba.cpu().numpy()
        return pred_proba

    @staticmethod
    def edge_sampling(
        edge_index: torch.Tensor,
        edge_sampling_proba: torch.Tensor,
        random_state: int = None,
    ):
        """
        Performs edge sampling based on probability.

        Parameters:
        - edge_index: torch.Tensor
            Edge indices.
        - edge_sampling_proba: torch.Tensor
            Edge sampling probabilities.
        - random_state: int or None, optional (default: None)
            Random seed for reproducibility.

        Returns:
        - sampled_edge_index: torch.Tensor
            Sampled edge indices.
        """
        assert edge_sampling_proba.min() >= 0 and edge_sampling_proba.max() <= 1
        seed_everything(random_state)
        edge_sample_mask = torch.rand_like(edge_sampling_proba) < edge_sampling_proba
        return edge_index[:, edge_sample_mask]

    @staticmethod
    def get_group_mean(
        values: torch.Tensor, labels: torch.Tensor, classes: torch.Tensor
    ):
        """
        Computes the mean of values within each class.

        Parameters:
        - values: torch.Tensor
            Values to compute the mean of.
        - labels: torch.Tensor
            Labels corresponding to values.
        - classes: torch.Tensor
            Classes for which to compute the mean.

        Returns:
        - new_values: torch.Tensor
            Mean values for each class.
        """
        new_values = torch.zeros_like(values)
        for i in classes:
            mask = labels == i
            new_values[mask] = values[mask].mean()
        return new_values

    @staticmethod
    def get_virtual_node_features(x: torch.Tensor, y_pred: torch.Tensor, classes: list):
        """
        Computes virtual node features based on predicted labels.

        Parameters:
        - x: torch.Tensor
            Node features.
        - y_pred: torch.Tensor
            Predicted labels.
        - classes: list
            Unique classes in the dataset.

        Returns:
        - virtual_node_features: torch.Tensor
            Virtual node features for each class.
        """
        return torch.stack([x[y_pred == label].mean(axis=0) for label in classes])

    def get_connectivity_distribution_sparse(
        self,
        y_pred: torch.Tensor,
        edge_index: torch.Tensor,
        n_class: int,
        n_node: int,
        n_edge: int,
    ):
        """
        Computes the distribution of connectivity labels.

        Parameters:
        - y_pred: torch.Tensor
            Predicted labels.
        - edge_index: torch.Tensor
            Edge indices (sparse).
        - n_class: int
            Number of classes.
        - n_node: int
            Number of nodes.
        - n_edge: int
            Number of edges.

        Returns:
        - neighbor_y_distr: torch.Tensor
            Normalized connectivity label distribution.
        """
        start_time = time.time()

        device = y_pred.device
        edge_dest_class = torch.zeros(
            (n_edge, n_class), dtype=torch.int, device=device
        ).scatter_(
            1, y_pred[edge_index[1]].unsqueeze(1), 1
        )  # [n_edges, n_class]
        neighbor_y_distr = (
            torch.zeros((n_node, n_class), dtype=torch.int, device=device)
            .scatter_add_(
                dim=0,
                index=edge_index[0].repeat(n_class, 1).T,
                src=edge_dest_class,
            )
            .float()
        )  # [n_nodes, n_class]

        # row-wise normalization
        neighbor_y_distr /= neighbor_y_distr.sum(axis=1).reshape(-1, 1)
        neighbor_y_distr = neighbor_y_distr.nan_to_num(0)

        self.time_neighbor_distr = time.time() - start_time
        return neighbor_y_distr

    def get_node_risk(self, y_pred_proba: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes node risk based on predicted probabilities.

        Parameters:
        - y_pred_proba: torch.Tensor
            Predicted class probabilities.
        - y_pred: torch.Tensor
            Predicted labels.

        Returns:
        - node_risk: torch.Tensor
            Node risk scores.
        """
        # compute node pred
        start_time = time.time()
        node_unc = 1 - y_pred_proba.max(axis=1).values
        self.time_unc_comp = time.time() - start_time
        # compute class-aware relative pred
        node_unc_class_mean = self.get_group_mean(node_unc, y_pred, self.classes)
        node_risk = (node_unc - node_unc_class_mean).clip(min=0)
        # calibrate node risk w.r.t class weights
        node_risk *= self.train_class_weights[y_pred]
        self.time_risk_comp = time.time() - start_time
        return node_risk

    def estimate_node_posterior_likelihood(
        self, y_pred_proba: torch.Tensor, y_neighbor_distr: torch.Tensor
    ):
        """
        Estimates node posterior likelihood for each class.

        Parameters:
        - y_pred_proba: torch.Tensor
            Predicted class probabilities.
        - y_neighbor_distr: torch.Tensor
            Connectivity label distribution.

        Returns:
        - node_posterior: torch.Tensor
            Node posterior likelihood.
        """
        mode = self.mode
        if mode == "bat0":
            node_posterior = y_pred_proba
        elif mode == "bat1":
            node_posterior = y_neighbor_distr
        else:
            raise NotImplementedError
        return node_posterior

    def get_virual_link_proba(self, node_posterior: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes virtual link probabilities based on node posterior likelihood.

        Parameters:
        - node_posterior: torch.Tensor
            Node posterior likelihood.
        - y_pred: torch.Tensor
            Predicted labels.

        Returns:
        - virtual_link_proba: torch.Tensor
            Virtual link probabilities.
        """
        # set likelihood to current predicted class as 0
        node_posterior *= 1 - F.one_hot(y_pred, num_classes=self.n_class)
        node_posterior = node_posterior.clip(min=0)
        # row-wise renormalize
        node_posterior /= node_posterior.sum(axis=1).reshape(-1, 1)
        virtual_link_proba = node_posterior.nan_to_num(0)
        return virtual_link_proba
