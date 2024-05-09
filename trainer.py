import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric as pyg
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from bat import BaseGraphAugmenter, DummyAugmenter
from tqdm import trange


class NodeClassificationTrainer:
    """
    A trainer class for node classification with Graph Augmenter.

    Parameters:
    -----------
    - model: torch.nn.Module
        The node classification model.
    - data: pyg.data.Data
        PyTorch Geometric data object containing graph data.
    - device: str or torch.device
        Device to use for computations (e.g., 'cuda' or 'cpu').
    - augmenter: BaseGraphAugmenter, optional
        Graph augmentation strategy.
    - learning_rate: float, optional
        Learning rate for optimization.
    - weight_decay: float, optional
        Weight decay (L2 penalty) for optimization.
    - train_epoch: int, optional
        Number of training epochs.
    - early_stop_patience: int, optional
        Number of epochs with no improvement to trigger early stopping.
    - eval_freq: int, optional
        Frequency of evaluation during training.
    - eval_metrics: dict, optional
        Dictionary of evaluation metrics and associated functions.
    - verbose_freq: int, optional
        Frequency of verbose logging.
    - verbose_config: dict, optional
        Configuration for verbose logging.
    - save_model_dir: str, optional
        Directory to save model checkpoints.
    - save_model_name: str, optional
        Name of the saved model checkpoint.
    - enable_tqdm: bool, optional
        Whether to enable tqdm progress bar.
    - random_state: int, optional
        Seed for random number generator.

    Methods:
    --------
    - model_update(self)
        Performs a single update step for the model.
    - model_eval(self)
        Evaluates the model on the validation and test sets.
    - train(self, train_epoch=None, eval_freq=None, verbose_freq=None)
        Trains the node classification model and performs evaluation.
    - print_best_results(self)
        Prints the evaluation results of the best model.
    - get_validation_score(self, eval_results)
        Computes the average validation score for model selection.
    - verbose(self, results, epoch, verbose_config, runtime: bool or str = None)
        Prints verbose training progress information.
    """

    default_eval_metrics = {
        "bacc": (balanced_accuracy_score, {}),
        "macro-f1": (f1_score, {"average": "macro"}),
    }
    default_verbose_config = {
        "metrics": list(default_eval_metrics.keys()),
        "datasets": ["train", "val", "test"],
        "runtime": "average",
    }

    def __init__(
        self,
        model,
        data,
        device,
        augmenter: BaseGraphAugmenter = DummyAugmenter(),
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        train_epoch: int = 1000,
        early_stop_patience: int = None,
        warmup_epoch: int = 0,
        eval_freq: int = 1,
        eval_metrics: dict = default_eval_metrics,
        verbose_freq: int = None,
        verbose_config: dict = default_verbose_config,
        save_model_dir: str = "saved_ckpt",
        save_model_name: str = None,
        enable_tqdm: bool = False,
        random_state: int = None,
    ):
        # parameter checks
        assert isinstance(model, torch.nn.Module), "model must be a PyTorch Module"
        assert isinstance(
            data, pyg.data.Data
        ), "data must be a PyTorch Geometric Data object"
        try:
            torch.device(device)
        except:
            raise ValueError(f"device must be a valid PyTorch device, got {device}")
        assert isinstance(
            augmenter, BaseGraphAugmenter
        ), "augmenter must be a BaseGraphAugmenter"
        assert isinstance(learning_rate, float), "learning_rate must be a float"
        assert isinstance(weight_decay, float), "weight_decay must be a float"
        assert (
            isinstance(train_epoch, int) and train_epoch > 0
        ), "train_epoch must be a positive integer"
        assert early_stop_patience is None or (
            isinstance(early_stop_patience, int) and early_stop_patience > 0
        ), "early_stop_patience must be None or a positive integer"
        assert (
            isinstance(eval_metrics, dict) or eval_metrics is None
        ), "eval_metrics must be a dictionary or None"
        assert (
            isinstance(eval_freq, int) and eval_freq > 0
        ), "eval_freq must be a positive integer"
        assert (
            isinstance(verbose_freq, int) or verbose_freq is None
        ), "verbose_freq must be a integer, or None"
        assert (
            isinstance(verbose_config, dict) or verbose_config is None
        ), "verbose_config must be a dictionary or None"
        assert isinstance(save_model_dir, str), "save_model_dir must be a string"
        assert os.path.exists(
            save_model_dir
        ), f"save_model_dir={save_model_dir} does not exist"
        assert (
            isinstance(save_model_name, str) or save_model_name is None
        ), "save_model_name must be a string or None"
        assert isinstance(enable_tqdm, bool), "enable_tqdm must be a boolean"
        assert random_state is None or isinstance(
            random_state, int
        ), "random_state must be an integer"

        # initialize
        self.model = model
        self.data = data
        self.device = device
        self.augmenter = augmenter.init_with_data(
            data
        )  # augmenter is a BaseGraphAugmenter object
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=100, verbose=False
        )

        # training settings
        self.train_epoch = train_epoch
        self.early_stop_flag = early_stop_patience is not None
        self.early_stop_patience = early_stop_patience
        self.warmup_epoch = warmup_epoch
        self.eval_freq = eval_freq
        self.eval_metrics = eval_metrics
        self.verbose_flag = verbose_freq is not None
        self.verbose_freq = verbose_freq
        self.verbose_config = verbose_config
        self.tqdm_flag = enable_tqdm

        # model saving settings
        save_model_name = (
            (f"{self.model.__class__.__name__}-{data.num_features}-{data.n_class}.pt")
            if save_model_name is None
            else save_model_name
        )
        self.save_model_dir = save_model_dir
        self.save_model_name = save_model_name
        self.save_model_path = f"{save_model_dir}/{save_model_name}"

        # evaluation settings
        self.data_masks = {
            "train": data.train_mask.cpu().numpy(),
            "val": data.val_mask.cpu().numpy(),
            "test": data.test_mask.cpu().numpy(),
        }

    def model_update(self, epoch: int):
        """
        Performs a single update step for the model.
        """
        # extract necessary variables
        data = self.data
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion
        augmenter = self.augmenter
        x, y, edge_index = data.x, data.y, data.edge_index
        train_mask = data.train_mask

        # perform graph augmentation
        if epoch > self.warmup_epoch:
            x, edge_index, aug_runtime_info = augmenter.augment(model, x, edge_index)
            y, train_mask = augmenter.adapt_labels_and_train_mask(y, train_mask)

        # record runtime
        start_time = time.time()

        # set model in training mode and zero out gradients
        model.train()
        optimizer.zero_grad()

        # compute model output for input and edge indices
        output = model(x, edge_index)

        # compute loss on training nodes
        loss = criterion(output[train_mask], y[train_mask])

        # backpropagate the loss and update the model parameters
        loss.backward()
        optimizer.step()

        # record runtime
        used_time = time.time() - start_time
        if epoch > self.warmup_epoch:
            update_runtime_info = {"update_time(ms)": used_time * 1000}
            update_runtime_info.update(aug_runtime_info)
            self.runtime_info.append(update_runtime_info)

        # evaluate on validation set and adjust learning rate
        with torch.no_grad():
            model.eval()
            output = model(data.x, data.edge_index)
            val_loss = criterion(output[data.val_mask], data.y[data.val_mask])
        scheduler.step(val_loss)

        return

    def model_eval(self):
        """
        Evaluates the model on the validation and test sets.

        Returns:
        - results: dict
            Evaluation results containing various metrics.
        """
        # extract necessary variables
        data = self.data
        model = self.model
        metrics = self.eval_metrics
        criterion = self.criterion

        # set model in evaluation mode and compute logits
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)

        # obtain predicted labels and true labels as numpy arrays
        pred = logits.argmax(dim=1)
        y_pred = pred.cpu().numpy()
        y_true = data.y.cpu().numpy()

        # initialize a dictionary to store evaluation results
        results = {
            "loss": {
                data_name: criterion(logits[data_mask], data.y[data_mask]).item()
                for data_name, data_mask in self.data_masks.items()
            }
        }

        # loop over each evaluation metric and compute the metric for each dataset (train/val/test)
        for metric_name, (metric_func, metric_kwargs) in metrics.items():
            results[metric_name] = {
                data_name: metric_func(
                    y_true[data_mask], y_pred[data_mask], **metric_kwargs
                )
                for data_name, data_mask in self.data_masks.items()
            }

        # return the evaluation results
        return results

    def train(
        self,
        train_epoch: int = None,
        eval_freq: int = None,
        verbose_freq: int = None,
        return_best_model: bool = True,
    ):
        """
        Trains the node classification model and performs evaluation.

        Parameters:
        - train_epoch: int, optional
            Number of training epochs.
        - eval_freq: int, optional
            Frequency of evaluation during training.
        - verbose_freq: int, optional
            Frequency of verbose logging.

        Returns:
        - model: torch.nn.Module
            Trained node classification model.
        """
        # extract necessary variables
        model = self.model
        verbose_config = self.verbose_config
        save_model_path = self.save_model_path
        early_stop_flag = self.early_stop_flag
        early_stop_patience = self.early_stop_patience
        # decide whether to use default values or user-specified values
        train_epoch = self.train_epoch if train_epoch is None else train_epoch
        verbose_freq = self.verbose_freq if verbose_freq is None else verbose_freq
        eval_freq = self.eval_freq if eval_freq is None else eval_freq
        verbose_flag = verbose_freq is not None

        # parameter checks
        assert (
            isinstance(train_epoch, int) and train_epoch > 0
        ), "train_epoch must be a positive integer"
        assert (
            isinstance(eval_freq, int) and eval_freq > 0
        ), "eval_freq must be a positive integer"
        if verbose_flag:
            assert (
                isinstance(verbose_freq, int) and verbose_freq > 0
            ), "verbose_freq must be a positive integer"
            assert (
                verbose_freq % eval_freq == 0
            ), "verbose_freq must be a multiple of eval_freq"

        # basic training information
        training_info = f"Epoch: Train {train_epoch}, Eval {eval_freq}"
        if verbose_flag:
            training_info += f", Verbose {verbose_freq}"
        if early_stop_flag:
            training_info += f", EarlyStop {early_stop_patience}"
        training_info += f" | EvalMetrics: {list(self.eval_metrics.keys())}"
        if verbose_flag:
            print(training_info)

        # initialize
        self.runtime_info = []
        self.eval_scores = []
        self.valid_scores = []
        self.best_valid_score = -np.inf
        self.best_epoch = 0

        # loop over each epoch in the training phase
        epoch_range = (
            trange(1, train_epoch + 1, desc="Training")
            if self.tqdm_flag
            else range(1, train_epoch + 1)
        )
        for epoch in epoch_range:
            # update the model parameters
            self.model_update(epoch)

            # evaluate the model on the validation set every eval_freq epochs
            if eval_freq and epoch % eval_freq == 0:
                eval_results = self.model_eval()
                self.eval_scores.append([epoch, eval_results])

                # compute the average validation score for model selection
                valid_score = self.get_validation_score(eval_results)
                self.valid_scores.append([epoch, valid_score])

                # save the best model parameters
                if valid_score > self.best_valid_score:
                    self.best_epoch = epoch
                    self.best_valid_score = valid_score
                    # print (f"///// Best model parameters updated at epoch {epoch} /////")
                    torch.save(model.state_dict(), save_model_path)

                if early_stop_flag:
                    # stop training if the validation score does not improve for early_stop_rounds epochs
                    if epoch - self.best_epoch >= early_stop_patience:
                        if verbose_flag:
                            print(f"///// Early stopping at epoch {epoch} /////")
                        if self.tqdm_flag:
                            epoch_range.set_postfix_str(
                                f"Early stopping: patience ({early_stop_patience}) reached"
                            )
                        break

                # print the evaluation results
                if verbose_flag and epoch % verbose_freq == 0:
                    self.verbose(eval_results, epoch, verbose_config)

        # load the best model parameters and save best results
        if return_best_model:
            model.load_state_dict(torch.load(save_model_path))
        self.best_eval_results = self.model_eval()

        if verbose_flag:
            # print the evaluation results of the best model
            print(f"///// Best model parameters saved to '{save_model_path}' /////")
            self.print_best_results()

        # convert the runtime information to pandas DataFrame
        self.runtime_info = pd.DataFrame(self.runtime_info)
        self.valid_scores = pd.DataFrame(self.valid_scores)

        return model

    def print_best_results(self):
        """
        Prints the evaluation results of the best model.
        """
        # extract necessary variables
        verbose_config = self.verbose_config
        best_epoch = self.best_epoch
        best_eval_results = self.best_eval_results
        # print the evaluation results
        print("Best ", end="")
        self.verbose(best_eval_results, best_epoch, verbose_config, runtime="average")

        return

    def get_validation_score(self, eval_results: dict):
        """
        Computes the average validation score for model selection.

        Parameters:
        - eval_results: dict
            Evaluation results.

        Returns:
        - score: float
            Average validation score.
        """
        # return the average validation scores of all evaluation metrics
        return np.mean(
            [
                eval_results[metric_name]["val"]
                for metric_name in self.eval_metrics.keys()
            ]
        )

    def verbose(
        self,
        results: dict,
        epoch: int,
        verbose_config: dict,
        runtime: bool or str = None,
    ):
        """
        Prints verbose training progress information.

        Parameters:
        - results: dict
            Evaluation results for the epoch.
        - epoch: int
            Current epoch number.
        - verbose_config: dict
            Configuration for verbose logging.
        - runtime: bool or str, optional
            Whether to include runtime information.
        """
        # create a log string to print the evaluation results
        log = f"Epoch: {epoch:>4d} | "
        for dataset in verbose_config["datasets"]:
            log += f"{dataset}/"
        log = log[:-1] + " "
        for metric in verbose_config["metrics"]:
            log += f"| {metric.upper()}: "
            for dataset in verbose_config["datasets"]:
                # extract the score for each metric and dataset
                if metric == "loss":
                    score = f"{results[metric][dataset]:.3f}/"
                else:
                    score = results[metric][dataset] * 100
                    score = f"{score:.2f}/" if score < 100 else f"{score:.1f}/"
                log += score
            log = log[:-1] + " "

        # runtime information
        runtime = verbose_config["runtime"] if runtime is None else runtime
        if runtime:
            if runtime == "latest" or runtime == True:
                # print the latest runtime information
                runtime_info = self.runtime_info[-1]
            elif runtime == "average":
                # print the average runtime information
                runtime_info = pd.DataFrame(self.runtime_info).mean().to_dict()
            else:
                raise ValueError('runtime must be either "latest" or "average"')

            if self.augmenter:
                log += f"| upd/aug time: {runtime_info['update_time(ms)']:.2f}/{runtime_info['time_aug(ms)']:.2f}ms "
                log += f"| node/edge ratio: {runtime_info['node_ratio(%)']:.2f}/{runtime_info['edge_ratio(%)']:.2f}% "
                # log += f"| unc time: {runtime_info['time_unc(ms)']:.4f}ms "
                # log += f"| sim time: {runtime_info['time_sim(ms)']:.4f}ms "
            else:
                log += f"| upd time: {runtime_info['update_time(ms)']:.2f}ms "

        # print the log string
        print(log)

        return
