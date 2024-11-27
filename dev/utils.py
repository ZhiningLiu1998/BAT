import torch
import random
import numpy as np
import pandas as pd
import os.path as osp
import os

from collections import Counter
from torch_scatter import scatter_add
from torch_geometric.utils import dense_to_sparse

from routing import edge_sampling

from tqdm import tqdm


def pc_softmax(logits, cls_num):
    sample_per_class = torch.tensor(cls_num)
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits - spc.log()
    return logits


def get_test_amp_dmp_acc(
    data, model, class_num_list, test_mask, train_mask, amp_threshold=0.5, use_dist=False,
):
    high_amp_mask = torch.tensor(get_node_amp(data, threshold=amp_threshold)).to(
        test_mask.device
    )
    if use_dist:
        high_dmp_mask = torch.tensor(get_node_dmp_dist(data, train_mask)).to(
            test_mask.device
        )
    else:
        high_dmp_mask = torch.tensor(get_node_dmp(data, train_mask)).to(test_mask.device)
    y_pred = pc_softmax(model(data.x, data.edge_index), class_num_list).max(1)[1]
    y_true = data.y
    test_high_amp_mask = high_amp_mask & test_mask
    test_low_amp_mask = ~high_amp_mask & test_mask
    test_high_dmp_mask = high_dmp_mask & test_mask
    test_low_dmp_mask = ~high_dmp_mask & test_mask

    test_low_amp_acc = (
        y_pred[test_low_amp_mask].eq(y_true[test_low_amp_mask]).sum().item()
        / test_low_amp_mask.sum().item()
    )
    test_high_amp_acc = (
        y_pred[test_high_amp_mask].eq(y_true[test_high_amp_mask]).sum().item()
        / test_high_amp_mask.sum().item()
    )
    test_low_dmp_acc = (
        y_pred[test_low_dmp_mask].eq(y_true[test_low_dmp_mask]).sum().item()
        / test_low_dmp_mask.sum().item()
    )
    test_high_dmp_acc = (
        y_pred[test_high_dmp_mask].eq(y_true[test_high_dmp_mask]).sum().item()
        / test_high_dmp_mask.sum().item()
    )

    return test_low_amp_acc, test_high_amp_acc, test_low_dmp_acc, test_high_dmp_acc


def get_node_amp(data, threshold=0.3, verbose=False):
    adj = index_to_adj(data.x, data.edge_index, add_self_loop=False)
    node_het = get_node_neighbor_het_rate(data.y, adj)
    node_amp = node_het > threshold
    if verbose:
        print(f"Avg Node Heterogeneity: {node_het.mean()}")
        print(f"Threrhold: {threshold}")
        print(f"Counts Node AMP: {torch.tensor(node_amp).unique(return_counts=True)}")
    return node_amp


def get_node_dmp(data, train_mask, verbose=False):
    from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask

    y = data.y.cpu().numpy()
    label_idx = mask_to_index(train_mask).cpu().numpy()

    node_nearest_label = np.full(len(y), -1)
    node_nearest_label[label_idx] = y[label_idx]

    n_update = len(label_idx)

    for num_hop in range(1, 10):
        for node in label_idx:
            nbs, _, _, _ = k_hop_subgraph(
                node_idx=int(node),
                num_hops=num_hop,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
            )
            nbs = nbs.cpu().numpy()
            nb_mask = (
                index_to_mask(torch.tensor(nbs), size=data.num_nodes).cpu().numpy()
            )

            unvisit_mask = node_nearest_label == -1
            node_nearest_label[unvisit_mask & nb_mask] = y[node]

            n_update += unvisit_mask.sum()

    node_dmp = node_nearest_label != y
    if verbose:
        print(torch.tensor(node_nearest_label).unique(return_counts=True))
        print(torch.tensor(node_dmp).unique(return_counts=True))

    return node_dmp


def get_node_dmp_dist(data, train_mask, verbose=False):
    from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask

    y = data.y.cpu().numpy()
    label_idx = mask_to_index(train_mask).cpu().numpy()

    node_nearest_label = np.full(len(y), -1)
    node_nearest_label[label_idx] = y[label_idx]

    n_update = len(label_idx)

    for num_hop in range(1, 10):
        for node in label_idx:
            nbs, _, _, _ = k_hop_subgraph(
                node_idx=int(node),
                num_hops=num_hop,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
            )
            nbs = nbs.cpu().numpy()
            nb_mask = (
                index_to_mask(torch.tensor(nbs), size=data.num_nodes).cpu().numpy()
            )

            unvisit_mask = node_nearest_label == -1
            node_nearest_label[unvisit_mask & nb_mask] = num_hop

            n_update += unvisit_mask.sum()

    node_dmp = node_nearest_label > 3
    if verbose:
        print(torch.tensor(node_nearest_label).unique(return_counts=True))
        print(torch.tensor(node_dmp).unique(return_counts=True))

    return node_dmp


def compute_hops_to_nearest_labeled_nodes(data, train_mask):
    from torch_geometric.utils import k_hop_subgraph, mask_to_index

    y = data.y.cpu().numpy()
    label_idx = mask_to_index(train_mask).cpu().numpy()
    num_hops = np.zeros(data.num_nodes).astype(int)
    for node in tqdm(
        range(data.num_nodes), desc="Computing hops to nearest labeled node"
    ):
        node_label = y[node]
        num_hop = 0
        while True:
            nbs, _, _, _ = k_hop_subgraph(
                node, num_hop, data.edge_index, num_nodes=data.num_nodes
            )
            labeled_nbs = set(nbs.cpu().numpy()).intersection(set(label_idx))
            if len(labeled_nbs) > 0 or num_hop >= 10:
                ngb_labels = y[list(labeled_nbs)]
                label_correct = node_label in ngb_labels
                if label_correct or num_hop >= 10:
                    # print (f'Node {node} label {node_label} hop {num_hop} n_ngb {len(nbs)} labeled_ngb {labeled_nbs} ngb_label {ngb_labels} isin {isin}')
                    break
            num_hop += 1
        num_hops[node] = num_hop
    return num_hops


def get_confused_class(y_pred_proba):
    y_pred_top2 = torch.topk(y_pred_proba, 2, dim=1).indices
    return y_pred_top2[:, 1]


def get_node_risk(node_unc, y_pred):
    node_unc_class_mean = get_group_mean(node_unc, y_pred, reduce=False, norm=False)
    node_unc_class_dev = node_unc - node_unc_class_mean
    return node_unc_class_dev.clip(min=0)


def add_edge_confuse(data, y_pred_proba, train_mask, node_risk):
    n_node, n_node_train = len(data.x), train_mask.sum().item()
    y_pred_confuse = get_confused_class(y_pred_proba)
    y_tile_train = torch.tile(data.y[train_mask], (n_node, 1)).T
    y_tile_confuse = torch.tile(y_pred_confuse, (n_node_train, 1))
    train_to_confuse_weight = (y_tile_train == y_tile_confuse).float()
    train_to_confuse_weight *= torch.tile(node_risk, (n_node_train, 1))
    weight_mat = torch.cuda.FloatTensor(n_node, n_node).fill_(0)
    weight_mat[train_mask] = train_to_confuse_weight
    edge_index_candidates, edge_sampling_weights = dense_to_sparse(weight_mat)
    add_edge_index = edge_sampling(
        sampling="indpt",
        edge_index=edge_index_candidates,
        weights=edge_sampling_weights,
        alpha=0,
        sparsity=0,
    )["edge_index"]
    return add_edge_index


def print_top_prediction_accuracy(model, data):
    y_true = data.y.clone()
    y_pred_proba = predict_proba_tensor(model, data.x, data.edge_index)
    y_pred_sorted = y_pred_proba.argsort(dim=1, descending=True)
    for rank in range(y_pred_sorted.shape[1]):
        y_pred = y_pred_sorted[:, rank]
        acc = (y_pred == y_true).float().mean()
        print(f"#{rank} confused class accuracy {acc:.2%}")


def add_edges(data, y_pred, train_mask, node_risk):
    n_node = len(data.x)
    adj_ds = index_to_adj(
        data.x,
        data.edge_index,
        add_self_loop=False,
        remove_self_loop=True,
        sparse=False,
    )
    adj_sp = index_to_adj(
        data.x, data.edge_index, add_self_loop=False, sparse=True
    ).float()
    adj_2hop = (adj_sp @ adj_sp).bool().to_dense()
    adj_2hop = (adj_2hop * ~adj_ds).fill_diagonal_(False)
    train_to_not_train_mask = torch.tile(train_mask, (n_node, 1)).T & torch.tile(
        ~train_mask, (n_node, 1)
    )
    adj_2hop = adj_2hop * train_to_not_train_mask
    candidate_edge_index, _ = dense_to_sparse(adj_2hop)
    edge_label_consis_mask = (
        y_pred[candidate_edge_index[0]] == y_pred[candidate_edge_index[1]]
    )
    candidate_edge_index = candidate_edge_index[:, edge_label_consis_mask]

    src_risk, tar_risk = (
        node_risk[candidate_edge_index[0]],
        node_risk[candidate_edge_index[1]],
    )
    edge_risk = torch.stack([src_risk, tar_risk]).max(0).values
    edge_sampling_weights = 1 - edge_risk
    add_edge_index = edge_sampling(
        sampling="indpt",
        edge_index=candidate_edge_index,
        weights=edge_sampling_weights,
        alpha=0,
        sparsity=0,
    )["edge_index"]
    return add_edge_index


def get_index_with_two_hop_consis_connections(adj, y):
    adj_two_hop = (
        adj.matmul(adj)
        .masked_fill_(torch.eye(len(adj), len(adj)).byte().to(adj.device), 0)
        .bool()
    )
    y_tile = torch.tile(y, (len(adj), 1))
    y_consis_mask = y_tile.T == y_tile
    adj_two_hop = adj_two_hop & y_consis_mask | adj.bool()
    new_edge_index, _ = dense_to_sparse(adj_two_hop)
    return new_edge_index


# def add_edges(adj, y_pred, train_mask, node_risk):
#     n_node = len(y_pred)
#     adj_ = set_diagonal_zero(adj).float()
#     adj_2hop = set_diagonal_zero(adj_ @ adj_)
#     train_to_not_train_mask = torch.tile(train_mask, (n_node, 1)).T & torch.tile(~train_mask, (n_node, 1))
#     adj_2hop = adj_2hop * train_to_not_train_mask
#     candidate_edge_index, _ = dense_to_sparse(adj_2hop)
#     edge_label_consis_mask = y_pred[candidate_edge_index[0]] == y_pred[candidate_edge_index[1]]
#     candidate_edge_index = candidate_edge_index[:, edge_label_consis_mask]

#     src_risk, tar_risk = node_risk[candidate_edge_index[0]], node_risk[candidate_edge_index[1]]
#     edge_risk = torch.stack([src_risk, tar_risk]).max(0).values
#     edge_sampling_weights = 1 - edge_risk
#     add_edge_index = edge_sampling(sampling='indpt', edge_index=candidate_edge_index,
#                                 weights=edge_sampling_weights, alpha=0, sparsity=0)['edge_index']
#     return add_edge_index


def get_class_weight(class_num_list, y, reduce=False):
    class_weight = 1 / (
        torch.tensor(class_num_list) / torch.tensor(class_num_list).min()
    )
    class_weight = class_weight.to(y.device)
    if reduce:
        return class_weight
    else:
        return class_weight[y]


def get_class_risk(class_num_list, y, reduce=False):
    class_risk = torch.tensor(class_num_list) / torch.tensor(class_num_list).max()
    class_risk = class_risk.to(y.device)
    if reduce:
        return class_risk
    else:
        return class_risk[y]


def get_group_mean(
    values: torch.Tensor,
    groups: torch.Tensor,
    reduce: bool = False,
    num_unique: int = None,
    norm=False,
):
    group_unique = groups.unique().cpu().numpy()
    if reduce:
        num_unique = len(group_unique) if num_unique is None else num_unique
        mean_values = torch.zeros(num_unique, dtype=torch.float32)
        for i in group_unique:
            mean_values[i] = values[groups == i].mean()
        return mean_values
    else:
        new_values = torch.zeros_like(values)
        for i in group_unique:
            mask = groups == i
            new_values[mask] = values[mask].mean()
        if norm:
            new_values /= new_values.mean()
        return new_values


def get_confused_class(y_pred_proba):
    y_pred_top2 = torch.topk(y_pred_proba, 2, dim=1).indices
    return y_pred_top2[:, 1]


@torch.no_grad()
def get_connectivity_distribution(y_pred, adj, n_cls):
    def batched_bincount(x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=torch.int, device=x.device)
        values = torch.ones_like(x, dtype=torch.int, device=x.device)
        target.scatter_add_(dim, x, values)
        return target

    y_pred_mat = y_pred.mul(adj)
    y_pred_mat[~(adj.bool())] = n_cls
    batched_count = batched_bincount(y_pred_mat, 1, n_cls + 1)[:, :n_cls].float()
    batched_count /= batched_count.sum(axis=1).reshape(-1, 1)
    return batched_count.nan_to_num(0)


def get_all_stats(model, data):
    n_cls = len(data.y.unique())
    adj = index_to_adj(data.x, data.edge_index, add_self_loop=True)
    y_pred = predict_tensor(model, data.x, data.edge_index)
    y_pred_proba = predict_proba_tensor(model, data.x, data.edge_index)

    connect_distr = get_connectivity_distribution(y_pred, adj, n_cls)
    pred_conn_discrepancy = (y_pred_proba - connect_distr).abs().sum(axis=1) / 2
    pred_conn_discrepancy_class_mean = get_group_mean(
        pred_conn_discrepancy, y_pred, reduce=False, norm=False
    )
    pred_conn_discrepancy_class_dev = (
        pred_conn_discrepancy - pred_conn_discrepancy_class_mean
    )
    pred_conn_discrepancy_class_dev_clip = pred_conn_discrepancy_class_dev.clip(min=0)

    y_pred_discrepancy = (
        -(y_pred_proba - connect_distr).gather(1, y_pred.reshape(-1, 1)).flatten()
    )
    y_pred_discrepancy_class_mean = get_group_mean(
        y_pred_discrepancy, y_pred, reduce=False, norm=False
    )
    y_pred_discrepancy_class_dev = y_pred_discrepancy - y_pred_discrepancy_class_mean
    y_pred_discrepancy_class_dev_clip = y_pred_discrepancy_class_dev.clip(min=0)

    node_ngb_het = get_node_neighbor_het_rate(data.y, adj)
    node_ngb_het_pred = get_node_neighbor_het_rate(y_pred, adj)
    node_ngb_het_pred_class_mean = get_group_mean(
        node_ngb_het_pred, y_pred, reduce=False, norm=False
    )
    node_ngb_het_pred_class_dev = node_ngb_het_pred - node_ngb_het_pred_class_mean
    node_ngb_het_pred_class_dev_clip = node_ngb_het_pred_class_dev.clip(min=0)
    node_ngb_het_pred_class_dev_abs = node_ngb_het_pred_class_dev.abs()
    node_ngb_label_uniq = get_node_neighbor_label_uniq(data.y, adj)
    node_ngb_label_uniq_pred = get_node_neighbor_label_uniq(y_pred, adj)
    node_acc = (y_pred == data.y).float()
    node_confused_acc = (get_confused_class(y_pred_proba) == data.y).float()
    node_ngb_acc = get_node_neighbor_acc(data.y, y_pred, adj)
    node_error = 1 - predict_proba_target_tensor(model, data.x, data.edge_index, data.y)
    node_unc = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="self", norm=False
    )
    node_unc_class_mean = get_group_mean(node_unc, y_pred, reduce=False, norm=False)
    node_unc_class_dev = node_unc - node_unc_class_mean
    node_unc_class_dev_clip = node_unc_class_dev.clip(min=0)
    node_unc_class_dev_abs = node_unc_class_dev.abs()
    node_unc_class_dev_pos = node_unc_class_dev - node_unc_class_dev.min()
    node_unc_class_dev_pos /= node_unc_class_dev_pos.max()
    node_unc_ngb = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="ngb-mean", norm=False
    )
    node_unc_ngb_consis = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="consis-ngb-mean", norm=False
    )
    node_unc_mean = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="mean", norm=False
    )
    node_unc_max = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="max", norm=False
    )
    node_unc_dev = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="deviation", norm=False
    )
    node_unc_dev_class_mean = get_group_mean(
        node_unc_dev, y_pred, reduce=False, norm=False
    )
    node_unc_dev_class_dev = node_unc_dev - node_unc_dev_class_mean
    node_unc_dev_class_dev_clip = node_unc_dev_class_dev.clip(min=0)
    node_unc_dev_class_dev_abs = node_unc_dev_class_dev.abs()
    node_unc_dev_abs = node_unc_dev.abs()
    node_unc_dev_abs_class_mean = get_group_mean(
        node_unc_dev_abs, y_pred, reduce=False, norm=False
    )
    node_unc_dev_abs_class_dev = node_unc_dev_abs - node_unc_dev_abs_class_mean
    node_unc_dev_abs_class_dev_clip = node_unc_dev_abs_class_dev.clip(min=0)
    node_unc_dev_abs_class_dev_abs = node_unc_dev_abs_class_dev.abs()
    node_unc_dev_consis = get_subgraph_unc(
        model, data.x, data.edge_index, reduce="consis-deviation", norm=False
    )
    node_unc_dev_consis_class_mean = get_group_mean(
        node_unc_dev_consis, y_pred, reduce=False, norm=False
    )
    node_unc_dev_consis_class_dev = node_unc_dev_consis - node_unc_dev_consis_class_mean
    node_unc_dev_consis_class_dev_clip = node_unc_dev_consis_class_dev.clip(min=0)
    node_unc_dev_consis_class_dev_abs = node_unc_dev_consis_class_dev.abs()
    node_unc_dev_consis_abs = node_unc_dev_consis.abs()
    het_pred_unc_dev = node_ngb_het_pred * node_unc_dev_abs
    label_uniq_unc_dev = node_ngb_label_uniq_pred * node_unc_dev_abs
    all_stats = {
        "pred_conn_discrepancy": pred_conn_discrepancy,
        "pred_conn_discrepancy_class_dev": pred_conn_discrepancy_class_dev,
        "pred_conn_discrepancy_class_dev_clip": pred_conn_discrepancy_class_dev_clip,
        "y_pred_discrepancy": y_pred_discrepancy,
        "y_pred_discrepancy_class_dev": y_pred_discrepancy_class_dev,
        "y_pred_discrepancy_class_dev_clip": y_pred_discrepancy_class_dev_clip,
        "node_ngb_het": node_ngb_het,
        "node_ngb_het_pred": node_ngb_het_pred,
        "node_ngb_het_pred_class_dev": node_ngb_het_pred_class_dev,
        "node_ngb_het_pred_class_dev_abs": node_ngb_het_pred_class_dev_abs,
        "node_ngb_het_pred_class_dev_clip": node_ngb_het_pred_class_dev_clip,
        "node_ngb_label_uniq": node_ngb_label_uniq,
        "node_ngb_label_uniq_pred": node_ngb_label_uniq_pred,
        "node_acc": node_acc,
        "node_confused_acc": node_confused_acc,
        "node_ngb_acc": node_ngb_acc,
        "node_error": node_error,
        "node_unc": node_unc,
        "node_unc_class_dev": node_unc_class_dev,
        "node_unc_class_dev_clip": node_unc_class_dev_clip,
        "node_unc_class_dev_abs": node_unc_class_dev_abs,
        "node_unc_class_dev_pos": node_unc_class_dev_pos,
        "node_unc_ngb": node_unc_ngb,
        "node_unc_ngb_consis": node_unc_ngb_consis,
        "node_unc_mean": node_unc_mean,
        "node_unc_max": node_unc_max,
        "node_unc_dev": node_unc_dev,
        "node_unc_dev_abs": node_unc_dev.abs(),
        "node_unc_dev_class_dev": node_unc_dev_class_dev,
        "node_unc_dev_class_dev_clip": node_unc_dev_class_dev_clip,
        "node_unc_dev_class_dev_abs": node_unc_dev_class_dev_abs,
        "node_unc_dev_abs_class_dev": node_unc_dev_abs_class_dev,
        "node_unc_dev_abs_class_dev_clip": node_unc_dev_abs_class_dev_clip,
        "node_unc_dev_abs_class_dev_abs": node_unc_dev_abs_class_dev_abs,
        "node_unc_dev_clip": node_unc_dev.clip(min=0),
        "node_unc_dev_consis": node_unc_dev_consis,
        "node_unc_dev_consis_class_dev": node_unc_dev_consis_class_dev,
        "node_unc_dev_consis_class_dev_clip": node_unc_dev_consis_class_dev_clip,
        "node_unc_dev_consis_class_dev_abs": node_unc_dev_consis_class_dev_abs,
        "node_unc_dev_consis_abs": node_unc_dev_consis.abs(),
        "node_unc_dev_consis_clip": node_unc_dev_consis.clip(min=0),
        "het_pred_unc_dev": het_pred_unc_dev,
        "label_uniq_unc_dev": label_uniq_unc_dev,
    }
    return pd.DataFrame({k: v.cpu().numpy() for k, v in all_stats.items()})


def get_maskes(model, data, data_train_mask):
    labels, counts = sort_by_count(data.y[data_train_mask])
    minor_labels = torch.tensor(labels[counts == counts.min()]).to(data.y.device)
    major_labels = torch.tensor(labels[counts == counts.max()]).to(data.y.device)
    assert len(minor_labels) + len(major_labels) == len(labels)
    y_pred = predict_tensor(model, data.x, data.edge_index)
    mask_space = {
        "labeled": data_train_mask,
        "unlabeled": ~data_train_mask,
        "unlabeled-minor": ~data_train_mask & torch.isin(data.y, minor_labels),
        "unlabeled-major": ~data_train_mask & torch.isin(data.y, major_labels),
        "unlabeled-minor-true": ~data_train_mask
        & torch.isin(data.y, minor_labels)
        & (y_pred == data.y),
        "unlabeled-minor-false": ~data_train_mask
        & torch.isin(data.y, minor_labels)
        & (y_pred != data.y),
        "unlabeled-minor-false-minor": ~data_train_mask
        & torch.isin(data.y, minor_labels)
        & (y_pred != data.y)
        & torch.isin(y_pred, minor_labels),
        "unlabeled-minor-false-major": ~data_train_mask
        & torch.isin(data.y, minor_labels)
        & (y_pred != data.y)
        & torch.isin(y_pred, major_labels),
        "unlabeled-major-true": ~data_train_mask
        & torch.isin(data.y, major_labels)
        & (y_pred == data.y),
        "unlabeled-major-false": ~data_train_mask
        & torch.isin(data.y, major_labels)
        & (y_pred != data.y),
        "unlabeled-major-false-minor": ~data_train_mask
        & torch.isin(data.y, major_labels)
        & (y_pred != data.y)
        & torch.isin(y_pred, minor_labels),
        "unlabeled-major-false-major": ~data_train_mask
        & torch.isin(data.y, major_labels)
        & (y_pred != data.y)
        & torch.isin(y_pred, major_labels),
    }
    return pd.DataFrame({k: v.cpu().numpy() for k, v in mask_space.items()})


def linear_conv_with_adptive_padding(values, window_size, padding_mean_range=5):
    if torch.is_tensor(values):
        values = values.cpu().numpy()
    if window_size == 1:
        return values
    padding_size = window_size // 2
    left_padding, right_padding = (
        values[:padding_mean_range].mean(),
        values[-padding_mean_range:].mean(),
    )
    padded_values = np.concatenate(
        [
            np.full(padding_size, left_padding),
            values,
            np.full(padding_size - 1, right_padding),
        ]
    )
    conv_values = np.convolve(
        padded_values, np.ones(window_size) / window_size, mode="valid"
    )
    return conv_values


def get_class_edge_connectivity_mat(y, edge_index, n_cls=None):
    if n_cls is None:
        n_cls = y.max().item() + 1
    y_edges = torch.stack((y[edge_index[0]], y[edge_index[1]])).T
    edge_type, edge_count = y_edges.unique(dim=0, return_counts=True)
    y_connect_mat = torch.zeros((n_cls, n_cls)).int()
    for index, value in zip(edge_type, edge_count):
        y_connect_mat[index[0]][index[1]] = value
    return y_connect_mat.to(edge_index.device)


def get_node_neighbor_acc(y_true, y_pred, adj):
    node_acc = (y_pred == y_true).float()
    acc_tile = torch.tile(node_acc, (len(node_acc), 1))
    return (acc_tile * adj).sum(axis=1) / adj.sum(axis=1)


def get_node_neighbor_het_rate(y, adj):
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    y = y.to(adj.device)
    y_tile = torch.tile(y, (len(y), 1))
    ngb_label_mat = (adj * y_tile).float()
    ngb_label_mat[adj == 0] = torch.nan
    node_ngb_consis = (ngb_label_mat == y_tile.T).sum(axis=1) / adj.sum(axis=1)
    node_ngb_consis = node_ngb_consis.nan_to_num(0)  # handle 0 degree nodes
    node_ngb_het = 1 - node_ngb_consis
    return node_ngb_het


def get_node_neighbor_label_uniq(y, adj):
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    y = y.to(adj.device)
    y_tile = torch.tile(y, (len(y), 1))
    ngb_label_mat = (adj * y_tile).float()
    ngb_label_mat[adj == 0] = torch.nan
    node_ngb_label_uniq = torch.tensor(
        [
            len(ngb_label_mat[i][~ngb_label_mat[i].isnan()].unique())
            for i in range(len(ngb_label_mat))
        ]
    ).to(adj.device)
    return node_ngb_label_uniq


def plot_sorted(values):
    import seaborn as sns

    if torch.is_tensor(values):
        data = torch.sort(values).values.cpu()
    else:
        data = sorted(values)
    sns.lineplot(data=data)


def get_confusion_matrix(y_true, y_pred, mask):
    from sklearn.metrics import confusion_matrix

    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    conf_mat = confusion_matrix(y_true[mask], y_pred[mask])
    return conf_mat


def get_class_wise_accuracy(y_true, y_pred, mask, debug=False):
    conf_mat = get_confusion_matrix(y_true, y_pred, mask)
    class_wise_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    class_wise_acc = np.nan_to_num(class_wise_acc, nan=0)
    info = "Class-wise Accuracy"
    for cls in range(len(class_wise_acc)):
        info += f" | {cls}: {class_wise_acc[cls]:.2%}"
    if debug:
        print(info)
    return class_wise_acc


def get_class_inverse_weight(labels, train_mask):
    labels = labels[train_mask]
    from sklearn.utils.class_weight import compute_class_weight

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights /= class_weights.mean()
    return torch.tensor(class_weights, dtype=torch.float32)


def predict_proba_target_tensor(model, x, edge_index, y):
    y_pred_proba = predict_proba_tensor(model, x, edge_index)
    return y_pred_proba.gather(1, y.reshape(-1, 1)).flatten()


def predict_proba_tensor(model, x, edge_index):
    out = model.forward(x, edge_index)
    return torch.softmax(out, dim=1).detach()


def predict_tensor(model, x, edge_index):
    out = model.forward(x, edge_index)
    return out.argmax(dim=1).detach()


def predict_proba(model, x, edge_index):
    out = model.forward(x, edge_index)
    return torch.softmax(out, dim=1).cpu().detach().numpy()


def predict(model, x, edge_index):
    out = model.forward(x, edge_index)
    return out.argmax(dim=1).cpu().numpy()


def get_pred_consistency_matrix(model, x, edge_index, debug=False):
    y_pred = predict_tensor(model, x, edge_index)
    edge_index = edge_index.clone().cpu()
    y_pred_mat = torch.tile(y_pred, (len(y_pred), 1))
    pred_consis_mat = y_pred_mat == y_pred_mat.T
    if debug:
        print(
            f"{pred_consis_mat.sum().item()/pred_consis_mat.numel():.3%} consistency in pred_consistency_matrix."
        )
    return pred_consis_mat


def get_edge_predict_consistency(model, x, edge_index, y, y_pred=None, debug=False):
    y_pred = predict_tensor(model, x, edge_index) if y_pred is None else y_pred
    edge_consis = (y_pred[edge_index[0]] == y_pred[edge_index[1]]).float()
    # if debug:
    #     true_edge_consis = (y[edge_index[0]] == y[edge_index[1]]).float()
    #     print (
    #         f'{edge_consis.sum()/edge_consis.shape[0]:.3%} predict consistent edges '
    #         f'({(edge_consis == true_edge_consis).sum()/true_edge_consis.shape[0]:.3%} estimation accuracy)'
    #         # f'{true_edge_consis.sum()/true_edge_consis.shape[0]:.3%} edges are indeed consistent\n'
    #     )
    return edge_consis


def get_unc_diff_matrix(node_unc):
    tar_unc_mat = torch.tile(node_unc.cpu(), (len(node_unc), 1))
    src_unc_mat = tar_unc_mat.T
    unc_diff_mat = tar_unc_mat - src_unc_mat
    unc_diff_mat = unc_diff_mat.clip(min=0)
    unc_diff_mat -= unc_diff_mat.min()
    unc_diff_mat /= unc_diff_mat.max()
    return unc_diff_mat


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


def get_train_class_num_list(y, train_mask):
    labels = y[train_mask].cpu().numpy()
    class_num_list = []
    for i in np.unique(labels):
        class_num_list.append((labels == i).sum())
    return class_num_list


def get_node_unc(model, x, edge_index, how="linear"):
    pred_proba = predict_proba_tensor(model, x, edge_index)
    if how == "linear":
        node_unc = 1 - pred_proba.max(axis=1).values
    elif how == "inverse":
        node_unc = 1 / pred_proba.max(axis=1).values
    else:
        raise NotImplementedError
    return node_unc


def index_to_adj(
    x, edge_index, add_self_loop=False, remove_self_loop=False, sparse=False
):
    from torch_geometric.utils import to_dense_adj

    assert not (add_self_loop == True and remove_self_loop == True)
    num_nodes = len(x)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].bool()
    if add_self_loop:
        adj.fill_diagonal_(True)
    if remove_self_loop:
        adj.fill_diagonal_(False)
    if sparse:
        adj = adj.to_sparse()
    return adj


def get_subgraph_unc(model, x, edge_index, reduce="mean", how="linear", norm=True):
    adj = index_to_adj(x, edge_index, add_self_loop=True)
    node_unc = get_node_unc(model, x, edge_index, how=how)
    subgraph_unc_tile = adj * torch.tile(node_unc, (len(node_unc), 1))
    if reduce == "mean":
        subgraph_unc = subgraph_unc_tile.sum(axis=1) / adj.sum(axis=1)
    elif reduce == "ngb-mean":
        subgraph_unc = (subgraph_unc_tile.sum(axis=1) - node_unc) / (
            adj.sum(axis=1) - 1
        )
    elif reduce == "max":
        subgraph_unc = subgraph_unc_tile.max(axis=1).values
    elif reduce == "self":
        subgraph_unc = node_unc
    elif reduce == "deviation":
        subgraph_unc = node_unc - (subgraph_unc_tile.sum(axis=1) - node_unc) / (
            adj.sum(axis=1) - 1
        )
    elif reduce == "consis-ngb-mean":
        y_pred = predict_tensor(model, x, edge_index)
        y_pred_tile = torch.tile(y_pred, (len(y_pred), 1))
        y_pred_consis_tile = y_pred_tile == y_pred_tile.T
        subgraph_unc_tile = subgraph_unc_tile * y_pred_consis_tile
        subgraph_unc = (subgraph_unc_tile.sum(axis=1) - node_unc) / (
            (adj * y_pred_consis_tile).sum(axis=1) - 1
        )
        subgraph_unc = subgraph_unc.nan_to_num(0)
    elif reduce == "consis-deviation":
        y_pred = predict_tensor(model, x, edge_index)
        y_pred_tile = torch.tile(y_pred, (len(y_pred), 1))
        y_pred_consis_tile = y_pred_tile == y_pred_tile.T
        subgraph_unc_tile = subgraph_unc_tile * y_pred_consis_tile
        subgraph_unc = node_unc - (subgraph_unc_tile.sum(axis=1) - node_unc) / (
            (adj * y_pred_consis_tile).sum(axis=1) - 1
        )
        subgraph_unc = subgraph_unc.nan_to_num(0)
    else:
        raise NotImplementedError
    if norm:
        subgraph_unc -= subgraph_unc.min()
        subgraph_unc /= subgraph_unc.mean()
    return subgraph_unc


class ResultsCollector:
    def __init__(self):
        self.results = []

    def parse(self):
        metrics = list(self.results[0].keys())
        sets = list(list(self.results[0].values())[0].keys())
        return metrics, sets

    def get_results_dataframe(self):
        metrics, sets = self.parse()
        result_inds = pd.MultiIndex.from_product(
            [metrics, sets], names=["metric", "set"]
        )
        result_values = []
        for ind in result_inds:
            result_values_row = []
            for record in self.results:
                result_values_row.append(record[ind[0]][ind[1]])
                # print (record[ind[0]][ind[1]])
            result_values.append(result_values_row)
        self.results_df = pd.DataFrame(result_values, index=result_inds)
        return self.results_df

    def get_test_results_dataframe(self):
        results_df = self.get_results_dataframe().T
        results_df = results_df.T.loc[(slice(None), "test"), :].T
        results_df.columns = [x[0] for x in results_df.columns.ravel()]
        return results_df

    def describe(self):
        results_df = self.get_results_dataframe()
        return results_df.T.describe().loc[["mean", "std"]]

    def describe_test(self, test_name="test", print_info=True):
        import warnings

        warnings.filterwarnings("ignore")
        describe_df = self.describe()
        describe_test_df = describe_df.T.loc[(slice(None), test_name), :].T
        describe_test_df.columns = [x[0] for x in describe_test_df.columns.ravel()]
        if print_info:
            info = "Best test scores:"
            for metric in describe_test_df.columns:
                res = describe_test_df[metric]
                mean, std = res["mean"], res["std"]
                info += f" | {metric} {mean:.3f}-{std:.3f}"
            print("////// " + info + " //////")
        return describe_test_df

    def collect(self, results):
        self.results.append(results)


def idx_to_mask(idx, n: int):
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def get_order(values, by="count", descending=True):
    if by == "count":
        sort_key = lambda x: x[1]
    elif by == "dict":
        sort_key = lambda x: x[0]
    else:
        raise NotImplementedError

    count_list = np.array(
        sorted(
            [[k, v] for k, v in Counter(values).items()],
            key=sort_key,
            reverse=descending,
        )
    )
    order, count = count_list[:, 0], count_list[:, 1].astype(int)
    return order, count


def make_few_shot(
    labels,
    head_sample_num,
    tail_class_num,
    imb_ratio,
    val_sample_num,
    test_sample_num,
    random_seed=None,
):
    labels, label_map = reorder_label_by_count(labels)
    c_train_num = get_manual_train_num(
        labels, head_sample_num, tail_class_num, imb_ratio
    )
    train_idx, val_idx, test_idx, c_num_mat = split_manual(
        labels, label_map, c_train_num, val_sample_num, test_sample_num, random_seed
    )
    train_mask = idx_to_mask(train_idx, len(labels))
    val_mask = idx_to_mask(val_idx, len(labels))
    test_mask = idx_to_mask(test_idx, len(labels))
    return train_mask, val_mask, test_mask, c_num_mat


def get_manual_train_num(labels, head_sample_num, tail_class_num, imb_ratio):
    c_train_num = []
    for i in range(labels.max().item() + 1):
        if (
            i > labels.max().item() - tail_class_num
        ):  # last classes belong to minority classes
            c_train_num.append(int(head_sample_num * imb_ratio))
        else:
            c_train_num.append(head_sample_num)
    return c_train_num


def sort_by_count(values, by="count", descending=True):
    if torch.is_tensor(values):
        values = values.cpu().numpy()
    if by == "count":
        sort_key = lambda x: x[1]
    elif by == "dict":
        sort_key = lambda x: x[0]
    else:
        raise NotImplementedError

    count_list = np.array(
        sorted(
            [[k, v] for k, v in Counter(values).items()],
            key=sort_key,
            reverse=descending,
        )
    )
    order, count = count_list[:, 0], count_list[:, 1].astype(int)
    return order, count


def reorder_label_by_count(labels, debug=False):
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


def split_manual(
    labels,
    label_map,
    c_train_num,
    val_sample_num,
    test_sample_num,
    random_seed=None,
    debug=False,
):
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    # cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)
    c_num_mat[:, 1] = val_sample_num
    c_num_mat[:, 2] = test_sample_num

    for i in range(num_classes):
        idx = list(label_map.keys())[list(label_map.values()).index(i)]
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        if debug:
            # print('ORG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
            print(
                f"ORG:{idx:d} -> NEW:{i:d} "
                f"train {c_train_num[i]} val {c_num_mat[i, 1]} test {c_num_mat[i, 2]}"
            )
        random.seed(random_seed)
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[: c_train_num[i]]
        c_num_mat[i, 0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i] : c_train_num[i] + c_num_mat[i, 1]]
        test_idx = (
            test_idx
            + c_idx[
                c_train_num[i]
                + c_num_mat[i, 1] : c_train_num[i]
                + c_num_mat[i, 1]
                + c_num_mat[i, 2]
            ]
        )

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat


def split_manual_lt(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)
    c_num_mat[:, 1] = 25
    c_num_mat[:, 2] = 55

    for i in range(num_classes):
        c_idx = (labels[idx_train] == i).nonzero()[:, -1].tolist()
        print("{:d}-th class sample number: {:d}".format(i, len(c_idx)))
        val_lists = list(map(int, idx_val[labels[idx_val] == i]))
        test_lists = list(map(int, idx_test[labels[idx_test] == i]))
        random.shuffle(val_lists)
        random.shuffle(test_lists)

        c_num_mat[i, 0] = len(c_idx)

        val_idx = val_idx + val_lists[: c_num_mat[i, 1]]
        test_idx = test_idx + test_lists[: c_num_mat[i, 2]]

    train_idx = torch.LongTensor(idx_train)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat


def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    # Check whether inv_indices is correct
    assert (
        torch.arange(len(n_data))[indices][torch.tensor(inv_indices)]
        - torch.arange(len(n_data))
    ).sum().abs() < 1e-12

    mu = np.power(1 / ratio, 1 / (n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        # Check whether the number of class is greater than or equal to 1
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(
            int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i]))
        )
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)

    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]
    # print(class_num_list);input()

    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) * original_mask])

    for i in indices.numpy():
        for r in range(1, n_round[i] + 1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list, [])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = (row_mask * col_mask).type(torch.bool)

            # Compute degree
            degree = scatter_add(torch.ones_like(row[edge_mask]), row[edge_mask]).to(
                row.device
            )
            if len(degree) < len(label):
                degree = torch.cat(
                    [degree, degree.new_zeros(len(label) - len(degree))], dim=0
                )
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            _, remove_idx = torch.topk(
                degree, (r * remove_class_num_list[i]) // n_round[i], largest=False
            )
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = (row_mask * col_mask).type(torch.bool)

    train_mask = (node_mask * train_mask).type(torch.bool)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) * train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def seed_everything(seed=0):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
