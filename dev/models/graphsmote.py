import numpy as np
import torch
import torch.nn.functional as F
import random
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GSMOTEDecoder(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nembed, dropout=0.1):
        super(GSMOTEDecoder, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):

        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))

        return adj_out


def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, tail_classes=[]):
    assert len(tail_classes) > 0
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None

    for i in tail_classes:
        target_label = i
        chosen = idx_train[(labels == target_label)[idx_train]]
        if chosen.shape[0] == 0:
            raise ValueError(f"Label {target_label} has 0 instances in given {labels}.")
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = (
                embed[chosen, :]
                + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            )

            new_labels = (
                labels.new(torch.Size((chosen.shape[0], 1)))
                .reshape(-1)
                .fill_(target_label)
            )
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(
                        torch.clamp_(
                            adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0
                        )
                    )
                else:
                    temp = adj.new(
                        torch.clamp_(
                            adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0
                        )
                    )
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(
            torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))
        ).fill_(0.0)
        new_adj[: adj.shape[0], : adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0] :, : adj.shape[0]] = adj_new[:, :]
        new_adj[: adj.shape[0], adj.shape[0] :] = torch.transpose(adj_new, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt, adj_mask=None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2

    neg_weight = edge_num / (total_num - edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss


from torch_geometric.utils import to_dense_adj, dense_to_sparse, mask_to_index


def src_upsample(adj, features, labels, idx_train, portion=1.0, tail_classes=[]):
    assert len(tail_classes) > 0
    c_largest = labels.max().item()
    adj_back = to_dense_adj(adj, max_num_nodes=features.shape[0])[0]
    chosen = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for target_label in tail_classes:
        new_chosen = idx_train[(labels == (target_label))[idx_train]]
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

            num = int(new_chosen.shape[0] * portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(
        torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num))
    )
    new_adj[: adj_back.shape[0], : adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0] :, : adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[: adj_back.shape[0], adj_back.shape[0] :] = adj_back[:, chosen]
    new_adj[adj_back.shape[0] :, adj_back.shape[0] :] = adj_back[chosen, :][:, chosen]

    # ipdb.set_trace()
    features_append = deepcopy(features[chosen, :])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train


def src_upsample_(adj, features, labels, idx_train, portion=1.0, tail_classes=[]):
    assert len(tail_classes) > 0
    c_largest = labels.max().item()
    chosen = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for target_label in tail_classes:
        new_chosen = idx_train[(labels == (target_label))[idx_train]]
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

            num = int(new_chosen.shape[0] * portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)

    add_num = chosen.shape[0]

    # ipdb.set_trace()
    features_append = deepcopy(features[chosen, :])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(len(labels), len(labels) + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return adj, features, labels, idx_train


def src_smote(adj, features, labels, idx_train, portion=1.0, tail_classes=[]):
    assert len(tail_classes) > 0
    c_largest = labels.max().item()
    adj_back = to_dense_adj(adj, max_num_nodes=features.shape[0])[0]
    chosen = None
    new_features = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for target_label in tail_classes:
        new_chosen = idx_train[(labels == (target_label))[idx_train]]
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            embed = (
                chosen_embed
                + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place
            )

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        if num > 0:
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            embed = (
                chosen_embed
                + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place
            )

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(
        torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num))
    )
    new_adj[: adj_back.shape[0], : adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0] :, : adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[: adj_back.shape[0], adj_back.shape[0] :] = adj_back[:, chosen]
    new_adj[adj_back.shape[0] :, adj_back.shape[0] :] = adj_back[chosen, :][:, chosen]

    # ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train


def src_smote_(adj, features, labels, idx_train, portion=1.0, tail_classes=[]):
    assert len(tail_classes) > 0
    c_largest = labels.max().item()
    # adj_back = to_dense_adj(adj, max_num_nodes=features.shape[0])[0]
    chosen = None
    new_features = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for target_label in tail_classes:
        new_chosen = idx_train[(labels == (target_label))[idx_train]]
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            embed = (
                chosen_embed
                + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place
            )

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        if num > 0:
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            embed = (
                chosen_embed
                + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place
            )

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

    add_num = chosen.shape[0]
    # new_adj = adj_back.new(
    #     torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num))
    # )
    # new_adj[: adj_back.shape[0], : adj_back.shape[0]] = adj_back[:, :]
    # new_adj[adj_back.shape[0] :, : adj_back.shape[0]] = adj_back[chosen, :]
    # new_adj[: adj_back.shape[0], adj_back.shape[0] :] = adj_back[:, chosen]
    # new_adj[adj_back.shape[0] :, adj_back.shape[0] :] = adj_back[chosen, :][:, chosen]

    # ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    # idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_new = np.arange(len(labels), len(labels) + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return adj, features, labels, idx_train
