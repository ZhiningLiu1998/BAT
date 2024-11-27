"""
Code Reference: https://github.com/victorchen96/ReNode/blob/main/transductive/network/gcn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_geometric.nn import GCNConv


class StandGCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=1):
        super(StandGCN1, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
        # x = F.relu(self.conv1(x, edge_index))

        return x


class StandGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2):
        super(StandGCN2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout_p = dropout

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def get_embed(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

    def embed_to_pred(self, x, adj):
        edge_index = adj
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, adj)

        return x


class StandGCNX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=3):
        super(StandGCNX, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid) for _ in range(nlayer - 2)])
        self.dropout_p = dropout

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def get_embed(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for iter_layer in self.convx:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x = iter_layer(x, edge_index)
            x = F.relu(x)
        return x

    def embed_to_pred(self, x, adj):
        edge_index = adj
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        dropout,
        nlayer,
    ):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(nlayer - 2):
            self.convs.append(GCNConv(nhid, nhid, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=True))

        self.dropout = dropout

        self.reg_params = self.convs[1:-1].parameters()
        self.non_reg_params = self.convs[-1].parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def create_gcn(nfeat, nhid, nclass, dropout, nlayer, batch_norm=False):
    if batch_norm == True:
        model = GCN(nfeat, nhid, nclass, dropout, nlayer)
    else:
        if nlayer == 1:
            model = StandGCN1(nfeat, nhid, nclass, dropout, nlayer)
        elif nlayer == 2:
            model = StandGCN2(nfeat, nhid, nclass, dropout, nlayer)
        else:
            model = StandGCNX(nfeat, nhid, nclass, dropout, nlayer)
    return model


class MLP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(nfeat, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(nlayer - 2):
            self.layers.append(torch.nn.Linear(nhid, nhid))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.layers.append(torch.nn.Linear(nhid, nclass))

        self.dropout = dropout

        self.reg_params = self.layers[1:-1].parameters()
        self.non_reg_params = self.layers[-1].parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x.log_softmax(dim=-1)


def create_mlp(nfeat, nhid, nclass, dropout, nlayer, batch_norm=False):
    model = MLP(nfeat, nhid, nclass, dropout, nlayer)
    return model
