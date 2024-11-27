import numpy as np
from collections import Counter
from operator import index
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import to_networkx, degree
import torch_geometric as pyg
import utils

def get_dataset(data_name, path='./data', transform=None, pre_transform=None, 
                is_normalize=False, add_self_loop=False, reorder_label=False,
                print_info=False):
    from torch_geometric.utils.loop import add_remaining_self_loops
    from torch_geometric.utils import degree
    import torch_geometric.transforms as T

    name = data_name.lower()
    if name in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path+'/planetoid', name, transform=transform, 
                            pre_transform=pre_transform, split='full')
    elif name == "cora_full":
        from torch_geometric.datasets import CitationFull
        dataset = CitationFull(path+'/citationfull', 'cora', transform=transform, 
                           pre_transform=pre_transform)
    elif name in ["computers", "photo"]:
        from torch_geometric.datasets import Amazon
        dataset = Amazon(path+'/amazon', name, transform=transform, 
                         pre_transform=pre_transform)
    else: raise NotImplementedError("Not Implemented Dataset!")

    if is_normalize:
        print (f'::get_dataset({data_name}):: Normalizing ...')
        dataset.data = T.NormalizeFeatures()(dataset.data)
    if add_self_loop:
        print (f'::get_dataset({data_name}):: Adding self-loop ...')
        dataset.data.edge_index = add_remaining_self_loops(dataset.data.edge_index)[0]
    if reorder_label:
        print (f'::get_dataset({data_name}):: Reordering label...')
        dataset.data.y, _ = utils.reorder_label_by_count(dataset.data.y)
    if print_info:
        describe_dataset(dataset)

    return dataset


def load_platenoid_lt(dataset_name:str, imb_ratio:int, subgraph=False, print_info=False, **kwargs):
    from torch import save, load
    is_subgraph = '-subgraph' if subgraph else ''
    file_path = f'data/LT/{dataset_name.lower()}-LT-IR{imb_ratio}{is_subgraph}.pt'
    try:
        dataset = load(file_path)
        print (f'Loaded from the saved file {file_path}')
    except:
        print (f'Processing ...')
        dataset = get_dataset(dataset_name, **kwargs)
        data = dataset[0]
        data.y, label_map = utils.reorder_label_by_count(data.y)    # reorder the labels by count
        labels = data.y.clone().cpu().numpy()
        unique_labels = np.unique(labels)
        n_cls = len(unique_labels)
        data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
        _, train_cls_sizes = utils.sort_by_count(labels[data_train_mask])
        train_cls_sizes, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
            utils.make_longtailed_data_remove(data.edge_index, data.y, train_cls_sizes, 
                                            n_cls, imb_ratio, data_train_mask.clone())
        dataset.data.train_mask = data_train_mask.clone()
        if subgraph:
            keep_node_mask = (data_train_mask | data.val_mask | data.test_mask)
            sub_data = data.clone()
            sub_data.x = data.x[keep_node_mask]
            sub_data.y = data.y[keep_node_mask]
            sub_data.edge_index, sub_data.edge_attr = pyg.utils.subgraph(subset=keep_node_mask, edge_index=data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)
            sub_data.train_mask = data.train_mask[keep_node_mask]
            sub_data.val_mask = data.val_mask[keep_node_mask]
            sub_data.test_mask = data.test_mask[keep_node_mask]
            dataset.data = sub_data
        # dataset.data.edge_index = dataset.data.edge_index[:,train_edge_mask]
        save(dataset, file_path)
        print (f'Saved to file {file_path}')
    print (f'////// DATASET {dataset_name} - ImbRatio {imb_ratio} //////')
    if print_info:
        describe_dataset(dataset)
    return dataset

def describe_dataset(dataset):
    data = dataset.data
    node_degrees = degree(dataset.data.edge_index[0], dtype=int).cpu().numpy()
    print(
        f'Number of nodes:     {data.num_nodes}\n'
        f'Number of features:  {dataset.num_features}\n'
        f'Number of edges:     {data.num_edges/2}\n'
        f'Number of classes:   {dataset.num_classes}\n'
        f'Average node degree: {np.mean(node_degrees)}\n'
        f'Median node degree:  {np.median(node_degrees)}\n'
        f'Is undirected:       {data.is_undirected()}\n'
        f'Has isolated nodes:  {data.has_isolated_nodes()}\n'
        f'Has self-loops:      {data.has_self_loops()}\n'
    )
    return

def load_planetoid_dataset(dataset_name:str, path='./data'):
    dataset = get_dataset(dataset_name, path, split_type='full')

    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges/2}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training/validataion/test nodes: {data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    return dataset