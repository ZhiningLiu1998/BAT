import numpy as np
from copy import copy

def normalize_weights(weights, alpha=.1, how='none'):
    if how == 'none':
        return weights
    elif how == 'unit':
        edge_sample_weights = copy(weights) + alpha
        edge_sample_weights /= edge_sample_weights.sum()
    elif how == 'max':
        edge_sample_weights = copy(weights) + alpha
        edge_sample_weights /= edge_sample_weights.max()
    else: raise NotImplementedError
    return edge_sample_weights

def edge_sampling(sampling, edge_index, weights, alpha, sparsity, norm=None,
                  random_seed=None, edge_weight=None, debug=False, **add_kwargs):
    kwargs = {
        'edge_index': edge_index.clone(),
        'weights': weights.cpu().numpy(),
        'alpha': alpha,
        'sparsity': sparsity,
        'random_seed': random_seed,
        'edge_weight': edge_weight,
        'debug': debug,
    }
    if norm is not None:
        kwargs['norm'] = norm
    if sampling == 'dummy':
        return edge_sampling_dummy(**kwargs)
    elif sampling == 'random':
        return edge_sampling_random(**kwargs)
    elif sampling == 'indpt':
        return edge_sampling_indpt(**kwargs)
    elif sampling == 'ratio':
        return edge_sampling_ratio(**kwargs)
    elif sampling == 'reweight':
        return edge_reweight(**kwargs)
    else: raise NotImplementedError

def edge_sampling_dummy(edge_index, weights, alpha, sparsity=None, norm='none',
                        random_seed=None, edge_weight=None, debug:bool=False):
    info = f'EdgeSamp Dummy | {len(weights)} (100.00%) edges are kept'
    if debug:
        print (info)
    return {
        'edge_index': edge_index, 
        'edge_weight': edge_weight, 
        'edge_mask': np.ones_like(weights, dtype=int),
        'info': info,
    }

def edge_sampling_random(edge_index, weights, alpha, sparsity, norm='none',
                         random_seed=None, edge_weight=None, debug:bool=False):
    """
    Sample n * sparsity edges randomly
    Consistency param: `alpha`
    """
    np.random.seed(random_seed)
    n_edges, n_keep_edges = len(weights), int(len(weights) * sparsity)
    edge_sample_idx = np.random.choice(range(n_edges), size=n_keep_edges, replace=False) # random
    edge_sample_mask = np.zeros_like(weights, dtype=int)
    edge_sample_mask[edge_sample_idx] = 1
    info = f'EdgeSamp Random | sparsity: {sparsity} | ' + \
           f'{edge_sample_mask.sum()} ({edge_sample_mask.sum() / len(weights):.2%}) edges are kept'
    if debug:
        print (info)
    return {
        'edge_index': edge_index[:, edge_sample_mask],
        'edge_weight': edge_weight[edge_sample_mask] if edge_weight is not None else None,
        'edge_mask': edge_sample_mask,
        'info': info,
    }

def edge_sampling_indpt(edge_index, weights, alpha, sparsity=None, norm='none',
                        random_seed=None, edge_weight=None, debug:bool=False):
    """
    Sample edges independently w.r.t. given weights (no fixed ratio)
    Consistency param: `sparsity`
    """
    np.random.seed(random_seed)
    edge_sample_prob = normalize_weights(weights, alpha=alpha, how=norm)
    edge_sample_mask = (np.random.sample(size=edge_sample_prob.shape) < edge_sample_prob)
    info = f'EdgeSamp Independent | alpha: {alpha} | ' + \
           f'{edge_sample_mask.sum()} ({edge_sample_mask.sum() / len(weights):.2%}) edges are kept'
    if debug:
        print (info)
    return {
        'edge_index': edge_index[:, edge_sample_mask],
        'edge_weight': edge_weight[edge_sample_mask] if edge_weight is not None else None,
        'edge_mask': edge_sample_mask,
        'info': info,
    }

def edge_sampling_ratio(edge_index, weights, alpha, sparsity, norm='unit',
                        random_seed=None, edge_weight=None, debug:bool=False):
    """
    Sample n * sparsity edges w.r.t. given weights (with fixed ratio)
    """
    np.random.seed(random_seed)
    n_edges = len(weights)
    n_keep_edges = int(n_edges * sparsity)
    edge_sample_prob = normalize_weights(weights, alpha=alpha, how='unit')
    edge_sample_idx = np.random.choice(range(n_edges), size=n_keep_edges, replace=False, p=edge_sample_prob)
    edge_sample_mask = np.zeros_like(weights, dtype=int)
    edge_sample_mask[edge_sample_idx] = 1
    info = f'EdgeSamp Ratio | alpha: {alpha} | sparsity: {sparsity} | ' + \
           f'{edge_sample_mask.sum()} ({edge_sample_mask.sum() / len(weights):.2%}) edges are kept'
    if debug:
        print (info)
    return {
        'edge_index': edge_index[:, edge_sample_mask],
        'edge_weight': edge_weight[edge_sample_mask] if edge_weight is not None else None,
        'edge_mask': edge_sample_mask,
        'info': info,
    }

def edge_reweight(edge_index, weights, alpha, sparsity=None, norm='max',
                  random_seed=None, edge_weight=None, debug:bool=False):
    """
    (Re)weight edges with the given weights
    Consistency param: `sparsity`
    """
    edge_sample_weights = normalize_weights(weights, alpha=alpha, how=norm)
    new_edge_weight = edge_weight * edge_sample_weights if edge_weight is not None else edge_sample_weights
    info = f'EdgeSamp Reweight | given edge_weight: {edge_weight is not None}'
    if debug:
        print (info)
    return {
        'edge_index': edge_index,
        'edge_weight': new_edge_weight.astype('float32'),
        'edge_mask': np.ones_like(weights, dtype=int),
        'info': info,
    }