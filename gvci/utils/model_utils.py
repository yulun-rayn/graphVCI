import math

import torch

from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

def init_randoms(tensor, method='uniform'):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        if method == 'uniform':
            tensor.data.uniform_(-stdv, stdv)
        elif method == 'normal':
            tensor.data.normal_(-stdv, stdv)
        else:
            pass

def init_zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
