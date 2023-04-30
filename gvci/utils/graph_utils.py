import pandas as pd
import networkx as nx

import torch
from torch_scatter import scatter

def parse_grn(grn_df, name_col) -> nx.Graph:
    '''Parse a GRN data frame to a graph.
    Assumes the file has targets as rows and sources as columns
    '''
    # build graph
    graph = nx.DiGraph()
    for _, row in grn_df.iterrows():
        target_gene = row[name_col]
        edges = [(src_gene, target_gene) for src_gene in row[row==1].index]
        graph.add_edges_from(edges)
    return graph

def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None, fill_value=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    if fill_value is None:
        adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)
    else:
        adj = fill_value * torch.ones(size, dtype=edge_attr.dtype, device=edge_index.device)
        edge_attr = edge_attr - fill_value

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    return adj

def masked_select(edge_index, mask, edge_mode="source_to_target"):
    assert edge_mode == "source_to_target" or edge_mode == "target_to_source"
    source = torch.select(edge_index, 0, int(edge_mode == "target_to_source"))
    mask_index = torch.nonzero(mask).squeeze(1)
    
    source_index = torch.sum(
        source.unsqueeze(1) == torch.repeat_interleave(
            mask_index.unsqueeze(0), source.size(0), 
        dim=0), 
    dim=1) > 0
    
    return edge_index[:, source_index]

def get_graph(graph=None, n_nodes=None, n_features=None, graph_mode="sparse",
              input_adj_mode="source_to_target", output_adj_mode="source_to_target"):
    assert graph_mode == "dense" or graph_mode == "sparse"
    assert input_adj_mode == "source_to_target" or input_adj_mode == "target_to_source"
    assert output_adj_mode == "source_to_target" or output_adj_mode == "target_to_source"
    if graph is None:
        assert n_nodes is not None
        assert n_features is not None

    if type(graph) == str:
        graph = torch.load(graph)

    # node
    if graph is None or graph.x is None:
        node_features = torch.Tensor(n_nodes, n_features)
    else:
        node_features = graph.x

    # edge
    if graph is None or graph.edge_index is None:
        if graph_mode == "dense":
            adjacency = torch.ones((n_nodes, n_nodes), dtype=torch.long)
            edge_features = torch.ones_like(adjacency)
        elif graph_mode == "sparse":
            adjacency = torch.stack((torch.arange(n_nodes), torch.arange(n_nodes)))
            edge_features = torch.ones(adjacency.size(1))
    else:
        adjacency = graph.edge_index
        edge_features = graph.edge_attr

        if graph_mode == "dense":
            if adjacency.size(0) != adjacency.size(1):
                edge_features = (to_dense_adj(adjacency, edge_attr=edge_features)[0]
                    if edge_features is not None else None)
                adjacency = to_dense_adj(adjacency)[0]
            if edge_features is None:
                edge_features = adjacency
            if input_adj_mode != output_adj_mode:
                edge_features = edge_features.transpose(0,1)
                adjacency = adjacency.t()
        if graph_mode == "sparse":
            if adjacency.size(0) != 2:
                adjacency = adjacency.nonzero().t()
                edge_features = (edge_features[adjacency[0], adjacency[1], ...] 
                    if edge_features is not None else None)
            if edge_features is None:
                edge_features = torch.ones(adjacency.size(1))
            if input_adj_mode != output_adj_mode:
                adjacency = torch.flip(adjacency, dims=(0,))

    return node_features, adjacency, edge_features
