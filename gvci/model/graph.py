from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax

from gvci.utils.math_utils import masked_softmax
from gvci.utils.graph_utils import masked_select


class MyGAT(nn.Module):
    def __init__(self, sizes, heads=2, edge_dim=None, final_act=None,
                 add=False, layer_norm=False, add_self_loops=False, dropout=0.0):
        super(MyGAT, self).__init__()
        layers = []
        norms = [] if layer_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyGATConv(sizes[s], sizes[s + 1]//heads,
                heads=heads, edge_dim=edge_dim,
                add_self_loops=add_self_loops, dropout=dropout
            )]
            if layer_norm:
                norms += [nn.LayerNorm(sizes[s + 1])]
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if norms is not None else None
        self.act = nn.ReLU()
        self.add = add

        self.final_layer = MyGATConv(sizes[-2], sizes[-1]//heads,
            heads=heads, edge_dim=edge_dim,
            add_self_loops=add_self_loops, dropout=dropout
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = nn.Softmax(dim=-1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, edge_index, edge_attr=None, mask=None):
        r"""
            x: [B, N, F] or [N, F]
            edge_index: [2, E] or list (B) of [2, E]
            edge_attr: [E] or [E, A] or list (B) of [E] or list (B) of [E, A]
            mask: [B, N] or [N]
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        edge_index = edge_index if isinstance(edge_index, list) else [edge_index]*x.size(0)

        if mask is not None:
            mask = mask.unsqueeze(0).repeat_interleave(x.size(0), dim=0) if mask.dim() == 1 else mask
            edge_index = [masked_select(edge_index[i], mask[i]) for i in range(x.size(0))]

        if edge_attr is None:
            edge_attr = [torch.ones((index.size(-1), 1), device=x.device) for index in edge_index]
        edge_attr = edge_attr if isinstance(edge_attr, list) else [edge_attr]*x.size(0)
        edge_attr = [attr.unsqueeze(-1) if attr.dim() == 1 else attr for attr in edge_attr]

        data_list = [Data(x[i], edge_index[i], edge_attr[i]) for i in range(x.size(0))]
        batch = Batch.from_data_list(data_list)
        x, edge_index, edge_attr = (
            batch.x, batch.edge_index, batch.edge_attr
        )

        for i, layer in enumerate(self.layers):
            out = layer(x, edge_index, edge_attr)
            x = out + x if self.add else out
            x = self.norms[i](x) if self.norms is not None else x
            x = self.act(x)
        x = self.final_layer(x, edge_index, edge_attr)
        x = self.final_act(x) if self.final_act is not None else x

        batch.x = x
        data_list = batch.to_data_list()
        return torch.stack([d.x for d in data_list])


class MyDenseGAT(nn.Module):
    def __init__(self, sizes, heads=2, edge_dim=None, final_act=None,
                 add=False, layer_norm=False, add_self_loops=False, dropout=0.1):
        super(MyDenseGAT, self).__init__()
        layers = []
        norms = [] if layer_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyDenseGATConv(sizes[s], sizes[s + 1]//heads,
                heads=heads, edge_dim=edge_dim,
                add_self_loops=add_self_loops, dropout=dropout
            )]
            if layer_norm:
                norms += [nn.LayerNorm(sizes[s + 1])]
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if norms is not None else None
        self.act = nn.ReLU()
        self.add = add

        self.final_layer = MyDenseGATConv(sizes[-2], sizes[-1]//heads,
            heads=heads, edge_dim=edge_dim,
            add_self_loops=add_self_loops, dropout=dropout
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = nn.Softmax(dim=-1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, adj, attr=None, mask=None):
        r"""
            x: [B, N, F] or [N, F]
            adj: [B, N, N] or [N, N]
            attr: [B, N, N, A] or [B, N, N] or [N, N, A] or [N, N]
            mask: [B, N] or [N]
        """
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        if mask is not None:
            mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
            adj = adj*mask[..., None, :]

        for i, layer in enumerate(self.layers):
            out = layer(x, adj)
            x = out + x if self.add else out
            x = self.norms[i](x) if self.norms is not None else x
            x = self.act(x)
        x = self.final_layer(x, adj)
        x = self.final_act(x) if self.final_act is not None else x
        return x


class MyGCN(nn.Module):
    def __init__(self, sizes, final_act=None,
                add=False, layer_norm=False,
                add_self_loops=False, dropout=0.0):
        super(MyGCN, self).__init__()
        layers = []
        norms = [] if layer_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyGCNConv(sizes[s], sizes[s + 1],
                add_self_loops=add_self_loops, normalize=False, dropout=dropout
            )]
            if layer_norm:
                norms += [nn.LayerNorm(sizes[s + 1])]
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if norms is not None else None
        self.act = nn.ReLU()
        self.add = add

        self.final_layer = MyGCNConv(sizes[-2], sizes[-1],
            add_self_loops=add_self_loops, normalize=False, dropout=dropout
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = nn.Softmax(dim=-1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, edge_index, edge_weight_logits=None, mask=None):
        r"""
            x: [B, N, F] or [N, F]
            edge_index: [2, E] or list (B) of [2, E]
            edge_weight_logits: [E] or list (B) of [E]
            mask: [B, N] or [N]
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        edge_index = edge_index if isinstance(edge_index, list) else [edge_index]*x.size(0)

        if mask is not None:
            mask = mask.unsqueeze(0).repeat_interleave(x.size(0), dim=0) if mask.dim() == 1 else mask
            edge_index = [masked_select(edge_index[i], mask[i]) for i in range(x.size(0))]

        if edge_weight_logits is None:
            edge_weight_logits = [torch.zeros(edge_index[i].size(-1), device=x.device) for i in range(x.size(0))]
        edge_weight_logits = edge_weight_logits if isinstance(edge_weight_logits, list) else [edge_weight_logits]*x.size(0)
        edge_weight = [softmax(edge_weight_logits[i], edge_index[i][1]) for i in range(x.size(0))]

        data_list = [Data(x[i], edge_index[i], edge_weight[i]) for i in range(x.size(0))]
        batch = Batch().from_data_list(data_list)
        x, edge_index, edge_weight = (
            batch.x, batch.edge_index, batch.edge_attr
        )

        for i, layer in enumerate(self.layers):
            out = layer(x, edge_index, edge_weight)
            x = out + x if self.add else out
            x = self.norms[i](x) if self.norms is not None else x
            x = self.act(x)
        x = self.final_layer(x, edge_index, edge_weight)
        x = self.final_act(x) if self.final_act is not None else x

        batch.x = x
        data_list = batch.to_data_list()
        return torch.stack([d.x for d in data_list])


class MyDenseGCN(nn.Module):
    def __init__(self, sizes, final_act=None,
                add=False, layer_norm=False,
                add_self_loops=False, dropout=0.1):
        super(MyDenseGCN, self).__init__()
        layers = []
        norms = [] if layer_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyDenseGCNConv(sizes[s], sizes[s + 1],
                add_self_loops=add_self_loops, normalize=False, dropout=dropout
            )]
            if layer_norm:
                norms += [nn.LayerNorm(sizes[s + 1])]
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if norms is not None else None
        self.act = nn.ReLU()
        self.add = add

        self.final_layer = MyDenseGCNConv(sizes[-2], sizes[-1],
            add_self_loops=add_self_loops, normalize=False, dropout=dropout
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = nn.Softmax(dim=-1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, adj, weight_logits=None, mask=None):
        r"""
            x: [B, N, F] or [N, F]
            adj: [B, N, N] or [N, N]
            weight_logits: [B, N, N] or [N, N]
            mask: [B, N] or [N]
        """
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        if mask is not None:
            mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
            adj = adj*mask[..., None, :]

        if weight_logits is None:
            weight_logits = torch.zeros_like(adj)
        weight_logits = weight_logits.unsqueeze(0) if weight_logits.dim() == 2 else weight_logits
        adj = masked_softmax(weight_logits, mask=adj, dim=-1)

        for i, layer in enumerate(self.layers):
            out = layer(x, adj)
            x = out + x if self.add else out
            x = self.norms[i](x) if self.norms is not None else x
            x = self.act(x)
        x = self.final_layer(x, adj)
        x = self.final_act(x) if self.final_act is not None else x
        return x


from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing

from gvci.utils.model_utils import init_randoms, init_zeros, gcn_norm


class MyGATConv(MessagePassing):
    r"""See :class:`torch_geometric.nn.conv.TransformerConv`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        add_self_loops: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        init_method: str = 'uniform',
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.edge_dim = edge_dim

        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_query = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.lin_key = Parameter(torch.Tensor(in_channels[0], heads * out_channels))
        self.lin_value = Parameter(torch.Tensor(in_channels[0], heads * out_channels))
        if edge_dim is not None:
            self.lin_edge = Parameter(torch.Tensor(edge_dim, heads * out_channels))
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.lin_query, init_method)
        init_randoms(self.lin_key, init_method)
        init_randoms(self.lin_value, init_method)
        init_randoms(self.lin_edge, init_method)
        init_zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.add_self_loops:
            edge_index, tmp_edge_attr = add_remaining_self_loops(
                edge_index, edge_attr, "mean", x[1].size(self.node_dim)
            )
            assert tmp_edge_attr is not None
            edge_attr = tmp_edge_attr

        query = torch.matmul(x[1], self.lin_query).view(-1, H, C)
        key = torch.matmul(x[0], self.lin_key).view(-1, H, C)
        value = torch.matmul(x[0], self.lin_value).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.matmul(edge_attr, self.lin_edge).view(
                -1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / self.out_channels**0.5
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        #if edge_attr is not None:
        #    out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MyDenseGATConv(MessagePassing):
    r"""Dense version of `torch_geometric.nn.conv.TransformerConv`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        add_self_loops: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        init_method: str = 'uniform',
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_query = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.lin_key = Parameter(torch.Tensor(in_channels[0], heads * out_channels))
        self.lin_value = Parameter(torch.Tensor(in_channels[0], heads * out_channels))
        if edge_dim is not None:
            self.lin_edge = Parameter(torch.Tensor(edge_dim, heads * out_channels))
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.lin_query, init_method)
        init_randoms(self.lin_key, init_method)
        init_randoms(self.lin_value, init_method)
        init_randoms(self.lin_edge, init_method)
        init_zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], adj: Tensor,
                return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(adj, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.add_self_loops:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        query = torch.matmul(x[1], self.lin_query).view(B, N, H, C)
        key = torch.matmul(x[0], self.lin_key).view(B, N, H, C)
        value = torch.matmul(x[0], self.lin_value).view(B, N, H, C)

        alpha = torch.einsum('bnhf,bmhf->bhnm', query, key) / self.out_channels**0.5
        alpha = masked_softmax(alpha, mask=adj, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum('bhnm,bmhf->bnhf', alpha, value)

        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            return out, (adj, alpha)
        else:
            return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MyGCNConv(MessagePassing):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 dropout: float = 0.0, bias: bool = True,
                 init_method: str = 'uniform', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.dropout = dropout

        self._cached_edge_index = None

        self.lin = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.lin, init_method)
        init_zeros(self.bias)
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.add_self_loops:
            fill_value = 2. if self.improved else 1.
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, x.size(self.node_dim)
            )
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(self.node_dim)
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        x = torch.matmul(x, self.lin)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:

        x_j = F.dropout(x_j, p=self.dropout, training=self.training)

        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class MyDenseGCNConv(nn.Module):
    r"""See :class:`torch_geometric.nn.conv.DenseGCNConv`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, dropout: float = 0.0,
                 bias: bool = True, init_method: str = 'uniform'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.dropout = dropout

        self.lin = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.lin, init_method)
        init_zeros(self.bias)

    def forward(self, x, adj):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if self.add_self_loops:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        if self.normalize:
            deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        adj = F.dropout(adj, p=self.dropout, training=self.training)

        x = torch.matmul(x, self.lin)

        out = torch.matmul(adj, x) # row target, col source

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
