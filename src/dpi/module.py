import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax

from utils.model_utils import masked_softmax


class GCNDecoder(torch.nn.Module):
    def __init__(self, sizes):
        super(GCNDecoder, self).__init__()
        gcn_size = len(sizes) // 2

        self.gcn = MyGCN(sizes[:gcn_size], final_act='relu')
        self.mlp = MLP(sizes[(gcn_size-1):-1], final_act='relu')

    def forward(self, x, edge_index, edge_weight_logits):
        x = self.gcn(x, edge_index, edge_weight_logits)
        x = x.sum(1)
        x = self.mlp(x)

        return x


class DenseGCNDecoder(torch.nn.Module):
    def __init__(self, sizes):
        super(DenseGCNDecoder, self).__init__()
        gcn_size = len(sizes) // 2

        self.gcn = MyDenseGCN(sizes[:gcn_size], final_act='relu')
        self.mlp = MLP(sizes[(gcn_size-1):-1], final_act='relu')

    def forward(self, x, edge_index, edge_weight_logits):
        x = self.gcn(x, edge_index, edge_weight_logits)
        x = x.sum(1)
        x = self.mlp(x)

        return x


class MyGAT(torch.nn.Module):
    def __init__(self, sizes,
                heads=2, add_self_loops=False,
                batch_norm=False, final_act=None):
        super(MyGAT, self).__init__()
        layers = []
        norms = [] if batch_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyGATConv(sizes[s], sizes[s + 1],
                heads=heads, add_self_loops=add_self_loops
            )]
            if batch_norm:
                norms += [torch.nn.BatchNorm1d(sizes[s + 1])]
        self.layers = torch.nn.ModuleList(layers)
        self.norms = torch.nn.ModuleList(norms) if norms is not None else None
        self.act = torch.nn.ReLU()

        self.final_layer = MyGATConv(sizes[-2], sizes[-1],
            heads=heads, add_self_loops=add_self_loops
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = torch.nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = torch.nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = torch.nn.Softmax(dim=1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, edge_index, edge_weight_logits=None):
        edge_weight = None

        x = x.unsqueeze(0) if x.dim() == 2 else x
        data_list = [Data(d, edge_index, edge_weight) for d in x]
        batch = Batch().from_data_list(data_list)
        x, edge_index, edge_weight = (
            batch.x, batch.edge_index, batch.edge_attr
        )

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = (
                self.norms[i](x.transpose(1, 2)).transpose(1, 2) 
                if self.norms is not None else x
            )
            x = self.act(x)
        x = self.final_layer(x, edge_index, edge_weight)
        x = self.final_act(x) if self.final_act is not None else x

        batch.x = x
        data_list = batch.to_data_list()
        return torch.stack([d.x for d in data_list])


class MyGCN(torch.nn.Module):
    def __init__(self, sizes,
                normalize=False, add_self_loops=False,
                batch_norm=False, final_act=None):
        super(MyGCN, self).__init__()
        layers = []
        norms = [] if batch_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyGCNConv(
                sizes[s], sizes[s + 1],
                normalize=normalize, add_self_loops=add_self_loops
            )]
            if batch_norm:
                norms += [torch.nn.BatchNorm1d(sizes[s + 1])]
        self.layers = torch.nn.ModuleList(layers)
        self.norms = torch.nn.ModuleList(norms) if norms is not None else None
        self.act = torch.nn.ReLU()

        self.final_layer = MyGCNConv(sizes[-2], sizes[-1],
            normalize=normalize, add_self_loops=add_self_loops
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = torch.nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = torch.nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = torch.nn.Softmax(dim=1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, edge_index, edge_weight_logits):
        edge_weight = softmax(edge_weight_logits, edge_index[1])

        x = x.unsqueeze(0) if x.dim() == 2 else x
        data_list = [Data(d, edge_index, edge_weight) for d in x]
        batch = Batch().from_data_list(data_list)
        x, edge_index, edge_weight = (
            batch.x, batch.edge_index, batch.edge_attr
        )

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = (
                self.norms[i](x.transpose(1, 2)).transpose(1, 2) 
                if self.norms is not None else x
            )
            x = self.act(x)
        x = self.final_layer(x, edge_index, edge_weight)
        x = self.final_act(x) if self.final_act is not None else x

        batch.x = x
        data_list = batch.to_data_list()
        return torch.stack([d.x for d in data_list])


class MyDenseGCN(torch.nn.Module):
    def __init__(self, sizes,
                normalize=False, add_self_loops=False,
                batch_norm=False, final_act=None):
        super(MyDenseGCN, self).__init__()
        layers = []
        norms = [] if batch_norm else None
        for s in range(len(sizes) - 2):
            layers += [MyDenseGCNConv(sizes[s], sizes[s + 1],
                normalize=normalize, add_self_loops=add_self_loops
            )]
            if batch_norm:
                norms += [torch.nn.BatchNorm1d(sizes[s + 1])]
        self.layers = torch.nn.ModuleList(layers)
        self.norms = torch.nn.ModuleList(norms) if norms is not None else None
        self.act = torch.nn.ReLU()

        self.final_layer = MyDenseGCNConv(sizes[-2], sizes[-1],
            normalize=normalize, add_self_loops=add_self_loops
        )
        if final_act is None:
            self.final_act = None
        elif final_act == 'relu':
            self.final_act = torch.nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = torch.nn.Sigmoid()
        elif final_act == 'softmax':
            self.final_act = torch.nn.Softmax(dim=1)
        else:
            raise ValueError("final_act not recognized")

    def forward(self, x, adj, weight_logits):
        weight = masked_softmax(weight_logits, mask=adj, dim=1)
        adj = adj*weight

        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            x = (
                self.norms[i](x.transpose(1, 2)).transpose(1, 2) 
                if self.norms is not None else x
            )
            x = self.act(x)
        x = self.final_layer(x, adj)
        x = self.final_act(x) if self.final_act is not None else x
        return x


class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, final_act=None):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU()
                if s < len(sizes) - 2
                else None
            ]
        if final_act is None:
            pass
        elif final_act == 'relu':
            layers += [torch.nn.ReLU()]
        elif final_act == 'sigmoid':
            layers += [torch.nn.Sigmoid()]
        elif final_act == 'softmax':
            layers += [torch.nn.Softmax(dim=1)]
        else:
            raise ValueError("final_act not recognized")

        layers = [l for l in layers if l is not None]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing

from utils.model_utils import init_randoms, init_zeros, gcn_norm


class MyGATConv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the static
    attention problem of the standard :class:`~torch_geometric.conv.GATConv`
    layer: since the linear layers in the standard GAT are applied right after
    each other, the ranking of attended nodes is unconditioned on the query
    node. In contrast, in GATv2, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
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
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.lin_dst = Parameter(torch.Tensor(in_channels, heads * out_channels))
        else:
            self.lin_src = Parameter(torch.Tensor(in_channels[0], heads * out_channels))
            self.lin_dst = Parameter(torch.Tensor(in_channels[1], heads * out_channels))

        # The learnable parameters to compute attention coefficients:
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.lin_src, init_method)
        init_randoms(self.lin_dst, init_method)
        init_randoms(self.att, init_method)
        init_zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, None) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = torch.matmul(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = torch.matmul(x_src, self.lin_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = torch.matmul(x_dst, self.lin_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if self.add_self_loops:
            # We only want to add self-loops for nodes that appear both as
            # source and target nodes:
            num_nodes = x_src.size(0)
            if x_dst is not None:
                num_nodes = min(num_nodes, x_dst.size(0))
            edge_index = remove_self_loops(edge_index)
            edge_index = add_self_loops(edge_index,
                fill_value=self.fill_value, num_nodes=num_nodes
            )

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j: Tensor, x_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # GATv2
        x = x_j if x_i is None else x_j + x_i

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MyGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, init_method: str = 'uniform',
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

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
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class MyDenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, add_self_loops: bool = True, 
                 normalize: bool = True, bias: bool = True,
                 init_method: str = 'uniform'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.lin, init_method)
        init_zeros(self.bias)

    def forward(self, x, adj, mask=None):
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
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
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

        x = torch.matmul(x, self.lin)
        out = torch.matmul(adj, x) # row target, col source

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


from torch.distributions import Distribution, Gamma, Poisson, constraints

from utils.math_utils import (
    logprob_nb_positive,
    logprob_zinb_positive,
    convert_counts_logits_to_mean_disp, 
    convert_mean_disp_to_counts_logits
)


class NegativeBinomial(Distribution):
    r"""
    Negative binomial distribution.
    One of the following parameterizations must be provided:
    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.
    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:
    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / (self.theta+self._eps)

    def sample(
        self, sample_shape: Union[torch.Size, Tuple] = torch.Size()
    ) -> torch.Tensor:
        with torch.no_grad():
            # Important remark: Gamma is parametrized by the rate = 1/scale!
            gamma_d = Gamma(concentration=self.theta, rate=self.theta/(self.mu+self._eps))
            p_means = gamma_d.sample(sample_shape)

            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train = torch.clamp(p_means, max=1e8)
            counts = Poisson(
                l_train
            ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
            return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )

        return logprob_nb_positive(value, mu=self.mu, theta=self.theta)


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""
    Zero-inflated negative binomial distribution.
    One of the following parameterizations must be provided:
    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.
    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:
    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    zi_logits
        Logits scale of zero inflation probability.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_probs": constraints.half_open_interval(0.0, 1.0),
        "zi_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):

        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            mu=mu,
            theta=theta,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(
            zi_logits, self.mu, self.theta
        )

    @property
    def mean(self):
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self):
        raise NotImplementedError

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    def sample(
        self, sample_shape: Union[torch.Size, Tuple] = torch.Size()
    ) -> torch.Tensor:
        with torch.no_grad():
            samp = super().sample(sample_shape=sample_shape)
            is_zero = torch.rand_like(samp) <= self.zi_probs
            samp[is_zero] = 0.0
            return samp

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
            )
        return logprob_zinb_positive(value, self.mu, self.theta, self.zi_logits)