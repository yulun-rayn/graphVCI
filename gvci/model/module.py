import torch
import torch.nn as nn

from .graph import MyDenseGCN, MyGCN, MyDenseGAT, MyGAT
from .aggr import (
    GraphAggrSUM, GraphAggrMLP, GraphAggrMLPv2,
    NodeAggrDOT, NodeAggrMLP, NodeAggrATT
)

from vci.model.module import MLP


class Enc_graphVCI(nn.Module):
    def __init__(self, mlp_sizes, gnn_sizes, num_nodes, edge_dim=None,
                 aggr_heads=1, graph_mode="sparse", aggr_mode="mlp",
                 final_act=None):
        super(Enc_graphVCI, self).__init__()
        self.graph_mode = graph_mode
        self.aggr_mode = aggr_mode

        self.encoder = MLP(mlp_sizes)

        if self.graph_mode == "dense":
            self.graph_encoder = MyDenseGCN(gnn_sizes, final_act=final_act,
                add=True, layer_norm=True, add_self_loops=True
            )
            '''
            self.graph_encoder = MyDenseGAT(gnn_sizes, edge_dim=edge_dim,
                final_act=final_act, add=True, layer_norm=True
            )
            '''
        elif self.graph_mode == "sparse":
            self.graph_encoder = MyGCN(gnn_sizes, final_act=final_act,
                add=True, layer_norm=True, add_self_loops=True
            )
            '''
            self.graph_encoder = MyGAT(gnn_sizes, edge_dim=edge_dim,
                final_act=final_act, add=True, layer_norm=True
            )
            '''
        else:
            raise ValueError("graph_mode not recognized")

        if self.aggr_mode == "sum":
            self.aggr = GraphAggrSUM(aggr_heads,
                input_size=gnn_sizes[-1],
                output_size=mlp_sizes[-1],
                final_act=final_act
            )
        elif self.aggr_mode == "mlp":
            self.aggr = GraphAggrMLP(aggr_heads,
                input_size=gnn_sizes[-1]+mlp_sizes[-1],
                output_size=mlp_sizes[-1],
                final_act=final_act
            )
            '''
            self.aggr = GraphAggrMLPv2(aggr_heads,
                input_size=num_nodes+mlp_sizes[-1],
                final_act=final_act
            )
            '''
        else:
            raise ValueError("aggr_mode not recognized")

    def forward(self, z, x, edge_index, edge_attr, return_graph=False):
        z = self.encoder(z)
        g = self.graph_encoder(x, edge_index, edge_attr)

        g = g.squeeze(0) if g.dim() > x.dim() else g
        z = self.aggr(z, g.mean(0)).squeeze(-1)
        if return_graph:
            return z, g
        else:
            return z


class Dec_graphVCI(nn.Module):
    def __init__(self, mlp_sizes, num_features,
                 aggr_heads=1, aggr_mode="dot",
                 final_act=None):
        super(Dec_graphVCI, self).__init__()
        self.aggr_mode = aggr_mode

        self.decoder = MLP(mlp_sizes)

        if self.aggr_mode == "dot":
            self.aggr = NodeAggrDOT(aggr_heads,
                input_size=num_features,
                output_size=mlp_sizes[-1],
                final_act=final_act
            )
        elif self.aggr_mode == "mlp":
            self.aggr = NodeAggrMLP(aggr_heads,
                input_size=num_features+mlp_sizes[-1],
                final_act=final_act
            )
        elif self.aggr_mode == "att":
            self.aggr = NodeAggrATT(aggr_heads,
                input_size=num_features+mlp_sizes[-1],
                output_size=mlp_sizes[-1],
                final_act=final_act
            )
        else:
            raise ValueError("aggr_mode not recognized")

    def forward(self, z, g):
        z = self.decoder(z)

        return self.aggr(z, g).squeeze(-1)
