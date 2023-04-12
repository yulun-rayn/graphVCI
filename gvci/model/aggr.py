import torch
import torch.nn as nn

from vci.model.module import MLP


class GraphAggrSUM(nn.Module):
    def __init__(self, heads=1,
                 input_size=None, output_size=None,
                 final_act=None):
        super(GraphAggrSUM, self).__init__()
        self.heads = heads

        self.emb = MLP(
            [input_size, heads*output_size], final_act=final_act
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

    def forward(self, z, x):
        z = z.unsqueeze(1)
        x = self.emb(x)
        x = x.repeat(z.size(0), 1)
        x = x.view(x.size(0), self.heads, -1)

        out = torch.transpose(z + x, 1, 2)
        return self.final_act(out) if self.final_act is not None else out


class GraphAggrMLP(nn.Module):
    def __init__(self, heads=1,
                 input_size=None, output_size=None,
                 final_act=None):
        super(GraphAggrMLP, self).__init__()
        self.heads = heads

        self.mlp = MLP(
            [input_size, heads*output_size], final_act=final_act
        )

    def forward(self, z, x):
        x = x.repeat(z.size(0), 1)

        out = self.mlp(torch.cat([z, x], -1))
        return out.view(z.size(0), -1, self.heads)


class GraphAggrMLPv2(nn.Module):
    def __init__(self, heads=1,
                 input_size=None, output_size=None,
                 final_act=None):
        super(GraphAggrMLPv2, self).__init__()
        self.heads = heads

        self.mlp = MLP(
            [input_size, heads], final_act=final_act
        )

    def forward(self, z, x):
        z = torch.repeat_interleave(
            z.unsqueeze(1), x.size(1), dim=1
        )
        x = x.t().repeat(z.size(0), 1, 1)

        return self.mlp(torch.cat([z, x], -1))


class NodeAggrDOT(nn.Module):
    def __init__(self, heads=1,
                 input_size=None, output_size=None,
                 final_act=None):
        super(NodeAggrDOT, self).__init__()
        self.heads = heads

        self.emb = MLP(
            [input_size, heads*output_size], final_act=final_act
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

    def forward(self, z, x):
        x = self.emb(x)
        x = x.view(x.size(0), -1, self.heads)

        out = torch.einsum('bd,ndm -> bnm', [z, x])
        return self.final_act(out) if self.final_act is not None else out


class NodeAggrMLP(nn.Module):
    def __init__(self, heads=1,
                 input_size=None, output_size=None,
                 final_act=None):
        super(NodeAggrMLP, self).__init__()

        self.mlp = MLP(
            [input_size, heads], final_act=final_act
        )

    def forward(self, z, x):
        z = torch.repeat_interleave(
            z.unsqueeze(1), x.size(0), dim=1
        )
        x = x.repeat(z.size(0), 1, 1)

        return self.mlp(torch.cat([z, x], -1))


class NodeAggrATT(nn.Module):
    def __init__(self, heads=1,
                 input_size=None, output_size=None,
                 final_act=None):
        super(NodeAggrATT, self).__init__()
        self.heads = heads

        self.att = MLP(
            [input_size, heads*output_size], final_act=None
        )
        self.softmax = nn.Softmax(dim=2)

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

    def forward(self, z, x):
        z = torch.repeat_interleave(
            z.unsqueeze(1), x.size(0), dim=1
        )
        x = x.repeat(z.size(0), 1, 1)

        alpha = self.att(torch.cat([z, x], -1))
        alpha = alpha.view(z.size(0), z.size(1), -1, self.heads)
        #alpha = self.softmax(alpha)

        out = torch.einsum('bnd,bndm -> bnm', [z, alpha])
        return self.final_act(out) if self.final_act is not None else out
