#!/usr/bin/env python3

from typing import Union
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GINConv as BaseGINConv
from torch_geometric.nn import SAGEConv as BaseSAGEConv
from torch_geometric.nn.models import MLP
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size


class GINConv(BaseGINConv):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        mlp = MLP([in_channels, out_channels, out_channels])
        # return GINConv(mlp, **kwargs)
        super().__init__(mlp, **kwargs)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SAGEConv(BaseSAGEConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, "lin"):
            x = (
                self.lin(x[0]).relu(),
                x[1],
            )  # Set project to false to prevent x being decided with MLP

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
