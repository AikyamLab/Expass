#!/usr/bin/env python3

import torch
from torch_geometric.nn import GCNConv, Sequential, LEConv, GraphConv, Linear
from torch_geometric.nn import global_mean_pool
from conv_layers import GINConv, SAGEConv

CONV_ARCHITECTURES = {
 "gcn": GCNConv,
 "leconv": LEConv,
 "graphconv": GraphConv,
 "sageconv": SAGEConv,
 "ginconv": GINConv,
}

class Model(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers, gnn_arch):
        super().__init__()
        self.dropout = dropout
        self.num_layer = num_layers
        self.gnn_arch = gnn_arch
        conv_arch = CONV_ARCHITECTURES.get(self.gnn_arch)
        if conv_arch is None:
            print("invalid architecture")
            exit(0)
        self.conv1 = conv_arch(nfeat, nhid)
        self.layerlist = []
        if self.num_layer > 2:
            for _ in range(self.num_layer - 2):
                self.layerlist.append(
                    (conv_arch(nhid, nhid), "x, edge_index, edge_weight -> x")
                )
                self.layerlist.append(torch.nn.ReLU())
                self.layerlist.append(torch.nn.Dropout(p=self.dropout))
            self.intermediate = Sequential("x, edge_index, edge_weight", self.layerlist)
        self.final = Linear(nhid, nclass)

    def forward(self, x, edge_index, edge_weight, batch=None, **kwargs):
        x = self.embed(x, edge_index, edge_weight, batch)
        x = self.final(x)
        return x
    
    def embed(self, x, edge_index, edge_weight, batch=None, **kwargs):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        if self.num_layer > 2:
            x = self.intermediate(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        return x

