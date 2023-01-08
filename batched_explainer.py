#!/usr/bin/env python3

import torch
from torch_geometric.nn.models.gnn_explainer import GNNExplainer

class BatchedGNNExplainer(GNNExplainer):
    r"""Slight modification of torch_geometric GNNExplainer"""
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    @torch.no_grad()
    def get_initial_prediction(self, x, edge_index, batch=None, edge_weight=None, **kwargs):
        out = self.model(x, edge_index, edge_weight, batch, **kwargs)
        if self.return_type == "regression":
            prediction = out
        else:
            log_logits = self._to_log_prob(out)
            prediction = log_logits.argmax(dim=-1)
        return prediction
