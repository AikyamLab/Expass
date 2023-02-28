#!/usr/bin/env python3

import torch
from torch_geometric.data import Data
from typing import Callable
from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation

device = "cuda" if torch.cuda.is_available() else "cpu"


class IntegratedGradExplainer(_BaseExplainer):
    """
    Integrated Gradient Explanation for GNNs from GraphXAI
    Args:
        model (torch.nn.Module): Model on which to make predictions.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__(model)
        self.criterion = criterion

    def get_explanation_graph(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        label: torch.Tensor = None,
        y: torch.Tensor = None,
        node_agg=torch.sum,
        steps: int = 40,
        forward_kwargs={},
    ):
        """
        Explain a whole-graph prediction.
        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            y (torch.Tensor): Same as `label`, provided for general
                compatibility in the arguments. (:default: :obj:`None`)
            node_agg :
            forward_args (tuple, optional): additional arguments to model.forward
                beyond x and edge_index
        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`None`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop,]`
                `edge_imp`: :obj:`torch.Tensor, [edge_index.shape[1],]`
                `graph`: :obj:`torch_geometric.data.Data`
        """

        if (label is None) and (y is None):
            raise ValueError(
                "Either label or y should be provided for Integrated Gradients"
            )

        label = y if label is None else label

        self.model.eval()
        grads = torch.zeros(steps + 1, *x.shape).to(x.device)
        baseline = torch.zeros_like(x).to(
            x.device
        )  # TODO: baseline all 0s, all 1s, ...?
        for i in range(steps + 1):
            with torch.no_grad():
                temp_x = baseline + (float(i) / steps) * (x.clone() - baseline)
            temp_x.requires_grad = True
            if forward_kwargs is None:
                output = self.model(temp_x, edge_index)
            else:
                output = self.model(temp_x, edge_index, **forward_kwargs)
            loss = self.criterion(output, label)
            loss.backward()
            grad = temp_x.grad
            grads[i] = grad

        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = torch.mean(grads, axis=0)
        integrated_gradients = (x - baseline) * avg_grads

        exp = Explanation(
            node_imp=node_agg(integrated_gradients, dim=1),
        )

        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp
