#!/usr/bin/env python3

import torch
import numpy as np
from time import time
from pathlib import Path
from typing import NamedTuple, Optional, Any
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn.functional import normalize
from tqdm import tqdm

from parser import argument_parser
from datasets import DATASET_LOADERS
from model import Model
# from torch_geometric.nn.models.gnn_explainer import GNNExplainer
from batched_explainer import BatchedGNNExplainer as GNNExplainer
from PGMEx import PGMExplainer
from intgrad import IntegratedGradExplainer

DEVICE = "cpu"
HERE = Path(__file__).parent
CONVERGENCE_DIR = HERE / "convergence_files"
CONVERGENCE_DIR.mkdir(exist_ok=True)

class PerformanceResults(NamedTuple):
    train_acc: float
    val_acc: float
    test_acc: float
    train_auroc: float
    val_auroc: float
    test_auroc: float
    test_f1_score: float

# region main ---------------------------------

def main(
    dataset: str,
    arch: str,
    explainer: str,
    num_layers: int = 3,
    batch_size: int = 200,
    seed: int = 912,
    epochs:int = 150,
    model_saving_lag: int = 25,
    vanilla_mode: bool = False,
    lr_gnn=0.01,
    explainer_iters=5,
    correct_sampling_percent=0.4,
    explanations_lag=20,
    explanation_topk_thresh=0.3,
    lr_gnnex=0.01,
    explainer_epochs=200,
):
    out_dir = HERE / f"{dataset}-{arch}"
    out_dir.mkdir(exist_ok = True)
    convergence_file_stem = f"loss-lrgnn_{lr_gnn}-seed_{seed}"
    best_model_path = out_dir / f"{convergence_file_stem}-best.pth"
    
    # Initialize all placeholder variables that are updated in the loop
    preds = None
    use_explanations = False
    best_auroc_val = 0
    oversmoothing = 0
    
    # Load explainer, data, model, optimizer, loss function
    dataset_loader = DATASET_LOADERS.get(dataset)
    if dataset_loader is None:
        raise ValueError("Invalid dataset")
    train_loader, val_loader, test_loader = dataset_loader(
        seed=seed, batch_size=batch_size, split_train_val_test=True
    )
    n_feat = guess_n_features(train_loader)
    model = Model(
        nhid=32,
        nfeat=n_feat,
        nclass=2,
        dropout=0.0,
        num_layers=num_layers,
        gnn_arch=arch,
    ).to(DEVICE)
    sample_weights = cal_weights_model(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_gnn)
    criterion = torch.nn.CrossEntropyLoss(weight=sample_weights)
    explainer = get_explainer(
        explainer=explainer, 
        model=model,
        explainer_epochs=explainer_epochs,
        lr_gnnex=lr_gnnex,
        criterion=criterion,
    )

    # Begin train-test-explanation loop
    for epoch in tqdm(range(epochs)):
        epoch_start_time = time()
        if not vanilla_mode and epoch > explanations_lag:
            use_explanations = True
        avg_loss = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            preds=preds,
            explainer=explainer,
            use_explanations=use_explanations,
            explainer_iters=explainer_iters,
            correct_sampling_percent=correct_sampling_percent,
            explanation_topk_thresh=explanation_topk_thresh,
        )
        output_train, performance = evaluate_performance(
            train_loader, val_loader, test_loader, model
        )
        preds = output_train
        model_saving_lag = 25 if model_saving_lag is None else model_saving_lag
        if epoch >= model_saving_lag and performance.val_auroc >= best_auroc_val:
            best_auroc_val = performance.val_auroc
            torch.save(
                model.state_dict(), 
                best_model_path
            )
        log_progress(
            epoch, avg_loss, performance, oversmoothing, convergence_file_stem, epoch_start_time
        )
    
    # Oversmoothing
    oversmoothing = calculate_oversmoothing(
        model=model,
        dataset_loader=dataset_loader,
        seed=seed,
        batch_size=batch_size,
        best_model_path=best_model_path,
    )
    log_progress(
        epoch, avg_loss, performance, oversmoothing, convergence_file_stem, epoch_start_time
    )

# endregion main

# region Functions ---------------------------------

def guess_n_features(train_loader) -> int:
    # TODO test that this works
    first_batch = train_loader.dataset[0]
    # print(
    #     "num_nodes", first_batch.num_nodes, 
    #     "num_edges", first_batch.num_edges, 
    #     "num_node_features", first_batch.num_node_features, 
    #     "num_edge_features", first_batch.num_edge_features
    # )
    return first_batch.num_node_features

def cal_weights_model(dataset):
    "Calculate weights for weighted cross entropy loss to address data imbalance"
    labels = []
    for data in dataset:
        labels += data.y.tolist()
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negative = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    weights = torch.tensor([n_full / (2 * n_negative), n_full / (2 * n_positive)])
    return weights

def get_explainer(
    explainer: str,
    model: Model,
    explainer_epochs: Optional[int] = None,
    lr_gnnex: Optional[float] = None,
    criterion: Optional[Any] = None
):
    if explainer == "gnn_explainer":
        return GNNExplainer(
            model, epochs=explainer_epochs, lr=lr_gnnex, return_type="raw", log=False
        )
        return
    if explainer == "pgmexplainer":
        return PGMExplainer(model=model, graph=None)
    if explainer == "intgradexplainer":
        return IntegratedGradExplainer(model, criterion)
    raise ValueError(
        '`explainer` must be one of: ("gnn_explainer", "pgmexplainer", "intgradexplainer")'
    )

def train(
    model,
    train_loader,
    optimizer,
    criterion,
    preds,
    explainer,
    use_explanations: bool,
    explainer_iters: int=5,
    correct_sampling_percent: float=0.05,
    explanation_topk_thresh: float=0.25,
):
    losses = []
    for idx, data in enumerate(train_loader):  
        model.eval()
        # NOTE: Use `scores_edges = weights_graphs[idx.item()]` if you want to
        # use the explanations that were obtained in the previous loop
        input_data = data.x
        scores = get_default_scores(data, explainer)
        if use_explanations and preds is not None:
            scores = []
            # Use the explanations that were obtained in the previous loop
            # Uses predictions for previous epoch from a selected batch through 'idx'
            sampled_correct_indices = sample_correct_indices(
                pred=preds[idx], 
                gtruth=data.y,
                correct_sampling_percent=correct_sampling_percent
            )
            scores = get_explainer_scores(
                data=data,
                model=model,
                explainer=explainer,
                sampled_correct_indices=sampled_correct_indices,
                explainer_iters=explainer_iters,
                use_explanations=use_explanations,
                explanation_topk_thresh=explanation_topk_thresh,
            )
            if isinstance(explainer, PGMExplainer) or isinstance(explainer, IntegratedGradExplainer):
                input_data = scores
                scores = None
            

        model.train() # Change to training mode
        optimizer.zero_grad()  # Clear gradients.
        out = model(input_data, data.edge_index, scores, data.batch)
        loss = criterion(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        losses.append(loss)
    avg_loss = sum(losses) / len(train_loader.dataset)
    return avg_loss

def get_default_scores(data, explainer):
    if isinstance(explainer, GNNExplainer):
        return torch.ones(data.edge_index.shape[1])
    if isinstance(explainer, PGMExplainer):
        return None
    if isinstance(explainer, IntegratedGradExplainer):
        return None
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")

def sample_correct_indices(pred, gtruth, correct_sampling_percent: float = 0.5) -> np.ndarray:
    """
    Takes predictions from model, returns the indices of a subset of the
    correct predictions
    """
    cor_idx = np.where(pred.cpu() == gtruth)[0]
    samples = int(correct_sampling_percent * cor_idx.size)
    if samples < 1:
        samples = 1
    if cor_idx.shape[0] == 0:
        return np.array([])
    else:
        sampled_idx = np.random.choice(cor_idx, samples)
        return sampled_idx

def get_explainer_scores(
    data,
    model,
    explainer,
    sampled_correct_indices,
    explainer_iters: int,
    use_explanations: bool, 
    explanation_topk_thresh: float,
):
    scores = []
    for i in range(data.num_graphs):
        if i in sampled_correct_indices:
            # Generating explanations for sampled graphs from batch
            graph_scores = _get_sampled_nodes_or_edge_scores(
                data=data,
                idx=i,
                model=model,
                explainer=explainer,
                explainer_iters=explainer_iters,
                use_explanations=use_explanations,
                explanation_topk_thresh=explanation_topk_thresh,

            )
        else:
            # Default weights for non-sampled graphs
            graph_scores = _get_remaining_nodes_or_edge_scores(
                data=data, idx=i, explainer=explainer,
            )
        scores.extend(graph_scores)

    if isinstance(explainer, GNNExplainer):
        scores = torch.Tensor(scores)
    if isinstance(explainer, PGMExplainer) or isinstance(explainer, IntegratedGradExplainer):
        # changing shape to match data.x nodes
        scores = torch.tensor(scores).view(data.x.shape[0], 1)  
        # applying weights to nodes
        scores = scores * data.x
    return scores
    

def _get_sampled_nodes_or_edge_scores(
    data,
    idx,
    model,
    explainer,
    explainer_iters: int,
    use_explanations: bool, 
    explanation_topk_thresh: float,
):
    if isinstance(explainer, GNNExplainer):
        scores_edges = normalized_explanation_median(
            data[idx], explainer_iters, explainer, use_explanations, explanation_topk_thresh
        )
        scores_edges = scores_edges.detach().cpu().numpy()
        return scores_edges
    if isinstance(explainer, PGMExplainer):
        explainer = PGMExplainer(model, data[idx])
        _, p_values, _ = explainer.explain(
            num_samples=1000,
            percentage=10,
            top_node=3,
            p_threshold=0.05,
            pred_threshold=0.1,
        )
        scores_nodes = [1 - j for j in p_values]  
        scores_nodes = torch.tensor(scores_nodes, dtype=data[idx].x.dtype)
        return scores_nodes
    if isinstance(explainer, IntegratedGradExplainer):
        model_kwargs = {"batch": data[idx].batch, "edge_weight": None}
        exp = explainer.get_explanation_graph(
            edge_index=data[idx].edge_index,
            x=data[idx].x,
            y=data[idx].y,
            forward_kwargs=model_kwargs,
        )
        scores_nodes = exp.node_imp
        scores_nodes = normalize(scores_nodes, dim=0)
        scores_nodes = scores_nodes.detach().cpu()
        return scores_nodes
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")

def _get_remaining_nodes_or_edge_scores(data, idx, explainer):
    if isinstance(explainer, GNNExplainer):
        remaining_edges = torch.ones_like(data[idx].edge_index[1])
        remaining_edges = remaining_edges.detach().cpu().numpy()
        return remaining_edges
    if isinstance(explainer, PGMExplainer):
        return torch.ones(data[idx].x.shape[0], dtype=data[idx].x.dtype)  
    if isinstance(explainer, IntegratedGradExplainer):
        remaining_nodes = torch.ones(data[idx].x.shape[0])
        remaining_nodes = remaining_nodes.detach().cpu()
        return remaining_nodes
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")

def normalized_explanation_median(
    data,
    iters: int,
    explainer: GNNExplainer,
    use_explanations: bool,
    explanation_topk_thresh: float,
):
    "Finds the normalized median of multiple explanations on the same data point"
    weigths_iters = []
    for it in range(iters):
        _, scores_edges = explainer.explain_graph(
            x = data.x, 
            edge_index = data.edge_index, 
            edge_weight=None, 
            use_explanations=use_explanations
        )
        weigths_iters.append(scores_edges)

    scores_edges = torch.stack(weigths_iters).median(0)[0]
    # Normalise weights
    scores_edges = (scores_edges - scores_edges.min()) / (
        scores_edges.max() - scores_edges.min()
    )
    thresh = scores_edges.topk(int(explanation_topk_thresh * data.edge_index.shape[1]))[0][-1]
    scores_edges = torch.where(scores_edges >= thresh, 1.0, 0.0)
    return scores_edges

def test(loader, model):
    model.eval()
    preds = []
    labels = []
    for data in loader:
        out = model(data.x, data.edge_index, None, data.batch)
        pred = out.argmax(dim=1)
        preds.append(pred)
        labels.append(data.y)
    preds  = torch.cat(preds)
    labels = torch.cat(labels)
    accuracy = (preds == labels).float().mean()
    return preds, labels, accuracy

def evaluate_performance(train_loader, val_loader, test_loader, model):
    output_train, labels_train, train_acc = test(train_loader, model)
    output_val, labels_val, val_acc = test(val_loader, model)
    output_test, labels_test, test_acc = test(test_loader, model)
    train_auroc = roc_auc_score(labels_train, output_train)
    val_auroc   = roc_auc_score(labels_val, output_val)
    test_auroc  = roc_auc_score(labels_test, output_test)
    test_f1_score = f1_score(labels_test, output_test)
    performance = PerformanceResults(
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        train_auroc=train_auroc,
        val_auroc=val_auroc,
        test_auroc=test_auroc,
        test_f1_score=test_f1_score,
    )
    return output_train, performance

def log_progress(
    epoch: int,
    avg_loss: float,
    performance: PerformanceResults,
    oversmoothing: float,
    convergence_file_stem: str,
    epoch_start_time: float,
):
    metrics = {
        "Epoch": epoch,
        "Train Loss": avg_loss,
        "Train Acc": performance.train_acc,
        "Test Acc": performance.test_acc,
        "Train AUROC": performance.train_auroc,
        "Val AUROC": performance.val_auroc,
        "Test AUROC": performance.test_auroc,
        "Test F1": performance.test_f1_score,
        "Val Acc": performance.val_acc,
        "Oversmoothing": oversmoothing,
    }
    metrics_formatted = [
        f"{metric_name}: {metric_value:.4f}"
        for metric_name, metric_value in metrics.items()
    ]
    progress_string = ", ".join(metrics_formatted)
    if epoch % 25 == 0:
        print(progress_string)
    with open(CONVERGENCE_DIR / f"{convergence_file_stem}.csv", "a") as f:
        f.write(progress_string + "\n")
    # print(f"Elapsed: {time() - epoch_start_time:.3f}s")

def calculate_oversmoothing(model, dataset_loader, seed, batch_size, best_model_path):
    graph_embedding = torch.Tensor()
    graph_label     = torch.Tensor()
    model.eval()
    dataset = dataset_loader(
        seed=seed, batch_size=batch_size, split_train_val_test=False
    )
    model.load_state_dict(torch.load(best_model_path))
    for data in dataset:
        embedding = model.embed(data.x, data.edge_index, None, data.batch)
        graph_embedding = torch.cat((graph_embedding, embedding))
        graph_label = torch.cat((graph_label, data.y))
    oversmoothing = calculate_gdr(graph_label, graph_embedding)
    return oversmoothing

def calculate_gdr(label, embedding):
    X_labels = []
    for i in label.unique():
        X_label = embedding[label == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.0] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    dis_intra = 0.0
    for i in label.unique():
        x2 = np.sum(np.square(X_labels[int(i)]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[int(i)], X_labels[int(i)].T)
        dis_intra += np.mean(dists)
    dis_intra /= label.unique().shape[0]

    dis_inter = 0.0
    for i in range(label.unique().shape[0] - 1):
        for j in range(i + 1, label.unique().shape[0]):
            x2_i = np.sum(np.square(X_labels[int(i)]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[int(j)]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(label.unique().shape[0] * (label.unique().shape[0] - 1) / 2)
    dis_inter /= num_inter

    return dis_inter / dis_intra

# endregion

if __name__ == "__main__":
    args = argument_parser.parse_known_args()[0]
    # print(args)
    main(**vars(args))
