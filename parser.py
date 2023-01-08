#!/usr/bin/env python3

import argparse
argument_parser = argparse.ArgumentParser()

# Core arguments --------------------------------

argument_parser.add_argument(
    "--dataset", type=str, default="alkane",
    choices=("mutag", "DD", "alkane", "PROTEIN"),
    help="Graph classification dataset",
)
argument_parser.add_argument(
    "--arch", type=str, default="gcn",
    choices=("gcn", "graphconv", "leconv"),
    help="GNN Architecutures",
)
argument_parser.add_argument(
    "--explainer", type=str, default="gnn_explainer",
    choices=("gnn_explainer", "pgmexplainer", "intgradexplainer"),
    help="Explainer method to use. Ignored if flag --vanilla_mode is used",
)

# Architecture hyperparams --------------------------------

argument_parser.add_argument(
    "--num_layers", type=int, default=3, 
    help="Number of GNN layers"
)
argument_parser.add_argument(
    "--batch_size", type=int, default=200, 
    help="Batch size for the dataloader"
)
argument_parser.add_argument(
    "--seed", type=int, default=912, 
    help="Random seed."
)

# Training hyperparams --------------------------------

argument_parser.add_argument(
    "--epochs", default=150, type=int,
    help = "Number of epochs of the top-level loop"
)
argument_parser.add_argument(
    "--lr_gnn", default=0.01, type=float,
    help = "Learning rate of the GNN"
)
argument_parser.add_argument(
    "--lr_gnnex", default=0.01, type=float,
    help = "Learning rate of the GNN Explainer"
)
argument_parser.add_argument(
    "--explainer_iters", default=5, type=int,
    help = ""
)
argument_parser.add_argument(
    "--explainer_epochs", default=200, type=int,
    help = ""
)
argument_parser.add_argument(
    "--correct_sampling_percent", default=0.4, type=float,
    help = ""
)
argument_parser.add_argument(
    "--explanation_topk_thresh", default=0.3, type=float,
    help = ""
)
argument_parser.add_argument(
    "--explanations_lag", default=20, type=int,
    help = ""
)
argument_parser.add_argument(
    "--model_saving_lag", type=int, default=25, 
    help="Number of epochs to wait before saving model checkopints."
)

# Boolean flags ---------------------------------

argument_parser.add_argument(
    "--vanilla_mode", action="store_true", default=False,
    help="Whether to run training in vanilla mode (i.e. not using explanation)",
)