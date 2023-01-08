#!/usr/bin/env python3

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_graphs(datapath: str):
    """
    Extracts datasets from a format consistent with that used by Sanchez-Lengeling et al., Neurips 2020
    TODO: replace path with Harvard Dataverse loading
    Args:
        dir_path (str): Path to directory containing all graphs
        smiles_df_path (str): Path to CSV file containing all information about SMILES
            representations of the molecules.
    :rtype: :obj:`(List[torch_geometric.data.Data], List[List[Explanation]], List[int])`
    Returns:
        all_graphs (list of `torch_geometric.data.Data`): List of all graphs in the
            dataset
    """

    # att = np.load(os.path.join(dir_path, 'true_raw_attribution_datadicts.npz'),
    #         allow_pickle = True)
    # X = np.load(os.path.join(dir_path, 'x_true.npz'), allow_pickle = True)
    # y = np.load(os.path.join(dir_path, 'y_true.npz'), allow_pickle = True)
    data = np.load(datapath, allow_pickle=True)
    att, X, y, df = data["attr"], data["X"], data["y"], data["smiles"]

    # ylist = [y['y'][i][0] for i in range(y['y'].shape[0])]
    ylist = [y[i][0] for i in range(y.shape[0])]

    # att = att['datadict_list']
    X = X[0]

    all_graphs = []

    for i in range(len(X)):
        x = torch.from_numpy(X[i]["nodes"])
        edge_attr = torch.from_numpy(X[i]["edges"])
        y = torch.tensor([ylist[i]], dtype=torch.long)

        # Get edge_index:
        e1 = torch.from_numpy(X[i]["receivers"]).long()
        e2 = torch.from_numpy(X[i]["senders"]).long()

        edge_index = torch.stack([e1, e2])

        data_i = Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index)

        all_graphs.append(data_i)  # Add to larger list

    return all_graphs


class GraphDataset:
    def __init__(self, name, split_sizes=(0.7, 0.2, 0.1), seed=None, device=None):

        self.name = name

        self.seed = seed
        self.device = device

        if split_sizes[1] > 0:
            self.train_index, self.test_index = train_test_split(
                torch.arange(start=0, end=len(self.graphs)),
                test_size=split_sizes[1] + split_sizes[2],
                random_state=self.seed,
                shuffle=True,
            )
        else:
            self.test_index = None
            self.train_index = torch.arange(start=0, end=len(self.graphs))

        if split_sizes[2] > 0:
            self.test_index, self.val_index = train_test_split(
                self.test_index,
                test_size=split_sizes[2] / (split_sizes[1] + split_sizes[2]),
                random_state=self.seed,
                shuffle=True,
            )

        else:
            self.val_index = None

        self.Y = torch.tensor([self.graphs[i].y for i in range(len(self.graphs))]).to(
            self.device
        )

    def get_data_list(
        self,
        index,
    ):
        data_list = [self.graphs[i].to(self.device) for i in index]

        return data_list

    def get_loader(self, index, batch_size=16, **kwargs):

        data_list = self.get_data_list(index)

        for i in range(len(data_list)):
            data_list[i].exp_key = [i]

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

        return loader

    def get_train_loader(self, batch_size=16):
        return self.get_loader(index=self.train_index, batch_size=batch_size)

    def get_train_list(self):
        return self.get_list(index=self.train_index)

    def get_test_loader(self):
        assert self.test_index is not None, "test_index is None"
        return self.get_loader(index=self.test_index, batch_size=1)

    def get_test_list(self):
        assert self.test_index is not None, "test_index is None"
        return self.get_list(index=self.test_index)

    def get_val_loader(self):
        assert self.test_index is not None, "val_index is None"
        return self.get_loader(index=self.val_index, batch_size=1)

    def get_val_list(self):
        assert self.val_index is not None, "val_index is None"
        return self.get_list(index=self.val_index)

    def get_train_w_label(self, label):
        inds_to_choose = (self.Y[self.train_index] == label).nonzero(as_tuple=True)[0]
        in_train_idx = inds_to_choose[
            torch.randint(low=0, high=inds_to_choose.shape[0], size=(1,))
        ]
        chosen = self.train_index[in_train_idx.item()]

        return self.graphs[chosen]

    def get_test_w_label(self, label):
        assert self.test_index is not None, "test_index is None"
        inds_to_choose = (self.Y[self.test_index] == label).nonzero(as_tuple=True)[0]
        in_test_idx = inds_to_choose[
            torch.randint(low=0, high=inds_to_choose.shape[0], size=(1,))
        ]
        chosen = self.test_index[in_test_idx.item()]

        return self.graphs[chosen]

    def download(self):
        pass

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
