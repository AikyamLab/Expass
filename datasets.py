#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from torch_geometric.data     import Data
from torch_geometric.loader   import DataLoader
from torch_geometric.datasets import TUDataset
from alkane   import AlkaneCarbonyl

DATA_DIR = Path(__file__).parent / "data/TUDataset"


# Helper functions ----------

def _append_idx(data):
    "Appends an index to each graph to track which explanations are used from the batch"
    idx = [
        Data(
            x=data[i].x,
            edge_index=data[i].edge_index,
            edge_attr=data[i].edge_attr,
            y=data[i].y,
            idx=i,
        )
        for i in range(len(data))
    ]
    return idx

def _split_tu_dataset(dataset, seed, batch_size) -> DataLoader:
    num_training = int(len(dataset) * 0.8)
    num_val      = int(len(dataset) * 0.1)
    num_test     = len(dataset) - (num_training + num_val)
    np.random.seed(seed)
    train_dataset = dataset[:num_training]
    val_dataset   = dataset[num_training : (num_training + num_val)]
    test_dataset  = dataset[(num_training + num_val) :]
    train_data = _append_idx(train_dataset)
    val_data   = _append_idx(val_dataset)
    test_data  = _append_idx(test_dataset)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def _split_molecule_dataset(dataset, seed, batch_size) -> DataLoader:
    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader  = dataset.get_test_loader()
    val_loader   = dataset.get_val_loader()
    train_data = _append_idx(train_loader.dataset)
    val_data   = _append_idx(val_loader.dataset)
    test_data  = _append_idx(test_loader.dataset)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# Data loading functions ----

def load_mutag(seed: int, batch_size: int, split_train_val_test: bool = True):
    dataset = TUDataset(root=DATA_DIR, name="MUTAG")
    dataset = dataset.shuffle()
    if not split_train_val_test:
        return DataLoader(dataset)
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    np.random.seed(seed)
    train_data = _append_idx(train_dataset)
    test_data = _append_idx(test_dataset)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = test_loader
    return train_loader, val_loader, test_loader

def load_dd(seed: int, batch_size: int, split_train_val_test: bool = True):
    dataset = TUDataset(root=DATA_DIR, name="DD")
    dataset = dataset.shuffle()
    if split_train_val_test:
        train_loader, val_loader, test_loader = _split_tu_dataset(
            dataset=dataset, seed=seed, batch_size=batch_size,
        )
        return train_loader, val_loader, test_loader
    return DataLoader(dataset)

def load_protein(seed: int, batch_size: int, split_train_val_test: bool = True):
    dataset = TUDataset(root=DATA_DIR, name="PROTEINS_full")
    dataset = dataset.shuffle()
    if split_train_val_test:
        train_loader, val_loader, test_loader = _split_tu_dataset(
            dataset=dataset, seed=seed, batch_size=batch_size,
        )
        return train_loader, val_loader, test_loader
    return DataLoader(dataset)

def load_alkane(seed: int, batch_size: int, split_train_val_test: bool = True):
    dataset = AlkaneCarbonyl(split_sizes=(0.8, 0.1, 0.1), downsample_seed=seed)
    if split_train_val_test:
        train_loader, val_loader, test_loader = _split_molecule_dataset(
            dataset=dataset, seed=seed, batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    return DataLoader(dataset)


DATASET_LOADERS = {
 "mutag": load_mutag,
 "DD": load_dd,
 "PROTEIN": load_protein,
 "alkane": load_alkane,
}
