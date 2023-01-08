#!/usr/bin/env python3

import random
import requests
from dataset import GraphDataset, load_graphs
from pathlib import Path

ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H", "*"]

here = Path(__file__).parent
alkane_url = "https://dataverse.harvard.edu/api/access/datafile/6405497"
alkane_datapath = here / "data/alkane_carbonyl.npz"
alkane_datapath.parent.mkdir(exist_ok=True, parents=True)
if not alkane_datapath.is_file():
    file_contents = requests.get(alkane_url)
    alkane_datapath.write_bytes(file_contents.content)

class AlkaneCarbonyl(GraphDataset):
    def __init__(
        self,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        data_path: str = alkane_datapath,
        device=None,
        downsample=True,
        downsample_seed=None,
    ):
        """
        Args:
            split_sizes (tuple):
            seed (int, optional):
            data_path (str, optional):
        """

        self.device = device
        self.downsample = downsample
        self.downsample_seed = downsample_seed

        self.graphs = load_graphs(data_path)

        # Downsample because of extreme imbalance:
        yvals = [self.graphs[i].y for i in range(len(self.graphs))]

        zero_bin = []
        one_bin = []

        if downsample:
            for i in range(len(self.graphs)):
                if self.graphs[i].y == 0:
                    zero_bin.append(i)
                else:
                    one_bin.append(i)

            # Sample down to keep the dataset balanced
            random.seed(downsample_seed)
            keep_inds = random.sample(zero_bin, k=2 * len(one_bin))

            self.graphs = [self.graphs[i] for i in (keep_inds + one_bin)]

        super().__init__(
            name="AklaneCarbonyl", seed=seed, split_sizes=split_sizes, device=device
        )
