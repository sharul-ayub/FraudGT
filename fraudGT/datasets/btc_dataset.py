import itertools
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .aml_dataset import ports, to_adj_nodes_with_times, z_norm
from .temporal_dataset import TemporalDataset


class BTCDataset(TemporalDataset):
    """BTC transaction edge-classification dataset.

    The processed graph follows the AML dataset contract:
    - node type: ``node``
    - forward edge type: ``('node', 'to', 'node')``
    - reverse edge type: ``('node', 'rev_to', 'node')``
    - labels and split masks live on the forward edge type
    """

    raw_csv_name = "df_final.csv"
    reserved_columns = {
        "input_address",
        "txId",
        "output_address",
        "Time step",
        "class",
    }
    valid_names = {"full", "feature_set1", "feature_set2", "feature_set3", "feature_set4", "feature_set5"}

    def __init__(self, root: str, name: str = "full", reverse_mp: bool = False,
                 add_ports: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        if self.name not in self.valid_names:
            raise ValueError(
                f"Unknown BTC dataset name '{self.name}'. Expected one of "
                f"{sorted(self.valid_names)}."
            )

        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0],weights_only=False)

        if not reverse_mp:
            for split in ["train", "val", "test"]:
                del self.data_dict[split]["node", "rev_to", "node"]

        if add_ports:
            self.ports_dict = torch.load(self.processed_paths[1])
            for split in ["train", "val", "test"]:
                self.data_dict[split] = self.add_ports_func(
                    self.data_dict[split], self.ports_dict[split]
                )

    def add_ports_func(self, data, split_ports):
        in_ports, out_ports = split_ports
        if not self.reverse_mp:
            data["node", "to", "node"].edge_attr = torch.cat(
                [data["node", "to", "node"].edge_attr, in_ports, out_ports],
                dim=1,
            )
        else:
            data["node", "to", "node"].edge_attr = torch.cat(
                [data["node", "to", "node"].edge_attr, in_ports],
                dim=1,
            )
            data["node", "rev_to", "node"].edge_attr = torch.cat(
                [data["node", "rev_to", "node"].edge_attr, out_ports],
                dim=1,
            )
        return data

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        if self.name == "full":
            return osp.join(self.root, "processed")
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return [self.raw_csv_name]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "ports.pt"]

    @property
    def raw_csv_path(self) -> str:
        return osp.join(self.root, self.raw_csv_name)

    @property
    def feature_list_path(self) -> str:
        return osp.join(self.root, self.name, "features.txt")

    def _read_header(self) -> List[str]:
        try:
            return pd.read_csv(self.raw_csv_path, nrows=0).columns.tolist()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"BTC raw CSV not found at '{self.raw_csv_path}'."
            ) from exc

    def _feature_columns(self, header: List[str]) -> List[str]:
        if self.name == "full":
            features = [col for col in header if col not in self.reserved_columns]
        else:
            if not osp.exists(self.feature_list_path):
                raise FileNotFoundError(
                    f"BTC feature set '{self.name}' requires '{self.feature_list_path}'. "
                    "Create this file with one exact CSV feature column name per line."
                )
            with open(self.feature_list_path, "r", encoding="utf-8") as handle:
                features = [
                    line.strip()
                    for line in handle
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            if not features:
                raise ValueError(
                    f"BTC feature set file '{self.feature_list_path}' does not "
                    "contain any feature columns."
                )

        missing = [col for col in features if col not in header]
        if missing:
            raise ValueError(
                f"BTC feature set '{self.name}' contains columns that are not in "
                f"{self.raw_csv_name}: {missing}"
            )
        reserved = [col for col in features if col in self.reserved_columns]
        if reserved:
            raise ValueError(
                f"BTC feature set '{self.name}' cannot use metadata/label columns "
                f"as edge features: {reserved}"
            )
        return features

    @staticmethod
    def _temporal_split_indices(timestamps: torch.Tensor):
        unique_times = torch.unique(timestamps, sorted=True)
        if unique_times.numel() < 3:
            raise ValueError("BTC temporal split requires at least three distinct time steps.")

        time_inds = [torch.where(timestamps == t)[0] for t in unique_times]
        time_totals = np.array([inds.numel() for inds in time_inds])
        target = [0.6, 0.2, 0.2]
        split_scores = {}

        for i, j in itertools.combinations(range(1, len(time_totals)), 2):
            split_totals = [
                time_totals[:i].sum(),
                time_totals[i:j].sum(),
                time_totals[j:].sum(),
            ]
            total = np.sum(split_totals)
            split_props = [v / total for v in split_totals]
            split_error = [abs(v - t) / t for v, t in zip(split_props, target)]
            split_scores[(i, j)] = max(split_error)

        i, j = min(split_scores, key=split_scores.get)
        split = [range(i), range(i, j), range(j, len(time_totals))]
        split_inds = []
        for split_range in split:
            split_inds.append(torch.cat([time_inds[idx] for idx in split_range]))
        return split_inds

    def process(self):
        header = self._read_header()
        feature_columns = self._feature_columns(header)
        required_columns = [
            "input_address",
            "txId",
            "output_address",
            "Time step",
            "class",
        ]
        missing_required = [col for col in required_columns if col not in header]
        if missing_required:
            raise ValueError(
                f"BTC raw CSV is missing required columns: {missing_required}"
            )

        usecols = required_columns + feature_columns
        dtype = {
            "input_address": "string",
            "output_address": "string",
            "txId": "int64",
            "Time step": "int64",
            "class": "int64",
        }
        dtype.update({col: "float32" for col in feature_columns})

        df_edges = pd.read_csv(self.raw_csv_path, usecols=usecols, dtype=dtype)
        if df_edges[["input_address", "output_address"]].isna().any().any():
            raise ValueError("BTC raw CSV contains missing input/output addresses.")
        df_edges = df_edges.sort_values(["Time step", "txId"], kind="mergesort").reset_index(drop=True)

        address_codes, unique_addresses = pd.factorize(
            pd.concat([df_edges["input_address"], df_edges["output_address"]], ignore_index=True),
            sort=False,
        )
        n_edges = len(df_edges)
        from_id = address_codes[:n_edges]
        to_id = address_codes[n_edges:]
        num_nodes = len(unique_addresses)

        labels = df_edges["class"].map({1: 1, 2: 0, 3: -1})
        if labels.isna().any():
            bad_labels = sorted(df_edges.loc[labels.isna(), "class"].unique().tolist())
            raise ValueError(f"BTC class column contains unsupported labels: {bad_labels}")

        timestamps = torch.tensor(df_edges["Time step"].to_numpy(), dtype=torch.long)
        y = torch.tensor(labels.to_numpy(), dtype=torch.long)
        edge_index = torch.tensor(np.vstack([from_id, to_id]), dtype=torch.long)
        edge_attr = torch.tensor(df_edges.loc[:, feature_columns].to_numpy(), dtype=torch.float32)
        tx_id = torch.tensor(df_edges["txId"].to_numpy(), dtype=torch.long)
        x = torch.ones((num_nodes, 1), dtype=torch.float32)

        known = int((y != -1).sum())
        illicit = int((y == 1).sum())
        licit = int((y == 0).sum())
        unknown = int((y == -1).sum())
        print(f"BTC feature set '{self.name}' uses {len(feature_columns)} edge features.")
        print(f"Number of BTC nodes = {num_nodes}")
        print(f"Number of BTC transactions = {n_edges}")
        print(f"Labels: illicit={illicit}, licit={licit}, unknown={unknown}, known={known}")

        train_inds, val_inds, test_inds = self._temporal_split_indices(timestamps)
        e_train = train_inds
        e_val = torch.cat([train_inds, val_inds])
        e_test = torch.cat([train_inds, val_inds, test_inds])

        self.ports_dict = {}
        self.data_dict = {}
        split_indices = {"train": train_inds, "val": val_inds, "test": test_inds}
        cumulative_indices = {"train": e_train, "val": e_val, "test": e_test}
        for split in ["train", "val", "test"]:
            inds = split_indices[split]
            e_mask = cumulative_indices[split]

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = z_norm(edge_attr[e_mask])
            masked_y = y[e_mask]
            masked_timestamps = timestamps[e_mask]
            masked_tx_id = tx_id[e_mask]
            split_mask = torch.isin(e_mask, inds)

            data = HeteroData()
            data["node"].x = x
            data["node"].num_nodes = int(x.shape[0])
            data["node", "to", "node"].edge_index = masked_edge_index
            data["node", "to", "node"].edge_attr = masked_edge_attr
            data["node", "to", "node"].y = masked_y
            data["node", "to", "node"].known_mask = masked_y != -1
            data["node", "to", "node"].split_mask = split_mask
            data["node", "to", "node"].timestamps = masked_timestamps
            data["node", "to", "node"].tx_id = masked_tx_id

            data["node", "rev_to", "node"].edge_index = masked_edge_index.flipud()
            data["node", "rev_to", "node"].edge_attr = masked_edge_attr
            data["node", "rev_to", "node"].timestamps = masked_timestamps
            data["node", "rev_to", "node"].tx_id = masked_tx_id

            data.feature_columns = feature_columns
            data.dataset_name = self.name

            adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            in_ports = ports(data["node", "to", "node"].edge_index, adj_list_in)
            out_ports = ports(data["node", "to", "node"].edge_index.flipud(), adj_list_out)
            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            for split in ["train", "val", "test"]:
                self.data_dict[split] = self.pre_transform(self.data_dict[split])

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])

    def __repr__(self) -> str:
        return f"BTC_Dataset(name={self.name})"
