"""DBSCAN with pyclustering's neighbor-count convention."""

from __future__ import annotations

import numpy as np

from pyclustering._lib import addr, exact_int, lib
from pyclustering.cluster._common import labels_to_clusters, points
from pyclustering.cluster.encoder import type_encoding


class dbscan:
    def __init__(self, data, eps, neighbors, ccore=True, **kwargs):
        self._data_type = kwargs.get("data_type", "points")
        if self._data_type not in ("points", "distance_matrix"):
            raise TypeError(f"Unknown type of data is specified '{self._data_type}'")
        self._data = points(data)
        if self._data_type == "distance_matrix" and self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Distance matrix must be square.")
        self._eps = float(eps)
        self._neighbors = exact_int(neighbors, "neighbors")
        if self._eps < 0:
            raise ValueError("Connectivity radius must be non-negative.")
        if self._neighbors < 0:
            raise ValueError("Minimum neighbor count must be non-negative.")
        self._clusters = []
        self._noise = []
        self._labels = None

    def process(self):
        rows = len(self._data)
        dimensions = self._data.shape[1]
        labels = np.empty(rows, dtype=np.int64)
        visited = np.empty(rows, dtype=np.int64)
        queue = np.empty(rows, dtype=np.int64)
        neighbors = np.empty(rows, dtype=np.int64)
        amount = lib().mpc_dbscan(
            addr(self._data),
            addr(labels),
            addr(visited),
            addr(queue),
            addr(neighbors),
            rows,
            dimensions,
            self._eps,
            self._neighbors,
            int(self._data_type == "distance_matrix"),
        )
        self._labels = labels
        self._clusters = labels_to_clusters(labels, amount)
        self._noise = np.flatnonzero(labels < 0).astype(int).tolist()
        return self

    def get_clusters(self):
        return self._clusters

    def get_noise(self):
        return self._noise

    def get_cluster_encoding(self):
        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION
