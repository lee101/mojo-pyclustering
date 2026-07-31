"""PAM k-medoids with point and distance-matrix inputs."""

from __future__ import annotations

import numpy as np

from pyclustering._lib import addr, exact_int, f64, i64, lib
from pyclustering.cluster._common import labels_to_clusters, metric_parts, points, predict
from pyclustering.cluster.encoder import type_encoding
from pyclustering.utils.metric import distance_metric, type_metric


class kmedoids:
    def __init__(self, data, initial_index_medoids, tolerance=0.0001, ccore=True, **kwargs):
        self._data_type = kwargs.get("data_type", "points")
        if self._data_type not in ("points", "distance_matrix"):
            raise TypeError(f"Unknown type of data is specified '{self._data_type}'")
        self._data = points(data)
        if self._data_type == "distance_matrix" and self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Distance matrix must be square.")
        self._medoids = i64(initial_index_medoids, copy=True)
        self._tolerance = float(tolerance)
        self._itermax = exact_int(kwargs.get("itermax", 200), "itermax")
        self._metric = kwargs.get(
            "metric", distance_metric(type_metric.EUCLIDEAN_SQUARE)
        )
        if len(self._medoids) == 0:
            raise ValueError("Initial medoids are empty.")
        if len(np.unique(self._medoids)) != len(self._medoids):
            raise ValueError("Initial medoid indexes must be unique.")
        if np.any(self._medoids < 0) or np.any(self._medoids >= len(self._data)):
            raise IndexError("Medoid index is outside the input data.")
        if self._tolerance < 0 or self._itermax < 0:
            raise ValueError("Tolerance and maximum iterations must be non-negative.")
        self._clusters = []
        self._labels = None

    def process(self):
        if self._itermax == 0:
            return self
        rows = len(self._data)
        dimensions = self._data.shape[1]
        labels = np.empty(rows, dtype=np.int64)
        first = np.empty(rows, dtype=np.float64)
        second = np.empty(rows, dtype=np.float64)
        costs = np.empty((len(self._medoids), rows), dtype=np.float64)
        kind, degree, ranges = metric_parts(
            self._metric, dimensions, None if self._data_type == "distance_matrix" else self._data
        )
        active = lib().mpc_kmedoids(
            addr(self._data),
            addr(self._medoids),
            addr(labels),
            addr(first),
            addr(second),
            addr(costs),
            rows,
            dimensions,
            len(self._medoids),
            self._itermax,
            self._tolerance,
            int(self._data_type == "distance_matrix"),
            kind,
            degree,
            addr(ranges),
        )
        self._medoids = self._medoids[:active].copy()
        self._labels = labels
        self._clusters = labels_to_clusters(labels, active)
        return self

    def predict(self, points_to_predict):
        if not self._clusters:
            return []
        if self._data_type != "points":
            raise TypeError("predict is only defined for point data")
        return predict(points_to_predict, self._data[self._medoids], self._metric)

    def get_clusters(self):
        return self._clusters

    def get_medoids(self):
        return self._medoids.astype(int).tolist()

    def get_cluster_encoding(self):
        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION
