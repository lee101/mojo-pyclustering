"""K-means with the pyclustering API."""

from __future__ import annotations

import numpy as np

from pyclustering._lib import addr, exact_int, f64, lib
from pyclustering.cluster._common import labels_to_clusters, metric_parts, points, predict
from pyclustering.cluster.encoder import type_encoding
from pyclustering.utils.metric import distance_metric, type_metric


class kmeans:
    def __init__(self, data, initial_centers, tolerance=0.001, ccore=True, **kwargs):
        self._data = points(data)
        self._centers = points(initial_centers).copy()
        if self._data.shape[1] != self._centers.shape[1]:
            raise ValueError("Dimension of data and initial centers must be equal.")
        self._tolerance = float(tolerance)
        self._itermax = exact_int(kwargs.get("itermax", 100), "itermax")
        self._metric = kwargs.get(
            "metric", distance_metric(type_metric.EUCLIDEAN_SQUARE)
        )
        self._observer = kwargs.get("observer")
        if self._observer is not None:
            raise NotImplementedError("iteration observers are not covered")
        if self._tolerance < 0 or self._itermax < 0:
            raise ValueError("Tolerance and maximum iterations must be non-negative.")
        self._clusters = []
        self._labels = None
        self._total_wce = 0.0

    def process(self):
        if self._itermax == 0:
            return self
        rows, dimensions = self._data.shape
        amount = len(self._centers)
        labels = np.empty(rows, dtype=np.int64)
        distances = np.empty(rows, dtype=np.float64)
        sums = np.empty((amount, dimensions), dtype=np.float64)
        counts = np.empty(amount, dtype=np.int64)
        result = np.empty(1, dtype=np.float64)
        kind, degree, ranges = metric_parts(self._metric, dimensions, self._data)
        active = lib().mpc_kmeans(
            addr(self._data),
            addr(self._centers),
            addr(labels),
            addr(distances),
            addr(sums),
            addr(counts),
            addr(result),
            rows,
            dimensions,
            amount,
            self._itermax,
            self._tolerance,
            kind,
            degree,
            addr(ranges),
        )
        self._centers = self._centers[:active].copy()
        self._labels = labels
        self._clusters = labels_to_clusters(labels, active)
        self._total_wce = float(result[0])
        return self

    def predict(self, points_to_predict):
        if not self._clusters:
            return []
        return predict(points_to_predict, self._centers, self._metric)

    def get_clusters(self):
        return self._clusters

    def get_centers(self):
        return self._centers.tolist()

    def get_total_wce(self):
        return self._total_wce

    def get_cluster_encoding(self):
        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION
