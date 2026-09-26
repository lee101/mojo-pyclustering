"""PAM k-medoids with point and distance-matrix inputs."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyclustering._lib import addr, exact_int, i64, lib
from pyclustering.cluster._common import labels_to_clusters, metric_parts, points, predict
from pyclustering.cluster.encoder import type_encoding
from pyclustering.utils.metric import distance_metric, type_metric

# Scoring one candidate costs O(rows * dimensions), so the sweep is
# compute-bound and fans out over a thread pool.  Small inputs stay serial:
# below this many candidates the hand-off costs more than the work it saves.
_PARALLEL_MIN_CANDIDATES = 256
_MAX_WORKERS = min(32, os.cpu_count() or 1)


def _candidate_parts(rows: int, parallel: bool) -> list[tuple[int, int]]:
    if not parallel or rows < _PARALLEL_MIN_CANDIDATES:
        return [(0, rows)]
    workers = min(_MAX_WORKERS, rows)
    parts = []
    for index in range(workers):
        first = (index * rows) // workers
        last = ((index + 1) * rows) // workers
        if first < last:
            parts.append((first, last))
    return parts


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
        active = len(self._medoids)
        labels = np.empty(rows, dtype=np.int64)
        first = np.empty(rows, dtype=np.float64)
        second = np.empty(rows, dtype=np.float64)
        costs = np.empty((active, rows), dtype=np.float64)
        kind, degree, ranges = metric_parts(
            self._metric, dimensions, None if self._data_type == "distance_matrix" else self._data
        )
        metric_args = (
            int(self._data_type == "distance_matrix"),
            kind,
            degree,
            addr(ranges),
        )
        common = (
            addr(self._data),
            addr(self._medoids),
            addr(labels),
            addr(first),
            addr(second),
            rows,
            dimensions,
            active,
        )
        parts = _candidate_parts(rows, _MAX_WORKERS > 1)
        # medoid_assign returns sum(first_distances) accumulated in row order,
        # which is exactly the `previous` total the fused kernel used as its
        # convergence baseline, so the two agree bit for bit.
        previous = lib().mpc_kmedoids_init(*common, *metric_args)
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            for _ in range(self._itermax):
                if len(parts) > 1:
                    list(pool.map(
                        lambda part: lib().mpc_kmedoids_evaluate(
                            addr(self._data),
                            addr(self._medoids),
                            addr(labels),
                            addr(first),
                            addr(second),
                            addr(costs),
                            rows,
                            dimensions,
                            active,
                            *metric_args,
                            part[0],
                            part[1],
                        ),
                        parts,
                    ))
                else:
                    lib().mpc_kmedoids_evaluate(
                        addr(self._data),
                        addr(self._medoids),
                        addr(labels),
                        addr(first),
                        addr(second),
                        addr(costs),
                        rows,
                        dimensions,
                        active,
                        *metric_args,
                        parts[0][0],
                        parts[0][1],
                    )
                if lib().mpc_kmedoids_select(
                    addr(self._medoids), addr(costs), rows, active
                ) < 0:
                    break
                current = lib().mpc_kmedoids_reassign(*common, *metric_args)
                changes = previous - current
                previous = current
                if changes <= self._tolerance:
                    break
        active = lib().mpc_kmedoids_compact(*common, *metric_args)
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
