from __future__ import annotations

import numpy as np

from pyclustering._lib import addr, f64, lib
from pyclustering.utils.metric import distance_metric, type_metric


def points(data) -> np.ndarray:
    array = f64(data)
    if array.ndim != 2 or len(array) == 0 or array.shape[1] == 0:
        raise ValueError("Input data must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Input data must contain only finite values.")
    return array


def metric_parts(metric: distance_metric, dimensions: int, data=None):
    kind = int(metric.get_type())
    if kind == int(type_metric.USER_DEFINED):
        raise NotImplementedError("user-defined metrics cannot cross the Mojo C ABI")
    arguments = metric.get_arguments()
    degree = float(arguments.get("degree", 2.0))
    if not np.isfinite(degree) or degree <= 0:
        raise ValueError("Metric degree must be a positive finite number.")
    ranges = arguments.get("max_range")
    if kind == int(type_metric.GOWER):
        if ranges is None:
            source = arguments.get("data", data)
            if source is None:
                raise ValueError("Gower distance requires 'data' or 'max_range'.")
            source = f64(source)
            ranges = np.max(source, axis=0) - np.min(source, axis=0)
            arguments["max_range"] = ranges
    if ranges is None:
        ranges = np.zeros(dimensions, dtype=np.float64)
    ranges = f64(ranges)
    if ranges.shape != (dimensions,):
        raise ValueError("Metric range dimensionality does not match input data.")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("Metric ranges must contain only finite values.")
    return kind, degree, ranges


def labels_to_clusters(labels: np.ndarray, amount: int) -> list[list[int]]:
    return [np.flatnonzero(labels == index).tolist() for index in range(amount)]


def predict(data, centers, metric):
    sample = points(data)
    centers = points(centers)
    if sample.shape[1] != centers.shape[1]:
        raise ValueError("Point and center dimensions must match.")
    labels = np.empty(len(sample), dtype=np.int64)
    distances = np.empty(len(sample), dtype=np.float64)
    kind, degree, ranges = metric_parts(metric, sample.shape[1], sample)
    lib().mpc_assign(
        addr(sample),
        addr(centers),
        addr(labels),
        addr(distances),
        len(sample),
        sample.shape[1],
        len(centers),
        kind,
        degree,
        addr(ranges),
    )
    return labels
