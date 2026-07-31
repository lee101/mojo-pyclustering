"""The built-in pyclustering distance metrics."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class type_metric(IntEnum):
    EUCLIDEAN = 0
    EUCLIDEAN_SQUARE = 1
    MANHATTAN = 2
    CHEBYSHEV = 3
    MINKOWSKI = 4
    CANBERRA = 5
    CHI_SQUARE = 6
    GOWER = 7
    USER_DEFINED = 1000


def euclidean_distance(point1, point2):
    return np.sqrt(euclidean_distance_square(point1, point2))


def euclidean_distance_square(point1, point2):
    delta = np.asarray(point1) - np.asarray(point2)
    return np.sum(delta * delta, axis=-1)


def manhattan_distance(point1, point2):
    return np.sum(np.abs(np.asarray(point1) - np.asarray(point2)), axis=-1)


def chebyshev_distance(point1, point2):
    return np.max(np.abs(np.asarray(point1) - np.asarray(point2)), axis=-1)


def minkowski_distance(point1, point2, degree=2):
    delta = np.asarray(point1) - np.asarray(point2)
    if delta.ndim <= 1:
        total = 0.0
        for value in delta:
            total += float(value) ** degree
        return total ** (1.0 / degree)
    return np.power(np.sum(np.power(delta, degree), axis=-1), 1.0 / degree)


def canberra_distance(point1, point2):
    first, second = np.asarray(point1), np.asarray(point2)
    denominator = np.abs(first) + np.abs(second)
    return np.sum(
        np.divide(
            np.abs(first - second),
            denominator,
            out=np.zeros_like(first - second, dtype=float),
            where=denominator != 0,
        ),
        axis=-1,
    )


def chi_square_distance(point1, point2):
    first, second = np.asarray(point1), np.asarray(point2)
    denominator = np.abs(first) + np.abs(second)
    return np.sum(
        np.divide(
            (first - second) ** 2,
            denominator,
            out=np.zeros_like(first - second, dtype=float),
            where=denominator != 0,
        ),
        axis=-1,
    )


def gower_distance(point1, point2, max_range):
    first, second = np.asarray(point1), np.asarray(point2)
    ranges = np.asarray(max_range)
    values = np.divide(
        np.abs(first - second),
        ranges,
        out=np.zeros_like(first - second, dtype=float),
        where=ranges != 0,
    )
    return np.sum(values, axis=-1) / first.shape[-1]


class distance_metric:
    def __init__(self, metric_type, **kwargs):
        self._type = type_metric(metric_type)
        self._arguments = kwargs
        self._numpy = bool(kwargs.get("numpy_usage", False))
        if self._type == type_metric.USER_DEFINED and not callable(kwargs.get("func")):
            raise ValueError("User-defined metric requires callable 'func'.")
        if self._type == type_metric.GOWER:
            self._gower_ranges()

    def _gower_ranges(self):
        ranges = self._arguments.get("max_range")
        if ranges is None:
            data = self._arguments.get("data")
            if data is None:
                raise ValueError("Gower distance requires 'data' or 'max_range'.")
            data = np.asarray(data)
            ranges = np.max(data, axis=0) - np.min(data, axis=0)
            self._arguments["max_range"] = ranges
        return ranges

    def __call__(self, point1, point2):
        calculators = {
            type_metric.EUCLIDEAN: euclidean_distance,
            type_metric.EUCLIDEAN_SQUARE: euclidean_distance_square,
            type_metric.MANHATTAN: manhattan_distance,
            type_metric.CHEBYSHEV: chebyshev_distance,
            type_metric.MINKOWSKI: lambda a, b: minkowski_distance(
                a, b, self._arguments.get("degree", 2)
            ),
            type_metric.CANBERRA: canberra_distance,
            type_metric.CHI_SQUARE: chi_square_distance,
            type_metric.GOWER: lambda a, b: gower_distance(a, b, self._gower_ranges()),
            type_metric.USER_DEFINED: self._arguments.get("func"),
        }
        return calculators[self._type](point1, point2)

    def get_type(self):
        return self._type

    def get_arguments(self):
        return self._arguments

    def get_function(self):
        return self._arguments.get("func")

    def enable_numpy_usage(self):
        self._numpy = True

    def disable_numpy_usage(self):
        self._numpy = False
