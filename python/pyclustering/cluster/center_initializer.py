"""Random and k-means++ center initialization."""

from __future__ import annotations

import random

import numpy as np


class random_center_initializer:
    def __init__(self, data, amount_centers, **kwargs):
        self._data = data
        self._amount = int(amount_centers)
        self._available = set(range(len(data)))
        random.seed(kwargs.get("random_state"))
        if self._amount <= 0:
            raise ValueError("Amount of cluster centers should be at least 1.")
        if self._amount > len(data):
            raise ValueError("Amount of cluster centers should not exceed data size.")

    def initialize(self, **kwargs):
        return_index = kwargs.get("return_index", False)
        if self._amount == len(self._data):
            return list(range(len(self._data))) if return_index else self._data[:]
        centers = []
        for _ in range(self._amount):
            index = random.randint(0, len(self._data))
            if index not in self._available:
                index = self._available.pop()
            else:
                self._available.remove(index)
            centers.append(index if return_index else self._data[index])
        return centers


class kmeans_plusplus_initializer:
    FARTHEST_CENTER_CANDIDATE = "farthest"

    def __init__(self, data, amount_centers, amount_candidates=None, **kwargs):
        self._data = np.asarray(data)
        self._amount = int(amount_centers)
        self._candidates = 3 if amount_candidates is None else amount_candidates
        if self._candidates != self.FARTHEST_CENTER_CANDIDATE:
            self._candidates = min(int(self._candidates), len(self._data))
        random.seed(kwargs.get("random_state"))
        if self._amount <= 0 or self._amount > len(self._data):
            raise ValueError("Amount of cluster centers must be within the data size.")
        if self._candidates != self.FARTHEST_CENTER_CANDIDATE and self._candidates <= 0:
            raise ValueError("Amount of center candidates should be at least 1.")

    def initialize(self, **kwargs):
        first = random.randint(0, len(self._data) - 1)
        centers = [first]
        free = set(range(len(self._data)))
        free.remove(first)
        for _ in range(1, self._amount):
            distances = np.vstack(
                [np.sum((self._data - self._data[index]) ** 2, axis=1) for index in centers]
            )
            shortest = np.min(distances, axis=0)
            if self._candidates == self.FARTHEST_CENTER_CANDIDATE:
                shortest[centers] = np.nan
                chosen = int(np.nanargmax(shortest))
            elif shortest.sum() == 0:
                chosen = min(free)
            else:
                cumulative = np.cumsum(shortest / shortest.sum())
                chosen = 0
                for _ in range(self._candidates):
                    candidate = -1
                    probability = random.random()
                    for index, boundary in enumerate(cumulative):
                        if probability < boundary:
                            candidate = index
                            break
                    if candidate == -1:
                        chosen = next(iter(free))
                    elif shortest[chosen] < shortest[candidate]:
                        chosen = candidate
            centers.append(chosen)
            free.remove(chosen)
        if kwargs.get("return_index", False):
            return centers
        return [self._data[index] for index in centers]
