"""Benchmarks against upstream pyclustering on identical inputs."""

from __future__ import annotations

import importlib
import math
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")


def load_upstream():
    original = sys.path[:]
    sys.path = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != os.path.abspath(PYTHON_DIR)
    ]
    modules = {
        name: importlib.import_module(name)
        for name in (
            "pyclustering",
            "pyclustering.cluster.kmeans",
            "pyclustering.cluster.kmedians",
            "pyclustering.cluster.kmedoids",
            "pyclustering.cluster.dbscan",
        )
    }
    reference = SimpleNamespace(
        version=modules["pyclustering"].__version__,
        kmeans=modules["pyclustering.cluster.kmeans"].kmeans,
        kmedians=modules["pyclustering.cluster.kmedians"].kmedians,
        kmedoids=modules["pyclustering.cluster.kmedoids"].kmedoids,
        dbscan=modules["pyclustering.cluster.dbscan"].dbscan,
    )
    for name in list(sys.modules):
        if name == "pyclustering" or name.startswith("pyclustering."):
            del sys.modules[name]
    sys.path = original
    return reference


upstream = load_upstream()

from pyclustering.cluster.dbscan import dbscan  # noqa: E402
from pyclustering.cluster.kmeans import kmeans  # noqa: E402
from pyclustering.cluster.kmedians import kmedians  # noqa: E402
from pyclustering.cluster.kmedoids import kmedoids  # noqa: E402


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def blob_data(count, dimensions, clusters, seed=17):
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=8.0, size=(clusters, dimensions))
    data = np.vstack(
        [rng.normal(mean, 0.7, size=(count // clusters, dimensions)) for mean in means]
    )
    return np.ascontiguousarray(data), np.ascontiguousarray(means)


def cases():
    data, centers = blob_data(100_000, 8, 6)
    yield (
        "kmeans.process (100k x 8, k=6)",
        lambda: kmeans(data, centers, itermax=30).process(),
        lambda: upstream.kmeans(data, centers, itermax=30).process(),
    )

    model_mojo = kmeans(data, centers, itermax=30).process()
    # CCORE returns centers as a list while leaving its metric in NumPy mode,
    # which makes upstream 0.10.1.2 predict raise AttributeError. The Python
    # processing path retains ndarray centers and exercises the same predictor.
    model_upstream = upstream.kmeans(
        data, centers, itermax=30, ccore=False
    ).process()
    queries, _ = blob_data(200_000, 8, 6, seed=91)
    yield (
        "kmeans.predict (200k x 8, k=6)",
        lambda: model_mojo.predict(queries),
        lambda: model_upstream.predict(queries),
    )

    data, centers = blob_data(60_000, 6, 5)
    yield (
        "kmedians.process (60k x 6, k=5)",
        lambda: kmedians(data, centers, itermax=30).process(),
        lambda: upstream.kmedians(data, centers, itermax=30).process(),
    )

    data, _ = blob_data(900, 5, 4)
    initial = [0, 225, 450, 675]
    yield (
        "kmedoids.process (900 x 5, k=4)",
        lambda: kmedoids(data, initial, itermax=30).process(),
        lambda: upstream.kmedoids(data, initial, itermax=30).process(),
    )

    data, _ = blob_data(3_000, 2, 5)
    yield (
        "dbscan.process (3k x 2)",
        lambda: dbscan(data, 1.0, 4).process(),
        lambda: upstream.dbscan(data, 1.0, 4).process(),
    )


def main():
    print(
        f"Machine: {platform.processor() or platform.machine()}; "
        f"Python {platform.python_version()}; pyclustering {upstream.version}"
    )
    print()
    print("| case | Mojo | upstream | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function in cases():
        mojo_function()
        upstream_function()
        mojo_time = best_time(mojo_function)
        upstream_time = best_time(upstream_function)
        ratio = upstream_time / mojo_time
        verdict = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{upstream_time * 1000:.2f} ms | {ratio:.2f}x {verdict} |"
        )


if __name__ == "__main__":
    main()
