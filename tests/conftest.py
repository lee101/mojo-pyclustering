import importlib
import os
import sys
from types import SimpleNamespace

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")


def _load_upstream():
    original_path = sys.path[:]
    sys.path = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != os.path.abspath(PYTHON_DIR)
    ]
    try:
        modules = {
            name: importlib.import_module(name)
            for name in (
                "pyclustering",
                "pyclustering.cluster.kmeans",
                "pyclustering.cluster.kmedians",
                "pyclustering.cluster.kmedoids",
                "pyclustering.cluster.dbscan",
                "pyclustering.cluster.center_initializer",
                "pyclustering.utils.metric",
            )
        }
        reference = SimpleNamespace(
            version=modules["pyclustering"].__version__,
            kmeans=modules["pyclustering.cluster.kmeans"].kmeans,
            kmedians=modules["pyclustering.cluster.kmedians"].kmedians,
            kmedoids=modules["pyclustering.cluster.kmedoids"].kmedoids,
            dbscan=modules["pyclustering.cluster.dbscan"].dbscan,
            random_center_initializer=modules[
                "pyclustering.cluster.center_initializer"
            ].random_center_initializer,
            kmeans_plusplus_initializer=modules[
                "pyclustering.cluster.center_initializer"
            ].kmeans_plusplus_initializer,
            distance_metric=modules["pyclustering.utils.metric"].distance_metric,
            type_metric=modules["pyclustering.utils.metric"].type_metric,
        )
    finally:
        for name in list(sys.modules):
            if name == "pyclustering" or name.startswith("pyclustering."):
                del sys.modules[name]
        sys.path = original_path
    return reference


UPSTREAM = _load_upstream()


@pytest.fixture(scope="session")
def upstream():
    return UPSTREAM
