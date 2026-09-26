"""Load and configure the Mojo shared library."""

from __future__ import annotations

import ctypes
import operator
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_PYCLUSTERING_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyclustering.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpc_metric": ([I, I, I, I, F, I], F),
    "mpc_assign": ([I, I, I, I, I, I, I, I, F, I], F),
    "mpc_kmeans": ([I] * 11 + [F, I, F, I], I),
    "mpc_kmedians": ([I] * 12 + [F, I, F, I], I),
    "mpc_kmedoids_init": ([I] * 10 + [F, I], F),
    "mpc_kmedoids_evaluate": ([I] * 11 + [F, I, I, I], None),
    "mpc_kmedoids_select": ([I, I, I, I], I),
    "mpc_kmedoids_reassign": ([I] * 10 + [F, I], F),
    "mpc_kmedoids_compact": ([I] * 10 + [F, I], I),
    "mpc_dbscan": ([I, I, I, I, I, I, I, F, I, I], I),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYCLUSTERING_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJO_PYCLUSTERING_MOJO")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_PYCLUSTERING_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(directory, name)
        for directory, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("Complex-valued data is not supported.")
    if np.issubdtype(source.dtype, np.floating) and source.dtype.itemsize > 8:
        raise TypeError("Floating-point values wider than float64 are not supported.")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.size == 0:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    if source.dtype.kind not in "iub":
        raise TypeError("Index buffers require integer values.")
    limits = np.iinfo(np.int64)
    if np.any(source < limits.min) or np.any(source > limits.max):
        raise OverflowError("Index value is outside the int64 range.")
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def exact_int(value, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool.")
    return result


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays.")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty.")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous.")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must use float64 or int64.")
    return int(array.ctypes.data)


if __name__ == "__main__":
    print(build(force=True))
