# mojo-pyclustering

`mojo-pyclustering` is a Mojo implementation of the compute-heavy core of
[pyclustering](https://github.com/annoviko/pyclustering), exposed to Python with
the same lowercase class names, core constructor arguments, `process()`
workflow, and result methods as upstream. It is an import-compatible
replacement for the explicitly covered subset:

```python
from pyclustering.cluster.kmeans import kmeans

sample = [[0.0, 0.0], [0.2, 0.1], [8.0, 8.0], [8.1, 8.2]]
model = kmeans(sample, [[0.0, 0.0], [8.0, 8.0]]).process()

print(model.get_clusters())
print(model.get_centers())
```

This prints two clusters containing indexes `[0, 1]` and `[2, 3]`, with their
final mean centers.

## Coverage

| upstream module | covered API |
| --- | --- |
| `cluster.kmeans` | `kmeans`, built-in metrics, `process`, `predict`, clusters, centers, total WCE |
| `cluster.kmedians` | `kmedians`, coordinate medians, built-in metrics, prediction and total WCE |
| `cluster.kmedoids` | PAM `kmedoids`, point data and precomputed distance matrices, prediction for point data |
| `cluster.dbscan` | `dbscan`, point data and precomputed distance matrices, clusters and noise |
| `cluster.center_initializer` | `random_center_initializer`, `kmeans_plusplus_initializer`, farthest-center mode |
| `utils.metric` | all eight built-in metric types and their public distance functions |
| `cluster.encoder` | `type_encoding` values returned by the covered algorithms |

The package intentionally does not cover pyclustering's neural-network models,
visualizers, sample loaders, cluster encoders/converters, or its other
clustering families such as X-Means, OPTICS, CURE, ROCK, BIRCH, CLIQUE and
SOMSC. Iteration observers and `USER_DEFINED` distance callbacks are also not
covered: arbitrary Python functions cannot safely execute through the compiled
Mojo C ABI. The `ccore` argument remains accepted for source compatibility but
the covered algorithms always use Mojo.

Parity is tested against the real conda-forge `pyclustering` 0.10.1.2 package.
The suite compares partitions, centers or medoids, total errors, predictions,
noise labels, distance-matrix modes, empty-cluster behavior, and all built-in
metrics. Comparisons use upstream's Python algorithm path so the same defined
iteration rules are exercised; benchmarks use its CCORE path where it works.

## Install

From a source checkout, Pixi installs the pinned Mojo compiler and Python
dependencies:

```bash
pixi install
pixi run build
pixi run python - <<'PY'
from pyclustering.cluster.kmeans import kmeans

data = [[0.0, 0.0], [0.2, 0.1], [8.0, 8.0], [8.1, 8.2]]
model = kmeans(data, [[0.0, 0.0], [8.0, 8.0]]).process()
assert model.get_clusters() == [[0, 1], [2, 3]]
print(model.get_centers())
PY
```

`pixi run build` creates `dist/libmojo-pyclustering.so`. The Python wrapper also
rebuilds a missing or stale library on first use. A deployed package can point
at a prebuilt library with `MOJO_PYCLUSTERING_LIB=/path/to/library.so`.

## Performance

This is the exact table produced by `pixi run bench` on the publication-gate
run on this machine. It reported x86_64, Python 3.13.14, and pyclustering
0.10.1.2. Times are the best warm-run times selected by the checked-in
benchmark, with identical contiguous `float64` input for both implementations.

| case | Mojo | upstream | upstream / Mojo |
| --- | ---: | ---: | ---: |
| kmeans.process (100k x 8, k=6) | 10.65 ms | 1715.69 ms | 161.07x faster |
| kmeans.predict (200k x 8, k=6) | 6.12 ms | 1956.98 ms | 319.91x faster |
| kmedians.process (60k x 6, k=5) | 11.52 ms | 1037.14 ms | 90.04x faster |
| kmedoids.process (900 x 5, k=4) | 9.16 ms | 222.39 ms | 24.28x faster |
| dbscan.process (3k x 2) | 27.08 ms | 191.39 ms | 7.07x faster |

Upstream's CCORE implementation is used for the processing benchmarks. In
pyclustering 0.10.1.2, a CCORE k-means fit leaves `predict()` with list centers
while its metric expects an ndarray and raises `AttributeError`; the prediction
benchmark therefore prepares the upstream model with `ccore=False` and times
only its public prediction implementation. `pixi run bench` holds a
machine-wide lock and prints the reproducible Markdown table.

No GPU path is included or claimed. The covered hot loops are distance scans
with about two arithmetic operations per 16 bytes of point data, well below the
roughly 2-flop-per-byte threshold where device transfer and launch costs can pay
off. They remain CPU kernels rather than adding a GPU path that loses.

## How it works

The Python layer keeps the public API and owns all data, output, and scratch
arrays. Inputs become C-contiguous `float64` arrays once; indexes and labels use
contiguous `int64`. `ctypes` passes their addresses and scalar dimensions to one
shared library call per clustering operation.

`src/capi.mojo` is the single C-ABI compilation entry point. Buffer addresses
cross as `Int` and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo. The kernels allocate
nothing: Python supplies assignment, distance, reduction, median-selection,
PAM, and DBSCAN traversal scratch space. Points and centers are row-major, while
precomputed distance matrices are dense row-major square arrays.

The inner Euclidean distance loop uses the machine SIMD width. Lloyd updates,
coordinate quick-selection, PAM swaps, and density expansion then operate
directly on those buffers without Python callbacks or per-point object
allocation. PAM evaluates one candidate distance for all swap clusters and
parallelizes candidate batches at 256 rows or more. DBSCAN uses SIMD distance
checks with scalar remainder handling, collects each visited point's
neighborhood once, and expands clusters through a caller-owned FIFO.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

## License

MIT
