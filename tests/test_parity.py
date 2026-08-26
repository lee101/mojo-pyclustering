import numpy as np
import pytest
import warnings

from pyclustering.cluster.center_initializer import (
    kmeans_plusplus_initializer,
    random_center_initializer,
)
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import (
    canberra_distance,
    chebyshev_distance,
    chi_square_distance,
    distance_metric,
    euclidean_distance,
    euclidean_distance_square,
    gower_distance,
    manhattan_distance,
    minkowski_distance,
    type_metric,
)
from pyclustering._lib import addr


def blobs(seed=7, count=40):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal((-4.0, -2.0), 0.3, (count, 2)),
            rng.normal((0.0, 4.0), 0.35, (count, 2)),
            rng.normal((5.0, -1.0), 0.25, (count, 2)),
        ]
    )


def normalized(clusters):
    return sorted(tuple(sorted(cluster)) for cluster in clusters)


def test_kmeans_default_parity(upstream):
    data = blobs()
    initial = data[[0, 40, 80]]
    ours = kmeans(data, initial, ccore=False).process()
    theirs = upstream.kmeans(data, initial, ccore=False).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert np.allclose(ours.get_centers(), theirs.get_centers(), atol=1e-12)
    assert ours.get_total_wce() == pytest.approx(theirs.get_total_wce(), abs=1e-10)
    assert ours.process() is ours
    assert ours.get_cluster_encoding() == type_encoding.CLUSTER_INDEX_LIST_SEPARATION


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        (type_metric.EUCLIDEAN, {}),
        (type_metric.EUCLIDEAN_SQUARE, {}),
        (type_metric.MANHATTAN, {}),
        (type_metric.CHEBYSHEV, {}),
        (type_metric.MINKOWSKI, {"degree": 4}),
        (type_metric.CANBERRA, {}),
        (type_metric.CHI_SQUARE, {}),
        (type_metric.GOWER, {"max_range": [10.0, 10.0]}),
    ],
)
def test_kmeans_builtin_metric_parity(upstream, kind, kwargs):
    data = blobs(count=15)
    initial = data[[0, 15, 30]]
    ours_metric = distance_metric(kind, **kwargs)
    theirs_metric = upstream.distance_metric(upstream.type_metric(int(kind)), **kwargs)
    ours = kmeans(data, initial, metric=ours_metric, ccore=False).process()
    theirs = upstream.kmeans(
        data, initial, metric=theirs_metric, ccore=False
    ).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert np.allclose(ours.get_centers(), theirs.get_centers(), atol=1e-10)
    assert ours.get_total_wce() == pytest.approx(theirs.get_total_wce(), rel=1e-10)


def test_kmeans_empty_center_compaction(upstream):
    data = np.array([[0.0], [0.1], [10.0], [10.1]])
    initial = [[0.0], [0.0], [10.0]]
    ours = kmeans(data, initial, ccore=False).process()
    theirs = upstream.kmeans(data, initial, ccore=False).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert np.allclose(ours.get_centers(), theirs.get_centers())


def test_kmeans_predict_before_and_after_process():
    data = blobs(count=4)
    model = kmeans(data, data[[0, 4, 8]])
    assert model.predict([[0.0, 0.0]]) == []
    model.process()
    assert np.array_equal(model.predict(data), np.repeat(np.arange(3), 4))


def test_kmedians_parity(upstream):
    data = blobs(count=25)
    initial = data[[0, 25, 50]]
    ours = kmedians(data, initial, ccore=False).process()
    theirs = upstream.kmedians(data, initial, ccore=False).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert np.allclose(ours.get_medians(), theirs.get_medians(), atol=1e-12)
    assert ours.get_total_wce() == pytest.approx(theirs.get_total_wce(), abs=1e-10)
    assert np.array_equal(ours.predict(data[:5]), theirs.predict(data[:5]))


def test_kmedians_even_cluster_median(upstream):
    data = np.array([[0.0, 8.0], [2.0, 4.0], [4.0, 2.0], [8.0, 0.0]])
    ours = kmedians(data, [[1.0, 1.0]], ccore=False).process()
    theirs = upstream.kmedians(data, [[1.0, 1.0]], ccore=False).process()
    assert ours.get_medians() == theirs.get_medians() == [[3.0, 3.0]]


def test_kmedoids_point_parity(upstream):
    data = blobs(count=10)
    initial = [0, 10, 20]
    ours = kmedoids(data, initial, ccore=False).process()
    theirs = upstream.kmedoids(data, initial, ccore=False).process()
    assert ours.get_medoids() == theirs.get_medoids()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert np.array_equal(ours.predict(data[:8]), theirs.predict(data[:8]))


def test_kmedoids_distance_matrix_parity(upstream):
    data = blobs(count=7)
    matrix = np.sqrt(np.sum((data[:, None] - data[None, :]) ** 2, axis=2))
    initial = [0, 7, 14]
    ours = kmedoids(
        matrix, initial, ccore=False, data_type="distance_matrix"
    ).process()
    theirs = upstream.kmedoids(
        matrix.tolist(), initial, ccore=False, data_type="distance_matrix"
    ).process()
    assert ours.get_medoids() == theirs.get_medoids()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())


@pytest.mark.parametrize("rows", [255, 256])
def test_kmedoids_parallel_threshold_parity(upstream, rows):
    rng = np.random.default_rng(41)
    sizes = [rows // 4] * 4
    for index in range(rows % 4):
        sizes[index] += 1
    data = np.vstack(
        [
            rng.normal((cluster * 6.0, cluster % 2 * 5.0), 0.25, (size, 2))
            for cluster, size in enumerate(sizes)
        ]
    )
    initial = np.cumsum([0, *sizes[:-1]]).tolist()
    ours = kmedoids(data, initial, itermax=1, ccore=False).process()
    theirs = upstream.kmedoids(
        data, initial, itermax=1, ccore=False
    ).process()
    assert ours.get_medoids() == theirs.get_medoids()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())


def test_dbscan_point_parity(upstream):
    data = np.vstack([blobs(count=12), [[20.0, 20.0], [-20.0, -20.0]]])
    ours = dbscan(data, 0.9, 2, ccore=False).process()
    theirs = upstream.dbscan(data.tolist(), 0.9, 2, ccore=False).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert ours.get_noise() == theirs.get_noise()


def test_dbscan_simd_tail_parity(upstream):
    rng = np.random.default_rng(53)
    data = np.vstack(
        [
            rng.normal(-3.0, 0.12, (12, 5)),
            rng.normal(3.0, 0.12, (12, 5)),
            [[20.0] * 5],
        ]
    )
    ours = dbscan(data, 0.8, 2, ccore=False).process()
    theirs = upstream.dbscan(data.tolist(), 0.8, 2, ccore=False).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert ours.get_noise() == theirs.get_noise()


def test_dbscan_distance_matrix_parity(upstream):
    data = np.vstack([blobs(count=7), [[20.0, 20.0]]])
    matrix = np.sqrt(np.sum((data[:, None] - data[None, :]) ** 2, axis=2))
    ours = dbscan(
        matrix, 0.8, 2, ccore=False, data_type="distance_matrix"
    ).process()
    theirs = upstream.dbscan(
        matrix.tolist(), 0.8, 2, ccore=False, data_type="distance_matrix"
    ).process()
    assert normalized(ours.get_clusters()) == normalized(theirs.get_clusters())
    assert ours.get_noise() == theirs.get_noise()


def test_initializers_seeded_parity(upstream):
    data = blobs(count=5).tolist()
    ours_random = random_center_initializer(data, 4, random_state=11).initialize(
        return_index=True
    )
    theirs_random = upstream.random_center_initializer(
        data, 4, random_state=11
    ).initialize(
        return_index=True
    )
    assert ours_random == theirs_random
    ours_plus = kmeans_plusplus_initializer(
        data, 3, random_state=11
    ).initialize(return_index=True)
    np.warnings = warnings
    try:
        theirs_plus = upstream.kmeans_plusplus_initializer(
            data, 3, random_state=11
        ).initialize(return_index=True)
    finally:
        del np.warnings
    ours_farthest = kmeans_plusplus_initializer(
        data, 3, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE,
        random_state=11,
    ).initialize(return_index=True)
    np.warnings = warnings
    try:
        theirs_farthest = upstream.kmeans_plusplus_initializer(
            data, 3, upstream.kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE,
            random_state=11,
        ).initialize(return_index=True)
    finally:
        del np.warnings
    assert ours_plus == theirs_plus
    assert ours_farthest == theirs_farthest


@pytest.mark.parametrize("algorithm,initial", [
    (kmeans, [[0.0]]),
    (kmedians, [[0.0]]),
    (kmedoids, [0]),
])
def test_zero_iterations_leave_result_unprocessed(algorithm, initial):
    model = algorithm([[0.0], [1.0]], initial, itermax=0).process()
    assert model.get_clusters() == []


@pytest.mark.parametrize("kind", list(type_metric)[:-1])
def test_metric_scalar_parity(upstream, kind):
    kwargs = {}
    if kind == type_metric.MINKOWSKI:
        kwargs["degree"] = 3
    if kind == type_metric.GOWER:
        kwargs["max_range"] = [5.0, 7.0, 9.0]
    first = [1.0, -2.0, 4.0]
    second = [-1.0, 3.0, 2.0]
    ours = distance_metric(kind, **kwargs)(first, second)
    theirs = upstream.distance_metric(upstream.type_metric(int(kind)), **kwargs)(
        first, second
    )
    assert ours == pytest.approx(theirs)


@pytest.mark.parametrize(
    "function,args",
    [
        (euclidean_distance, ([1.0, 2.0], [3.0, -1.0])),
        (euclidean_distance_square, ([1.0, 2.0], [3.0, -1.0])),
        (manhattan_distance, ([1.0, 2.0], [3.0, -1.0])),
        (chebyshev_distance, ([1.0, 2.0], [3.0, -1.0])),
        (minkowski_distance, ([1.0, 2.0], [3.0, -1.0], 3)),
        (canberra_distance, ([1.0, 2.0], [3.0, -1.0])),
        (chi_square_distance, ([1.0, 2.0], [3.0, -1.0])),
        (gower_distance, ([1.0, 2.0], [3.0, -1.0], [4.0, 6.0])),
    ],
)
def test_public_metric_functions_return_finite_values(function, args):
    assert np.isfinite(function(*args))


def test_user_metric_reports_ffi_limit():
    metric = distance_metric(type_metric.USER_DEFINED, func=lambda a, b: 0.0)
    with pytest.raises(NotImplementedError, match="C ABI"):
        kmeans([[0.0], [1.0]], [[0.0]], metric=metric).process()


def test_minkowski_odd_degree_uses_absolute_deltas(upstream):
    ours = distance_metric(type_metric.MINKOWSKI, degree=3)
    theirs = upstream.distance_metric(upstream.type_metric.MINKOWSKI, degree=3)
    first = [-5.0, 2.0]
    second = [1.0, -1.0]
    assert ours(first, second) == pytest.approx(theirs(first, second))
    model = kmeans(
        [first, second, [10.0, 10.0]],
        [first, [10.0, 10.0]],
        metric=ours,
    ).process()
    assert normalized(model.get_clusters()) == [(0, 1), (2,)]


@pytest.mark.parametrize("dimensions", [1, 3, 4, 5, 7, 8, 9])
def test_noncontiguous_float32_inputs_and_simd_tails(dimensions):
    base = np.arange(12 * dimensions * 2, dtype=np.float32).reshape(12, dimensions * 2)
    data = base[:, ::2]
    assert not data.flags.c_contiguous
    model = kmeans(data, data[[0, 11]], itermax=2).process()
    assert sum(map(len, model.get_clusters())) == len(data)
    assert np.asarray(model.get_centers()).shape == (2, dimensions)


def test_ffi_address_rejects_wrong_layout_dtype_and_empty():
    with pytest.raises(ValueError, match="non-empty"):
        addr(np.empty(0, dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((2, 2), dtype=np.float64)[:, ::-1])
    with pytest.raises(TypeError, match="float64 or int64"):
        addr(np.ones(2, dtype=np.float32))


def test_contiguous_float64_input_remains_zero_copy():
    data = np.arange(24, dtype=np.float64).reshape(12, 2)
    model = kmeans(data, data[[0, 11]])
    assert np.shares_memory(model._data, data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kmeans([[complex(1, 1)]], [[0.0]]),
        lambda: kmeans([[np.inf]], [[0.0]]),
        lambda: kmeans([[1.0]], [[]]),
        lambda: kmedoids([[0.0], [1.0]], [0, 0]),
        lambda: dbscan([[0.0]], 1.0, -1),
    ],
)
def test_unsafe_boundary_inputs_are_rejected(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kmeans([[0.0]], [[0.0]], itermax=1.5),
        lambda: kmedoids([[0.0], [1.0]], [0.5]),
        lambda: dbscan([[0.0]], 1.0, 1.5),
    ],
)
def test_integer_arguments_are_not_silently_narrowed(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "model",
    [
        lambda data: kmeans(data, [[0.0]]),
        lambda data: kmedians(data, [[0.0]]),
        lambda data: kmedoids(data, [0]),
        lambda data: dbscan(data, 2.0, 1),
    ],
)
def test_cluster_encoding_for_each_covered_algorithm(model):
    result = model([[0.0], [1.0]]).process()
    assert result.get_cluster_encoding() == type_encoding.CLUSTER_INDEX_LIST_SEPARATION


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kmeans([], [[0.0]]),
        lambda: kmedians([[0.0]], []),
        lambda: kmedoids([[0.0]], []),
        lambda: dbscan([[0.0]], -1.0, 1),
    ],
)
def test_invalid_arguments(factory):
    with pytest.raises(ValueError):
        factory()
