"""Stable C ABI for the Python bindings."""

from clustering import (
    assign,
    dbscan_points,
    kmeans_process,
    kmedians_process,
    kmedoids_compact,
    kmedoids_evaluate_range,
    kmedoids_init,
    kmedoids_reassign,
    kmedoids_select,
    metric_distance,
)

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


@export("mpc_metric")
def mpc_metric(
    a: Int, b: Int, dimensions: Int, metric: Int, degree: Float64, ranges: Int
) abi("C") -> Float64:
    return metric_distance(fp(a), fp(b), dimensions, metric, degree, fp(ranges))


@export("mpc_assign")
def mpc_assign(
    data: Int,
    centers: Int,
    labels: Int,
    distances: Int,
    rows: Int,
    dimensions: Int,
    clusters: Int,
    metric: Int,
    degree: Float64,
    ranges: Int,
) abi("C") -> Float64:
    return assign(
        fp(data),
        fp(centers),
        ip(labels),
        fp(distances),
        rows,
        dimensions,
        clusters,
        metric,
        degree,
        fp(ranges),
    )


@export("mpc_kmeans")
def mpc_kmeans(
    data: Int,
    centers: Int,
    labels: Int,
    distances: Int,
    sums: Int,
    counts: Int,
    result: Int,
    rows: Int,
    dimensions: Int,
    clusters: Int,
    itermax: Int,
    tolerance: Float64,
    metric: Int,
    degree: Float64,
    ranges: Int,
) abi("C") -> Int:
    return kmeans_process(
        fp(data),
        fp(centers),
        ip(labels),
        fp(distances),
        fp(sums),
        ip(counts),
        fp(result),
        rows,
        dimensions,
        clusters,
        itermax,
        tolerance,
        metric,
        degree,
        fp(ranges),
    )


@export("mpc_kmedians")
def mpc_kmedians(
    data: Int,
    medians: Int,
    labels: Int,
    distances: Int,
    updates: Int,
    counts: Int,
    work: Int,
    result: Int,
    rows: Int,
    dimensions: Int,
    clusters: Int,
    itermax: Int,
    tolerance: Float64,
    metric: Int,
    degree: Float64,
    ranges: Int,
) abi("C") -> Int:
    return kmedians_process(
        fp(data),
        fp(medians),
        ip(labels),
        fp(distances),
        fp(updates),
        ip(counts),
        fp(work),
        fp(result),
        rows,
        dimensions,
        clusters,
        itermax,
        tolerance,
        metric,
        degree,
        fp(ranges),
    )


@export("mpc_kmedoids_init")
def mpc_kmedoids_init(
    data: Int,
    medoids: Int,
    labels: Int,
    first_distances: Int,
    second_distances: Int,
    rows: Int,
    dimensions: Int,
    active: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: Int,
) abi("C") -> Float64:
    return kmedoids_init(
        fp(data),
        ip(medoids),
        ip(labels),
        fp(first_distances),
        fp(second_distances),
        rows,
        dimensions,
        active,
        data_type,
        metric,
        degree,
        fp(ranges),
    )


@export("mpc_kmedoids_evaluate")
def mpc_kmedoids_evaluate(
    data: Int,
    medoids: Int,
    labels: Int,
    first_distances: Int,
    second_distances: Int,
    costs: Int,
    rows: Int,
    dimensions: Int,
    active: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: Int,
    first_candidate: Int,
    last_candidate: Int,
) abi("C"):
    kmedoids_evaluate_range(
        fp(data),
        ip(medoids),
        ip(labels),
        fp(first_distances),
        fp(second_distances),
        fp(costs),
        rows,
        dimensions,
        active,
        data_type,
        metric,
        degree,
        fp(ranges),
        first_candidate,
        last_candidate,
    )


@export("mpc_kmedoids_select")
def mpc_kmedoids_select(
    medoids: Int,
    costs: Int,
    rows: Int,
    active: Int,
) abi("C") -> Int:
    return kmedoids_select(ip(medoids), fp(costs), rows, active)


@export("mpc_kmedoids_reassign")
def mpc_kmedoids_reassign(
    data: Int,
    medoids: Int,
    labels: Int,
    first_distances: Int,
    second_distances: Int,
    rows: Int,
    dimensions: Int,
    active: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: Int,
) abi("C") -> Float64:
    return kmedoids_reassign(
        fp(data),
        ip(medoids),
        ip(labels),
        fp(first_distances),
        fp(second_distances),
        rows,
        dimensions,
        active,
        data_type,
        metric,
        degree,
        fp(ranges),
    )


@export("mpc_kmedoids_compact")
def mpc_kmedoids_compact(
    data: Int,
    medoids: Int,
    labels: Int,
    first_distances: Int,
    second_distances: Int,
    rows: Int,
    dimensions: Int,
    active: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: Int,
) abi("C") -> Int:
    return kmedoids_compact(
        fp(data),
        ip(medoids),
        ip(labels),
        fp(first_distances),
        fp(second_distances),
        rows,
        dimensions,
        active,
        data_type,
        metric,
        degree,
        fp(ranges),
    )


@export("mpc_dbscan")
def mpc_dbscan(
    data: Int,
    labels: Int,
    visited: Int,
    queue: Int,
    neighbors: Int,
    rows: Int,
    dimensions: Int,
    eps: Float64,
    minimum_neighbors: Int,
    data_type: Int,
) abi("C") -> Int:
    return dbscan_points(
        fp(data),
        ip(labels),
        ip(visited),
        ip(queue),
        ip(neighbors),
        rows,
        dimensions,
        eps,
        minimum_neighbors,
        data_type,
    )
