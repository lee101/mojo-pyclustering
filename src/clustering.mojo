"""Dense clustering kernels over caller-owned buffers."""

from std.algorithm import parallelize
from std.math import pow, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime INF = 1.7976931348623157e308


@always_inline
def metric_distance(
    a: FPtr,
    b: FPtr,
    dimensions: Int,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Float64:
    if metric == 0 or metric == 1:
        var lanes = SIMD[DType.float64, W](0.0)
        var column = 0
        while column + W <= dimensions:
            var delta = a.load[width=W](column) - b.load[width=W](column)
            lanes += delta * delta
            column += W
        var total = lanes.reduce_add()
        while column < dimensions:
            var delta = a[column] - b[column]
            total += delta * delta
            column += 1
        return sqrt(total) if metric == 0 else total

    if metric == 2:
        var total = 0.0
        for column in range(dimensions):
            total += abs(a[column] - b[column])
        return total

    if metric == 3:
        var largest = 0.0
        for column in range(dimensions):
            largest = max(largest, abs(a[column] - b[column]))
        return largest

    if metric == 4:
        var total = 0.0
        for column in range(dimensions):
            total += pow(abs(a[column] - b[column]), degree)
        return pow(total, 1.0 / degree)

    if metric == 5:
        var total = 0.0
        for column in range(dimensions):
            var denominator = abs(a[column]) + abs(b[column])
            if denominator != 0.0:
                total += abs(a[column] - b[column]) / denominator
        return total

    if metric == 6:
        var total = 0.0
        for column in range(dimensions):
            var denominator = abs(a[column]) + abs(b[column])
            if denominator != 0.0:
                var delta = a[column] - b[column]
                total += delta * delta / denominator
        return total

    var total = 0.0
    for column in range(dimensions):
        if ranges[column] != 0.0:
            total += abs(a[column] - b[column]) / ranges[column]
    return total / Float64(dimensions)


def assign(
    data: FPtr,
    centers: FPtr,
    labels: IPtr,
    distances: FPtr,
    rows: Int,
    dimensions: Int,
    clusters: Int,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Float64:
    var total = 0.0
    for row in range(rows):
        var best = metric_distance(
            data + row * dimensions, centers, dimensions, metric, degree, ranges
        )
        var best_cluster = 0
        for cluster in range(1, clusters):
            var candidate = metric_distance(
                data + row * dimensions,
                centers + cluster * dimensions,
                dimensions,
                metric,
                degree,
                ranges,
            )
            if candidate < best:
                best = candidate
                best_cluster = cluster
        labels[row] = Int64(best_cluster)
        distances[row] = best
        total += best
    return total


def kmeans_process(
    data: FPtr,
    centers: FPtr,
    labels: IPtr,
    distances: FPtr,
    sums: FPtr,
    counts: IPtr,
    result: FPtr,
    rows: Int,
    dimensions: Int,
    cluster_count: Int,
    itermax: Int,
    tolerance: Float64,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Int:
    var active = cluster_count
    for _ in range(itermax):
        _ = assign(
            data,
            centers,
            labels,
            distances,
            rows,
            dimensions,
            active,
            metric,
            degree,
            ranges,
        )
        for index in range(active * dimensions):
            sums[index] = 0.0
        for cluster in range(active):
            counts[cluster] = 0
        for row in range(rows):
            var cluster = Int(labels[row])
            counts[cluster] += 1
            for column in range(dimensions):
                sums[cluster * dimensions + column] += data[
                    row * dimensions + column
                ]

        var cluster = 0
        while cluster < active:
            if counts[cluster] == 0:
                for later in range(cluster + 1, active):
                    counts[later - 1] = counts[later]
                    for column in range(dimensions):
                        centers[(later - 1) * dimensions + column] = centers[
                            later * dimensions + column
                        ]
                        sums[(later - 1) * dimensions + column] = sums[
                            later * dimensions + column
                        ]
                active -= 1
            else:
                cluster += 1

        var largest_change = 0.0
        for current in range(active):
            for column in range(dimensions):
                sums[current * dimensions + column] /= Float64(counts[current])
            largest_change = max(
                largest_change,
                metric_distance(
                    centers + current * dimensions,
                    sums + current * dimensions,
                    dimensions,
                    metric,
                    degree,
                    ranges,
                ),
            )
            for column in range(dimensions):
                centers[current * dimensions + column] = sums[
                    current * dimensions + column
                ]
        if largest_change <= tolerance:
            break

    result[0] = assign(
        data,
        centers,
        labels,
        distances,
        rows,
        dimensions,
        active,
        metric,
        degree,
        ranges,
    )
    return active


def select_kth(values: FPtr, length: Int, kth: Int) -> Float64:
    var left = 0
    var right = length - 1
    while left < right:
        var pivot = values[(left + right) // 2]
        var low = left
        var high = right
        while low <= high:
            while values[low] < pivot:
                low += 1
            while values[high] > pivot:
                high -= 1
            if low <= high:
                var temporary = values[low]
                values[low] = values[high]
                values[high] = temporary
                low += 1
                high -= 1
        if kth <= high:
            right = high
        elif kth >= low:
            left = low
        else:
            break
    return values[kth]


def kmedians_process(
    data: FPtr,
    medians: FPtr,
    labels: IPtr,
    distances: FPtr,
    updates: FPtr,
    counts: IPtr,
    work: FPtr,
    result: FPtr,
    rows: Int,
    dimensions: Int,
    cluster_count: Int,
    itermax: Int,
    tolerance: Float64,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Int:
    var active = cluster_count
    for _ in range(itermax):
        _ = assign(
            data,
            medians,
            labels,
            distances,
            rows,
            dimensions,
            active,
            metric,
            degree,
            ranges,
        )
        for cluster in range(active):
            counts[cluster] = 0
        for row in range(rows):
            counts[Int(labels[row])] += 1

        var cluster = 0
        while cluster < active:
            if counts[cluster] == 0:
                for later in range(cluster + 1, active):
                    counts[later - 1] = counts[later]
                    for column in range(dimensions):
                        medians[(later - 1) * dimensions + column] = medians[
                            later * dimensions + column
                        ]
                active -= 1
                _ = assign(
                    data,
                    medians,
                    labels,
                    distances,
                    rows,
                    dimensions,
                    active,
                    metric,
                    degree,
                    ranges,
                )
                for current in range(active):
                    counts[current] = 0
                for row in range(rows):
                    counts[Int(labels[row])] += 1
                cluster = 0
            else:
                cluster += 1

        for current in range(active):
            for column in range(dimensions):
                var length = 0
                for row in range(rows):
                    if Int(labels[row]) == current:
                        work[length] = data[row * dimensions + column]
                        length += 1
                var lower = (length - 1) // 2
                var value = select_kth(work, length, lower)
                if length % 2 == 0:
                    value = (value + select_kth(work, length, lower + 1)) / 2.0
                updates[current * dimensions + column] = value

        var largest_change = 0.0
        for current in range(active):
            largest_change = max(
                largest_change,
                metric_distance(
                    medians + current * dimensions,
                    updates + current * dimensions,
                    dimensions,
                    metric,
                    degree,
                    ranges,
                ),
            )
            for column in range(dimensions):
                medians[current * dimensions + column] = updates[
                    current * dimensions + column
                ]
        if largest_change <= tolerance:
            break

    result[0] = assign(
        data,
        medians,
        labels,
        distances,
        rows,
        dimensions,
        active,
        metric,
        degree,
        ranges,
    )
    return active


@always_inline
def medoid_distance(
    data: FPtr,
    first: Int,
    second: Int,
    rows: Int,
    dimensions: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Float64:
    if data_type == 1:
        return data[first * rows + second]
    return metric_distance(
        data + first * dimensions,
        data + second * dimensions,
        dimensions,
        metric,
        degree,
        ranges,
    )


def medoid_assign(
    data: FPtr,
    medoids: IPtr,
    labels: IPtr,
    first_distances: FPtr,
    second_distances: FPtr,
    rows: Int,
    dimensions: Int,
    clusters: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Float64:
    var deviation = 0.0
    for row in range(rows):
        var closest = INF
        var second = INF
        var best_cluster = -1
        for cluster in range(clusters):
            var distance = medoid_distance(
                data,
                row,
                Int(medoids[cluster]),
                rows,
                dimensions,
                data_type,
                metric,
                degree,
                ranges,
            )
            if distance < closest:
                second = closest
                closest = distance
                best_cluster = cluster
            elif distance < second:
                second = distance
        labels[row] = Int64(best_cluster)
        first_distances[row] = closest
        second_distances[row] = second
        deviation += closest
    return deviation


def medoid_evaluate_candidate(
    data: FPtr,
    medoids: IPtr,
    labels: IPtr,
    first_distances: FPtr,
    second_distances: FPtr,
    costs: FPtr,
    candidate: Int,
    rows: Int,
    dimensions: Int,
    active: Int,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
):
    var invalid = first_distances[candidate] == 0.0
    for current in range(active):
        if Int(medoids[current]) == candidate:
            invalid = True
    for cluster in range(active):
        costs[
            cluster * rows + candidate
        ] = INF if invalid else -first_distances[candidate]
    if invalid:
        return
    for row in range(rows):
        if row == candidate:
            continue
        var candidate_distance = medoid_distance(
            data,
            row,
            candidate,
            rows,
            dimensions,
            data_type,
            metric,
            degree,
            ranges,
        )
        var assigned = Int(labels[row])
        for cluster in range(active):
            if assigned == cluster:
                costs[cluster * rows + candidate] += (
                    min(candidate_distance, second_distances[row])
                    - first_distances[row]
                )
            elif candidate_distance < first_distances[row]:
                costs[cluster * rows + candidate] += (
                    candidate_distance - first_distances[row]
                )


def kmedoids_process(
    data: FPtr,
    medoids: IPtr,
    labels: IPtr,
    first_distances: FPtr,
    second_distances: FPtr,
    costs: FPtr,
    rows: Int,
    dimensions: Int,
    cluster_count: Int,
    itermax: Int,
    tolerance: Float64,
    data_type: Int,
    metric: Int,
    degree: Float64,
    ranges: FPtr,
) -> Int:
    var active = cluster_count
    if itermax > 0:
        _ = medoid_assign(
            data,
            medoids,
            labels,
            first_distances,
            second_distances,
            rows,
            dimensions,
            active,
            data_type,
            metric,
            degree,
            ranges,
        )
    var changes = INF
    var iteration = 0
    while changes > tolerance and iteration < itermax:
        if rows >= 256:
            var task_count = min(36, (rows + 31) // 32)
            var batch_size = (rows + task_count - 1) // task_count

            def evaluate_batch(task: Int) capturing:
                var begin = task * batch_size
                var end = min(rows, begin + batch_size)
                for candidate in range(begin, end):
                    medoid_evaluate_candidate(
                        data,
                        medoids,
                        labels,
                        first_distances,
                        second_distances,
                        costs,
                        candidate,
                        rows,
                        dimensions,
                        active,
                        data_type,
                        metric,
                        degree,
                        ranges,
                    )

            parallelize[evaluate_batch](task_count, task_count)
        else:
            for candidate in range(rows):
                medoid_evaluate_candidate(
                    data,
                    medoids,
                    labels,
                    first_distances,
                    second_distances,
                    costs,
                    candidate,
                    rows,
                    dimensions,
                    active,
                    data_type,
                    metric,
                    degree,
                    ranges,
                )

        var best_cost = INF
        var best_cluster = -1
        var best_candidate = -1
        for cluster in range(active):
            for candidate in range(rows):
                var cost = costs[cluster * rows + candidate]
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = cluster
                    best_candidate = candidate
        if best_cluster < 0:
            break
        medoids[best_cluster] = Int64(best_candidate)
        var previous = 0.0
        for row in range(rows):
            previous += first_distances[row]
        var current = medoid_assign(
            data,
            medoids,
            labels,
            first_distances,
            second_distances,
            rows,
            dimensions,
            active,
            data_type,
            metric,
            degree,
            ranges,
        )
        changes = previous - current
        iteration += 1

    var cluster = 0
    while cluster < active:
        var found = False
        for row in range(rows):
            if Int(labels[row]) == cluster:
                found = True
                break
        if not found:
            for later in range(cluster + 1, active):
                medoids[later - 1] = medoids[later]
            active -= 1
            _ = medoid_assign(
                data,
                medoids,
                labels,
                first_distances,
                second_distances,
                rows,
                dimensions,
                active,
                data_type,
                metric,
                degree,
                ranges,
            )
        else:
            cluster += 1
    return active


@always_inline
def dbscan_connected(
    data: FPtr,
    first: Int,
    second: Int,
    rows: Int,
    dimensions: Int,
    squared_eps: Float64,
    eps: Float64,
    data_type: Int,
) -> Bool:
    if data_type == 1:
        return data[first * rows + second] <= eps
    var a = data + first * dimensions
    var b = data + second * dimensions
    if dimensions == 1:
        var delta = a[0] - b[0]
        return delta * delta <= squared_eps
    if dimensions == 2:
        var delta0 = a[0] - b[0]
        var delta1 = a[1] - b[1]
        return delta0 * delta0 + delta1 * delta1 <= squared_eps
    var lanes = SIMD[DType.float64, W](0.0)
    var column = 0
    while column + W <= dimensions:
        var delta = a.load[width=W](column) - b.load[width=W](column)
        lanes += delta * delta
        column += W
    var squared = lanes.reduce_add()
    while column < dimensions:
        var delta = a[column] - b[column]
        squared += delta * delta
        column += 1
    return squared <= squared_eps


def dbscan_collect_neighbors(
    data: FPtr,
    neighbors: IPtr,
    point: Int,
    rows: Int,
    dimensions: Int,
    squared_eps: Float64,
    eps: Float64,
    data_type: Int,
) -> Int:
    var count = 0
    for candidate in range(rows):
        if candidate != point and dbscan_connected(
            data,
            point,
            candidate,
            rows,
            dimensions,
            squared_eps,
            eps,
            data_type,
        ):
            neighbors[count] = Int64(candidate)
            count += 1
    return count


def dbscan_points(
    data: FPtr,
    labels: IPtr,
    visited: IPtr,
    queue: IPtr,
    neighbors: IPtr,
    rows: Int,
    dimensions: Int,
    eps: Float64,
    minimum_neighbors: Int,
    data_type: Int,
) -> Int:
    for row in range(rows):
        labels[row] = -1
        visited[row] = 0
    var cluster = 0
    var squared_eps = eps * eps
    for seed in range(rows):
        if visited[seed] != 0:
            continue
        visited[seed] = 1
        var neighbor_count = dbscan_collect_neighbors(
            data, neighbors, seed, rows, dimensions, squared_eps, eps, data_type
        )
        if neighbor_count < minimum_neighbors:
            continue

        labels[seed] = Int64(cluster)
        var head = 0
        var tail = 0
        for index in range(neighbor_count):
            var candidate = Int(neighbors[index])
            if labels[candidate] < 0:
                labels[candidate] = Int64(cluster)
                queue[tail] = Int64(candidate)
                tail += 1

        while head < tail:
            var point = Int(queue[head])
            head += 1
            if visited[point] == 0:
                visited[point] = 1
                var next_count = dbscan_collect_neighbors(
                    data,
                    neighbors,
                    point,
                    rows,
                    dimensions,
                    squared_eps,
                    eps,
                    data_type,
                )
                if next_count >= minimum_neighbors:
                    for index in range(next_count):
                        var candidate = Int(neighbors[index])
                        if labels[candidate] < 0:
                            labels[candidate] = Int64(cluster)
                            queue[tail] = Int64(candidate)
                            tail += 1
        cluster += 1
    return cluster
