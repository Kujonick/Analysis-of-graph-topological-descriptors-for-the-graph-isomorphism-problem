import numpy as np

from src.collision_tests.utils import TestParameters, open_test_enviroment
import xxhash

from typing import Dict, List, Tuple


def select_problematic_ids(
    parameters: TestParameters, **function_kwargs
) -> Dict[str, list[int]]:
    """goes through all the graphs and selects only the ones that have collisions on embedding"""

    graph_reader, embedding_function = open_test_enviroment(
        parameters, **function_kwargs
    )

    collisions: dict[str, list[int]] = {}
    hashes: dict[str, int] = {}
    for graph_id, graph in enumerate(graph_reader):

        embedding = embedding_function(graph)
        h = xxhash.xxh128_hexdigest(embedding.tobytes())

        if h not in hashes:
            hashes[h] = graph_id
        else:
            if h in collisions:
                collisions[h].append(graph_id)
            else:
                collisions[h] = [hashes[h], graph_id]
    return collisions


def find_optimal_histogram_ranges(
    parameters: TestParameters, **function_kwargs
) -> List[Tuple[float, float]]:
    """function that goes through all descriptor values per graphs and finds minimum and maximum of each feature value"""

    dataset = parameters.dataset
    graph_reader, embedding_function = open_test_enviroment(
        parameters, **function_kwargs
    )
    metadata = dataset.get_metadata()

    first_graph = next(graph_reader)
    function_values: List[np.ndarray] = embedding_function(first_graph)
    function_values = [arr[~np.isnan(arr)] for arr in function_values]
    hist_ranges: List[Tuple[float, float]] = [
        (np.min(arr), np.max(arr)) if len(arr) else (np.inf, -np.inf)
        for arr in function_values
    ]
    for graph in graph_reader:
        function_values = embedding_function(graph)
        function_values = [arr[~np.isnan(arr)] for arr in function_values]
        hist_ranges = [
            (
                (min(ranges[0], np.min(values)), max(ranges[1], np.max(values)))
                if len(values)
                else ranges
            )
            for ranges, values in zip(hist_ranges, function_values)
        ]

    # in situation that range is too small and there is no way to fit all bins into
    for i, range in enumerate(hist_ranges):
        right, left = range[1], range[0]
        if not np.isfinite(right):
            right = 0
        if not np.isfinite(left):
            left = 0

        x_min = np.float32(left)
        x_max = np.float32(right)

        scale = max(abs(x_min), abs(x_max))
        ulp = np.spacing(
            np.float32((scale))
        )  # Unit in the Last Place - distance between two representable numbers

        d_min = metadata["number_of_nodes"] ** 2 * ulp
        if d_min <= right - left:
            hist_ranges[i] = (left, right)
        else:
            avg = (right - left) // 2
            hist_ranges[i] = (avg + d_min, avg + d_min)

    return hist_ranges
