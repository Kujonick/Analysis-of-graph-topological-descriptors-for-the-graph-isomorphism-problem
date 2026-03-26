from typing import Callable, List, Optional, Tuple

import networkit as nk
import numpy as np

from .edge_descriptors import edge_descriptors_dict
from .node_descriptors import node_descriptors_dict
from .transforms import get_transform


def get_function(name: str) -> Callable[[nk.Graph], np.ndarray | list[np.ndarray]]:

    parts = name.split(":")
    transforms = parts[1:]
    name = parts[0]

    functions = [get_transform(transform_str) for transform_str in transforms]

    if name in edge_descriptors_dict:
        functions.append(edge_descriptors_dict[name])
    elif name in node_descriptors_dict:
        functions.append(node_descriptors_dict[name])

    def chained(x):
        for f in functions:
            x = f(x)
        return x

    return chained

    raise ValueError(f"Unknown function name: {name}")


def normalize_features(features: Tuple[str, ...]) -> Tuple[str, ...]:
    distinct_features = []
    for feature in features:
        parts = feature.split(":")
        transforms = ":".join(parts[1:])
        name = parts[0]

        if name == "moltop":
            new_features = ["ari", "scan", "edge_betweenness"]
        elif name == "moltop_normalized":
            new_features = ["ari", "scan", "edge_betweenness_normalized"]
        elif name == "ltp":
            new_features = ["jaccard_index", "edge_betweenness", "lds"]
        elif name == "ltp_normalized":
            new_features = [
                "jaccard_index_normalized",
                "edge_betweenness_normalized",
                "lds",
            ]
        elif name == "ldp":
            new_features = ["ldp_degree", "ldp_min", "ldp_max", "ldp_mean", "ldp_std"]
        else:
            new_features = [name]

        distinct_features.extend(
            map(lambda x: ":".join([x, transforms]) if transforms else x, new_features)
        )
    return tuple(sorted(set(distinct_features)))


def create_embedding_function(
    features: Tuple[str, ...],
    bins_per_feature: int,
    histogram_ranges: Optional[
        List[Tuple[int, int]]
    ] = None,  # if histogram ranges are given, it means we want a histogram, else raw vector is returned
) -> Callable[[nk.Graph], np.ndarray | List[np.ndarray]]:

    distinct_features = normalize_features(features)

    feature_functions = list(map(lambda x: get_function(x), distinct_features))

    def combined_features(graph: nk.Graph) -> np.ndarray | List[np.ndarray]:

        edge_features = list(map(lambda f: f(graph), feature_functions))

        edge_features_count = len(edge_features)
        if histogram_ranges is not None:
            edge_histograms = [
                np.histogram(edge_feature, bins=bins_per_feature, range=hrange)[0]
                for edge_feature, hrange in zip(
                    edge_features, histogram_ranges[:edge_features_count]
                )
            ]
            embedding = np.concatenate(edge_histograms)
            return embedding

        return edge_features

    return combined_features
