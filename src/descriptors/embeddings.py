from typing import Callable, List, Optional, Tuple

import networkit as nk
import numpy as np

from .edge_descriptors import edge_descriptors_dict
from .node_descriptors import node_descriptors_dict


def get_function(name: str) -> Callable[[nk.Graph], np.ndarray | list[np.ndarray]]:
    if name in edge_descriptors_dict:
        return edge_descriptors_dict[name]
    elif name in node_descriptors_dict:
        return node_descriptors_dict[name]

    raise ValueError(f"Unknown function name: {name}")


def normalize_features(features):
    distinct_features = []
    for feature in features:
        if feature == "moltop":
            distinct_features.extend(["ari", "scan", "edge_betweenness"])
        elif feature == "moltop_normalized":
            distinct_features.extend(["ari", "scan", "edge_betweenness_normalized"])
        elif feature == "ltp":
            distinct_features.extend(["jaccard_index", "edge_betweenness", "lds"])
        elif feature == "ltp_normalized":
            distinct_features.extend(
                ["jaccard_index_normalized", "edge_betweenness_normalized", "lds"]
            )
        elif feature == "ldp":
            distinct_features.extend(
                ["ldp_degree", "ldp_min", "ldp_max", "ldp_mean", "ldp_std"]
            )
        else:
            distinct_features.append(feature)
    return sorted(set(distinct_features))


def create_embedding_function(
    features: list[str],
    bins_per_feature: int,
    histogram_ranges: Optional[List[Tuple[int, int]]] = None,
    embeddings: bool = True,  # if set to False, function returns raw values of function
) -> Callable[[nk.Graph], np.ndarray | List[np.ndarray]]:

    distinct_features = normalize_features(features)

    feature_functions = list(map(lambda x: get_function(x), distinct_features))

    def combined_features(graph: nk.Graph) -> np.ndarray | List[np.ndarray]:
        graph.indexEdges()

        edge_features = list(map(lambda f: f(graph), feature_functions))

        edge_features_count = len(edge_features)
        if embeddings:
            edge_histograms = [
                np.histogram(edge_feature, bins=bins_per_feature, range=hrange)[0]
                for edge_feature, hrange in zip(
                    edge_features, histogram_ranges[:edge_features_count]
                )
            ]
            embedding = np.concatenate(edge_histograms)

        return embedding if embeddings else edge_features

    return combined_features
