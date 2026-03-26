import networkit as nk
import numpy as np
from functools import partial
from typing import Callable


def get_transform(transform_str: str) -> Callable[[nk.Graph], nk.Graph]:
    if transform_str.find("k_graph") == 0:
        k = int(transform_str.replace("k_graph", ""))
        return partial(k_graph_transform, k=k)

    elif transform_str.find("modk_graph") == 0:
        k = int(transform_str.replace("modk_graph", ""))
        return partial(modified_k_graph_transform, k=k)

    else:
        raise ValueError("wrong transform name")


# @cache
def distance_matrix(graph: nk.Graph) -> np.ndarray:
    APSP = nk.distance.APSP(graph)
    APSP.run()
    distances = np.array(APSP.getDistances())
    distances[distances > 10_000] = (
        10_000  # this one is just for infinity, which means that path is not available
    )

    return distances


# @cache
def k_graph_transform(graph: nk.Graph, k: int) -> nk.Graph:
    distances = distance_matrix(graph)
    n = distances.shape[0]
    new_graph = nk.Graph(n)
    new_edges = distances == k

    edges = np.argwhere(new_edges & np.triu(np.ones_like(distances)).astype(bool)).T
    new_graph.addEdges((edges[0], edges[1]))
    return new_graph


# @cache
def modified_k_graph_transform(graph: nk.Graph, k: int) -> nk.Graph:
    distances = distance_matrix(graph)
    n = distances.shape[0]
    new_graph = nk.Graph(n)
    new_edges = distances <= k  # modified creates more dense graph with more edges

    edges = np.argwhere(new_edges & np.triu(np.ones_like(distances), 1).astype(bool)).T
    new_graph.addEdges((edges[0], edges[1]))
    return new_graph
