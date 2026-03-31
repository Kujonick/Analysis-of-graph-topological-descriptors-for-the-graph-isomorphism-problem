import networkit as nk
from functools import partial
from typing import Callable

from ..graph_utils.dataset import Graph


def get_transform(transform_str: str) -> Callable[[Graph], nk.Graph]:
    if transform_str.find("k_graph") == 0:
        k = int(transform_str.replace("k_graph", ""))
        return partial(k_graph_transform, k=k)

    elif transform_str.find("modk_graph") == 0:
        k = int(transform_str.replace("modk_graph", ""))
        return partial(modified_k_graph_transform, k=k)

    else:
        raise ValueError("wrong transform name")


def identity_transform(graph: Graph) -> nk.Graph:
    return graph.graph


def k_graph_transform(graph: Graph, k: int) -> nk.Graph:
    return graph.get_k_graph(k=k, modified=False)


def modified_k_graph_transform(graph: Graph, k: int) -> nk.Graph:
    return graph.get_k_graph(k=k, modified=True)
