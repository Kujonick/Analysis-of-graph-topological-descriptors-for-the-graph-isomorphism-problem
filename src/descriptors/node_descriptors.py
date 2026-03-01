from functools import wraps, partial

import networkit as nk
import networkx as nx
import numpy as np

from networkit.centrality import (
    EigenvectorCentrality,
    Closeness,
    DegreeCentrality,
    KatzCentrality,
    LocalClusteringCoefficient,
)

node_descriptors_dict = {}


def add_to_dict(name, can_be_normalized=False):
    def decorator(f):
        if can_be_normalized:
            node_descriptors_dict[name] = partial(f, normalize=False)
            node_descriptors_dict[name + "_normalized"] = partial(f, normalize=True)
        else:
            node_descriptors_dict[name] = f

        return f

    return decorator


def _calculate_degress(graph):
    adj = nx.to_scipy_sparse_array(nk.nxadapter.nk2nx(graph), format="coo", dtype=float)
    return adj.sum(axis=1), adj


def _calculate_degree_matrix(graph, normalize):
    degrees, adj = _calculate_degress(graph)
    degree_matrix = adj * degrees
    if normalize:
        degree_matrix = degree_matrix / degrees.shape[0]

    return degree_matrix, degrees


def change_to_numpy(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return np.array(function(*args, **kwargs), dtype=np.float32)

    return wrapper


def local_degree_profile(graph: nk.Graph, normalize: bool = True) -> list[np.ndarray]:
    adj = nx.to_scipy_sparse_array(nk.nxadapter.nk2nx(graph), format="coo", dtype=float)
    degrees = adj.sum(axis=1)

    dn = adj * degrees
    if normalize:
        dn = dn / degrees.shape[0]

    min_dn = dn.min(axis=1, explicit=True).toarray()
    max_dn = dn.max(axis=1, explicit=True).toarray()
    mean_dn = dn.sum(axis=1) / degrees

    std_dn = (dn**2).sum(axis=1) / degrees - mean_dn**2

    ldp = [
        np.array(degrees / degrees.shape[0], np.float32),
        np.array(min_dn, np.float32),
        np.array(max_dn, np.float32),
        np.array(mean_dn, np.float32),
        np.array(std_dn, np.float32),
    ]
    return ldp


@add_to_dict("ldp_degree")
def degree_ldp(graph: nk.Graph) -> np.ndarray:
    degrees, _ = _calculate_degress(graph)
    return degrees / degrees.shape[0]


@add_to_dict("ldp_min", can_be_normalized=True)
@change_to_numpy
def min_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, _ = _calculate_degree_matrix(graph, normalize)
    return degree_matrix.min(axis=1, explicit=True).toarray()


@add_to_dict("ldp_max", can_be_normalized=True)
@change_to_numpy
def max_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, _ = _calculate_degree_matrix(graph, normalize)
    return degree_matrix.max(axis=1, explicit=True).toarray()


@add_to_dict("ldp_mean", can_be_normalized=True)
@change_to_numpy
def mean_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, degrees = _calculate_degree_matrix(graph, normalize)
    return degree_matrix.sum(axis=1) / degrees


@add_to_dict("ldp_std", can_be_normalized=True)
def std_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, degrees = _calculate_degree_matrix(graph, normalize)
    mean_degree = degree_matrix.sum(axis=1) / degrees
    return (degree_matrix**2).sum(axis=1) / degrees - mean_degree**2


# =========== centrality


@add_to_dict("eigenvector_centrality")
def calculate_eigenvector_centrality(graph: nk.Graph) -> np.ndarray:
    desc = EigenvectorCentrality(graph)
    desc.run()
    return np.array(desc.scores(), np.float32)


@add_to_dict("closeness", can_be_normalized=True)
def calculate_closeness(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    desc = Closeness(
        graph, normalize, nk.centrality.ClosenessVariant.GENERALIZED
    )  # don't add argument names, it breaks __cinit__ for whatever reason
    desc.run()
    return np.array(desc.scores(), np.float32)


@add_to_dict("degree_centrality", can_be_normalized=True)
def calculate_degree_centrality(graph: nk.Graph, normalize: bool) -> np.ndarray:
    desc = DegreeCentrality(graph, normalized=normalize)
    desc.run()
    return np.array(desc.scores(), np.float32)


@add_to_dict("katz_centrality")
def calculate_katz_centrality(graph: nk.Graph) -> np.ndarray:
    desc = KatzCentrality(graph)
    desc.run()
    return np.array(desc.scores(), np.float32)


@add_to_dict("lcc")
def calculate_lcc(graph: nk.Graph) -> np.ndarray:
    desc = LocalClusteringCoefficient(graph)
    desc.run()
    return np.array(desc.scores(), np.float32)
