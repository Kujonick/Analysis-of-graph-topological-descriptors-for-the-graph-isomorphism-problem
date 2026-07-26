import pytest
from typing import List
import networkit as nk
import numpy as np
import random
import os
import json
from src.settings import Settings
from src.graph_utils.reading import read_graph6

from src.descriptors.edge_descriptors import edge_descriptors_dict
from src.descriptors.node_descriptors import node_descriptors_dict
from src.descriptors.embeddings import get_function
from src.graph_utils.dataset import Dataset, Graph
from src.descriptors.embeddings import create_embedding_function


g8_iter = list(read_graph6("graph8c"))
g7_iter = list(read_graph6("graph7c"))
g5_iter = list(read_graph6("graph5"))


test_graphs: List[nk.Graph] = [
    g8_iter[0],
    g8_iter[2],
    g8_iter[200],
    g7_iter[0],
    g7_iter[10],
    g7_iter[100],
    g5_iter[1],
    g5_iter[10],
]

histograms_path = os.path.join(
    Settings.processed_datasets_dir, "histograms_ranges.json"
)
if os.path.exists(histograms_path):
    with open(histograms_path, "r") as f:
        read_data = json.load(f)

    stored_histogram_ranges = {key: value for key, value in read_data.items()}

dataset_name = "graph4c"
test_dataset = Dataset(dataset_name)

node_counts = [g.numberOfNodes() for g in test_graphs]
edge_counts = [g.graph.numberOfEdges() for g in test_dataset]


@pytest.mark.parametrize(
    "descriptor_name,descriptor_function", list(node_descriptors_dict.items())
)
def test_node_descriptors(descriptor_name, descriptor_function):
    for graph, node_count in zip(test_graphs, node_counts):
        assert descriptor_function(graph).shape[0] == node_count


@pytest.mark.parametrize(
    "descriptor_name,descriptor_function", list(edge_descriptors_dict.items())
)
def test_edge_descriptors(descriptor_name, descriptor_function):
    for graph, edge_count in zip(test_dataset, edge_counts):
        assert descriptor_function(graph.graph).shape[0] == edge_count


def permute_graph(graph: nk.Graph) -> nk.Graph:
    n = graph.numberOfNodes()

    permutation = list(range(n))
    random.shuffle(permutation)

    mapping = dict(zip(range(n), permutation))
    graph_perm = nk.graph.Graph(n)

    for u, v in graph.iterEdges():
        graph_perm.addEdge(mapping[u], mapping[v])

    return graph_perm


@pytest.mark.parametrize(
    "descriptor_name,descriptor_function",
    list(edge_descriptors_dict.items()) + list(node_descriptors_dict.items()),
)
def test_get_function(descriptor_name, descriptor_function):
    test_function = get_function(descriptor_name)
    for G in test_dataset:
        # test if correct function loads
        assert np.allclose(
            test_function(G), descriptor_function(G.graph), equal_nan=True
        )


@pytest.mark.parametrize(
    "descriptor_name",
    list(edge_descriptors_dict.keys()) + list(node_descriptors_dict.keys()),
)
def test_permutation_invariancy(descriptor_name):
    nk.setNumberOfThreads(1)
    test_function = get_function(descriptor_name)
    random.seed(1234)
    for index, G in enumerate(test_dataset):
        embeddings = [np.sort(np.array(test_function(G)))]

        for i in range(10):
            G_perm = Graph("", 1, permute_graph(G.graph), k_graphs={})
            emb = np.sort(np.array(test_function(G_perm)))
            for j in range(i):
                assert np.allclose(
                    embeddings[j], emb
                ), f"Descriptor '{descriptor_name}' is not permutation-invariant index: [{index}] from {dataset_name} {i,j} {embeddings[j], emb}"
            embeddings.extend([emb])


@pytest.mark.parametrize(
    "descriptor_name",
    list(edge_descriptors_dict.keys()) + list(node_descriptors_dict.keys()),
)
def test_histogram_permutation_invariancy(descriptor_name):
    nk.setNumberOfThreads(1)

    ranges = stored_histogram_ranges[dataset_name][descriptor_name]
    print(ranges)
    embedding_fun = create_embedding_function(
        (descriptor_name,), 81, histogram_ranges=[tuple(ranges)]
    )

    # test of descriptor being permutation-invariant
    random.seed(1234)
    for G in test_dataset:
        embeddings = [np.array(embedding_fun(G))]

        for i in range(10):
            G_perm = Graph("", 1, permute_graph(G.graph), k_graphs={})
            emb = np.array(embedding_fun(G_perm))
            for j in range(i):
                assert np.allclose(
                    embeddings[j], emb
                ), f"Descriptor '{descriptor_name}' is not permutation-invariant {i,j} {embeddings[j], emb}"
            embeddings.extend([emb])
