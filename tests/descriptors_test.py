import pytest
from typing import List
from networkit import Graph

from src.graph_utils.reading import read_graph6
from src.descriptors.edge_descriptors import edge_descriptors_dict
from src.descriptors.node_descriptors import node_descriptors_dict

g8_iter = list(read_graph6('graph8c'))
g7_iter = list(read_graph6('graph7c'))
g5_iter = list(read_graph6('graph5'))


test_graphs: List[Graph] = [
    g8_iter[0],
    g8_iter[2],
    g8_iter[200],
    g7_iter[0],
    g7_iter[10],
    g7_iter[100],
    g5_iter[1],
    g5_iter[10],
]

node_counts = [g.numberOfNodes() for g in test_graphs]
edge_counts = [g.numberOfEdges() for g in test_graphs]



@pytest.mark.parametrize("descriptor_name,descriptor_function", list(node_descriptors_dict.items()))
def test_node_descriptors(descriptor_name, descriptor_function):
    for graph, node_count in zip(test_graphs, node_counts):
        assert descriptor_function(graph).shape[0] == node_count


@pytest.mark.parametrize("descriptor_name,descriptor_function", list(edge_descriptors_dict.items()))
def test_edge_descriptors(descriptor_name, descriptor_function):
    for graph, edge_count in zip(test_graphs, edge_counts):
        assert descriptor_function(graph).shape[0] == edge_count

