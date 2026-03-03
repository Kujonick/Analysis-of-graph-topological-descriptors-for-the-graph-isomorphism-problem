import pytest
import networkit
from src.graph_utils.reading import read_graph6, read_metadata


@pytest.mark.parametrize("input_file,expected_len", [('graph5', 34), ('graph7', 1044)])
def test_read_graph6(input_file, expected_len):
    graphs = [g for g in read_graph6(input_file)]
    assert len(graphs) == expected_len
    assert all(map(lambda x: isinstance(x, networkit.Graph), graphs))


def test_read_metadata():
    metadata = read_metadata()
    assert isinstance(metadata, dict)