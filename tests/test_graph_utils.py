import pytest
import networkit
import os
from pathlib import Path

from src.settings import Settings
from src.graph_utils.reading import read_graph6, read_metadata
from src.graph_utils.dataset import Dataset


## ------ reading ------
@pytest.mark.parametrize("input_file,graph_count", [("graph5", 34), ("graph7", 1044)])
def test_read_graph6(input_file, graph_count):
    graphs = [g for g in read_graph6(input_file)]
    assert len(graphs) == graph_count
    assert all(map(lambda x: isinstance(x, networkit.Graph), graphs))


def test_read_metadata():
    metadata = read_metadata()
    assert isinstance(metadata, dict)


## ------ dataset ------
@pytest.mark.parametrize(
    "input_file,graph_count,number_of_nodes", [("graph5", 34, 5), ("graph7", 1044, 7)]
)
def test_dataset(input_file, graph_count, number_of_nodes):
    dataset = Dataset(input_file)

    metadata = dataset.get_metadata()
    assert metadata["number_of_nodes"] == number_of_nodes
    assert metadata["graph_count"] == graph_count


@pytest.fixture
def temp_file():
    file_name = Settings.raw_datasets_dir / "test_dataset.g6"

    with open(file_name, "w") as f:
        f.write("DTw\n")
        f.write("DQw\n")
        f.write("E?~o\n")

    yield file_name

    os.remove(file_name)


def test_k_graph(temp_file: Path):
    # creation of dataset
    dataset = Dataset(temp_file.stem)
    # create k_graphs
    dataset.generate_k_graph_cache(2)
    dataset.generate_k_graph_cache(3)

    # are they correct?

    # create another dataset

    # check if it reading the cached files
