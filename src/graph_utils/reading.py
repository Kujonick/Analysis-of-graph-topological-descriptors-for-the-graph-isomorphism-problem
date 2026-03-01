from typing import Generator, Dict, Any

import networkit as nk
import networkx as nx
import numpy as np
import json
import os

READ_PATH = "raw_datasets"
METADATA_FILE = "metadata.json"


def read_graph6(
    name: str, output_format: str = "networkit"
) -> Generator[nk.Graph, None, None]:

    def output_mapper(graph: nx.Graph):
        return nk.nxadapter.nx2nk(graph) if output_format == "networkit" else graph

    path = os.path.join(READ_PATH, f"{name}.g6" if ".g6" not in name else name)
    with open(path, "r") as f:
        for line in map(str.strip, f):
            if not line:
                continue

            graph = nx.from_graph6_bytes(line.encode())
            graph = output_mapper(graph)
            yield graph


def evaluate_matedata(name: str) -> Dict[str, Any]:

    graph_reader = read_graph6(name)
    node_count: np.ndarray = np.array([graph.numberOfNodes() for graph in graph_reader])
    graph_count = node_count.shape[0]
    return {"number_of_nodes": int(np.median(node_count)), "graph_count": graph_count}


def read_metadata() -> Dict[str, dict]:
    metadata_path = os.path.join(READ_PATH, METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata


def read_dataset_properties(name) -> Dict[str, Any]:

    metadata = read_metadata()

    if name in metadata:
        return metadata[name]

    metadata[name] = evaluate_matedata(name)

    metadata_path = os.path.join(READ_PATH, METADATA_FILE)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata[name]
