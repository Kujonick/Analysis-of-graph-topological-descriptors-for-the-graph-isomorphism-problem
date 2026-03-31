from typing import Dict, Any, List
import numpy as np
from pathlib import Path
import json
import networkit as nk
import networkx as nx

from .reading import read_graph6, read_metadata
from ..settings import Settings


class Graph:
    def __init__(self, dataset_dir: Path, index: int, graph: nk.Graph):
        self.index = index
        self.graph = graph
        self.dataset_dir = dataset_dir

        self.k_graph_cache_dir = dataset_dir / str(index)
        if not self.k_graph_cache_dir.exists():
            self.k_graph_cache_dir.mkdir()

        self.k_graphs = {}
        for path in self.k_graph_cache_dir.iterdir():
            graph = list(read_graph6(path.stem, dir=path.parents[0]))[0]
            graph.indexEdges()
            self.k_graphs[path.stem] = graph

        self.dmatrix = self.distance_matrix()

    def distance_matrix(self) -> np.ndarray:
        APSP = nk.distance.APSP(self.graph)
        APSP.run()
        distances = np.array(APSP.getDistances())
        distances[distances > 10_000] = (
            10_000  # this one is just for infinity, which means that path is not available
        )
        return distances

    def _create_k_graph(self, k: int) -> nk.Graph:
        distances = self.dmatrix
        n = distances.shape[0]
        new_graph = nk.Graph(n)
        new_edges = distances == k

        edges = np.argwhere(new_edges & np.triu(np.ones_like(distances)).astype(bool)).T
        new_graph.addEdges((edges[0], edges[1]))
        return new_graph

    def _create_modified_k_graph(self, k: int) -> nk.Graph:
        distances = self.dmatrix
        n = distances.shape[0]
        new_graph = nk.Graph(n)
        new_edges = distances <= k  # modified creates more dense graph with more edges

        edges = np.argwhere(
            new_edges & np.triu(np.ones_like(distances), 1).astype(bool)
        ).T
        new_graph.addEdges((edges[0], edges[1]))
        return new_graph

    @staticmethod
    def _write_kgraph(graph: nk.Graph, path: Path):
        G_nx = nk.nxadapter.nk2nx(graph)
        G_nx = nx.convert_node_labels_to_integers(G_nx)
        nx.write_graph6(G_nx, path)

    def generate_kgraph(self, k: int, modified: bool):
        name = ("mod" if modified else "") + str(k)
        path = self.k_graph_cache_dir / f"{name}.g6"
        if path.exists():
            return
        if modified:
            graph = self._create_modified_k_graph(k)
        graph = self._create_k_graph(k)
        self._write_kgraph(graph, path)
        self.k_graphs[name] = graph

    def get_k_graph(self, k: int, modified: bool = False) -> nk.Graph:
        name = ("mod" if modified else "") + str(k)

        graph = self.k_graphs.get(name, None)
        if graph is None:
            raise ValueError(
                f"Not precomputed graph parameters for {self.dataset_dir=}, {self.index=}, {k=}, {modified=}"
            )
        return graph


class Dataset:
    def __init__(
        self,
        name: str,
        output_format: str = "networkit",
    ) -> None:

        dataset_dir = Settings.k_graph_dir / name
        if not dataset_dir.exists():
            dataset_dir.mkdir()

        self.name = name
        nk_graphs: List[nk.Graph] = list(read_graph6(name, output_format=output_format))
        self.graphs: List[Graph] = []
        for i, g_nk in enumerate(nk_graphs):
            self.graphs.append(Graph(dataset_dir, i, g_nk))
        self.len = len(self.graphs)

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self) -> Graph:
        if self.num >= self.len:
            raise StopIteration
        result = self.graphs[self.num]
        self.num += 1
        return result

    def _evaluate_matedata(self) -> Dict[str, Any]:

        node_count: np.ndarray = np.array(
            [graph.numberOfNodes() for graph in self.graphs]
        )
        graph_count = node_count.shape[0]
        return {
            "number_of_nodes": int(np.median(node_count)),
            "graph_count": graph_count,
        }

    def get_metadata(self):
        metadata = read_metadata()
        name = self.name
        if name in metadata:
            return metadata[name]

        metadata[name] = self._evaluate_matedata()

        with open(Settings.raw_metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata[name]

    def generate_k_graph_cache(self, k: int, modified: bool = False):
        for graph in self.graphs:
            graph.generate_kgraph(k, modified)
