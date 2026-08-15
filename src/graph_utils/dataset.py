from typing import Dict, Any, List
import numpy as np
import json
import networkit as nk
import networkx as nx

from .reading import read_graph6, read_metadata
from ..settings import Settings


class Graph:
    def __init__(
        self,
        dataset_name: str,
        index: int,
        graph: nk.Graph,
        k_graphs: Dict[str, nk.Graph],
    ):
        self.index = index
        graph.indexEdges()
        self.graph = graph
        self.dataset_name = dataset_name

        self.k_graphs = k_graphs
        for g in self.k_graphs.values():
            g.indexEdges()
        self.dmatrix = None

    @staticmethod
    def get_kgraph_name(k: int, modified: bool) -> str:
        return ("mod" if modified else "") + str(k)

    def distance_matrix(self) -> np.ndarray:
        if self.dmatrix is not None:
            return self.dmatrix

        APSP = nk.distance.APSP(self.graph)
        APSP.run()
        distances = np.array(APSP.getDistances())
        distances[distances > 10_000] = (
            10_000  # this one is just for infinity, which means that path is not available
        )
        return distances

    def _create_k_graph(self, k: int) -> nk.Graph:
        distances = self.distance_matrix()
        n = distances.shape[0]
        new_graph = nk.Graph(n)
        new_edges = distances == k

        edges = np.argwhere(new_edges & np.triu(np.ones_like(distances)).astype(bool)).T
        new_graph.addEdges((edges[0], edges[1]))
        return new_graph

    def _create_modified_k_graph(self, k: int) -> nk.Graph:
        distances = self.distance_matrix()
        n = distances.shape[0]
        new_graph = nk.Graph(n)
        new_edges = distances <= k  # modified creates more dense graph with more edges

        edges = np.argwhere(
            new_edges & np.triu(np.ones_like(distances), 1).astype(bool)
        ).T
        new_graph.addEdges((edges[0], edges[1]))
        return new_graph

    def generate_kgraph(self, k: int, modified: bool) -> nk.Graph:
        name = self.get_kgraph_name(k, modified)
        if name in self.k_graphs:
            return
        if modified:
            graph = self._create_modified_k_graph(k)
        else:
            graph = self._create_k_graph(k)
        self.k_graphs[name] = graph
        return graph

    def get_k_graph(self, k: int, modified: bool = False) -> nk.Graph:
        name = self.get_kgraph_name(k, modified)

        graph = self.k_graphs.get(name, None)
        if graph is None:
            raise ValueError(
                f"Not precomputed graph parameters for {self.dataset_name=}, {self.index=}, {k=}, {modified=}"
            )
        return graph


def serialize_graph(G_nk: nk.Graph) -> str:
    G_nx = nk.nxadapter.nk2nx(G_nk)
    G_nx = nx.convert_node_labels_to_integers(G_nx)

    g6_bytes = nx.to_graph6_bytes(G_nx, header=False)
    g6_string = g6_bytes.decode().strip()
    return g6_string


class Dataset:
    def __init__(
        self,
        name: str,
        output_format: str = "networkit",
    ) -> None:
        # base graphs read
        self.name = name
        nk_graphs: List[nk.Graph] = list(read_graph6(name, output_format=output_format))

        # k-graph cache read
        self.k_graph_cache_dir = Settings.k_graph_dir / name
        if not self.k_graph_cache_dir.exists():
            self.k_graph_cache_dir.mkdir()

        k_graphs_datasets = {}
        for path in self.k_graph_cache_dir.iterdir():
            k_graphs_datasets[path.stem] = list(
                read_graph6(path.stem, dir=path.parents[0])
            )

        self.existing_caches = set(k_graphs_datasets.keys())

        self.graphs: List[Graph] = []
        for i, g_nk in enumerate(nk_graphs):
            self.graphs.append(
                Graph(
                    name,
                    i,
                    g_nk,
                    {
                        k_name: k_graph_list[i]
                        for k_name, k_graph_list in k_graphs_datasets.items()
                    },
                )
            )
        self.len = len(self.graphs)

    # def __iter__(self):   currently there are thread race problems, maybe TODO solve later
    #     self.num = 0
    #     return self

    # def __next__(self) -> Graph:
    #     if self.num >= self.len:
    #         raise StopIteration
    #     result = self.graphs[self.num]
    #     self.num += 1
    #     return result
    def __len__(self):
        return self.len

    def __getitem__(self, key: int):
        return self.graphs[key]

    def _evaluate_matedata(self) -> Dict[str, Any]:

        node_count: np.ndarray = np.array(
            [graph.graph.numberOfNodes() for graph in self.graphs]
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
        name = Graph.get_kgraph_name(k, modified)
        if name in self.existing_caches:
            return
        kgraphs = []
        for graph in self.graphs:
            kgraphs.append(graph.generate_kgraph(k, modified))

        serialized = "\n".join(map(serialize_graph, kgraphs))
        cache_path = self.k_graph_cache_dir / (name + ".g6")
        with open(cache_path, "w") as f:
            f.write(serialized)
        self.existing_caches.add(name)
