from typing import Dict, Any
import numpy as np
import json

from .reading import read_graph6, read_metadata
from ..settings import Settings


class Dataset:
    def __init__(
        self,
        name: str,
        output_format: str = "networkit",
    ) -> None:

        self.name = name
        self.graphs = list(read_graph6(name, output_format=output_format))
        self.len = len(self.graphs)

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
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
