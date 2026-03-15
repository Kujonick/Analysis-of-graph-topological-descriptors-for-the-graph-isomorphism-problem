from typing import Generator, Dict, Any, Optional
from src.settings import Settings

import networkit as nk
import networkx as nx
import numpy as np
import json
import os


METADATA_PATH = Settings.raw_metadata_path


def read_graph6(
    name: str, 
    output_format: str = "networkit",
) -> Generator[nk.Graph, None, None]:

    def output_mapper(graph: nx.Graph):
        return nk.nxadapter.nx2nk(graph) if output_format == "networkit" else graph

    filename = (f"{name}.g6" if ".g6" not in name else name)
    path = Settings.raw_datasets_dir / filename
        
    with open(path, "r") as f:
        for line in map(str.strip, f):
            if not line:
                continue

            graph = nx.from_graph6_bytes(line.encode())
            graph = output_mapper(graph)
            yield graph


def read_metadata() -> Dict[str, dict]:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata

