from typing import Generator, Dict
from pathlib import Path
import networkit as nk
import networkx as nx
import json
import os

from src.settings import Settings


METADATA_PATH = Settings.raw_metadata_path


def read_graph6(
    name: str, output_format: str = "networkit", dir: Path = Settings.raw_datasets_dir
) -> Generator[nk.Graph, None, None]:

    def output_mapper(graph: nx.Graph):
        return nk.nxadapter.nx2nk(graph) if output_format == "networkit" else graph

    filename = f"{name}.g6" if ".g6" not in name else name
    path = dir / filename
    with open(path, "r") as f:
        content = f.read()
    content = content.split("\n")
    for line in map(str.strip, content):
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
