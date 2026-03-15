from typing import Union, Optional, Dict

import networkit as nk
import networkx as nx


def plot_graph(
        graph: Union[nx.Graph, nk.Graph],
        layout: Optional[Dict]=None,
        ax=None
        ) -> None:
    
    if isinstance(graph, nk.Graph):
        graph = nk.nxadapter.nk2nx(graph)

    if layout is None:
        layout = nx.kamada_kawai_layout(graph)
    nx.draw(
        graph,
        with_labels=True,
        node_color="lightgreen",
        node_size=500,
        font_size=10,
        font_color="black",
        pos=layout,
        ax=ax,
    )
