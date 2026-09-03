from typing import Optional, Dict

import networkit as nk
import networkx as nx


def plot_graph(
    graph: nk.Graph, layout: Optional[Dict] = None, ax=None, label_edges=True
) -> None:
    edge_labels = None

    edge_labels = {}
    for u, v in graph.iterEdges():
        eid = graph.edgeId(u, v)
        edge_labels[(u, v)] = str(eid)

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

    if edge_labels is not None and label_edges:
        nx.draw_networkx_edge_labels(
            graph,
            pos=layout,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax,
        )
