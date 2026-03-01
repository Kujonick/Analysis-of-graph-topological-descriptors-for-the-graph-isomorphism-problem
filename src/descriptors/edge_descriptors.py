from functools import partial
from typing import Type

import networkit as nk
import numpy as np

from networkit.linkprediction import (
    LinkPredictor,
    AdjustedRandIndex,
    CommonNeighborsIndex,
    JaccardIndex,
    AdamicAdarIndex,
    KatzIndex,
    NeighborhoodDistanceIndex,
    NeighborsMeasureIndex,
    PreferentialAttachmentIndex,
    ResourceAllocationIndex,
    SameCommunityIndex,
    TotalNeighborsIndex,
    AlgebraicDistanceIndex,
)
from networkit.sparsification import (
    LocalDegreeScore,
    SCANStructuralSimilarityScore,
    TriangleEdgeScore,
    ChibaNishizekiQuadrangleEdgeScore,
    ChibaNishizekiTriangleEdgeScore,
    LocalSimilarityScore,
    SimmelianSparsifierNonParametric,
)
from networkit.centrality import Betweenness, SpanningEdgeCentrality


edge_descriptors_dict = {}


def link_predictor_template(predictor: Type[LinkPredictor], graph: nk.Graph, *args):
    descriptor = predictor(graph, *args)
    return np.array([descriptor.run(*e) for e in graph.iterEdges()], np.float32)


def add_to_dict(name, can_be_normalized=False):
    def decorator(f):
        if can_be_normalized:
            edge_descriptors_dict[name] = partial(f, normalize=False)
            edge_descriptors_dict[name + "_normalized"] = partial(f, normalize=True)
        else:
            edge_descriptors_dict[name] = f

        return f

    return decorator


# =========== linkprediction


@add_to_dict("jaccard_index", can_be_normalized=True)
def jaccard_index(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    if normalize:
        return link_predictor_template(JaccardIndex, graph)
    else:
        return link_predictor_template(CommonNeighborsIndex, graph)


@add_to_dict("adamic_adar")
def calculate_adamic_adar_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(AdamicAdarIndex, graph)


@add_to_dict("katz_index")
def calculate_katz_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(KatzIndex, graph)


@add_to_dict("neighborhood_distance")
def calculate_neighborhood_distance_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(NeighborhoodDistanceIndex, graph)


@add_to_dict("neighborhood_measure")
def calculate_neighborhood_measure_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(NeighborsMeasureIndex, graph)


@add_to_dict("preferential_attachment")
def calculate_preferential_attachment_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(PreferentialAttachmentIndex, graph)


@add_to_dict("resource_allocation")
def calculate_resource_allocation_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(ResourceAllocationIndex, graph)


@add_to_dict("same_community")
def calculate_same_community_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(SameCommunityIndex, graph)


@add_to_dict("total_nieghbors")
def calculate_total_nieghbors_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(TotalNeighborsIndex, graph)


@add_to_dict("algebraic_distance")
def calculate_algebraic_distance_index(graph: nk.Graph) -> np.ndarray:
    descriptor = AlgebraicDistanceIndex(
        graph, 20, 200  # numberSystems
    )  # numberIterations
    descriptor.preprocess()
    return np.array([descriptor.run(*e) for e in graph.iterEdges()], np.float32)


@add_to_dict("ari")
def calculate_adjusted_rand_index(graph: nk.Graph) -> np.ndarray:
    return link_predictor_template(AdjustedRandIndex, graph)


# ========= sparsification


@add_to_dict("lds")
def local_degree_score(graph: nk.Graph) -> np.ndarray:
    graph.indexEdges()
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, np.float32)


@add_to_dict("scan")
def calculate_scan_structural_similarity_score(graph: nk.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = SCANStructuralSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    return np.array(scores, np.float32)


@add_to_dict("cn_quadrangle")
def calculate_CN_quadrangle_edge_score(graph: nk.Graph) -> np.ndarray:
    desc = ChibaNishizekiQuadrangleEdgeScore(graph)
    desc.run()
    return np.array(desc.scores(), np.float32)


@add_to_dict("cn_triangle")
def calculate_CN_triangle_edge_score(graph: nk.Graph) -> np.ndarray:
    desc = ChibaNishizekiTriangleEdgeScore(graph)
    desc.run()
    return np.array(desc.scores(), np.float32)


@add_to_dict("lss")
def calculate_local_similarity_sparsification(graph: nk.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = LocalSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    return np.array(scores, np.float32)


@add_to_dict("simmelian_sparsifier_np")  # Np - non parametric
def calculate_simmelian_sparsifier(graph: nk.Graph) -> np.ndarray:
    score = SimmelianSparsifierNonParametric()
    scores = score.scores(graph)
    return np.array(scores, np.float32)


# =========== centrality


@add_to_dict("edge_betweenness", can_be_normalized=True)
def calculate_edge_betweenness(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    graph.indexEdges()
    betweeness = Betweenness(graph, normalized=normalize, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    return np.array(scores, np.float32)


@add_to_dict("spanning_edge")
def calculate_spanning_edge_centrality(graph: nk.Graph) -> np.ndarray:
    graph.indexEdges()
    betweeness = SpanningEdgeCentrality(graph)
    betweeness.run()
    scores = betweeness.scores()
    return np.array(scores, np.float32)
