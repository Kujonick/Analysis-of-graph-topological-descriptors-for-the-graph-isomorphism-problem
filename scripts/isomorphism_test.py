from itertools import permutations
import numpy as np
import networkit as nk
import random
import sys
from time import perf_counter

from src.graph_utils.dataset import Dataset, Graph
from src.descriptors.embeddings import get_function
from src.descriptors.edge_descriptors import edge_descriptors_dict
from src.descriptors.node_descriptors import node_descriptors_dict
from src.settings import Settings
from tqdm import tqdm

settings = Settings()
settings.ensure_paths()
nk.setNumberOfThreads(1)
save_path = settings.isomorphism_test_dir / "isomorphism-and-time-test.csv"
if not save_path.exists():
    with open(save_path, "w") as f:
        f.write("dataset_name,descriptor_name,mod_same_vector,time\n")


def permute_graph(graph: nk.Graph, layout=None, permutation=None) -> nk.Graph:
    n = graph.numberOfNodes()

    if permutation is None:
        permutation = list(range(n))
        random.shuffle(permutation)

    mapping = dict(zip(range(n), permutation))
    rever_mapping = {v: k for k, v in mapping.items()}
    graph_perm = nk.graph.Graph(n)

    for u, v in graph.iterEdges():
        graph_perm.addEdge(mapping[u], mapping[v])
    if layout is not None:
        new_layout = {i: layout[rever_mapping[i]] for i in permutation}
    else:
        new_layout = None
    return graph_perm, new_layout


def recursive_size(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            [recursive_size(k, seen) + recursive_size(v, seen) for k, v in obj.items()]
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum([recursive_size(item, seen) for item in obj])

    return size


n = int(sys.argv[1])
dataset_name = f"graph{n}"
dataset = Dataset(dataset_name)

possible_perms = list(permutations(range(n)))
all_graphs = []
for i, G in enumerate(tqdm(dataset, desc="generating_graphs")):

    graphs = []

    for perm in possible_perms:
        perm_graph, perm_layout = permute_graph(G.graph, permutation=perm)

        G_perm = Graph("", 1, perm_graph, k_graphs={})
        graphs.append(G_perm)

    all_graphs.append(graphs)

print(f"SIZE OF ALL GRAPHS: {recursive_size(all_graphs) / (1024 * 1024):.2f} MB ")

descriptors = list(edge_descriptors_dict.keys()) + list(node_descriptors_dict.keys())

###
descriptors = list(filter(lambda x: "rounded4" in x, descriptors))
print(sorted(descriptors))
# raise SystemExit
###
for descriptor_name in tqdm(descriptors, desc="isomorphism_test"):
    # print(descriptor_name)

    test_function = get_function(descriptor_name)
    results = []
    times = []
    for i, graphs in enumerate(all_graphs):
        embeddings = []
        count = []
        timer = 0
        for G_perm in graphs:
            start_time = perf_counter()
            emb_raw = np.array(test_function(G_perm))
            timer += perf_counter() - start_time
            emb = np.sort(emb_raw)
            for j, prev_emb in enumerate(embeddings):
                if np.allclose(embeddings[j], emb):
                    count[j] += 1
                    break

            else:
                embeddings.append(emb)
                count.append(1)

        results.append(max(count) / len(graphs))
        times.append(timer / len(graphs))
    with open(save_path, "a") as f:
        f.write(
            f"{dataset_name},{descriptor_name},{min(results)},{float(np.mean(times))}\n"
        )
