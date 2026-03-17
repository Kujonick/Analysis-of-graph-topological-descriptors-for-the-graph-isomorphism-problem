import functools
import json
import os
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Tuple
from joblib import Parallel, delayed
import networkit as nk
import psutil
import numpy as np
import pandas as pd
import xxhash
from dataclasses import dataclass

from .graph_utils.dataset import Dataset
from .descriptors.embeddings import create_embedding_function, normalize_features
from .graph_utils.reading import read_graph6
from .settings import Settings

SAVING_PATH = Settings.processed_datasets_dir

CPU_COUNT: int = psutil.cpu_count()
ORDER = ["features", "dataset_name"]

@dataclass
class TestParameters():
    features: Tuple[str]
    dataset: Dataset

    def create_dict(self):
        return {
            'features' : np.array(self.features),
            'dataset_name' : self.dataset.name
        }
    

def open_test_enviroment(parameters: TestParameters, **function_kwargs):

    metadata = parameters.dataset.get_metadata()

    graph_reader = iter(parameters.dataset)

    embedding_function = create_embedding_function(
        parameters.features,
        bins_per_feature=metadata["number_of_nodes"] ** 2,
        **function_kwargs,
    )
    return graph_reader, embedding_function



def select_problematic_ids(
    parameters: TestParameters, **function_kwargs
) -> Dict[str, list[int]]:
    """goes through all the graphs and selects only the ones that have collisions on embedding"""

    graph_reader, embedding_function = open_test_enviroment(parameters, **function_kwargs)

    collisions: dict[str, list[int]] = {}
    hashes: dict[str, int] = {}
    for graph_id, graph in enumerate(graph_reader):

        embedding = embedding_function(graph)
        h = xxhash.xxh128_hexdigest(embedding.tobytes())

        if h not in hashes:
            hashes[h] = graph_id
        else:
            if h in collisions:
                collisions[h].append(graph_id)
            else:
                collisions[h] = [hashes[h], graph_id]
    return collisions


def find_optimal_histogram_ranges(
    parameters: TestParameters, **function_kwargs
) -> List[Tuple[float, float]]:
    """function that goes through all descriptor values per graphs and finds minimum and maximum of each feature value"""

    dataset = parameters.dataset
    graph_reader, embedding_function = open_test_enviroment(parameters, **function_kwargs)
    metadata = dataset.get_metadata()
    
    first_graph = next(graph_reader)
    function_values: List[np.ndarray] = embedding_function(first_graph)
    function_values = [arr[~np.isnan(arr)] for arr in function_values]
    hist_ranges: List[Tuple[float, float]] = [
        (np.min(arr), np.max(arr)) if len(arr) else (np.inf, -np.inf)
        for arr in function_values
    ]
    i = 0
    for graph in graph_reader:
        i += 1
        function_values = embedding_function(graph)
        function_values = [arr[~np.isnan(arr)] for arr in function_values]
        hist_ranges = [
            (
                (min(ranges[0], np.min(values)), max(ranges[1], np.max(values)))
                if len(values)
                else ranges
            )
            for ranges, values in zip(hist_ranges, function_values)
        ]
    # in situation that range is too small and there is no way to fit all bins into
    for i, range in enumerate(hist_ranges):
        right, left = range[1], range[0]
        step = 1e-4
        limit = 2 * metadata["number_of_nodes"] ** 2
        while True:
            a = np.linspace(left, right, limit, dtype=np.float32)
            if len(np.unique(a)) == len(a):
                break
            right += step

        hist_ranges[i] = (left, right)
    return hist_ranges


def reduce_number_of_features(
    stored_ranges_dict, parameters: TestParameters, **other_features
) -> Tuple[str, ...]:  
    name = parameters.dataset.name
    if name not in stored_ranges_dict:
        stored_ranges_dict[name] = {}
        return parameters.features

    features_to_be_used = tuple(
        feature
        for feature in parameters.features
        if feature not in stored_ranges_dict[name]
    )
    return features_to_be_used


def update_histogram_ranges(
    stored_ranges_dict,
    features_to_update,
    histogram_ranges,
    parameters: TestParameters,
):
    
    for feature, ranges in zip(features_to_update, histogram_ranges):
        stored_ranges_dict[parameters.dataset.name][feature] = tuple(map(float, ranges))


def read_histogram_ranges(
    stored_ranges_dict, parameters: TestParameters
) -> List[Tuple[int, int]]:
    return [stored_ranges_dict[parameters.dataset.name][feature] for feature in parameters.features]


def _values_equal(a, b):
    if isinstance(a, (np.ndarray, list)) and isinstance(b, (np.ndarray, list)):
        return np.array_equal(a, b)
    return a == b


def single_test(parameters: TestParameters, histogram_ranges):
    nk.setNumberOfThreads(1)

    result = select_problematic_ids(parameters, embeddings=True, histogram_ranges=histogram_ranges)  # type: ignore
    result = (
        [item for sublist in result.values() for item in sublist]
        if result
        else np.array([-1])
    )
    return result


def single_histogram_range_calc(parameters: TestParameters, features_to_be_used) -> List[Tuple[float, float]]:
    nk.setNumberOfThreads(1)

    parameters2 = TestParameters(features=features_to_be_used, dataset=parameters.dataset)
    histogram_ranges = find_optimal_histogram_ranges(parameters2, embeddings=False)
    return histogram_ranges


def _row_matches(row, criteria):
    return all(_values_equal(row[k], v) for k, v in criteria.items())


def tests(parameters_list: List[TestParameters]):

    # reading outputs file, having parameters values and list of all
    output_path = os.path.join(SAVING_PATH, "table.parquet")
    if os.path.exists(output_path):
        outputs_df = pd.read_parquet(output_path, engine="pyarrow")
    else:
        outputs_df = pd.DataFrame(columns=ORDER + ["result"])

    histograms_path = os.path.join(SAVING_PATH, "histograms_ranges.json")
    if os.path.exists(histograms_path):
        with open(histograms_path, "r") as f:
            read_data = json.load(f)

        stored_histogram_ranges = {key: value for key, value in read_data.items()}
    else:
        stored_histogram_ranges = {}

    filtered_parameters: List[TestParameters] = []
    for parameters in parameters_list:
        parameters.features = normalize_features(parameters.features)
        # check if this set of parameters already was run
        exists = outputs_df.apply(
            lambda row: _row_matches(row, parameters.create_dict()),
            axis=1,
        ).any()
        if len(outputs_df) > 0 and exists:
            continue
        filtered_parameters.append(parameters)

    try:
        # calculating histogram ranges

        with tqdm(total=len(filtered_parameters)) as progress_bar:
            progress_bar.set_postfix(phase="histogram_ranges")
            features_for_histogram_calc: List[Tuple[TestParameters, Tuple[str, ...]]] = []

            for parameters in filtered_parameters:

                # reusing already calculated histogram ranges
                features_to_be_used = reduce_number_of_features(
                    stored_histogram_ranges, parameters=parameters
                )
                if len(features_to_be_used) > 0:
                    features_for_histogram_calc.append((parameters, features_to_be_used))
                else:
                    progress_bar.update(1)
                    continue

                if len(features_for_histogram_calc) == CPU_COUNT:
                    histogram_ranges_batch = Parallel(n_jobs=CPU_COUNT)(
                        delayed(single_histogram_range_calc)(
                            parameters, features_to_be_used
                        )
                        for parameters, features_to_be_used in features_for_histogram_calc
                    )
                    for histogram_ranges, (parameters, features_to_be_used) in zip(
                        histogram_ranges_batch, features_for_histogram_calc
                    ):
                        update_histogram_ranges(
                            stored_histogram_ranges,
                            features_to_be_used,
                            histogram_ranges,
                            parameters,
                        )
                    progress_bar.update(CPU_COUNT)

                    features_for_histogram_calc.clear()

            if features_for_histogram_calc:
                histogram_ranges_batch = Parallel(
                    n_jobs=len(features_for_histogram_calc)
                )(
                    delayed(single_histogram_range_calc)(kwargs, features_to_be_used)
                    for kwargs, features_to_be_used in features_for_histogram_calc
                )
                for histogram_ranges, (parameters, features_to_be_used)  in zip(
                    histogram_ranges_batch, features_for_histogram_calc
                ):
                    update_histogram_ranges(
                        stored_histogram_ranges,
                        features_to_be_used,
                        histogram_ranges,
                        parameters,
                    )
                progress_bar.update(len(features_for_histogram_calc))

        # calculating colisions

        with tqdm(total=len(filtered_parameters)) as progress_bar:
            for i in range(0, len(filtered_parameters), CPU_COUNT):
                parameters_batch = filtered_parameters[i : i + CPU_COUNT]
                histogram_ranges_batch = [
                    read_histogram_ranges(stored_histogram_ranges, parameters)
                    for parameters in parameters_batch
                ]
                results = Parallel(n_jobs=CPU_COUNT)(
                    delayed(single_test)(parameters, histogram_ranges)
                    for parameters, histogram_ranges in zip(
                        parameters_batch, histogram_ranges_batch
                    )
                )

                outputs_df = pd.concat(
                    [
                        outputs_df,
                        pd.DataFrame(
                            [
                                dict(
                                    **parameters.create_dict(),
                                    result=[int(n) for n in result],
                                )
                                for result, parameters in zip(results, parameters_batch)
                            ]
                        ),
                    ]
                )
                print(outputs_df)
                print(outputs_df.dtypes)
                progress_bar.update(len(parameters_batch))

    except KeyboardInterrupt:
        pass

    finally:
        outputs_df.to_parquet(output_path, engine="pyarrow")
        with open(histograms_path, "w") as f:
            json.dump(stored_histogram_ranges, f, indent=4)
