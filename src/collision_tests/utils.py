from src.descriptors.embeddings import create_embedding_function
import networkit as nk

from src.graph_utils.dataset import Dataset


import numpy as np


from dataclasses import dataclass
from typing import Callable, Generator, Tuple


@dataclass
class TestParameters():
    features: Tuple[str, ...]
    dataset: Dataset

    def create_dict(self):
        return {
            'features' : np.array(self.features),
            'dataset_name' : self.dataset.name
        }


def open_test_enviroment(parameters: TestParameters, **function_kwargs) -> Tuple[Dataset, Callable[[nk.Graph],np.ndarray]]:

    metadata = parameters.dataset.get_metadata()

    graph_reader = iter(parameters.dataset)

    embedding_function = create_embedding_function(
        parameters.features,
        bins_per_feature=metadata["number_of_nodes"] ** 2,
        **function_kwargs,
    )
    return graph_reader, embedding_function


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


def _values_equal(a, b):
    if isinstance(a, (np.ndarray, list)) and isinstance(b, (np.ndarray, list)):
        return np.array_equal(a, b)
    return a == b


def _row_matches(row, criteria):
    return all(_values_equal(row[k], v) for k, v in criteria.items())


