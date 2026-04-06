import json
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import networkit as nk
import psutil
import numpy as np
import pandas as pd
import traceback

from src.collision_tests.tasks import select_problematic_ids
from src.collision_tests.tasks import find_optimal_histogram_ranges
from src.collision_tests.utils import (
    TestParameters,
    _row_matches,
    reduce_number_of_features,
)

from ..descriptors.embeddings import normalize_features
from ..settings import Settings

SAVING_PATH = Settings.processed_datasets_dir

CPU_COUNT: int = psutil.cpu_count()
ORDER = ["features", "dataset_name"]


def single_test(parameters: TestParameters, histogram_ranges) -> np.ndarray:
    nk.setNumberOfThreads(1)

    result = select_problematic_ids(parameters, histogram_ranges=histogram_ranges)  # type: ignore
    result = (
        [item for sublist in result.values() for item in sublist]
        if result
        else np.array([-1])
    )
    return result


def single_histogram_range_calc(
    parameters: TestParameters, features_to_be_used: Tuple[str, ...]
) -> List[Tuple[float, float]]:
    try:
        nk.setNumberOfThreads(1)

        parameters2 = TestParameters(
            features=features_to_be_used, dataset=parameters.dataset
        )
        histogram_ranges = find_optimal_histogram_ranges(parameters2)
        return histogram_ranges
    except (
        Exception
    ) as e:  # Errors in multi-thread run are not visible, so we print them here
        print("ERROR:", e)
        traceback.print_exc()
        raise


class TestOperator:
    def __init__(self, single_thread: bool = False, cpu_use: int = CPU_COUNT):
        self.single_thread = single_thread
        self.cpu_count = cpu_use
        output_path = os.path.join(SAVING_PATH, "table.parquet")
        self.output_path = output_path

        if os.path.exists(output_path):
            outputs_df = pd.read_parquet(output_path, engine="pyarrow")
        else:
            outputs_df = pd.DataFrame(columns=ORDER + ["result"])
        self.outputs_df = outputs_df

        histograms_path = os.path.join(SAVING_PATH, "histograms_ranges.json")
        self.histograms_path = histograms_path
        if os.path.exists(histograms_path):
            with open(histograms_path, "r") as f:
                read_data = json.load(f)

            stored_histogram_ranges = {key: value for key, value in read_data.items()}
        else:
            stored_histogram_ranges = {}
        self.histogram_ranges: Dict[str, Dict[str, Tuple[float, float]]] = (
            stored_histogram_ranges
        )

    def filter_parameters(self, parameters_list: List[TestParameters]):
        filtered_parameters: List[TestParameters] = []
        for parameters in parameters_list:
            parameters.features = normalize_features(parameters.features)
            # check if this set of parameters already was run
            exists = self.outputs_df.apply(
                lambda row: _row_matches(row, parameters.create_dict()),
                axis=1,
            ).any()
            if len(self.outputs_df) > 0 and exists:
                continue
            filtered_parameters.append(parameters)
        return filtered_parameters

    def update_histogram_ranges(
        self,
        features_to_update,
        histogram_ranges: List[Tuple[float, float]],
        parameters: TestParameters,
    ):

        for feature, ranges in zip(features_to_update, histogram_ranges):
            self.histogram_ranges[parameters.dataset.name][feature] = (
                float(ranges[0]),
                float(ranges[1]),
            )

    def read_histogram_ranges(
        self, parameters: TestParameters
    ) -> List[Tuple[float, float]]:
        return [
            self.histogram_ranges[parameters.dataset.name][feature]
            for feature in parameters.features
        ]

    def _run_parralell_histograms(
        self, features_for_histogram_calc: List[Tuple[TestParameters, Tuple[str, ...]]]
    ):
        n_jobs = len(features_for_histogram_calc)
        histogram_ranges_batch: List[List[Tuple[float, float]]] = Parallel(
            n_jobs=n_jobs, backend="threading"
        )(
            delayed(single_histogram_range_calc)(parameters, features_to_be_used)
            for parameters, features_to_be_used in features_for_histogram_calc
        )
        for histogram_ranges, (parameters, features_to_be_used) in zip(
            histogram_ranges_batch, features_for_histogram_calc
        ):
            self.update_histogram_ranges(
                features_to_be_used,
                histogram_ranges,
                parameters,
            )

    def _run_single_histogram(self, parameters, features_to_be_used):
        parameters2 = TestParameters(
            features=features_to_be_used, dataset=parameters.dataset
        )
        histogram_ranges = find_optimal_histogram_ranges(parameters2)

        self.update_histogram_ranges(
            features_to_be_used,
            histogram_ranges,
            parameters,
        )

    def calculate_histogram(
        self,
        filtered_parameters: List[TestParameters],
    ):
        with tqdm(total=len(filtered_parameters)) as progress_bar:
            progress_bar.set_postfix(phase="histogram_ranges")
            features_for_histogram_calc: List[
                Tuple[TestParameters, Tuple[str, ...]]
            ] = []

            for parameters in filtered_parameters:

                # reusing already calculated histogram ranges
                features_to_be_used = reduce_number_of_features(
                    self.histogram_ranges, parameters=parameters
                )
                if len(features_to_be_used) > 0:
                    features_for_histogram_calc.append(
                        (parameters, features_to_be_used)
                    )
                else:
                    progress_bar.update(1)
                    continue

                if self.single_thread:
                    self._run_single_histogram(*features_for_histogram_calc[0])
                    progress_bar.update(1)
                    features_for_histogram_calc.clear()

                elif len(features_for_histogram_calc) == self.cpu_count:
                    self._run_parralell_histograms(features_for_histogram_calc)
                    progress_bar.update(self.cpu_count)

                    features_for_histogram_calc.clear()

            if features_for_histogram_calc:
                self._run_parralell_histograms(features_for_histogram_calc)
                progress_bar.update(len(features_for_histogram_calc))

    def calculate_collisions(self, filtered_parameters: List[TestParameters]):
        step = 1 if self.single_thread else self.cpu_count

        with tqdm(total=len(filtered_parameters)) as progress_bar:
            for i in range(0, len(filtered_parameters), step):
                parameters_batch = filtered_parameters[i : i + step]
                histogram_ranges_batch = [
                    self.read_histogram_ranges(parameters)
                    for parameters in parameters_batch
                ]

                if self.single_thread:
                    result = select_problematic_ids(parameters_batch[0], histogram_ranges=histogram_ranges_batch[0])  # type: ignore
                    result = (
                        [item for sublist in result.values() for item in sublist]
                        if result
                        else np.array([-1])
                    )
                    results = [result]
                else:
                    results = Parallel(n_jobs=self.cpu_count, backend="threading")(
                        delayed(single_test)(parameters, histogram_ranges)
                        for parameters, histogram_ranges in zip(
                            parameters_batch, histogram_ranges_batch
                        )
                    )

                self.outputs_df = pd.concat(
                    [
                        self.outputs_df,
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

                progress_bar.update(len(parameters_batch))

    def tests(self, parameters_list: List[TestParameters]):

        filtered_parameters = self.filter_parameters(parameters_list)
        try:
            self.calculate_histogram(filtered_parameters)
            self.calculate_collisions(filtered_parameters)

        except KeyboardInterrupt:
            pass

        finally:
            self.outputs_df.to_parquet(self.output_path, engine="pyarrow")
            with open(self.histograms_path, "w") as f:
                json.dump(self.histogram_ranges, f, indent=4)
