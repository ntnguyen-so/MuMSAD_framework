#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from typing import Optional
from dataclasses import dataclass
from pyod.models.pca import PCA
import datetime
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

@dataclass
class CustomParameters:
    n_components: Optional[int] = None
    n_selected_components: Optional[int] = None
    whiten: bool = False
    svd_solver: str = 'auto'
    tol: float = 0.0
    max_iter: Optional[int] = None
    random_state: int = 42
    contamination: float = 0.


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        if "runtimeOutput" not in args.keys():
            args["runtimeOutput"] = None
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(config: AlgorithmArgs) -> (np.ndarray, float):
    df = pd.read_csv(config.dataInput)
    data = MinMaxScaler().fit_transform(df.iloc[:, 1:-1].values)
    labels = df.iloc[:, -1].values
    contamination = min(.49, labels.sum() / len(labels))
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination
    return data, contamination


def main(config: AlgorithmArgs):
    data, contamination = load_data(config)
    if config.runtimeOutput:
        start_process_time = datetime.datetime.now()

    clf = PCA(
        contamination=contamination,
        n_components=config.customParameters.n_components,
        n_selected_components=config.customParameters.n_selected_components,
        whiten=config.customParameters.whiten,
        svd_solver=config.customParameters.svd_solver,
        tol=config.customParameters.tol,
        iterated_power=config.customParameters.max_iter or "auto",
        random_state=config.customParameters.random_state,
        copy=True,
        weighted=True,
        standardization=True,
    )
    clf.fit(data)
    
    # output overall anomaly scores
    scores = clf.decision_scores_
    np.savetxt(config.dataOutput, scores, delimiter=",")

    # output dimensional anomaly scores
    scores_per_var = cdist(data, clf.selected_components_) / clf.selected_w_components_
    pd.DataFrame(scores_per_var).to_csv(config.anomalyScorePerVarOutput, index=False, header=None)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")

    if config.executionType == "train":
        print("Nothing to train, finished!")
    elif config.executionType == "execute":
        main(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
