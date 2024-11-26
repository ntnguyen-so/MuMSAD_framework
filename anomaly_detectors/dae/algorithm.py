import argparse
import json
import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from model import AutoEn
from dataclasses import dataclass, asdict
import shutil
import datetime
from sklearn.preprocessing import MinMaxScaler


@dataclass
class CustomParameters:
    latent_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.005
    noise_ratio: float = 0.1
    split: float = 0.8
    early_stopping_delta: float = 1e-2
    early_stopping_patience: int = 10
    random_state: int = 42


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


def load_data(args):
    df = pd.read_csv(args.dataInput)
    data = MinMaxScaler().fit_transform(df.iloc[:, 1:-1].values)
    labels = df.iloc[:, -1].values
    return data, labels


def train(args):
    xtr, ytr = load_data(args)
    ii = (ytr == 0)
    not_anamoly_data = xtr[ii]
    params = asdict(args.customParameters)
    del params["random_state"]
    model = AutoEn(**params)
    model.fit(not_anamoly_data, args.modelOutput)
    shutil.make_archive(args.modelOutput, "zip", "check")


def pred(args):
    xte, _ = load_data(args)
    if args.runtimeOutput:
        start_process_time = datetime.datetime.now()    
    shutil.unpack_archive(args.modelInput+".zip", "m", "zip")
    model = keras.models.load_model("m")
    pred = model.predict(xte)
    pred_per_var = pred.copy()
    pred = np.mean(np.abs(pred - xte), axis=1)
    
    # output overall anomaly scores
    anomaly_scores_per_var_ranking = np.argsort(-anomaly_scores_per_var, axis=1)
    np.savetxt(args.dataOutput, pred, delimiter= ",")
    
    # output dimensional anomaly scores
    anomaly_scores_per_var = np.abs(pred_per_var - xte)
    pd.DataFrame(anomaly_scores_per_var).to_csv(args.anomalyScorePerVarOutput, index=False, header=None)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random, tensorflow
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)


if __name__=="__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    else:
        pred(args)
