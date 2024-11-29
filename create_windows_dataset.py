import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.scores_loader import ScoresLoader
from utils.config import *


def create_tmp_dataset(
    name,
    save_dir,
    data_path,
    metric_path,
    window_size,
    metric, 
    data_normalization
):
    """Generates a new dataset from the given dataset. The time series
    in the generated dataset have been divided in windows.

    :param name: the name of the experiment
    :param save_dir: directory in which to save the new dataset
    :param data_path: path to dataset to be divided
    :param window_size: the size of the window timeseries will be split to
    :param metric: the specific metric to read
    """

    # Form new dataset's name
    name = '{}_{}'.format(name, window_size)

    # Load datasets
    dataloader = DataLoader(data_path)
    datasets = dataloader.get_dataset_names()
    x, y, fnames = dataloader.load(datasets)
    print(len(x))

    # Load metrics
    metricsloader = MetricsLoader(metric_path)
    metrics_data = metricsloader.read(metric)

    # Delete any data not in metrics (some timeseries metric scores were not computed)
    idx_to_delete = [i for i, x in enumerate(fnames) if x not in metrics_data.index]

    # Delete any time series shorter than requested window
    idx_to_delete_short = [i for i, ts in enumerate(x) if ts.shape[0] < window_size]
    if len(idx_to_delete_short) > 0:
        print(">>> Window size: {} too big for some timeseries. Deleting {} timeseries"
                .format(window_size, len(idx_to_delete_short)))
        idx_to_delete.extend(idx_to_delete_short)
        
    if len(idx_to_delete) > 0:
        for idx in sorted(idx_to_delete, reverse=True):
            del x[idx]
            del y[idx]
            del fnames[idx]
    metrics_data = metrics_data.loc[fnames] 
    assert(
        list(metrics_data.index) == fnames
    )

    # Keep only the metrics of the detectors (remove oracles)
    metrics_data = metrics_data[detector_names]

    # Split timeseries and compute labels
    ts_list, labels, fnames = split_and_compute_labels(x, metrics_data, window_size, fnames, data_normalization)
    
    # Create subfolder for each dataset
    for dataset in datasets:
        dir_path = os.path.join(save_dir, name, dataset)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        Path(os.path.join(save_dir, name, dataset)).mkdir(parents=True, exist_ok=True)

    # Save new dataset
    for ts, label, fname in tqdm(zip(ts_list, labels, fnames), total=len(ts_list), desc='Save dataset'):
        num_col = ts.shape[1]
        fname_split = fname.split('/')
        dataset_name = fname_split[-2]
        ts_name = fname_split[-1]
        
        np.save(os.path.join(save_dir, name, dataset_name, ts_name), ts)
        np.save(os.path.join(save_dir, name, dataset_name, ts_name + '_label'), label[:, np.newaxis])


def split_and_compute_labels(x, metrics_data, window_size, fnames, data_normalization=True):
    '''Splits the timeseries, computes the labels and returns 
    the segmented timeseries and the new labels.

    :param x: list of the timeseries to be segmented (as np arrays)
    :param metrics_data: df with the scores of all the detectors for every time series
    :param window_size: the size of the windows that will be created
    :return ts_list: list of n 2D arrays (n is number of time series in x)
    :return labels: labels for every created window
    '''
    ts_list = []
    labels = []

    assert(
        len(x) == metrics_data.shape[0]
    ), "Lengths and shapes do not match. Please check"

    old_indices = metrics_data.index.values.tolist()
    metrics_data.fillna(0, inplace=True)
    acc = metrics_data.iloc[:, 1:]
    acc = acc[acc.sum(axis=1) > 0]
    metrics_data = metrics_data.loc[acc.index.values.tolist(), :]
    metrics_data.to_csv('metrics.csv')

    #print(len(x), len(metrics_data))
    x = [x[old_indices.index(new_i)] for new_i in metrics_data.index.values.tolist()]
    fnames = [fnames[old_indices.index(new_i)] for new_i in metrics_data.index.values.tolist()]

    for ts_fnames, metric_label in tqdm(zip(fnames, metrics_data.idxmax(axis=1)), total=len(x), desc="Create dataset"):
        ts = pd.read_csv(data_path + ts_fnames, header=None)
        ts = ts.iloc[:, :-1].values

        # Split time series into windows
        ts_split = split_ts(ts, window_size, data_normalization)
        
        # Save everything to lists
        ts_list.append(ts_split)
        labels.append(np.ones(len(ts_split)) * detector_names.index(metric_label))

    assert(
        len(x) == len(ts_list) == len(labels)
    ), "Timeseries split and labels computation error, lengths do not match"
    
    # print(len(ts_list), len(labels), len(fnames))
    return ts_list, labels, fnames


def z_normalization(ts, decimals=5):
    ts = (ts - np.mean(ts)) / np.std(ts)
    ts = np.around(ts, decimals=decimals)

    # Test normalization
    assert(
        np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
    ), "After normalization it should: mean == 0 and std == 1"

    return ts

def split_ts(data, window_size, data_normalization=True):
    '''Split a timeserie into windows according to window_size.
    If the timeserie can not be divided exactly by the window_size
    then the first window will overlap the second.

    :param data: the timeserie to be segmented
    :param window_size: the size of the windows
    :return data_split: an 2D array of the segmented time series
    '''

    # Compute the modulo
    modulo = data.shape[0] % window_size
    
    # Normalize the data, if indicated
    if data_normalization:
        data = (data - data_stat_mean) / data_stat_std


    # Compute the number of windows
    k = data[modulo:].shape[0] / window_size
    assert(math.ceil(k) == k)

    # Split the timeserie
    data_split = np.split(data[modulo:], k)
    if modulo != 0:
        data_split.insert(0, list(data[:window_size]))
    data_split = np.asarray(data_split)

    return data_split




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Create temporary/experiment-specific dataset',
        description='This function creates a dataset of the size you want.  The data that will be used are set into the config file',
        epilog='Be careful where you save the generated dataset'
    )

    parser.add_argument('-n', '--name', type=str, help='path to save the dataset', default="")
    parser.add_argument('-s', '--save_dir', type=str, help='path to save the dataset', required=True)
    parser.add_argument('-p', '--path', type=str, help='path of the dataset to divide', required=True)
    parser.add_argument('-mp', '--metric_path', type=str, help='path to the metrics of the dataset given', default=metrics_path)
    parser.add_argument('-w', '--window_size', type=str, help='window size to segment the timeseries to', required=True)
    parser.add_argument('-m', '--metric', type=str, help='metric to use to produce the labels', default='AUC_PR')
    parser.add_argument('-d', '--data_normalization', type=bool, default=False, help='whether to normalize the data or not')

    args = parser.parse_args()

    if args.window_size == "all":
        window_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]

        for size in window_sizes:
            create_tmp_dataset(
                name=args.name,
                save_dir=args.save_dir,
                data_path=args.path,
                metric_path=args.metric_path,
                window_size=size, 
                metric=args.metric,
                data_normalization=args.data_normalization
            )
    else:        
        create_tmp_dataset(
            name=args.name,
            save_dir=args.save_dir,
            data_path=args.path,
            metric_path=args.metric_path,
            window_size=int(args.window_size), 
            metric=args.metric,
            data_normalization=args.data_normalization
        )


