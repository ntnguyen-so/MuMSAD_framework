########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : timeseries_dataset
#
########################################################################

import os
from tqdm import tqdm

from utils.config import *

import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset

import copy
import random
import torch


def create_splits(data_path, split_per=0.7, seed=None, read_from_file=None):
    """Creates the splits of a single dataset to train, val, test subsets.
    This is done either randomly, or with a seed, or read the split from a
    file. Please see such files (the ones we used for our experiments) in 
    the directory "experiments/supervised_splits" or 
    "experiments/unsupervised_splits".

    Note: The test set will be created only when reading the splits
        from a file, otherwise only the train, val set are generated.
        The train, val subsets share the same datasets/domains. 
        The test sets that we used in the unsupervised experiments 
        do not (thus the supervised, unsupervised notation).

    :param data_path: path to the initial dataset to be split
    :param split_per: the percentage in which to create the splits
        (skipped when read_from_file)
    :param seed: the seed to use to create the 'random' splits
        (we strongly advise you to use small numbers)
    :param read_from_file: file to read fixed splits from

    :return train_set: list of strings of time series file names
    :return val_set: list of strings of time series file names
    :return test_set: list of strings of time series file names
    """
    train_set = []
    val_set = []
    test_set = []
    # dir_path = os.path.split(data_path)[0]
    dir_path = data_path
    
    # Set seed if provided
    if seed: 
        np.random.seed(seed)

    # Read splits from file if provided
    if read_from_file is not None:
        df = pd.read_csv(read_from_file, index_col=0)
        subsets = list(df.index)
        
        if 'train_set' in subsets and 'val_set' in subsets:
            print('static')
            train_set = [x for x in df.loc['train_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]
            val_set = [x for x in df.loc['val_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]

            return train_set, val_set, test_set
        elif 'train_set' in subsets and 'test_set' in subsets:
            train_val_set = [x for x in df.loc['train_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]
            test_set = [x for x in df.loc['test_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]

            datasets = list(set([x.split('/')[0] for x in train_val_set]))
            datasets.sort()
        else:
            raise ValueError('Did not expect this type of file.')
    else:
        datasets = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
        datasets.sort()

    if not os.path.isdir(dir_path):
        dir_path = '/'.join(dir_path.split('/')[:-1])
    
    # Random split of train & val sets
    for dataset in datasets:
        # Read file names
        fnames = os.listdir(os.path.join(dir_path, dataset))

        # Decide on the size of each subset
        n_timeseries = len(fnames)
        train_split = math.ceil(n_timeseries * split_per)
        val_split = n_timeseries - train_split

        # Select random files for each subset
        train_idx = np.random.choice(
            np.arange(n_timeseries), 
            size=train_split, 
            replace=False
        )
        val_idx = np.asarray([x for x in range(n_timeseries) if x not in train_idx])

        # Replace indexes with file names
        train_set.extend([os.path.join(dataset, fnames[x]) for x in train_idx])
        val_set.extend([os.path.join(dataset, fnames[x]) for x in val_idx])
    
    return train_set, val_set, test_set


def read_files(data_path):
    """Returns a list of names of the csv files in the 
    directory given.

    :param data_path: path to the directory/-ies with csv time series files
    :return fnames: list of strings
    """

    # Load everything you can find
    fnames = [x for x in os.listdir(data_path) if ".csv" in x and "tsfresh" not in x.lower()]
    
    if len(fnames) > 0:
        pass
        # dataset = data_path.split('/')[-1]
        # dataset = dataset if len(dataset) > 0 else data_path.split('/')[-2]
        # fnames = [os.path.join(dataset, x) for x in fnames]
    else:
        datasets = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]
        for dataset in datasets:
            
            # Read file names
            curr_fnames = os.listdir(os.path.join(data_path, dataset))
            curr_fnames = [os.path.join(dataset, x) for x in curr_fnames]
            fnames.extend(curr_fnames)

    return fnames

def amplitude_scaling(sample, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(*scale_range)
    scaled_sample = sample * scale_factor
    return scaled_sample

def time_warping(sample, warp_factor_range=(0.9, 1.1)):
    warp_factor = np.random.uniform(*warp_factor_range)
    num_points = sample.shape[0]
    warped_indices = np.arange(0, num_points, warp_factor)
    warped_sample = np.interp(np.arange(num_points), warped_indices, sample)
    return warped_sample

def noise_injection(sample, noise_level=0.3):
    noise = np.random.normal(loc=0, scale=noise_level, size=sample.shape)
    noisy_sample = sample + noise
    return noisy_sample

def z_normalization(ts, decimals=5):
    ts = (ts - np.mean(ts)) / np.std(ts)
    ts = np.around(ts, decimals=decimals)

    # Test normalization
    assert(
        np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
    ), "After normalization it should: mean == 0 and std == 1"

    return ts

class TimeseriesDataset(Dataset):
    def __init__(self, data_path, fnames, verbose=True, transform=False):
        self.data_path = data_path
        self.fnames = fnames
        self.labels = []
        self.samples = []
        self.indexes = []
        self.transform = transform

        if len(self.fnames) == 0:
            return
        
        # print('DATA TO TRAIN', self.fnames)
        # Read datasets
        for fname in tqdm(self.fnames, disable=not verbose, desc="Loading dataset"):
            fname_data = copy.deepcopy(fname)
            fname_data = fname_data.replace('.csv', '.npy')
            
            fname_label = copy.deepcopy(fname)
            fname_label = fname_label.replace('.csv', '_label.npy')
            # print(os.path.join(self.data_path, fname))
            # data = pd.read_csv(os.path.join(self.data_path, fname), index_col=0)
            # data = np.load(os.path.join(self.data_path, fname))
            # dataset = fname.split('/')[0]
            # curr_idxs = list(data.index)
            # curr_idxs = [os.path.join(dataset, x) for x in curr_idxs]

            
            # self.indexes.extend(curr_idxs)    
            
            label = np.load(os.path.join(self.data_path, fname_label)).ravel()
            self.labels.extend(label.tolist())
            
            data_data = np.load(os.path.join(self.data_path, fname_data))
            median_value = np.nanmedian(data_data)
            data_data[np.isnan(data_data)] = median_value
            
            # print(data_data.shape, fname_data, fname_label, fname)
            data_data = np.swapaxes(data_data, 1, 2)
            #data_data = z_normalization(data_data, 7)
            self.samples.append(data_data)
        
        # Concatenate samples and labels
        self.labels = np.asarray(self.labels)
        self.samples = np.concatenate(self.samples, axis=0)

        # Add channels dimension
        # self.samples = self.samples[:, np.newaxis, :]
        
    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        sample = copy.copy(self.samples[idx])
        if self.transform:
            #sample = time_warping(sample)
            sample = amplitude_scaling(sample)
            sample = noise_injection(sample)
        
        return sample, self.labels[idx]

    def __getallsamples__(self):
        if self.transform:
            for i in range(len(self.samples)):
                self.samples[i] = noise_injection(self.samples[i])
        return self.samples

    def __getalllabels__(self):
        return self.labels

    def getallindex(self):
        return self.indexes

    def __getlabel__(self, idx):
        return self.labels[idx]

    def get_weights_subset(self, device):
        '''Compute and return the class weights for the dataset'''

        # Count labels within those indices
        labels = np.fromiter(map(self.__getlabel__, range(self.__len__())), dtype=np.int16)

        # Compute weights
        labels_exist = np.unique(labels)
        labels_not_exist = [x for x in np.arange(len(detector_names)) if x not in labels_exist]
        sklearn_class_weights = list(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels))
        for i in labels_not_exist:
            sklearn_class_weights.insert(i, 1)

        # Test
        if False:
            print('------------------------------------------')
            counter = Counter(labels)
            for detector, weight in zip(detector_names, sklearn_class_weights):
                print(f'{detector} : {counter[detector_names.index(detector)]}, {weight:.3f}')

        return torch.Tensor(sklearn_class_weights).to(device)
