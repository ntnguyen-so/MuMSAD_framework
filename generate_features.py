import numpy as np
import pandas as pd
import argparse
import re
import os

from utils.data_loader import DataLoader

from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformations.panel.catch22 import Catch22
from sktime.transformations.panel.rocket._rocket import Rocket
import copy
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.feature_extraction import extract_features
import random
import pickle
from config import *

random.seed(42)


def generate_features(path):
    """Given a dataset it computes the TSFresh automatically extracted 
    features and saves the new dataset (which does not anymore contain
    time series but tabular data) into one .csv in the folder of the
    original dataset

    :param path: path to the dataset to be converted
    """
    for default_fc_parameters in ['tsfresh', 'tsfresh_minimal', 'catch22']:
        window_size = int(re.search(r'\d+', path).group())

        # Create name of new dataset
        dataset_name = [x for x in path.split('/') if str(window_size) in x][0]
        new_name = f"feature_{dataset_name}_{default_fc_parameters}.npy"
        new_name_label = f"feature_{dataset_name}_label_{default_fc_parameters}.npy"
        index_path = f"feature_{dataset_name}_index_{default_fc_parameters}.pkl"
        feature_extractor_path = f"feature_{dataset_name}_FE_{default_fc_parameters}.pkl"
        
        # Load datasets 
        dataloader = DataLoader(path)
        datasets = dataloader.get_dataset_names()
        if num_dimensions == 1: # univariate time series
            df = dataloader.load_df(datasets) 

            # Divide df
            labels = df.pop("label")
            x = df.to_numpy()[:, np.newaxis]
            index = df.index
        else: # multivariate time series
            data, label, index = dataloader.load_npy(datasets)
        


        # Setup the TSFresh feature extractor (too costly to use any other parameter)
        if default_fc_parameters == 'catch22':            
            fe = Catch22(
                catch24=True
            )
        elif default_fc_parameters == 'tsfresh_minimal':
            fe = TSFreshFeatureExtractor(
                default_fc_parameters=default_fc_parameters, 
                show_warnings=False, 
                n_jobs=-1
            )
        elif default_fc_parameters == 'tsfresh':
            efficient_set = EfficientFCParameters()
            minimal_set = MinimalFCParameters()            
            selected_features = list(efficient_set.keys())
            for feature in list(minimal_set.keys()) + ['mean_abs_change', 'mean_change', 'number_cwt_peaks', 'benford_correlation']:
                if feature not in selected_features:
                    selected_features.append(feature)

            features2use = copy.deepcopy(efficient_set)
            for feature in list(features2use.keys()):
                if feature not in selected_features:
                    del features2use[feature]

            fe = TSFreshFeatureExtractor(
                default_fc_parameters=features2use, 
                show_warnings=False, 
                n_jobs=-1
            )
        
        # Compute features
        if num_dimensions == 1:            	
            X_transformed = fe.fit_transform(x)

            # Create new dataframe
            X_transformed.index = index
            X_transformed = pd.merge(labels, X_transformed, left_index=True, right_index=True)
            
            # Save new features
            X_transformed.to_csv(os.path.join(path, new_name))
        else:
            X_transformed = fe.fit_transform(data)
            np.save(os.path.join(path, new_name), X_transformed)
            np.save(os.path.join(path, new_name_label), label)

            print('Done, saved to', os.path.join(path, new_name_label))
            with open(os.path.join(path, feature_extractor_path), 'wb') as output:
                pickle.dump(fe, output, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, index_path), 'wb') as output:
                pickle.dump(index, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_features',
        description='Transform a dataset of time series (of equal length) to tabular data with a specified feature extractor'
    )
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use')
    parser.add_argument('-f', '--feature', type=str, choices=['catch22', 'tsfresh', 'tsfresh_minimal'],
                                help=("a feature extractor to be used (choose from: catch22, tsfresh, tsfresh_minimal);\n"
                                      "Note: tsfresh_minimal may not extract useful features for multivariate time series"))
    
    args = parser.parse_args()
    generate_features(
        path=args.path, 
    )
