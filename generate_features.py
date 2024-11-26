########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : generate_features
#
########################################################################

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

random.seed(42)


def generate_features(path):
    """Given a dataset it computes the TSFresh automatically extracted 
    features and saves the new dataset (which does not anymore contain
    time series but tabular data) into one .csv in the folder of the
    original dataset

    :param path: path to the dataset to be converted
    """
    for default_fc_parameters in ['tsfresh', 'catch22']:# ['minimal', 'efficient']:
        window_size = int(re.search(r'\d+', path).group())

        # Create name of new dataset
        dataset_name = [x for x in path.split('/') if str(window_size) in x][0]
        # new_name = f"TSFRESH_{dataset_name}.csv"
        new_name = f"TSFRESH_{dataset_name}_{default_fc_parameters}.npy"
        new_name_label = f"TSFRESH_{dataset_name}_label_{default_fc_parameters}.npy"
        index_path = f"TSFRESH_{dataset_name}_index_{default_fc_parameters}.pkl"
        feature_extractor_path = f"TSFRESH_{dataset_name}_FE_{default_fc_parameters}.pkl"
        print(os.path.join(path, new_name))

        # Load datasets 
        dataloader = DataLoader(path)
        datasets = dataloader.get_dataset_names()
        # df = dataloader.load_df(datasets) 
        data, label, index = dataloader.load_npy(datasets)
        print(data.shape, label.shape, len(index))
        
        # Divide df
        # labels = df.pop("label")
        # x = df.to_numpy()[:, np.newaxis]
        # index = df.index

        # Setup the TSFresh feature extractor (too costly to use any other parameter)
        if default_fc_parameters == 'minimal':
            fe = TSFreshFeatureExtractor(
                default_fc_parameters=default_fc_parameters, 
                show_warnings=False, 
                n_jobs=-1
            )
        elif default_fc_parameters == 'catch22':            
            fe = Catch22(
                catch24=True
            )
        elif default_fc_parameters == 'rocket':            
            fe = Rocket(
                n_jobs=-1
            )
        elif default_fc_parameters == 'tsfresh':
            for percentage in [.25]:
            
                efficient_set = EfficientFCParameters()
                minimal_set = MinimalFCParameters()            
                selected_features = list(efficient_set.keys())
                selected_features = random.sample(selected_features, int(len(selected_features)*percentage))
                for feature in list(minimal_set.keys()) + ['mean_abs_change', 'mean_change', 'number_cwt_peaks']:
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
                # fe = extract_features(data, default_fc_parameters=features2use)
                # print(fe)
                
                new_name = new_name.replace('tsfresh', 'tsfresh'+str(percentage))
                new_name_label = new_name_label.replace('tsfresh', 'tsfresh'+str(percentage))
                index_path = index_path.replace('tsfresh', 'tsfresh'+str(percentage))
                feature_extractor_path = feature_extractor_path.replace('tsfresh', 'tsfresh'+str(percentage))
        
        # Compute features
        # X_transformed = fe.fit_transform(x)
        X_transformed = fe.fit_transform(data)
        np.save(os.path.join(path, new_name), X_transformed)
        np.save(os.path.join(path, new_name_label), label)

        print('Done, saved to', os.path.join(path, new_name_label))
        with open(os.path.join(path, feature_extractor_path), 'wb') as output:
            pickle.dump(fe, output, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, index_path), 'wb') as output:
            pickle.dump(index, output, pickle.HIGHEST_PROTOCOL)
        # print(X_transformed.shape)

        # Create new dataframe
        # X_transformed.index = index
        # X_transformed = pd.merge(labels, X_transformed, left_index=True, right_index=True)
        
        # Save new features
        # X_transformed.to_csv(os.path.join(path, new_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_features',
        description='Transform a dataset of time series (of equal length) to tabular data\
        with TSFresh'
    )
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use')
    
    args = parser.parse_args()
    generate_features(
        path=args.path, 
    )
