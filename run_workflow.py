import pandas as pd
import numpy as np

import argparse
import re
import os
from collections import Counter

import torch

from utils.config import *
from utils.train_deep_model_utils import json_file
from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator, load_classifier
from utils.config import detector_names
import os
import math
import subprocess
from sklearn.decomposition import PCA
import time
import traceback
import random
# from utils.evaluator import Evaluator, load_classifier
import pickle
import sys
import os.path
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"]=""


def split_ts(data, window_size):
    '''Split a timeserie into windows according to window_size.
    If the timeserie can not be divided exactly by the window_size
    then the first window will overlap the second.

    :param data: the timeserie to be segmented
    :param window_size: the size of the windows
    :return data_split: an 2D array of the segmented time series
    '''

    # Compute the modulo
    modulo = data.shape[0] % window_size

    # Compute the number of windows
    k = data[modulo:].shape[0] / window_size
    assert(math.ceil(k) == k)

    # Split the timeserie
    data_split = np.split(data[modulo:], k)
    if modulo != 0:
        data_split.insert(0, list(data[:window_size]))
    data_split = np.asarray(data_split)

    return data_split

def z_normalization(ts, decimals=5):
    ts = (ts - np.mean(ts)) / np.std(ts)
    ts = np.around(ts, decimals=decimals)

    # Test normalization
    assert(
        np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
    ), "After normalization it should: mean == 0 and std == 1"

    return ts

def select_AD_model(model_path, model_name, model_parameters_file, window_size, ts_data, top_k, batch_size, is_deep=False):
    """Evaluate a deep learning model on time series data and predict the time series.

    :param model: Preloaded model instance.
    :param model_path: Path to the pretrained model weights.
    :param model_parameters_file: Path to the JSON file containing model parameters.
    :param window_size: the size of the window timeseries will be split to (must align with the model)
    :param model: Preloaded model instance.
    :param ts_data: Time series data
    :param top_k: k AD models yielding the highest acc
    Returns:
    """
    if is_deep:
        # load model parameters
        model_parameters = json_file(model_parameters_file)
        if 'original_length' in model_parameters:
            model_parameters['original_length'] = window_size * num_dimensions
        if 'timeseries_size' in model_parameters:
            model_parameters['timeseries_size'] = window_size * num_dimensions
        
        # print('model_name', model_name)
        # load model
        model = deep_models[model_name](**model_parameters)    
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to('cuda')
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            model.to('cpu')
        
        preds = model(sequence.float()).argmax(dim=1).tolist()
    else:
        model = load_classifier(model_path)
        preds = model.predict(sequence)    
        preds = list(preds)
        preds = [int(x) for x in preds]
    
    print(preds)
    # Majority voting
    counter = Counter(preds)
    most_k_voted = counter.most_common(top_k)
    top_k_detectors = [x[0] for x in most_k_voted]
    suggested_detectors = [detector_names[d] for d in top_k_detectors]
    
    return top_k_detectors


if __name__ == "__main__":
    data_types = ['not_normalized', 'normalized'] # ['normalized'] #
    feature_models = ['custom0.25', 'catch22']
    
    working_on_deep = True
    
    test_data_files = pd.read_csv('./reproducibility_guide/train_val_split.csv')
    test_data_files = test_data_files.iloc[2, :].tolist()[1:]
    test_data_files = [str(x) for x in test_data_files if 'OBSEA' in str(x)]
    
    for data_type in data_types:
        # MSAD_model_base_paths = ['results/weights/weights_custom0.25_' + data_type + '/']
        # votes_folders = ['results/votes/votes_custom_' + data_type + '/']
        deep_models_shortform = ['convnet', 'inception', 'resnet', 'sit']
        # fe_base = 'tsfresh0.25'
        np_mean = np.array([37.79987549, 4.92507308, 18.03925078])
        np_std = np.array([0.14562043, 0.03604426, 0.30548436])
        if len(sys.argv) == 1:
            i = 0            
        else:
            i = int(sys.argv[1])
            
        for feature_model in feature_models:
            if working_on_deep:
                MSAD_model_base_path =   'results/weights/' #
                votes_folder =  'results/votes/votes_deep/' #
            else:
                MSAD_model_base_path =   'results/weights_feature/weights_' + feature_model + '_' + data_type + '/' # 'results/weights/' #            
                votes_folder =  'results/votes/votes_' + feature_model + '_' + data_type + '/' # 'results/votes/votes_deep_2/' #
            
            fe_base = feature_model
            # if 'tsfresh0.25' in MSAD_model_base_path:
                # fe_base = 'custom0.25'
            jump = 300
            for model in sorted(os.listdir(MSAD_model_base_path), reverse=True)[i*jump:(i+1)*jump]:
                print('model:', model)
                is_deep_model = False
                for deep_model_name in deep_models_shortform:
                    if deep_model_name in model:
                        is_deep_model = True
                        
                # if is_deep_model:
                    # continue
                for saved_model in sorted(os.listdir(MSAD_model_base_path + model)):
                    if "model" not in saved_model:
                        continue
                    print(MSAD_model_base_path, model, saved_model, is_deep_model)
                    try:
                        # model_path='./results/weights/sit_stem_original_32/model_06072024_134554'
                        model_path = MSAD_model_base_path + model + '/' + saved_model
                        model_name=model_path.split('/')[-2].split('_')[0]
                        model_parameters_file='./models/configuration/' + "_".join(model_path.split('/')[-2].split('_')[:-1]) + '.json' # sit_stem_original.json'#sit_linear_patch.json'
                        # print(model_parameters_file)
                        path_to_data = './data/OBSEA/data/OBSEA/'
                        path2save = votes_folder + '/' + model_path.split('/')[-2] + '-' + saved_model + '-' + '.csv'
                        if os.path.exists(path2save):
                            print('exists ', path2save)
                            continue
                        print(MSAD_model_base_path, votes_folder)
                        num_k = 4
                        processes_running = list()
                        max_process = 10
                        # metadata_df = read_metadata()
                        curr_path = os.getcwd()
                        data_files = os.listdir(path_to_data)
                        files_checked = list()
                        files_choice = list()
                        #random.shuffle(data_files)
                        data_files = data_files[::1]
                        
                        window_size = int(model_path.split('/')[-2].split('_')[-1])
                        # if window_size != 16:
                            # continue
                                
                        if not is_deep_model:
                            fe_name = 'TSFRESH_OBSEA_' + str(window_size) + '_FE_' + fe_base + '.pkl'
                            fe_path = './data/ood_example/' + data_type + '/OBSEA_' + str(window_size) + '/' + fe_name
                            with open(f'{fe_path}', 'rb') as input:
                                fe = pickle.load(input)
                        
                        
                        for data_file in data_files:
                            # try:
                                print(data_file, test_data_files)
                                # if ("2021" not in data_file and "2021" not in data_file) and "_data" not in data_file:
                                    # continue
                                if not any(data_file in file for file in test_data_files):
                                    continue
                                if "unsupervised" in data_file:
                                    continue
                                
                                uploaded_ts = path_to_data + data_file
                                
                                
                                ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
                                ts_data = ts_data_raw[:, :-1].astype(float)
                                if data_type == "normalized":
                                    ts_data = (ts_data - np_mean) / (np_std)
                                sequence = ts_data
                                # sequence = z_normalization(ts_data, decimals=7)
                                # print(sequence)
                                
                                # Split timeseries and load to cpu
                                sequence = split_ts(sequence, window_size)#[:, np.newaxis]
                                if is_deep_model:
                                    sequence = np.swapaxes(sequence, 1, 2)
                                if not is_deep_model:
                                    sequence = fe.transform(sequence)
                                    sequence = np.where(np.isnan(sequence), 0, sequence)
                                    
                                if data_type == "not_normalized":
                                    scaler_path = MSAD_model_base_path + model + '/' +  saved_model.replace('model', 'scaler')
                                    scaler_ = load_classifier(scaler_path, scaler=True)
                                    sequence = scaler_.transform(sequence)
                                    # print('scaled', sequence)
                                print(sequence.shape)
                                
                                if is_deep_model:
                                    if torch.cuda.is_available():
                                        sequence = torch.from_numpy(sequence).to('cuda')
                                    else:
                                        sequence = torch.from_numpy(sequence).to('cpu')
                                        
                                pred_detector = select_AD_model(model_name=model_name, 
                                                                model_path=model_path, 
                                                                model_parameters_file=model_parameters_file, 
                                                                window_size=window_size, 
                                                                ts_data=ts_data, 
                                                                top_k=num_k, 
                                                                batch_size=64,
                                                                is_deep=is_deep_model)
                                                                
                                print(pred_detector)
                                files_checked.append('OBSEA/' + data_file)
                                files_choice.append(pred_detector[0])
                                # decisions['OBSEA/' + data_file] = pred_detector[0]
                                
                                # break
                            # except:# Exception e:
                                # pass
                            
                            
                        decisions = {'files': files_checked, 'choice': files_choice}
                        df = pd.DataFrame.from_dict(decisions)#, columns=['name', 'choice'])
                        # print(df)
                        df.to_csv(path2save)    
                        print('save ' + path2save)
                    except:
                        traceback.print_exc()
                        pass
