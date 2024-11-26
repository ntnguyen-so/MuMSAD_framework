########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : eval_deep_model
#
########################################################################

import argparse
import re
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader

from utils.config import *
from utils.train_deep_model_utils import json_file
from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator


def eval_deep_model(
    data_path, 
    model_name, 
    model_path=None, 
    model_parameters_file=None, 
    path_save=None, 
    fnames=None,
    read_from_file=None,
    model=None
):
    """Evaluate a deep learning model on time series data and predict the time series.

    :param data_path: Path to the time series data.
    :param model_name: Name of the model to be evaluated.
    :param model_path: Path to the pretrained model weights.
    :param model_parameters_file: Path to the JSON file containing model parameters.
    :param path_save: Path to save the evaluation results.
    :param fnames: List of file names (time series) to evaluate.
    :param read_from_file: Path to the file containing split information.
    :param model: Preloaded model instance.

    Returns:
    DataFrame: A DataFrame containing the predicted time series.
    """
    window_size = int(re.search(r'\d+', str(data_path)).group())
    batch_size = 128
    
    assert(
        (model is not None) or \
        (model_path is not None and model_parameters_file is not None)
    ), "You should provide the model or the path to the model, not both"

    assert(
        not (fnames is not None and read_from_file is not None)
    ), "You should provide either the fnames or the path to the specific splits, not both"

    # Load the model only if not provided
    if model == None:
        # Read models parameters
        model_parameters = json_file(model_parameters_file)
        
        # Change input size according to input
        if 'original_length' in model_parameters:
            model_parameters['original_length'] = window_size
        if 'timeseries_size' in model_parameters:
            model_parameters['timeseries_size'] = window_size
        
        # Load model
        model = deep_models[model_name](**model_parameters)

        # Check if model_path is specific file or dir
        if os.path.isdir(model_path):
            # Check the number of files in the directory
            files = os.listdir(model_path)
            if len(files) == 1:
                # Load the single file from the directory
                model_path = os.path.join(model_path, files[0])
            else:
                raise ValueError("Multiple files found in the 'model_path' directory. Please provide a single file or specify the file directly.")

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to('cuda')
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()

    # Load the splits
    if read_from_file is not None:
        _, val_set, test_set = create_splits(
            data_path,
            read_from_file=read_from_file,
        )
        fnames = test_set if len(test_set) > 0 else val_set
    else:
        # Read data (single csv file or directory with csvs)
        if '.csv' == data_path[-len('.csv'):]:
            tmp_fnames = [data_path.split('/')[-1]]
            data_path = data_path.split('/')[:-1]
            data_path = '/'.join(data_path)
            data_path = data_path.replace('.csv', '_data.npy')
        else:
            tmp_fnames = read_files(data_path)

        # Keep specific time series if fnames is given
        if fnames is not None:
            fnames_len = len(fnames)
            fnames = [x for x in tmp_fnames if x in fnames]
            if len(fnames) != fnames_len:
                raise ValueError("The data path does not include the time series in fnames")
        else:
            fnames = tmp_fnames

    # Uncomment for testing
    # fnames = fnames[:10]

    # Specify classifier name for saving results
    if model_path is not None:
        if "sit_conv" in model_path:
            model_name = "sit_conv"
        elif "sit_linear" in model_path:
            model_name = "sit_linear"
        elif "sit_stem_relu" in model_path:
            model_name = "sit_stem_relu"
        elif "sit_stem" in model_path:
            model_name = "sit_stem"
    classifier_name = f"{model_name}_{window_size}"
    if read_from_file is not None and "unsupervised" in read_from_file:
        classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
    elif "unsupervised" in path_save:
        extra = model_path.split('/')[-2].replace(classifier_name, "")
        classifier_name += extra
    
    print('before evaluating', fnames, data_path)
    # Evaluate model
    evaluator = Evaluator()
    results = evaluator.predict(
        model=model,
        fnames=fnames,
        data_path=data_path,
        batch_size=batch_size,
        deep_model=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    results = results.sort_index()
    results.columns = [f"{classifier_name}_{x}" for x in results.columns.values]
    
    # Print results
    print(results)
    counter = dict(Counter(results[f"{classifier_name}_class"]))
    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
    print(counter)
    
    # Save the results
    if path_save is not None:
        file_name = os.path.join(path_save, f"{classifier_name}_preds.csv")
        results.to_csv(file_name)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Evaluate deep learning models',
        description='Evaluate all deep learning architectures on a single or multiple time series \
            and save the results'
    )
    
    parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
    parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
    parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
    parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
    parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")
    parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)

    args = parser.parse_args()
    eval_deep_model(
        data_path=args.data, 
        model_name=args.model, 
        model_path=args.model_path, 
        model_parameters_file=args.params,
        path_save=args.path_save,
        read_from_file=args.file
    )