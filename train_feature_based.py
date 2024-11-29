import argparse
import os
from time import perf_counter
import re
from collections import Counter
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from eval_feature_based import eval_feature_based
from utils.evaluator import save_classifier
from utils.config import *
import copy
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler

names = {
        "knn": "Nearest Neighbors",
        "svc_linear": "Linear SVM",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "mlp": "Neural Net",
        "ada_boost": "AdaBoost",
        "bayes": "Naive Bayes",
        "qda": "QDA",
}

classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "svc_linear": LinearSVC(C=0.025, verbose=True),
        "decision_tree": DecisionTreeClassifier(max_depth=5),
        "random_forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1, verbose=True),
        "mlp": MLPClassifier(alpha=1, max_iter=1000, verbose=True),
        "ada_boost": AdaBoostClassifier(),
        "bayes": GaussianNB(),
        "qda": QuadraticDiscriminantAnalysis(),
}

def retrieve_indices(specified_set, data_index):
    result = []
    for _, fname in enumerate(specified_set):
        indices = [ind for ind, data_index_fname in enumerate(data_index) if data_index_fname == fname]
        result.extend(indices)
    return result


def train_feature_based(data_path, classifier_name, split_per=0.7, seed=None, read_from_file=None, eval_model=False, path_save=None):
    # Set up
    window_size = int(re.search(r'\d+', data_path).group())
    training_stats = {}
    original_dataset = data_path.split('/')[:-1]
    original_dataset = '/'.join(original_dataset)
    
    is_not_normalized = False
    if "not_normalized" in data_path:
        is_not_normalized = True
    print('is_not_normalized:', is_not_normalized)
    
    # Load the splits
    train_set, val_set, test_set = create_splits(
        original_dataset,
        split_per=split_per,
        seed=seed,
        read_from_file=read_from_file,
    )
    
    if False:
        train_indexes = [x[:-4] for x in train_set]
        val_indexes = [x[:-4] for x in val_set]
        test_indexes = [x[:-4] for x in test_set]

        # Read tabular data
        data = pd.read_csv(data_path, index_col=0)

        # Reindex them
        data_index = list(data.index)
        new_index = [tuple(x.rsplit('.', 1)) for x in data_index]
        new_index = pd.MultiIndex.from_tuples(new_index, names=["name", "n_window"])
        data.index = new_index
        
        # Create subsets
        training_data = data.loc[data.index.get_level_values("name").isin(train_indexes)]
        val_data = data.loc[data.index.get_level_values("name").isin(val_indexes)]
        test_data = data.loc[data.index.get_level_values("name").isin(test_indexes)]
        
        # Split data from labels
        y_train, X_train = training_data['label'], training_data.drop('label', 1)
        y_val, X_val = val_data['label'], val_data.drop('label', 1)
        y_test, X_test = test_data['label'], test_data.drop('label', 1)
    
    # load data    
    data = np.load(data_path)
    data = np.where(np.isnan(data), 0, data)# np.nan_to_num(data)
    if is_not_normalized: # use only when not normalized the data    
        data_scaler = StandardScaler()
        data = data_scaler.fit_transform(data)

    label_path = copy.deepcopy(data_path)
    label_file_name = label_path.split('/')[-1].replace(str(window_size), str(window_size) + '_label')
    label_path = '/'.join(label_path.split('/')[:-1]) + '/' + label_file_name    
    label = np.load(label_path)

    index_path = copy.deepcopy(data_path)
    index_file_name = index_path.split('/')[-1].replace(str(window_size), str(window_size) + '_index')
    index_file_name = index_file_name.replace('.npy', '.pkl')
    index_path = '/'.join(index_path.split('/')[:-1]) + '/' + index_file_name    
    with open(index_path, 'rb') as index_file:
        indices = pickle.load(index_file)

    if read_from_file is not None:
        train_set = [x for x in train_set if '.csv' in x]
        train_set = [x.split('/')[1].replace('.csv', '.npy') for x in train_set]

        val_set = [x for x in val_set if '.csv' in x]
        val_set = [x.split('/')[1].replace('.csv', '.npy') for x in val_set]
        
        test_set = [x for x in test_set if '.csv' in x]
        test_set = [x.split('/')[1].replace('.csv', '.npy') for x in test_set]

        train_indices = retrieve_indices(train_set, indices)
        val_indices = retrieve_indices(val_set, indices)
    
    data_len = len(data)
    X_train, X_val = data[train_indices], data[val_indices] # data[:int(data_len * split_per)], data[int(data_len * split_per):]
    y_train, y_val = label[train_indices], label[val_indices] # label[:int(data_len * split_per)], label[int(data_len * split_per):]

    # Select the classifier
    classifier = classifiers[classifier_name]
    clf_name = classifier_name

    # For svc_linear use only a random subset of the dataset to train
    if 'svc' in classifier_name and len(y_train) > 200000:
        rand_ind = np.random.randint(low=0, high=len(y_train), size=200000)
        X_train = X_train[rand_ind]
        y_train = y_train[rand_ind]

    # Fit the classifier
    print(f'----------------------------------\nTraining {names[classifier_name]}...')
    tic = perf_counter()
    classifier.fit(X_train, y_train)
    toc = perf_counter()

    # Print training time
    training_stats["training_time"] = toc - tic
    print(f"training time: {training_stats['training_time']:.3f} secs")
    
    # Print valid accuracy and inference time
    tic = perf_counter()
    classifier_score = classifier.score(X_val, y_val)
    toc = perf_counter()
    training_stats["val_acc"] = classifier_score
    training_stats["avg_inf_time"] = ((toc-tic)/X_val.shape[0]) * 1000
    print(f"valid accuracy: {training_stats['val_acc']:.3%}")
    print(f"inference time: {training_stats['avg_inf_time']:.3} ms")

    # Save training stats
    classifier_name = f"{clf_name}_{window_size}"
    if read_from_file is not None and "unsupervised" in read_from_file:
        classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    df = pd.DataFrame.from_dict(training_stats, columns=["training_stats"], orient="index")
    df.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

    # Save pipeline
    saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
    saved_model_path = save_classifier(classifier, saving_dir, fname=None)
    if is_not_normalized:
        save_classifier(data_scaler, saving_dir, fname=None, scaler=True)

    # Evaluate on test set or val set
    if eval_model:
        eval_set = test_indexes if len(test_indexes) > 0 else val_indexes
        eval_feature_based(
            data_path=data_path, 
            model_name=classifier_name,
            model_path=saved_model_path,
            path_save=path_save_results,
            fnames=eval_set,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_feature_based',
        description='Script for training the traditional classifiers',
    )
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
    parser.add_argument('-c', '--classifier', type=str, help='classifier to run', required=True)
    parser.add_argument('-sp', '--split_per', type=float, help='split percentage for train and val sets', default=0.7)
    parser.add_argument('-s', '--seed', type=int, help='seed for splitting train, val sets (use small number)', default=None)
    parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
    parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
    parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights")

    args = parser.parse_args()

    # Option to all classifiers
    if args.classifier == 'all':
        clf_list = list(classifiers.keys())
    else:
        clf_list = [args.classifier]

    for classifier in clf_list:
        train_feature_based(
            data_path=args.path,
            classifier_name=classifier,
            split_per=args.split_per, 
            seed=args.seed,
            read_from_file=args.file,
            eval_model=args.eval_true,
            path_save=args.path_save
        )
