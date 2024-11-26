########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : scores_loader
#
########################################################################

import multiprocessing
import multiprocessing.pool as mpp

from utils.metrics_loader import MetricsLoader
from utils.data_loader import DataLoader
# from utils.config import *
from utils.metrics import generate_curve

from numba import jit
import os, glob
import time
from pathlib import Path
import warnings
from tqdm import tqdm
# import itertools, operator
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import copy

from sklearn.metrics import precision_recall_curve, auc
def calc_auc_pr(score, label):
    precision, recall, _ = precision_recall_curve(label, score)
    auc_pr = auc(recall, precision)
    return auc_pr
    
from sklearn.metrics import precision_recall_curve
def calc_reverse_fpr(score, label, th):
    scores = list()
    for th in np.linspace(0, 1, 1000):    
        y_pred = np.array(score)
        y_true = np.array(label)
        # Convert predicted probabilities to binary predictions based on the threshold
        y_pred_binary = (y_pred >= th).astype(int)
    
        # Calculate True Positives (TP) and False Negatives (FN)
        TP = np.sum((y_true == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true == 0) & (y_pred_binary == 0))
        FN = np.sum((y_true == 1) & (y_pred_binary == 0))
        FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    
        # Calculate False Negative Rate (FNR)
        fpr = FP / (FP + TN)

        scores.append(1-fpr)

    return np.mean(np.array(scores))
    
from sklearn.metrics import precision_recall_curve
def calc_tpr(score, label, th):
    scores = list()
    
    for th in np.linspace(0, 1,       1000):
        y_pred = np.array(score)
        y_true = np.array(label)
        
        # Convert predicted probabilities to binary predictions based on the threshold
        y_pred_binary = (y_pred >= th).astype(int)
    
        # Calculate True Positives (TP) and False Negatives (FN)
        TP = np.sum((y_true == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true == 0) & (y_pred_binary == 0))
        FN = np.sum((y_true == 1) & (y_pred_binary == 0))
        FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    
        # Calculate False Negative Rate (FNR)
        tpr = TP / (TP + FN)

        scores.append(tpr)
    return np.mean(np.array(scores))
    
def calc_acc(score, label, th):
    if sum(label) > 0 and sum(label) < len(label):
        acc = calc_auc_pr(score, label)
    elif sum(label) == len(label):
        acc = calc_tpr(score, label, th)
    elif sum(label) == 0:
        acc = calc_reverse_fpr(score, label, th)
    else:
        raise 'error'

    return acc

def safe_float_conversion(arr):
    try:
        return np.array(arr, dtype=float)
    except ValueError:
        return np.nan

class ScoresLoader:
    def __init__(self, scores_path):
        self.scores_path = scores_path

    def get_detector_names(self):
        '''Returns the names of the detectors

        :return: list of strings
        '''
        detectors = []

        for dataset in os.listdir(self.scores_path):
            curr_detectors = []
            for name in os.listdir(os.path.join(self.scores_path, dataset)):
                curr_detectors.append(name)
            if len(detectors) < 1:
                detectors = curr_detectors.copy()
            elif not detectors == curr_detectors:
                detectors = [x for x in detectors if x in curr_detectors]
                raise ValueError('detectors are not the same in this dataset \'{}\''.format(dataset))
        detectors.sort()

        return detectors
        
    def calc_scores_stat(self, detector_scores_paths):
        self.max_scores, self.min_scores = dict(), dict()
        detectors = self.get_detector_names()
        
        for detector in tqdm(detectors, desc='Calculating statistics of scores'):
            total_scores = list()
            
            for score_path in detector_scores_paths:
                score = pd.read_csv(score_path).to_numpy()
                score = np.vectorize(safe_float_conversion)(score)
                score = list(score)
                total_scores.extend(score)
                
            self.max_scores[detector] = max(total_scores)
            self.min_scores[detector] = min(total_scores)

    def load(self, file_names):
        '''
        Load the score for the specified files/timeseries. If a time series has no score for all 
        the detectors (e.g. the anomaly score has been computed for 10/12 detectors) the this
        time series is skipped. Its index is returned in the idx_failed for the user to remove 
        it from any other places if needed.

        :param dataset: list of files
        :return scores: the loaded scores
        :return idx_failed: list with indexes of not loaded time series
        '''
        detectors = self.get_detector_names()
        scores = []
        idx_ok = []
        idx_failed = []
        
        for i, name in enumerate(tqdm(file_names, desc='Loading scores for statistics')):
            name_split = name.split('/')[-2:]
            paths = [os.path.join(self.scores_path, name_split[0], detector, 'score', name_split[1]) for detector in detectors]
        
        self.calc_scores_stat(paths)
        

        for i, name in enumerate(tqdm(file_names, desc='Loading scores')):
            name_split = name.split('/')[-2:]
            paths = [os.path.join(self.scores_path, name_split[0], detector, 'score', name_split[1]) for detector in detectors]
            data = []
            try:
                for path in paths:
                    score = pd.read_csv(path).to_numpy()
                    score = np.vectorize(safe_float_conversion)(score)
                    detector_name = path.split('/')[-3]
                    score = (score - self.min_scores[detector_name]) / (self.max_scores[detector_name] - self.min_scores[detector_name])
                    data.append(score) 
                    # print('Loaded scores from', path)
            except Exception as e:
                idx_failed.append(i)
                print(idx_failed, paths, e, path)
                continue
                # pass

            if len(data) > 0:
                min_length_data = min([len(df) for df in data])
                data = [df[-min_length_data:] for df in data]
                scores.append(np.concatenate(data, axis=1))
                idx_ok.append(i)
                
        #print(idx_failed)
        idx_failed = [i for i in range(len(file_names)) if i not in idx_ok]
        # Delete ts which failed to load
        if len(idx_failed) > 0:
            #print('failed to load')
            for idx in sorted(idx_failed, reverse=True):
                pass

        return scores, idx_failed


    def write(self, file_names, detector, score, metric):
        '''Write some scores for a specific detector

        :param files_names: list of names (list of strings)
        :param detector: name of the detector (string)
        :param score: 1D arrays (as many as file names)
        '''
        for fname in file_names:
            dataset, ts_name = fname.split('/')[-2:]
            
            Path(os.path.join(self.scores_path, dataset, detector))\
            .mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.scores_path, dataset, detector, metric))\
            .mkdir(parents=True, exist_ok=True)

            # np.save(
            #     os.path.join(self.scores_path, dataset, detector, metric, ts_name), 
            #     score[:100])
            np.savetxt(
                os.path.join(self.scores_path, dataset, detector, metric, ts_name), 
                score, 
                fmt='%.2f', 
                delimiter='\n')
            

# -----------------------------------------------------
    
    #@jit
    def compute_metric(self, labels, scores, metric, verbose=1, n_jobs=1):
        '''Computes desired metric for all labels and scores pairs.

        :param labels: list of arrays each representing the labels of a timeseries/sample
        :param scores: list of 2D arrays representing the scores of each detector on a
                        timeseries/sample.
        :param metric: str, name of metric to produce
        :param verbose: to print or not to print info
        :return: metric values
        '''
        n_files = len(labels)
        results = []

        if len(labels) != len(scores):
            raise ValueError("length of labels and length of scores not the same")
        
        #if len(scores) == 0:
        #    return results

        if scores[0].ndim == 1 or scores[0].shape[-1] == 1:
            args = [x + (metric,) for x in list(zip(labels, scores))]
            pool = multiprocessing.Pool(n_jobs)
            
            results = []
            for label, score in zip(labels, scores):
                result = self.compute_single_sample(label, score, metric)
                #for result in  tqdm(pool.istarmap(self.compute_single_sample, args), total=len(args)):
                results.append(result)

            results = np.asarray([x.tolist() for x in results])
        else:
            for i, x, y in tqdm(zip(range(n_files), labels, scores), total=n_files, desc='Compute {}'.format(metric), disable=not verbose):
                results.append(self.compute_single_sample(x, y, metric))
            results = np.asarray(results)

        return results


    def compute_single_sample(self,    label, score, metric):
        '''Compute a metric for a single sample and multiple scores.

        :param label: 1D array of 0, 1 labels, (len_ts)
        :param score: 2D array, (len_ts, n_det)
        :param metric: string to which metric to compute
        :return: an array of values, one for each score
        '''
        if len(label) > len(score):
            label = label[-len(score):]
        else:
            score = score[-len(label):]
        if label.shape[0] != score.shape[0]:
            raise ValueError("label and score first dimension do not match. {} != {}".format(label.shape[0], score.shape[0]))

        if label.ndim > 1:
            raise ValueError("label has more dimensions than expected.")

        tick = time.process_time()
        result = np.apply_along_axis(self.compute_single_metric, 0, score, label, metric)
        
        '''
        # Evaluate the computed metrics
        fig, ax = plt.subplots(3, 4, figsize=(15, 10))
        best = np.argmax(result)
        x = np.linspace(0, label.shape[0], label.shape[0])
        for i, axis in enumerate(ax):
            for j, axs in enumerate(axis):
                if i*4 + j == best:
                    axs.patch.set_alpha(0.3)
                    axs.patch.set_facecolor('green')
                axs.title.set_text('{:.1f}%'.format(result[i*4 + j]*100))
                axs.set_xlim([0, x[-1]])
                axs.set_ylim([0, 1])
                axs.plot(label, label='label', color='k', linewidth=2)
                axs.plot(score[:, i*4 + j], label='score')
                axs.legend()
                axs.fill_between(x, label, score[:, i*4 + j])
                plt.tight_layout()
        plt.show()
        '''

        return result.T

    @jit
    def estimate_max_length(self, label):
        """Computes the maximum length of '1's in an anomaly label

        :param label: array of 1s and 0s
        :return  max_len: the length of the maximum continuous series of 1s in label 
        """
        max_len = 0
        counter = 0

        for val in label:
            if val:
                counter += 1
            else:
                max_len = counter if counter > max_len else max_len
                counter = 0
        
        return max_len if max_len > 10 else 10

    def compute_single_metric(self, score, label, metric):
        '''Compute a metric for a single sample and score.

        :param label: 1D array of 0, 1 labels
        :param score: 1D array same length as label
        :param metric: string to which metric to compute
        :return: a single value
        '''
        label = label[-len(score):]
        if label.shape != score.shape:
            raise ValueError("label and metric should have the same length.")
        
        metric = metric.lower()
        if metric == 'naive':
            combined = np.vstack((label, score)).T
            diff = np.abs(np.diff(combined))
            result = 1 - np.mean(diff)
        elif metric == 'Recommendation_ACC'.lower():
            y_pred = copy.deepcopy(score)
            y_true = copy.deepcopy(label)
            # remove the nan score appearing at the beginning of metrics
            first_non_nan =  np.where(~np.isnan(y_pred))[0][0]
            y_pred = y_pred[first_non_nan:]
            y_true = y_true[-y_pred.shape[0]:]

            # do calculation
            precision, recall, ths = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
            best_f1 = f1[np.argmax(f1)]
            best_th = ths[np.argmax(f1)]

            # Convert predicted probabilities to binary predictions based on the threshold
            y_pred_binary = (y_pred >= best_th).astype(int)

            # Calculate True Positives (TP) and False Negatives (FN)
            TP = np.sum((y_true == 1) & (y_pred_binary == 1))
            FN = np.sum((y_true == 1) & (y_pred_binary == 0))

            # Calculate False Negative Rate (FNR)
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

            result = best_f1 - fnr if best_f1 - fnr > 0 else 0
        elif metric == 'pr_auc':
            result = calc_acc(score, label, 0) 
        elif np.all(0 == label):
            fpr, tpr, thresholds = metrics.roc_curve(label, score)
            thresholds[0] = 1
            result = 1 - metrics.auc(thresholds, fpr)
        elif np.all(1 == label):
            fpr, tpr, thresholds = metrics.roc_curve(label, score)
            thresholds[0] = 1
            result = metrics.auc(thresholds, tpr)
        elif metric == 'fscore':
            thresholds = np.linspace(1, 0, 20)            
            fscores = [self.compute_fscore(x, score, label) for x in thresholds]
            result = metrics.auc(thresholds, fscores)
        elif metric == 'auc_roc':
            fpr, tpr, _ = metrics.roc_curve(label, score)
            result = metrics.auc(fpr, tpr)
        elif metric == 'auc_pr':
            precision, recall, _ = metrics.precision_recall_curve(label, score)
            result = metrics.auc(recall, precision)
        elif metric == 'vus_roc':
            result, _ = generate_curve(label, score, 2*10)
        elif metric == 'vus_pr':
            _, result = generate_curve(label, score, 2*10)
        elif metric == 'vus':
            result = generate_curve(label, score, 2*10)        
        else:
            raise ValueError("can't recognize metric requested")

        # result = round(result, 5)
        # if result > 1 or result < 0:
        #     print('>> Pottentially faulty metrics result {}'.format(result))

        return result

    def compute_fscore(self, threshold, score, label):
        score = score > threshold
        return metrics.f1_score(label, score)



def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
