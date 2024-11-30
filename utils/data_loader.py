import os, glob

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
import copy

class DataLoader:
    """This class is used to read and load data from the benchmark.
    When the object is created the path to the benchmark directory
    should be given.
    """

    def __init__(self, data_path, rootcause_data_path=None):
        self.data_path = data_path
        self.rootcause_data_path = rootcause_data_path


    def get_dataset_names(self):
        '''Returns the names of existing datasets. 
        Careful, this function will not return any files in the given
        directory but only the names of the sub-directories
        as they are the datasets (not the timeseries).

        :return: list of datasets' names (list of strings)
        '''
        names = os.listdir(self.data_path)
        self.dataset_names = [x for x in names if os.path.isdir(os.path.join(self.data_path, x))]
        return [x for x in names if os.path.isdir(os.path.join(self.data_path, x))]
    
    def calc_data_characteristics(self):
        ret_max_vals, ret_min_vals = None, None
        
        for dataset_name in self.dataset_names:
            for fname in glob.glob(os.path.join(self.data_path, dataset_name, '*.out')):
                #print(fname)
                df = pd.read_csv(fname, header=None)
                max_vals = df.max().tolist()[:-1]
                min_vals = df.min().tolist()[:-1]

                if not ret_max_vals:
                    ret_max_vals = max_vals
                if not ret_min_vals:
                    ret_min_vals = min_vals

                for i in range(len(max_vals)):
                    if max_vals[i] > ret_max_vals[i]:
                        ret_max_vals[i] = max_vals[i]

                for i in range(len(min_vals)):
                    if min_vals[i] < ret_min_vals[i]:
                        ret_min_vals[i] = min_vals[i]
                    
        self.ret_max_vals = np.array(ret_max_vals)
        self.ret_min_vals = np.array(ret_min_vals)
        
    def calc_data_characteristics_std(self):
        ret_mean_vals, ret_std_vals = None, None
        std_list, mean_list = [], []
        
        for dataset_name in self.dataset_names:
            for fname in glob.glob(os.path.join(self.data_path, dataset_name, '*.out')):
                df = pd.read_csv(fname, header=None)
                mean_list.append(df.mean().tolist()[:-1])
                std_list.append(df.std().tolist()[:-1])
                    
        self.ret_mean_vals = np.array(mean_list).mean(axis=0)
        self.ret_std_vals = np.array(std_list).std(axis=0)

    def load(self, dataset):
        '''
        Loads the specified datasets

        :param dataset: list of datasets
        :return x: timeseries
        :return y: corresponding labels
        :return fnames: list of names of the timeseries loaded
        '''
        x = []
        y = []
        fnames = []
        y_column = []
        
        self.calc_data_characteristics_std()
        print(self.ret_mean_vals, self.ret_std_vals)


        if not isinstance(dataset, list):
            raise ValueError('only accepts list of str')

        pbar = tqdm(dataset)
        for name in pbar:
            pbar.set_description('Loading ' + name)
            for fname in glob.glob(os.path.join(self.data_path, name, '*.out')):
                curr_data = pd.read_csv(fname, header=None).to_numpy()
                
                if curr_data.ndim != 2:
                    raise ValueError('did not expect this shape of data: \'{}\', {}'.format(fname, curr_data.shape))

                curr_data[:, :-1] = (curr_data[:, :-1] - self.ret_mean_vals) / (self.ret_std_vals)
                x.append(np.sum(curr_data[:, :-1], axis=1))
                y.append(curr_data[:, -1])
                
                # Remove path from file name, keep dataset, time series name
                fname = '/'.join(fname.split('/')[-2:])        
                fnames.append(fname.replace(self.data_path, ''))

            # for root cause data
            for fname in glob.glob(os.path.join(self.rootcause_data_path, name, '*.out')):
                # print(fname)
                curr_data = pd.read_csv(fname, header=None).to_numpy()
                
                y_column.append(curr_data[:, :])
                    
        return x, y, fnames, y_column


    def load_df(self, dataset):
        '''
        Loads the time series of the given datasets and returns a dataframe

        :param dataset: list of datasets
        :return df: a single dataframe of all loaded time series
        '''
        df_list = []
        pbar = tqdm(dataset)

        if not isinstance(dataset, list):
            raise ValueError('only accepts list of str')

        for name in pbar:
            pbar.set_description(f'Loading {name}')
            
            for fname in glob.glob(os.path.join(self.data_path, name, '*.csv')):
                curr_df = pd.read_csv(fname, index_col=0)
                curr_index = [os.path.join(name, x) for x in list(curr_df.index)]
                curr_df.index = curr_index

                df_list.append(curr_df)
                
        df = pd.concat(df_list)

        return df
        
    def load_npy(self, dataset):
        '''
        Loads the time series of the given datasets and returns a dataframe

        :param dataset: list of datasets
        :return df: a single dataframe of all loaded time series
        '''
        np_data_list = []
        np_label_list = []
        np_index_list = []
        
        pbar = tqdm(dataset)

        if not isinstance(dataset, list):
            raise ValueError('only accepts list of str')

        for name in pbar:
            pbar.set_description(f'Loading {name}')
            
            for fname in glob.glob(os.path.join(self.data_path, name, '*.out.npy')):
                curr_np_data = np.load(fname)#, index_col=0)
                median_value = np.nanmedian(curr_np_data)
                curr_np_data[np.isnan(curr_np_data)] = median_value
                
                label_fname = copy.deepcopy(fname)
                label_fname = label_fname.replace('.out', '.out_label')
                curr_np_label = np.load(label_fname)                
                np_index_list += [fname.split('/')[-1]]*len(curr_np_label)

                np_data_list.append(curr_np_data)
                np_label_list.append(curr_np_label)
                
        np_data = np.concatenate(np_data_list)
        np_label = np.concatenate(np_label_list)

        return np_data, np_label, np_index_list


    def load_timeseries(self, timeseries):
        '''
        Loads specified timeseries

        :param fnames: list of file names
        :return x: timeseries
        :return y: corresponding labels
        :return fnames: list of names of the timeseries loaded
        '''
        x = []
        y = []
        fnames = []
        self.calc_data_characteristics()

        for fname in tqdm(timeseries, desc='Loading timeseries'):
            curr_data = pd.read_csv(os.path.join(self.data_path, fname), header=None).to_numpy()
            
            if curr_data.ndim != 2:
                raise ValueError('did not expect this shape of data: \'{}\', {}'.format(fname, curr_data.shape))

            curr_data[:, :-1] = (curr_data[:, :-1] - self.ret_min_vals) / (self.ret_max_vals - self.ret_min_vals)
            x.append(np.mean(curr_data[:, :-1]))
            y.append(curr_data[:, -1])
            fnames.append(fname)

        return x, y, fnames
