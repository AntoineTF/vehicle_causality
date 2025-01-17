from .base_dataset import BaseDataset
import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool
import h5py
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from unitraj.datasets import common_utils
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
from unitraj.datasets.types import object_type, polyline_type
from unitraj.utils.visualization import check_loaded_data
from functools import lru_cache

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)

class FilteredDataset(BaseDataset):
    def load_data(self):
        self.data_loaded = {}
        if self.is_validation:
            print('Loading factuals for the validation data...')
        else:
            print('Loading factuals for the training data ...')

        for cnt, data_path in enumerate(self.data_path):
            phase, dataset_name = data_path.split('/')[-2], data_path.split('/')[-1]
            self.cache_path = os.path.join(self.config['cache_path'], dataset_name, phase)

            data_usage_this_dataset = self.config['max_data_num'][cnt]
            self.starting_frame = self.config['starting_frame'][cnt]
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
                    print('Warning: cache path {} already exists, skip '.format(self.cache_path))
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:
                    _, summary_list, mapping = read_dataset_summary(data_path)

                    print("We selected",len(summary_list),"files")
                    
                    # Filter out files containing 'child' in their names
                    summary_list = [file for file in summary_list if 'child' not in file]
                    # Filter the mapping to only include keys that are in the filtered summary_list
                    mapping = {key: value for key, value in mapping.items() if key in summary_list}
                    # print("Document we loaded: ", summary_list)
                    print("Be we are going to load : ",len(summary_list),"files, we excluded the counter factuals")
                    print(f"Total files in summary_list for {data_path}: {len(summary_list)}")

                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    os.makedirs(self.cache_path, exist_ok=True)
                    process_num = os.cpu_count() // 2
                    print('Using {} processes to load data...'.format(process_num))

                    data_splits = np.array_split(summary_list, process_num)

                    data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
                    
                    # Print details about the data splits
                    print(f"Number of splits: {len(data_splits)}")
                    for idx, split in enumerate(data_splits):
                        print(f"Split {idx + 1}:")
                        print(f"  Data Path: {split[0]}")
                        print(f"  Mapping: {type(split[1])} with length {len(split[1]) if hasattr(split[1], '__len__') else 'N/A'}")
                        print(f"  Data Chunk: {type(split[2])} with length {len(split[2])}")
                        print(f"  Dataset Name: {split[3]}")
                        break
                    
                    os.makedirs('tmp', exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp', '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)
                    print("We opened ",i,"files, the process_num is: ", process_num)

                    with Pool(processes=process_num) as pool:
                        results = pool.map(self.process_data_chunk, list(range(process_num)))

                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    print(f"Final file_list contains {len(file_list)} files: {list(file_list.keys())[:10]}")
                    
                    unique_files = len(set(file_list.keys()))
                    print(f"Unique files in file_list: {unique_files}")
                    
                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)

                    data_list = list(file_list.items())
                    np.random.shuffle(data_list)
                    if not self.is_validation:
                        file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list), data_path))
            self.data_loaded.update(file_list)

            if self.config['store_data_in_memory']:
                print('Loading data into memory...')
                for data_path in file_list.keys():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                print('Loaded {} data into memory'.format(len(self.data_loaded_memory)))

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')
