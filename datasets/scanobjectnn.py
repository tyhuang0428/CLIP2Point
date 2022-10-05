import os
import h5py
from torch.utils.data import Dataset
import random
import torch
import numpy as np

from datasets.utils import pc_normalize


class ScanObjectNN(Dataset):
    def __init__(self, partition='test', few_num=0, num_points=1024):
        assert partition in ('test', 'training')
        self._load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition
        self.few_num = few_num
        self._preprocess()        

    def __getitem__(self, index):
        point, label = self.points[index], self.labels[index]
        point = pc_normalize(point)
        if self.partition == 'train':
            pt_idxs = np.arange(point.shape[0])
            np.random.shuffle(pt_idxs)
            point = point[pt_idxs]
            return point[: self.num_points], label
        return point, label

    def _load_ScanObjectNN(self, partition):
        BASE_DIR = '/data1/hty/h5_files/'
        DATA_DIR = os.path.join(BASE_DIR, 'main_split')
        h5_name = os.path.join(DATA_DIR, f'{partition}_objectdataset.h5')
        f = h5py.File(h5_name)
        self.points = torch.from_numpy(f['data'][:].astype('float32')) 
        self.labels = torch.from_numpy(f['label'][:].astype('int64')) 

    def _preprocess(self):
        if self.partition == 'training' and self.few_num > 0:
            num_dict = {i: 0 for i in range(15)}
            self.few_points = []
            self.few_labels = []
            random_list = [k for k in range(len(self.labels))]
            random.shuffle(random_list)
            for i in random_list:
                label = self.labels[i].item()
                if num_dict[label] == self.few_num:
                    continue
                self.few_points.append(self.points[i])
                self.few_labels.append(self.labels[i])
                num_dict[label] += 1
        else:
            self.few_points = self.points
            self.few_labels = self.labels

    def __len__(self):
        return len(self.few_labels)
