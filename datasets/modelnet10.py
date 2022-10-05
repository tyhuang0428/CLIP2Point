import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils import pc_normalize, offread_uniformed

cats = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9}


class ModelNet10(Dataset):
    def __init__(self, partition='test', few_num=0, num_points=1024):
        assert partition in ('test', 'train')
        super().__init__()
        self.partition = partition
        self.few_num = few_num
        self.num_points = num_points
        self._load_data()
        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()

    def _load_data(self):
        DATA_DIR = '/data/ModelNet10'
        self.paths = []
        self.labels = []
        for cat in os.listdir(DATA_DIR):
            cat_path = os.path.join(DATA_DIR, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(cats[cat])

    def _few(self):
        num_dict = {i: 0 for i in range(10)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i].item()
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels
    
    def __getitem__(self, index):
        point = torch.from_numpy(offread_uniformed(self.paths[index], 1024)).to(torch.float32)
        label = self.labels[index]
        point = pc_normalize(point)
        if self.partition == 'train':
            pt_idxs = np.arange(point.shape[0])
            np.random.shuffle(pt_idxs)
            point = point[pt_idxs]
            return point[: self.num_points], label
        return point, label
    
    def __len__(self):
        return len(self.labels)
