import os
import torch
import numpy as np
import torch.utils.data as data
import h5py
from typing import Tuple
import collections
from pytorch3d.io import load_obj
import random
from torchvision.transforms import Normalize, ToTensor
from PIL import Image

from render.render import Renderer


cat_labels = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5, '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11, '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16, 
'02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33, 
'03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50, 
'04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]


def torch_center_and_normalize(points,p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds 
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points


class ShapeNet(data.Dataset):
    def __init__(self, partition='train', whole=False, num_points=1024):
        assert partition in ['train', 'test']
        self.data_root = './data/ShapeNet55/ShapeNet-55'
        self.pc_path = './data/ShapeNet55/shapenet_pc'
        self.subset = partition
        self.npoints = 8192
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = num_points
        self.whole = whole

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            lines = test_lines + lines
        self.file_list = []
        check_list = ['03001627-udf068a6b', '03001627-u6028f63e', '03001627-uca24feec', '04379243-', '02747177-', '03001627-u481ebf18', '03001627-u45c7b89f', '03001627-ub5d972a1', '03001627-u1e22cc04', '03001627-ue639c33f']
        
        # flag = False
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]

            if taxonomy_id + '-' + model_id not in check_list:
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })

        self.permutation = np.arange(self.npoints)

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(model_path, create_texture_atlas=True, texture_wrap='clamp')
        textures = aux.texture_atlas    
        return verts, faces.verts_idx, textures

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        points = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        # points = self.random_sample(points, self.sample_points_num)
        points = self.pc_norm(points)
        points = torch.from_numpy(points).float()

        verts, faces, textures = self._load_mesh(os.path.join('/data/ShapeNetCore.v2', sample['taxonomy_id'], sample['model_id'], 'models', 'model_normalized.obj'))
        verts = torch_center_and_normalize(verts.to(torch.float), '2.0')
        mesh = dict()
        mesh["verts"] = verts
        mesh["faces"] = faces
        mesh["textures"] = textures
        # label = cat_labels[sample['taxonomy_id']]
        label = sample['taxonomy_id'] + '_' + sample['model_id']
        # return points, mesh, sample['taxonomy_id'], sample['model_id']
        return points, mesh, label

    def __len__(self):
        return len(self.file_list)


class ShapeNetDebug(ShapeNet):
    def __init__(self, partition='train', whole=False):
        super().__init__(partition, whole)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        return sample['taxonomy_id'] + '_' + sample['model_id']


class ShapeNetRender(ShapeNet):
    def __init__(self, partition='train', whole=False, num_points=1024):
        super().__init__(partition, whole, num_points)
        self.partition = partition
        self.views_dist = torch.ones((10), dtype=torch.float, requires_grad=False)
        self.views_elev = torch.asarray((0, 90, 180, 270, 225, 225, 315, 315, 0, 0), dtype=torch.float, requires_grad=False)
        self.views_azim = torch.asarray((0, 0, 0, 0, -45, 45, -45, 45, -90, 90), dtype=torch.float, requires_grad=False)
        self.render = Renderer()
        self.totensor = ToTensor()
        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        points = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        points = self.random_sample(points, self.sample_points_num)
        points = self.pc_norm(points)
        points = torch.from_numpy(points).float()

        if self.partition == 'test':
            return points, cat_labels[sample['taxonomy_id']]
        
        name = sample['taxonomy_id'] + '_' + sample['model_id']
        rand_idx = random.randint(0, 9)
        image = Image.open('./data/rendering/%s/%d.png' % (name, rand_idx))
        image = self.norm(self.totensor(image))
        return image, points, self.views_azim[rand_idx], self.views_elev[rand_idx], self.views_dist[rand_idx]


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'pytorch3d.structures.meshes':
        return batch
    elif isinstance(elem, dict):
        return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (int)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):

        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]
