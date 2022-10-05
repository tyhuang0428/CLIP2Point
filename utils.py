import numpy as np
import torch
from plyfile import PlyData


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def read_state_dict(path):
    ckpt = torch.load(path)
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    for key in list(base_ckpt.keys()):
        if key.startswith('point_model.'):
            base_ckpt[key[len('point_model.'):]] = base_ckpt[key]
        del base_ckpt[key]
    return base_ckpt


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array
