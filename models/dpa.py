from copy import deepcopy
import torch.nn as nn
import clip

from .adapter import SimplifiedAdapter
from render import Renderer, Selector
from utils import read_state_dict

clip_model, _ = clip.load("ViT-B/32", device='cpu')


class DPA(nn.Module):
    def __init__(self, args, eval=False):
        super().__init__()
        self.views = args.views
        self.selector =  Selector(self.views, args.dim, args.model)
        self.renderer = Renderer(points_radius=0.02)
        self.pre_model = deepcopy(clip_model.visual)
        self.ori_model = deepcopy(clip_model.visual)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.pre_model.load_state_dict(read_state_dict(args.ckpt))
        self.adapter1 = SimplifiedAdapter(num_views=args.views, in_features=512)
        self.adapter2 = SimplifiedAdapter(num_views=args.views, in_features=512)
    
    def forward(self, points):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=True)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        img_feat1 = self.adapter1(self.pre_model(imgs))
        img_feat2 = self.adapter2(self.ori_model(imgs))
        img_feats = (img_feat1 + img_feat2) * 0.5
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats
