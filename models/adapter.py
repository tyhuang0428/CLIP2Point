import torch
import torch.nn as nn


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.reshape(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.reshape(s1, s2, s3)


class SimplifiedAdapter(nn.Module):
    def __init__(self, num_views=10, in_features=512):
        super().__init__()

        self.num_views = num_views
        self.in_features = in_features
        self.adapter_ratio = 0.6
        self.fusion_init = 0.5
        self.dropout = 0.075
        
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        self.global_f = nn.Sequential(
                BatchNormPoint(self.in_features),
                nn.Dropout(self.dropout),
                nn.Flatten(),
                nn.Linear(in_features=self.in_features * self.num_views,
                          out_features=self.in_features),
                nn.BatchNorm1d(self.in_features),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.in_features, out_features=self.in_features))

    def forward(self, feat):
        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        
        # Global feature
        return self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1))
