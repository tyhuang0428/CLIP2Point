import torch
from torch import nn

from render.blocks import MLP, PointNet, SimpleDGCNN, load_point_ckpt


class ViewSelector(nn.Module):
    def __init__(self, nb_views, canonical_distance=1., transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones((self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        if self.nb_views == 10:
            views_elev = torch.asarray((0, 90, 180, 270, 225, 225, 315, 315, 0, 0), dtype=torch.float, requires_grad=False)
            views_azim = torch.asarray((0, 0, 0, 0, -45, 45, -45, 45, -90, 90), dtype=torch.float, requires_grad=False)
        elif self.nb_views == 6:
            views_elev = torch.asarray((0, 0, 0, 0, 90, -90), dtype=torch.float, requires_grad=False)
            views_azim = torch.asarray((0, 90, 180, 270, 0, 180), dtype=torch.float, requires_grad=False)
        
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, c_batch_size):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class LearnedViewSelector(ViewSelector):
    def __init__(self, nb_views, shape_features_size=512, canonical_distance=1., transform_distance=False, input_view_noise=0.0):
        ViewSelector.__init__(self, nb_views, canonical_distance, transform_distance, input_view_noise)
        self.view_transformer = nn.Sequential(
            MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 * self.nb_views, 3*self.nb_views], dropout=0.5, norm=True),
            MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False),
            nn.Tanh()) if self.transform_distance \
                else nn.Sequential(
            MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 * self.nb_views, 2*self.nb_views], dropout=0.5, norm=True),
            MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False),
            nn.Tanh())
    
    def forward(self, shape_features):
        c_batch_size = shape_features.shape[0]
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class FeatureExtractor(nn.Module):
    def __init__(self, shape_features_size, shape_extractor, screatch_feature_extractor):
        super().__init__()
        if shape_extractor == "PointNet":
            print('build PointNet selector')
            self.fe_model = PointNet(shape_features_size, alignment=True)
        elif shape_extractor == "DGCNN":
            print('build DGCNN selector')
            self.fe_model = SimpleDGCNN(shape_features_size)
        if screatch_feature_extractor:
            load_point_ckpt(self.fe_model, shape_extractor,
                            ckpt_dir='./checkpoint')
            self.features_order = {"logits": 0,
                                   "post_max": 1, "transform_matrix": 2}

    def forward(self, points):
        batch_size, _, _ = points.shape
        points = points.transpose(1, 2)
        features = self.fe_model(points)
        return features[0].view(batch_size, -1)


class Selector(nn.Module):
    def __init__(self, nb_views, shape_features_size=512, shape_extractor="PointNet", canonical_distance=1., transform_distance=False, input_view_noise=0.0, screatch_feature_extractor=False):
        super().__init__()
        self.learned = True if shape_features_size > 0 else False
        self.view_selector = LearnedViewSelector(nb_views, shape_features_size, canonical_distance, transform_distance, input_view_noise) if self.learned else ViewSelector(nb_views, canonical_distance, transform_distance, input_view_noise)
        if self.learned:
            self.feature_extractor = FeatureExtractor(shape_features_size=shape_features_size, shape_extractor=shape_extractor, screatch_feature_extractor=screatch_feature_extractor)

    def forward(self, points):
        if self.learned:
            shape_features = self.feature_extractor(points)
            return self.view_selector(shape_features)
        return self.view_selector(points.shape[0])
