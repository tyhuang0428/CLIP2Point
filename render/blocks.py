import os
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    activation layer
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

        
# Now, let's implement a sharedMLP layer. It is implmented by using Conv1d with kernel size equals to 1. 
class Conv1dLayer(Seq):
    def __init__(self, channels, act='relu', norm=True, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv1d(channels[i - 1], channels[i], 1, bias=bias))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
        super(Conv1dLayer, self).__init__(*m)


class MLP(Seq):
    """
    Given input with shape [B, C_in]
    return output with shape [B, C_out] 
    """

    def __init__(self, channels, act='relu', norm=True, bias=True, dropout=0.5):
        # todo:
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias=bias))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
            if dropout > 0:
                m.append(nn.Dropout(dropout))
        super(MLP, self).__init__(*m)


def knn(x, k):
    """
    Given point features x [B, C, N, 1], and number of neighbors k (int)
    Return the idx for the k neighbors of each point. 
    So, the shape of idx: [B, N, k]
    """
    with torch.no_grad():
        x = x.squeeze(-1)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        inner = -xx - inner - xx.transpose(2, 1)

        idx = inner.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def batched_index_select(x, idx):
    """
    This can be used for neighbors features fetching
    Given a pointcloud x, return its k neighbors features indicated by a tensor idx.
    :param x: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(
        0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k,
                           num_dims).permute(0, 3, 1, 2)
    return feature


def get_center_feature(x, k):
    """
    Given you a point cloud, and neighbors k, return the center features.
    :param x: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param k: int
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """
    x = x.repeat(1, 1, 1, k)
    return x


class Transformation(nn.Module):
    def __init__(self, k=3):
        super(Transformation, self).__init__()
        self.k = k
        # Task 2.2.1 T-Net architecture

        # self.convs consists of 3 convolution layer.
        # please look at the description above.

        self.convs = Seq(*[Conv1dLayer([self.k, 64], act='relu', norm=True, bias=True), Conv1dLayer(
            [64, 128], act='relu', norm=True, bias=True), Conv1dLayer([128, 1024], act=None, norm=False, bias=True)])
        self.fcs = Seq(*[Conv1dLayer([1024, 512], act='relu', norm=True, bias=True), Conv1dLayer([512, 256], act='relu', norm=True,
                                                                                                 bias=True), Conv1dLayer([256, self.k*self.k], act=None, norm=False, bias=True)])  # no relu or BN at the last layer.

    def forward(self, x):
        # Forward of T-Net architecture

        B, K, N = x.shape  # batch-size, dim, number of points
        ## forward of shared mlp
        # input - B x K x N
        # output - B x 1024 x N

        x = self.convs(x)

        ## global max pooling
        # input - B x 1024 x N
        # output - B x 1024

        x, _ = torch.max(x, 2, keepdim=True)
#         print(x.size())

        ## mlp
        # input - B x 1024
        # output - B x (K*K)

        x = self.fcs(x)

        ## reshape the transformation matrix to B x K x K
        identity = torch.eye(self.k, device=x.device)
        x = x.view(B, self.k, self.k) + identity[None]
        return x


def stn(x, transform_matrix=None):
    # spatial transformantion network. this is the matrix multiplication part inside the joint alignment network.
    x = x.transpose(2, 1)
    x = torch.bmm(x, transform_matrix)
    x = x.transpose(2, 1)
    return x


class OrthoLoss(nn.Module):
    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, x):
        ## hint: useful function `torch.bmm` or `torch.matmul`

        ## TASK 2.2.2
        ## compute the matrix product
        #         print(x.size(),torch.transpose(x,1,2).size())
        prod = torch.bmm(x, torch.transpose(x, 1, 2))

        prod = torch.stack([torch.eye(prod.size()[1]) for ii in range(
            prod.size()[0])]).to(x.device) - prod    # minus
        norm = torch.norm(prod, 'fro')**2
        return norm


class PointNet(nn.Module):
    def __init__(self, num_classes=40, alignment=False):
        super(PointNet, self).__init__()
        # look at the description under 2.2 or refer to the paper if you need more details

        self.alignment = alignment

        ## `input_transform` calculates the input transform matrix of size `3 x 3`
        if self.alignment:
            self.input_transform = Transformation(3)

        ## TASK 2.3.1
        ## define your network layers here
        ## local feature
        ## one shared mlp layer (shared MLP is actually 1x1 convolution. You can use our conv1dLayer)
        ## input size: B x 3 x N
        ## output size: B x 64 x N

        self.conv1 = Conv1dLayer([3, 64], act='relu', norm=True, bias=True)

        ## `feature_transform` calculates the feature transform matrix of size `64 x 64`
        if self.alignment:
            ## TASK 2.3.2 transormation layer
            self.feature_transform = Transformation(64)

        ## TASK 2.3.3
        ## define your network layers here
        ## global feature
        ## 2 layers of shared mlp.  64 -> 128 -> 1024
        ## input size: B x 64 x N
        ## output size: B x 1024 x N

        self.conv2s = Conv1dLayer(
            [64, 128, 1024], act='relu', norm=True, bias=True)

        # Task 2.3.4 classification layer
        # 3 MLP layers. 1024 -> 512 -> 256 -> num_classes.
        # there is a dropout in the second layer. dropout ratio = 05
        # no relu or BN at the last layer.
        # self.classifier = MLP([1024, 512, 256, num_classes],
        #                       act='relu', norm=True, bias=True, dropout=0.5)
        self.classifier = MLP([1024, 512],
                              act='relu', norm=True, bias=True, dropout=0.5)

    def forward(self, x):

        ## task 2.3.5 apply the input transform in the coordinate domain
        if self.alignment:
            # get transformation matrix then apply to x
            transform = self.input_transform(x)
            # apply transorm into the input feature x
            x = torch.bmm(transform, x)

        ## forward of shared mlp
        # input - B x K x N
        # output - B x 64 x N
        x = self.conv1(x)

        ## task 2.3.7 another transform in the feauture domain
        if self.alignment:
            transform = self.feature_transform(x)
            x = torch.bmm(transform, x)
        else:
            transform = None
#         local_feature = x  # this can be used in segmentation task. we comment it out here.

        ## TASK 2.3.8
        ## forward of shared mlp
        # input - B x 64 x N
        # output - B x 1024 x N
        x = self.conv2s(x)

        ## global max pooling
        # input - B x 1024 x N
        # output - B x 1024
        x = torch.max(x, dim=2, keepdim=True)[0]
        global_feature = x.view(-1, 1024)

        ## summary:
        ## global_feature: B x 1024
        ## local_feature: B x 64 x N
        ## transform: B x K x K

        # 2.3.10 classification
        out = self.classifier(global_feature)
        return out, global_feature, transform


class Conv2dLayer(Seq):
    def __init__(self, channels, act='relu', norm=True, bias=False, kernel_size=1, stride=1, dilation=1, drop=0., groups=1):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], bias=bias,
                               kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups))
            if norm:
                m.append(nn.BatchNorm2d(channels[i]))
            if act:
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super(Conv2dLayer, self).__init__(*m)


class EdgeConv2d(nn.Module):
    """
    Static EdgeConv graph convolution layer (with activation, batch normalization) for point cloud [B, C, N, 1]. 
    This operation perform the EdgeConv given the knn idx. 
    input: B, C, N, 1
    return: B, C, N, 1
    """

    def __init__(self, in_channels, out_channels, act='leakyrelu', norm=True, bias=False, aggr='max', groups=1):
        super(EdgeConv2d, self).__init__()
        self.nn = Conv2dLayer([in_channels * 2, out_channels],
                              act, norm, bias, groups=groups)
        if aggr == 'mean':
            self.aggr = torch.mean
        else:
            self.aggr = torch.max

    def forward(self, x, edge_index):
        # TASK3.3: Write the forwad pass of EdgeConv.
        # use x_j to indicate neighbor features.
        x_j = batched_index_select(x, edge_index)
        # use x_i to indicate center features.
        x_i = get_center_feature(x, edge_index.size()[-1])
        x = self.aggr(
            self.nn(torch.cat([x_i, x_i-x_j], dim=1)), dim=3, keepdim=True)[0]
        return x


class DynEdgeConv2d(EdgeConv2d):
    """
        Dynamic EdgeConv graph convolution layer (with activation, batch normalization) for point cloud [B, C, N, 1]
        This operaiton will build the knn graph at first, then perform the static EdgeConv
        input: B, C, N, 1
        return: B, C, N, 1
    """

    def __init__(self, in_channels, out_channels, k=9, act='relu',
                 norm=True, bias=False, aggr='max'):
        super(DynEdgeConv2d, self).__init__(in_channels,
                                            out_channels, act=act, norm=norm, bias=bias, aggr=aggr)
        self.k = k

    def forward(self, x):
        idx = knn(x, self.k)
        x = super(DynEdgeConv2d, self).forward(x, idx)
        return x
#


class SimpleDGCNN(nn.Module):
    def __init__(self, num_classes=40, k=9):
        super(SimpleDGCNN, self).__init__()
        self.k = k

        # Look at PointNet backbone.
        # There are conv1d layer: 3 --> 64 --> 128 -->1024.
        # Then MLP classifier.

        # Here we keep the classifier part the same. But change the backbone into dynamic EdgeConv.
        # k=9, use relu and bachnormalization. Other parameters keep the default.
        self.convs = Seq(*[DynEdgeConv2d(3, 64, k=self.k), DynEdgeConv2d(64,
                                                                         128, k=self.k), DynEdgeConv2d(128, 1024, k=self.k)])
        # self.classifier = Seq(*[MLP([1024, 512, 256], act='relu', norm=True, bias=True, dropout=0.5),
        #                         MLP([256, num_classes], act=None, norm=False, bias=True, dropout=0)])
        self.classifier = Seq(*MLP([1024, 512], act='relu', norm=True, bias=True, dropout=0))

    def forward(self, x):
        # x should be [B, C, N, 1]
        if len(x.shape) < 4:
            x = x.unsqueeze(-1)

        # dynamic edgeConvolution layers
        x = self.convs(x)

        # max pooling layer
        x = torch.max(x, dim=2, keepdim=True)[0]
        global_feature = x.view(-1, 1024)
        out = self.classifier(global_feature)
        return out, global_feature , None


def load_point_ckpt(model,  network_name,  ckpt_dir='./checkpoint', verbose=True):
    # ------------------ load ckpt
    filename = '{}/{}_model.pth'.format(ckpt_dir, network_name)
    if not os.path.exists(filename):
        print("No such checkpoint file as:  {}".format(filename))
        return None
    state = torch.load(filename)
    state['state_dict'] = {k: v.cuda() for k, v in state['state_dict'].items()}
    model.load_state_dict(state['state_dict'], strict=False)
    # optimizer.load_state_dict(state['optimizer_state_dict'])
    # scheduler.load_state_dict(state['scheduler_state_dict'])
    if verbose:
        print('Succeefullly loaded model from {}'.format(filename))