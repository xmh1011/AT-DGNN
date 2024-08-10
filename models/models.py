import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from config.config import *
from einops.layers.torch import Rearrange
import torch.nn.init as init
from torch_geometric.nn import SGConv, global_add_pool
from torch_geometric.nn import HeteroConv, GATConv, GCNConv, SAGEConv
from torch_geometric.utils import trim_to_layer, dense_to_sparse
import scipy.signal

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# DeepConvNet
# [Deep learning with convolutional neural networks for EEG decoding and visualization]
# (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
class DeepConvNet(nn.Module):
    def convBlock(self, inF, outF, droup_rate, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=droup_rate),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, droup_rate, kernalSize, channels, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(25, 25, (channels, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs))

    def calculateOutSize(self, model, channels, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, channels, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, channels, nTime, n_classes=2, droup_rate=0.25, *args, **kwargs):
        super(DeepConvNet, self).__init__()

        kernalSize = (1, 5)  # Please note that the kernel size in the origianl paper is (1, 10), we found when the segment length is shorter than 4s (1s, 2s, 3s) larger kernel size will
        # cause network error. Besides using (1, 5) when EEG segment is 4s gives slightly higher ACC and F1 with a smaller model size.
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, droup_rate, kernalSize, channels)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, droup_rate, kernalSize)
                                       for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, channels, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], n_classes, (1, self.fSize[1]))

    def forward(self, x):
        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x


# ShallowConvNet
# [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization]
# (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
class ShallowConvNet(nn.Module):
    def convBlock(self, inF, outF, droup_rate, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=droup_rate),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, droup_rate, kernalSize, channels, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(40, 40, (channels, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
        )

    def calculateOutSize(self, channels, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, channels, nTime)
        block_one = self.firstLayer
        avg = self.avgpool
        dp = self.dp
        out = torch.log(block_one(data).pow(2))
        out = avg(out)
        out = dp(out)
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, channels, nTime, n_classes=2, droup_rate=0.25, *args, **kwargs):
        super(ShallowConvNet, self).__init__()

        kernalSize = (1, 25)
        nFilt_FirstLayer = 40

        self.firstLayer = self.firstBlock(nFilt_FirstLayer, droup_rate, kernalSize, channels)
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dp = nn.Dropout(p=droup_rate)
        self.fSize = self.calculateOutSize(channels, nTime)
        self.lastLayer = nn.Linear(self.fSize[-1], n_classes)

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        x = self.lastLayer(x)

        return x


class EEGNetModule(nn.Module):
    def __init__(self, channels, F1, D, kernLength, dropout, input_size):
        super(EEGNetModule, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = D * F1
        self.T = input_size[2]
        self.kernLength = int(kernLength)

        # 第一阶段
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(channels, self.kernLength), groups=1,
                               padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1)
        # 深度卷积
        self.depthwiseConv = nn.Conv2d(in_channels=F1, out_channels=self.F2, kernel_size=(channels, 1), groups=F1,
                                       bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.F2)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        # 可分离卷积
        self.separableConv = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=(1, 16), padding='same',
                                       bias=False, groups=self.F2)
        self.conv2 = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=self.F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.activation(self.batchnorm2(self.depthwiseConv(x)))
        x = self.dropout1(self.avg_pool1(x))
        x = self.separableConv(x)
        x = self.conv2(x)
        x = self.activation(self.batchnorm3(x))
        x = self.dropout2(self.avg_pool2(x))
        return x


# EEGNet
# [EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces]
# (https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta?casa_token=lv6qPlB_YWgAAAAA:9c1FVN1Co6ae3vT6bjTh4VctC1sJLQPbv7uES2QtElX6JoAD2ICg4tndyvhaciMRSch51He_CszifyM0v1ZjBgp51WIW)
class EEGNet(nn.Module):
    def __init__(self, n_classes, input_size, sampling_rate, F1=8, kernLength=64, D=2, channels=32, dropout=0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = D * F1
        self.T = input_size[2]
        self.sampling_rate = sampling_rate
        self.kernLength = int(kernLength)
        # EEG数据处理模块
        self.EEGNet_sep = EEGNetModule(channels=channels, F1=F1, D=D, kernLength=kernLength, dropout=dropout,
                                       input_size=input_size)
        # 分类器
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.F2 * math.ceil(self.T / 32), n_classes)  # 这里T/32是因为有两次池化，每次都是T减少4倍

    def forward(self, x):
        x = self.EEGNet_sep(x)
        # 分类器
        x = self.flatten(x)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)


nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, bias=bias)
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float, activation: str = "relu"):
        super(_TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nonlinearity_dict[activation]
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nonlinearity_dict[activation]
        self.drop2 = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.project_channels = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.project_channels = nn.Identity()
        self.final_nonlinearity = nonlinearity_dict[activation]

    def forward(self, x):
        residual = self.project_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        return self.final_nonlinearity(out + residual)


class EEGTCNet(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 32, layers: int = 2, kernel_s: int = 4, filt: int = 12,
                 dropout: float = 0.3, activation: str = 'relu', F1: int = 8, D: int = 2, kernLength: int = 32,
                 dropout_eeg: float = 0.2
                 ):
        super(EEGTCNet, self).__init__()
        regRate = 0.25
        numFilters = F1
        F2 = numFilters * D

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=0.001),
            Conv2dWithConstraint(F1, F2, (in_channels, 1), bias=False, groups=F1,
                                 max_norm=1),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            Rearrange("b c 1 t -> b c t")
        )

        in_channels = [F2] + (layers - 1) * [filt]
        dilations = [2 ** i for i in range(layers)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])

        self.classifier = LinearWithConstraint(filt, n_classes, max_norm=regRate)
        # 初始化函数
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if hasattr(module, "weight") and module.weight is not None:
                if "norm" not in module.__class__.__name__.lower():
                    init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.eegnet(x)
        for blk in self.tcn_blocks:
            x = blk(x)
        x = self.classifier(x[:, :, -1])
        return x


# TCNet_Fusion
# [Electroencephalography-based motor imagery classification using temporal convolutional network fusion]
# (https://www.sciencedirect.com/science/article/abs/pii/S1746809421004237)
class TCNet_Fusion(nn.Module):
    def __init__(self, input_size, n_classes, channels, sampling_rate, kernel_s=3,
                 dropout=0.3, F1=24, D=2, dropout_eeg=0.3, layers=1, filt=12, activation='elu'):
        super(TCNet_Fusion, self).__init__()
        self.kernLength = int(0.25 * sampling_rate)
        F2 = F1 * D
        self.n_classes = n_classes

        self.EEGNet_sep = EEGNetModule(channels=channels, F1=F1, D=D, kernLength=self.kernLength, dropout=dropout_eeg,
                                       input_size=input_size)

        in_channels = [F2] + (layers - 1) * [filt]
        dilations = [2 ** i for i in range(layers)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        size = self.get_size_temporal(input_size)
        self.dense = nn.Linear(size[-1], n_classes)
        self.softmax = nn.Softmax(dim=1)

    def get_size_temporal(self, input_size):
        data = torch.randn((1, input_size[0], input_size[1], input_size[2]))
        x = self.EEGNet_sep(data)
        eeg_output = torch.squeeze(x, 2)
        for blk in self.tcn_blocks:
            tcn_output = blk(eeg_output)
        con1_output = torch.cat((eeg_output, tcn_output), dim=1)  # 沿特征维度拼接
        fc1_output = torch.flatten(eeg_output, start_dim=1)
        fc2_output = torch.flatten(con1_output, start_dim=1)
        # 再次Concatenation
        con2_output = torch.cat((fc1_output, fc2_output), dim=1)
        size = con2_output.size()
        return size

    def forward(self, x):
        x = self.EEGNet_sep(x)
        eeg_output = torch.squeeze(x, 2)
        for blk in self.tcn_blocks:
            tcn_output = blk(eeg_output)
        con1_output = torch.cat((eeg_output, tcn_output), dim=1)  # 沿特征维度拼接
        fc1_output = torch.flatten(eeg_output, start_dim=1)
        fc2_output = torch.flatten(con1_output, start_dim=1)
        # 再次Concatenation
        con2_output = torch.cat((fc1_output, fc2_output), dim=1)
        # Dense and Softmax
        dense_output = self.dense(con2_output)
        output = self.softmax(dense_output)
        return output


class PowerLayer(nn.Module):
    """
    The power layer: calculates the log-transformed power of the data
    """

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


# LGGNet
# [LGGNet: Learning from local-global-graph representations for brain–computer interface]
# (https://ieeexplore.ieee.org/abstract/document/10025569)
class LGGNet(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        # input_size: EEG frequency x channel x datapoint
        super(LGGNet, self).__init__()
        self.idx = idx_graph
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.channel = input_size[1]
        self.brain_area = len(self.idx)

        # by setting the convolutional kernel being (1,lenght) and the strids being 1, we can use conv2d to
        # achieve the 1d convolution operation.
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        # 表示局部滤波器的权重。它被定义为一个形状为(self.channel, size[-1])的浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        # 用来对local_filter_weight进行初始化，采用的是Xavier均匀分布初始化方法
        nn.init.xavier_uniform_(self.local_filter_weight)
        # 表示局部滤波器的偏置。它被定义为一个形状为(1, self.channel, 1)的浮点型张量，初始值为全零，并设置为需要梯度计算
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], out_graph)

        self.fc = nn.Sequential(  # 组合神经网络模块
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_s(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        adj = self.get_adj(out)
        out = self.bn(out)
        out = self.GCN(out, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_s(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        """
        x：输入的特征矩阵，大小为(b, node, feature)，其中b为批次大小，node为节点数目，feature为每个节点的特征向量维度。
        self_loop：一个布尔值，表示是否在邻接矩阵中加入自环（自己到自己的连接）。
        """
        # x: b, node, feature
        # 利用模型中的self_similarity方法计算输入特征矩阵x的自相似度矩阵。结果为一个大小为(b, n, n)的张量，其中n为节点数目
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        # 将自相似度矩阵与全局邻接矩阵的和进行逐元素相乘，并经过ReLU激活函数处理。这一步可以用来控制邻接矩阵中的连接关系
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            # 在邻接矩阵中添加自环，即在对角线上设置为1，表示每个节点与自己相连接
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        # 计算邻接矩阵每一行的元素之和，得到一个大小为(b, n)的张量
        rowsum = torch.sum(adj, dim=-1)
        # 创建一个与rowsum大小相同的全零张量mask，并将rowsum中和为0的位置置为1。这一步是为了处理邻接矩阵中存在度为0的节点，避免除以0的错误
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        # 将mask添加到rowsum中，实现对邻接矩阵的修正。避免除以0的错误，并保证每个节点的度至少为1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        # 将逆平方根得到的张量转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # 通过矩阵乘法和广播机制，将度矩阵的逆平方根与邻接矩阵相乘，得到归一化后的邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def compute_l2_regularization(self):
        l2_reg = torch.tensor(0.0).to(DEVICE)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)  # 计算每个可优化参数的L2范数
        return l2_reg

    def loss_fn(self, logits, labels):
        l2_reg = self.compute_l2_regularization()  # 计算L2正则化项
        loss = F.cross_entropy(logits, labels) + self.weight_decay * l2_reg  # 将L2正则化项添加到损失函数中
        return loss


# TSception
# [Tsception: a deep learning framework for emotion detection using EEG]
# (https://ieeexplore.ieee.org/abstract/document/9206750/)
class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depth_multiplier = depth_multiplier
        # depth_multiplier决定了每个输入通道应该扩展到多少个输出通道
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * depth_multiplier, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=0  # 设置padding为0来减少高度
        )

    def forward(self, x):
        return self.depthwise(x)


class ATC_Conv(nn.Module):
    def __init__(self, n_channel, in_channels, F1, D, KC, P2, dropout=0.3):
        super(ATC_Conv, self).__init__()
        F2 = F1 * D  # Output dimension
        # 第一层常规卷积: 时间卷积层
        self.temporal_conv = nn.Conv2d(in_channels, F1, (1, KC), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        # 第二层卷积: 深度卷积层
        self.depthwise_conv = DepthwiseConv2d(in_channels=F1, depth_multiplier=D, kernel_size=(n_channel, 1))
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.avgpool1 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        # 第三层卷积
        self.spatial_conv = nn.Conv2d(F1 * D, F2, (1, KC), padding='same', bias=False)  # 修改核的尺寸以适配 KC
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, P2))  # 池化尺寸由P2控制
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        # 时间卷积
        x = self.temporal_conv(x)
        x = self.batchnorm1(x)
        x = self.elu(x)
        x = self.dropout1(x)
        # 深度卷积
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout2(x)
        # 空间卷积
        x = self.spatial_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout3(x)
        return x


# ATCNet
# [Physics-informed attention temporal convolutional network for EEG-based motor imagery classification]
# (https://ieeexplore.ieee.org/abstract/document/9852687/)
class ATCNet(nn.Module):
    def __init__(self, input_size, n_channel, n_classes, n_windows=8,
                 eegn_F1=24, eegn_D=2, eegn_kernelSize=50, eegn_poolSize=8, eegn_dropout=0.3, num_heads=2,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3, fuse='average',
                 activation='elu'):
        super(ATCNet, self).__init__()
        self.n_windows = n_windows
        self.conv_block = ATC_Conv(n_channel, 1, eegn_F1, eegn_D, eegn_kernelSize, eegn_poolSize, eegn_dropout)
        self.fuse = fuse

        in_channels = [eegn_F1 * eegn_D] + (tcn_depth - 1) * [tcn_filters]
        dilations = [2 ** i for i in range(tcn_depth)]

        self.attention_block = MultiHeadSelfAttention(eegn_F1 * eegn_D, num_heads)

        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, tcn_filters, kernel_size=tcn_kernelSize, dilation=dilation,
                      dropout=tcn_dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        self.fuse_layer = nn.Linear(tcn_filters, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # (64,1,32,800)
        x = self.conv_block(x)  # (64,100,1,12)
        x = torch.flatten(x, start_dim=2)  # Flatten the channel and height dimensions (64,100,12)

        outputs = []
        for i in range(self.n_windows):
            windows_data = x[:, :, i:x.shape[2] - self.n_windows + i + 1]  # Sliding window
            # Attention block
            tcn_input = self.attention_block(windows_data)  # (batch_size, channels, T_w)
            for blk in self.tcn_blocks:
                tcn_output = blk(tcn_input)
                tcn_input = tcn_output
            # (64,32,5)
            tcn_output = tcn_output[:, :, -1]  # Last timestep
            outputs.append(tcn_output)
        # (64,32)
        if self.fuse == 'average':
            output = torch.mean(torch.stack(outputs, dim=1), dim=1)
        elif self.fuse == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            raise ValueError("Invalid fuse method")

        output = self.fuse_layer(output)
        output = self.softmax(output)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, d, T_w = x.size()

        # Permute to (batch_size, T_w, d)
        x = x.permute(0, 2, 1)

        # Linear transformations
        Q = self.query_linear(x)  # (batch_size, T_w, d_model)
        K = self.key_linear(x)  # (batch_size, T_w, d_model)
        V = self.value_linear(x)  # (batch_size, T_w, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)
        K = K.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)
        V = V.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, num_heads, T_w, T_w)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, T_w, T_w)
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, T_w, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, T_w, self.d_model)  # (batch_size, T_w, d_model)

        # Final linear transformation
        output = self.out_linear(context)  # (batch_size, T_w, d_model)
        output = self.layer_norm(output + x)  # Add & Norm

        # Permute back to (batch_size, d, T_w)
        output = output.permute(0, 2, 1)

        return output


class AdjacencyProcessor:
    def __init__(self, device=DEVICE):
        self.device = device

    def normalize_features(self, x):
        """
        对输入特征进行归一化处理
        x: (b, node, feature)
        返回: 归一化后的特征矩阵 (b, node, feature)
        """
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-6
        return (x - mean) / std

    def compute_similarity(self, x):
        """
        计算输入特征矩阵 x 的余弦相似度矩阵。
        x: (b, node, feature)
        返回: (b, node, node)
        """
        x = F.normalize(x, p=2, dim=2)  # 对每个特征向量进行L2归一化
        x_ = x.permute(0, 2, 1)  # 调整维度顺序为 (b, feature, node)
        s = torch.bmm(x, x_)  # 计算余弦相似度矩阵 (b, node, node)
        s = (s + 1) / 2  # 将相似度从 [-1, 1] 转换到 [0, 1]
        return s

    def normalize_adjacency_matrix(self, x):
        """
        归一化邻接矩阵。
        x：输入的特征矩阵，大小为(b, node, feature)。
        返回值：归一化后的邻接矩阵，大小为(b, node, node)。
        """
        # 计算自相似度矩阵
        adj = self.compute_similarity(x)  # (b, node, node)

        num_nodes = adj.shape[-1]
        # 加入自环
        adj = adj + torch.eye(num_nodes).to(DEVICE)
        # 计算度矩阵
        rowsum = torch.sum(adj, dim=-1)  # 计算度矩阵
        # 避免除以0
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # (b, node)
        # 将逆平方根转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (b, node, node)
        # 归一化邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)  # (b, node, node)
        return adj

    def process_batch(self, x_batch):
        """
        处理输入批次的特征矩阵 x_batch，返回相应的 edge_index 和 edge_weight。
        x_batch：大小为(b, node, feature)的输入特征矩阵。
        返回：
        - edge_index：大小为 (2, num_edges) 的张量。
        - edge_weight：大小为 (num_edges,) 的张量。
        """
        # 归一化邻接矩阵
        adj = self.normalize_adjacency_matrix(x_batch)  # (b, node, node)

        edge_index, edge_weight = dense_to_sparse(adj[0])

        return edge_index, edge_weight


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()  # https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def generate_cheby_adj(self, A, K, device):
        support = []
        for i in range(K):
            if i == 0:
                # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
                temp = torch.eye(A.shape[1])
                temp = temp.to(device)
                support.append(temp)
            elif i == 1:
                support.append(A)
            else:
                temp = torch.matmul(support[-1], A)
                support.append(temp)
        return support

    def forward(self, x, L):
        device = x.device
        adj = self.generate_cheby_adj(L, self.K, device)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


# DGCNN
# [EEG emotion recognition using dynamical graph convolutional neural networks]
# (https://ieeexplore.ieee.org/abstract/document/8320798)
class DGCNN(nn.Module):
    def __init__(self, input_size, batch_size, k_adj, num_out=64, nclass=2):
        super(DGCNN, self).__init__()
        self.batch_size = batch_size
        xdim = [batch_size] + [32, 5]
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # 对第二维（第一维为batch_size)进行标准化
        self.fc1 = Linear(xdim[1] * num_out, 32)
        self.fc2 = Linear(32, 8)
        self.fc3 = Linear(8, nclass)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        nn.init.xavier_normal_(self.A)

    def normalize_A(self, A, symmetry=False):
        A = F.relu(A)
        if symmetry:
            A = A + torch.transpose(A, 0, 1)  # A+ A的转置
            d = torch.sum(A, 1)  # 对A的第1维度求和
            d = 1 / torch.sqrt(d + 1e-10)  # d的-1/2次方
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        return L

    def forward(self, x):
        x = x.squeeze(1)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 因为第三维 才为特征维度
        L = self.normalize_A(self.A)  # A是自己设置的可训练参数及邻接矩阵
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = F.relu(self.fc2(result))
        result = self.fc3(result)
        return result
