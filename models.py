import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from einops.layers.torch import Rearrange
import torch.nn.init as init

_, os.environ['CUDA_VISIBLE_DEVICES'] = config.set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, channels, nTime, n_classes=2, droup_rate=0.25, *args, **kwargs):
        super(DeepConvNet, self).__init__()

        kernalSize = (1,
                      5)  # Please note that the kernel size in the origianl paper is (1, 10), we found when the segment length is shorter than 4s (1s, 2s, 3s) larger kernel size will
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
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
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
        # 定义前向传播
        # (64, 1, 32, 800)
        x = self.batchnorm1(self.conv1(x))  # (64,8,32,800)
        x = self.activation(self.batchnorm2(self.depthwiseConv(x)))  # (64,16,1,800)
        x = self.dropout1(self.avg_pool1(x))  # (64,16,1,200)
        x = self.separableConv(x)
        x = self.conv2(x)
        x = self.activation(self.batchnorm3(x))
        x = self.dropout2(self.avg_pool2(x))  # (64,16,1,25)
        return x


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
        # (64,1,32,800)
        x = self.EEGNet_sep(x)
        # (64,16,1,25)
        # 分类器
        x = self.flatten(x)  # (64,400)
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
        # (1,48,1,25)
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

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))
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

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output
