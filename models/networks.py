import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import *
from models.models import GraphNeuralNetwork, Aggregator, PowerLayer

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ATDGNN(nn.Module):

    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        super(ATDGNN, self).__init__()

        self.num_T = num_T
        self.out_graph = out_graph
        self.dropout_rate = dropout_rate
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.pool_step_rate = pool_step_rate
        self.idx = idx_graph
        self.channel = input_size[1]
        self.brain_area = len(self.idx)
        ###################
        # 多头注意力相关参数
        self.model_dim = round(num_T / 2)
        self.num_heads = 8
        if sampling_rate == 200:
            self.window_size = 100
            self.stride = 20
        else:
            self.window_size = 64
            self.stride = 16
        ###################
        hidden_features = input_size[2]

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
        # Batch normalization layers
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        #######################################
        # 特征整合、滑动窗口相关配置
        self.feature_integrator = FeatureIntegrator(in_channels=32, out_channels=self.model_dim)
        self.sliding_window_processor = SlidingWindowProcessor(model_dim=self.model_dim, num_heads=self.num_heads,
                                                               window_size=self.window_size, stride=self.stride)
        #######################################
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

        # Dynamic Graph Convolution Layers
        # self.dynamic_gcn = DynamicGraphNeuralNetwork(size[-1], out_graph)
        self.dynamic_gcn = StackedDynamicGraphNeuralNetwork(size[-1], hidden_features, out_graph, num_layers=3)
        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)

        # Fully connected layer for classification
        self.fc = nn.Sequential(  # 组合神经网络模块
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        #######################################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def forward(self, x):
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        ##############################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        out = self.bn(out)
        out = self.dynamic_gcn(out)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


class DynamicGraphNeuralNetwork(GraphNeuralNetwork):
    """
    Dynamic Graph Neural Network Layer.
    Extends the GraphNeuralNetwork layer with a dynamic adjacency matrix based on feature similarity.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicGraphNeuralNetwork, self).__init__(in_features, out_features, bias)

    def forward(self, x, adj=None):
        if adj is None:
            # Compute adjacency matrix dynamically based on feature similarity, for example:
            adj = self.normalize_adjacency_matrix(x)

        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output += self.bias
        output = F.relu(torch.matmul(adj, output))
        return output

    def compute_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def normalize_adjacency_matrix(self, x):
        """
        x：输入的特征矩阵，大小为(b, node, feature)，其中b为批次大小，node为节点数目，feature为每个节点的特征向量维度。
        self_loop：一个布尔值，表示是否在邻接矩阵中加入自环（自己到自己的连接）。
        """
        # x: b, node, feature
        # 利用模型中的self_similarity方法计算输入特征矩阵x的自相似度矩阵。结果为一个大小为(b, n, n)的张量，其中n为节点数目
        adj = self.compute_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes).to(DEVICE)
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


class StackedDynamicGraphNeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3, bias=True):
        super(StackedDynamicGraphNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(DynamicGraphNeuralNetwork(in_features, hidden_features, bias=bias))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(DynamicGraphNeuralNetwork(hidden_features, hidden_features, bias=bias))
        # Last layer
        self.layers.append(DynamicGraphNeuralNetwork(hidden_features, out_features, bias=bias))

    def forward(self, x, adj=None):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class FeatureIntegrator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # 假设输入x的形状为 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # 你想将feature和length维度相结合
        # 首先，将x变形为 (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 然后，应用1D卷积
        x = self.conv(x)  # 卷积后的形状为 (batch_size, out_channels, new_length)

        return x


class SlidingWindowProcessor(nn.Module):
    def __init__(self, model_dim, num_heads, window_size, stride):
        super(SlidingWindowProcessor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.layer_norm1 = nn.LayerNorm([window_size, model_dim])
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=32)
        # 定义融合层，使用1D卷积以保留32通道的结构，卷积核大小和步长可以根据需要调整
        self.fusion_conv = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, _, length = x.shape
        # 使用列表收集所有窗口的输出
        window_outputs = []

        for window_start in range(0, length - self.window_size + 1, self.stride):
            window_end = window_start + self.window_size
            window = x[:, :, window_start:window_end]
            window = window.permute(0, 2, 1)

            window = self.layer_norm1(window)
            attn_output, _ = self.multi_head_attention(window, window, window)
            attn_output = self.layer_norm2(attn_output + window)
            tcn_input = attn_output.permute(0, 2, 1)
            tcn_output = self.tcn_block(tcn_input)

            window_outputs.append(tcn_output)

        # 将所有窗口的输出沿着时间维度堆叠起来，形成一个新的维度
        stacked_outputs = torch.stack(window_outputs, dim=2)
        # 重新排列维度以匹配卷积层的输入要求
        stacked_outputs = stacked_outputs.permute(0, 3, 1, 2).reshape(batch_size, 32, -1)
        # 通过融合层整合所有窗口的输出
        fused_output = self.fusion_conv(stacked_outputs)

        return fused_output

