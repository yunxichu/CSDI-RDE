"""v1 legacy — CSDI score network（来自 ../csdi/diff_models.py）。

包含 diff_CSDI：Transformer 架构的 diffusion score network，供 CSDI_base 调用。
v2 版本见 methods/dynamics_csdi.py，在此基础上增加了噪声条件化和延迟 attention mask。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
# 导入必要的库


def get_torch_trans(heads=8, layers=1, channels=64):
    # 创建标准的Transformer编码器
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    # 创建Transformer编码器层
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)
    # 返回Transformer编码器

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):
    # 创建线性注意力Transformer
    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )
    # 返回线性注意力Transformer实例

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    # 创建1D卷积层并进行初始化
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # 创建1D卷积层
    nn.init.kaiming_normal_(layer.weight)
    # 使用Kaiming正态分布初始化权重
    return layer


class DiffusionEmbedding(nn.Module):
    # 扩散步骤嵌入模块
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        # 调用父类初始化
        if projection_dim is None:
            projection_dim = embedding_dim
            # 如果未指定投影维度，使用嵌入维度
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        # 注册缓冲区，存储预计算的嵌入表
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        # 第一个线性投影层
        self.projection2 = nn.Linear(projection_dim, projection_dim)
        # 第二个线性投影层

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        # 根据扩散步骤索引获取嵌入
        x = self.projection1(x)
        # 第一个投影
        x = F.silu(x)
        # SiLU激活函数
        x = self.projection2(x)
        # 第二个投影
        x = F.silu(x)
        # SiLU激活函数
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        # 创建步骤张量
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        # 创建频率张量
        table = steps * frequencies  # (T,dim)
        # 计算位置编码表
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        # 拼接正弦和余弦编码
        return table


class diff_CSDI(nn.Module):
    # 主要的CSDI扩散模型
    def __init__(self, config, inputdim=2):
        super().__init__()
        # 调用父类初始化
        self.channels = config["channels"]
        # 设置通道数

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        # 创建扩散步骤嵌入模块

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        # 输入投影层
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # 输出投影层1
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        # 输出投影层2
        nn.init.zeros_(self.output_projection2.weight)
        # 将输出层权重初始化为0

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )
        # 创建残差层列表

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        # 获取输入形状

        x = x.reshape(B, inputdim, K * L)
        # 重塑输入
        x = self.input_projection(x)
        # 输入投影
        x = F.relu(x)
        # ReLU激活
        x = x.reshape(B, self.channels, K, L)
        # 重塑回原始形状

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        # 获取扩散嵌入

        skip = []
        # 初始化跳跃连接列表
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            # 通过残差层
            skip.append(skip_connection)
            # 收集跳跃连接

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        # 对跳跃连接求平均并缩放
        x = x.reshape(B, self.channels, K * L)
        # 重塑
        x = self.output_projection1(x)  # (B,channel,K*L)
        # 输出投影1
        x = F.relu(x)
        # ReLU激活
        x = self.output_projection2(x)  # (B,1,K*L)
        # 输出投影2
        x = x.reshape(B, K, L)
        # 重塑为最终输出形状
        return x


class ResidualBlock(nn.Module):
    # 残差块模块
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        # 调用父类初始化
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # 扩散嵌入投影层
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        # 条件信息投影层
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        # 中间投影层
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        # 输出投影层

        self.is_linear = is_linear
        # 是否使用线性注意力
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            # 使用线性注意力时间层
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            # 使用线性注意力特征层
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            # 使用标准Transformer时间层
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            # 使用标准Transformer特征层


    def forward_time(self, y, base_shape):
        # 时间维度前向传播
        B, channel, K, L = base_shape
        # 获取基础形状
        if L == 1:
            return y
            # 如果时间长度为1，直接返回
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        # 重塑以便在时间维度应用注意力

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            # 线性注意力：输入形状(B*K, L, channel)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
            # 标准注意力：输入形状(L, B*K, channel)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        # 重塑回原始形状
        return y


    def forward_feature(self, y, base_shape):
        # 特征维度前向传播
        B, channel, K, L = base_shape
        # 获取基础形状
        if K == 1:
            return y
            # 如果特征数为1，直接返回
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # 重塑以便在特征维度应用注意力

        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            # 线性注意力：输入形状(B*L, K, channel)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
            # 标准注意力：输入形状(K, B*L, channel)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        # 重塑回原始形状
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        # 获取输入形状
        base_shape = x.shape
        # 保存基础形状
        x = x.reshape(B, channel, K * L)
        # 重塑输入

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        # 投影扩散嵌入并增加维度
        y = x + diffusion_emb
        # 添加扩散嵌入

        y = self.forward_time(y, base_shape)
        # 时间维度处理
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        # 特征维度处理
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        # 中间投影

        _, cond_dim, _, _ = cond_info.shape
        # 获取条件信息维度
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        # 重塑条件信息
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        # 条件信息投影
        y = y + cond_info
        # 添加条件信息

        gate, filter = torch.chunk(y, 2, dim=1)
        # 将输出分割为门控和过滤部分
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        # 应用门控机制

        y = self.output_projection(y)
        # 输出投影

        residual, skip = torch.chunk(y, 2, dim=1)
        # 将输出分割为残差和跳跃连接
        x = x.reshape(base_shape)
        # 重塑输入回原始形状
        residual = residual.reshape(base_shape)
        # 重塑残差回原始形状
        skip = skip.reshape(base_shape)
        # 重塑跳跃连接回原始形状
        return (x + residual) / math.sqrt(2.0), skip
        # 返回残差连接结果和跳跃连接