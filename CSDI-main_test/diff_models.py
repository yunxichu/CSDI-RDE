import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

def get_torch_trans(heads=8, layers=1, channels=64):
    """创建标准的PyTorch Transformer编码器"""
    # 创建Transformer编码器层
    # d_model: 输入特征的维度（等于channels）
    # nhead: 多头注意力的头数
    # dim_feedforward: 前馈网络的隐藏层维度
    # activation: 使用GELU激活函数
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    # 创建包含多个编码器层的Transformer编码器
    # num_layers: 编码器层的堆叠层数
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    """创建线性注意力Transformer，计算效率更高"""
    # 使用线性注意力Transformer，计算复杂度从O(n^2)降到O(n)
    return LinearAttentionTransformer(
        dim=channels,           # 输入维度
        depth=layers,           # Transformer层数
        heads=heads,            # 注意力头数
        max_seq_len=256,        # 最大序列长度
        n_local_attn_heads=0,   # 局部注意力头数（0表示不使用）
        local_attn_window_size=0, # 局部注意力窗口大小
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """创建1D卷积层并使用Kaiming初始化"""
    # 创建1D卷积层
    # in_channels: 输入通道数
    # out_channels: 输出通道数  
    # kernel_size: 卷积核大小
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # 使用Kaiming正态分布初始化权重，适合ReLU等激活函数
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    """扩散过程的时间步嵌入：为扩散模型的每个时间步生成位置编码"""
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        # 如果未指定投影维度，使用嵌入维度
        if projection_dim is None:
            projection_dim = embedding_dim
            
        # 注册缓冲区：存储不需要梯度但需要保存的参数
        # embedding: 预先计算好的正弦余弦位置编码表
        # persistent=False: 不保存到state_dict中
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),  # dim参数是总维度的一半
            persistent=False,
        )
        # 第一层投影：将嵌入维度映射到投影维度
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        # 第二层投影：进一步变换特征
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        # diffusion_step: 当前扩散步骤的索引，形状为[B]或标量
        # 从预计算的嵌入表中获取对应步骤的嵌入向量
        x = self.embedding[diffusion_step]  # 形状: [B, embedding_dim] 或 [embedding_dim]
        # 第一层线性变换
        x = self.projection1(x)
        # SiLU激活函数（Swish）：x * sigmoid(x)，比ReLU更平滑
        x = F.silu(x)
        # 第二层线性变换
        x = self.projection2(x)
        # 再次使用SiLU激活
        x = F.silu(x)
        # 返回扩散步骤的嵌入表示，形状: [B, projection_dim] 或 [projection_dim]
        return x

    def _build_embedding(self, num_steps, dim=64):
        """构建正弦余弦位置编码表"""
        # steps: 从0到num_steps-1的序列，形状为[T, 1]，T=num_steps
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        # frequencies: 频率向量，从10^0到10^4，形状为[1, dim]
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        # table: 步骤和频率的外积，形状为[T, dim]
        table = steps * frequencies  # (T,dim)
        # 将正弦和余弦部分拼接，形状变为[T, dim*2]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    """主要的CSDI模型：基于条件扩散模型的时间序列插补"""
    def __init__(self, config, inputdim=2):
        super().__init__()
        # channels: 模型内部的主要特征维度
        self.channels = config["channels"]

        # 扩散时间步嵌入模块
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],           # 扩散过程的总步数
            embedding_dim=config["diffusion_embedding_dim"],  # 嵌入维度
        )

        # 输入投影：将输入维度映射到内部通道数
        # 使用1x1卷积，相当于全连接层但保持空间结构
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        # 第一层输出投影：将内部通道数映射回自身
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # 第二层输出投影：最终输出，映射到1个通道
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        # 将最后一层的权重初始化为0，训练开始时输出接近0
        nn.init.zeros_(self.output_projection2.weight)

        # 残差层堆叠：多个残差块构成主要网络
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],      # 条件信息的特征维度
                    channels=self.channels,           # 内部通道数
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],  # 扩散嵌入维度
                    nheads=config["nheads"],          # 注意力头数
                    is_linear=config["is_linear"],    # 是否使用线性注意力
                )
                for _ in range(config["layers"])      # 残差块的数量
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        # x: 输入张量，形状为 [B, inputdim, K, L]
        #   B: batch_size，批次大小
        #   inputdim: 输入维度，通常为2（观测值+掩码）
        #   K: 特征数量（如不同传感器的数量）
        #   L: 序列长度（时间步数）
        # cond_info: 条件信息，形状为 [B, side_dim, K, L]
        # diffusion_step: 扩散步骤，形状为 [B] 或标量
        B, inputdim, K, L = x.shape

        # 输入投影前的reshape：将特征和序列维度展平
        # 从 [B, inputdim, K, L] 变为 [B, inputdim, K*L]
        x = x.reshape(B, inputdim, K * L)
        # 输入投影：1x1卷积，改变通道数
        x = self.input_projection(x)  # 输出: [B, channels, K*L]
        # ReLU激活函数
        x = F.relu(x)
        # 恢复原始形状：[B, channels, K, L]
        x = x.reshape(B, self.channels, K, L)

        # 获取扩散步骤的嵌入表示
        # diffusion_emb 形状: [B, projection_dim] 或 [projection_dim]
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # 通过所有残差层，收集跳跃连接
        skip = []  # 存储每个残差块的跳跃连接输出
        for layer in self.residual_layers:
            # 每个残差层返回两个值：
            # - 经过残差连接的主路径输出
            # - 跳跃连接输出（用于最终聚合）
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)  # 收集跳跃连接

        # 跳跃连接聚合：将所有跳跃连接求和并归一化
        # torch.stack(skip): 将列表中的张量堆叠，形状为 [num_layers, B, channels, K, L]
        # torch.sum(..., dim=0): 沿层数维度求和，形状为 [B, channels, K, L]
        # 除以 sqrt(num_layers): 归一化，保持数值稳定性
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        
        # 输出投影前的reshape：展平特征和序列维度
        x = x.reshape(B, self.channels, K * L)
        # 第一层输出投影
        x = self.output_projection1(x)  # (B, channels, K*L)
        # ReLU激活
        x = F.relu(x)
        # 第二层输出投影：映射到最终输出维度
        x = self.output_projection2(x)  # (B, 1, K*L)
        # 恢复目标形状：[B, K, L]
        x = x.reshape(B, K, L)
        return x

class ResidualBlock(nn.Module):
    """残差块：包含时间和特征两个维度的注意力机制"""
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        # 扩散嵌入投影：将扩散嵌入映射到通道维度
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # 条件信息投影：将条件信息映射到2倍通道数
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        # 中间投影：在注意力后进一步变换特征
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        # 输出投影：生成残差和跳跃连接
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # 是否使用线性注意力（计算效率更高）
        self.is_linear = is_linear
        # 根据配置选择注意力类型
        if is_linear:
            # 时间维度注意力层（线性注意力）
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            # 特征维度注意力层（线性注意力）
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            # 时间维度注意力层（标准Transformer）
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            # 特征维度注意力层（标准Transformer）
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """时间维度注意力：在序列长度维度应用自注意力"""
        # y: 输入张量，形状为 [B, channel, K*L]
        # base_shape: 原始形状 [B, channel, K, L]
        B, channel, K, L = base_shape
        
        # 如果序列长度L=1，不需要时间注意力，直接返回
        if L == 1:
            return y
            
        # 重塑张量以在时间维度应用注意力：
        # 1. 恢复原始形状: [B, channel, K, L]
        # 2. 置换维度: [B, K, channel, L] (将特征维度K放到批次位置)
        # 3. 重塑: [B*K, channel, L] (将批次和特征合并，形成虚拟批次)
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        # 应用时间注意力
        if self.is_linear:
            # 线性注意力期望输入形状: [batch, seq_len, features]
            # 所以需要 permute(0, 2, 1): [B*K, L, channel]
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # 输出: [B*K, channel, L]
        else:
            # 标准Transformer期望输入形状: [seq_len, batch, features]  
            # 所以需要 permute(2, 0, 1): [L, B*K, channel]
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # 输出: [B*K, channel, L]
            
        # 恢复原始形状：
        # 1. 重塑: [B, K, channel, L]
        # 2. 置换维度: [B, channel, K, L] (恢复通道优先)
        # 3. 重塑: [B, channel, K*L] (重新展平)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """特征维度注意力：在特征维度应用自注意力"""
        # y: 输入张量，形状为 [B, channel, K*L]
        # base_shape: 原始形状 [B, channel, K, L]
        B, channel, K, L = base_shape
        
        # 如果特征数K=1，不需要特征注意力，直接返回
        if K == 1:
            return y
            
        # 重塑张量以在特征维度应用注意力：
        # 1. 恢复原始形状: [B, channel, K, L]  
        # 2. 置换维度: [B, L, channel, K] (将序列维度L放到批次位置)
        # 3. 重塑: [B*L, channel, K] (将批次和序列合并，形成虚拟批次)
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)

        # 应用特征注意力
        if self.is_linear:
            # 线性注意力：输入 [B*L, K, channel]，输出 [B*L, K, channel]
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # 输出: [B*L, channel, K]
        else:
            # 标准Transformer：输入 [K, B*L, channel]，输出 [K, B*L, channel]
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # 输出: [B*L, channel, K]
            
        # 恢复原始形状：
        # 1. 重塑: [B, L, channel, K]
        # 2. 置换维度: [B, channel, K, L] (恢复通道优先)
        # 3. 重塑: [B, channel, K*L] (重新展平)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        # x: 输入张量，形状为 [B, channel, K, L]
        # cond_info: 条件信息，形状为 [B, side_dim, K, L]  
        # diffusion_emb: 扩散嵌入，形状为 [B, diffusion_embedding_dim]
        B, channel, K, L = x.shape
        base_shape = x.shape  # 保存原始形状 [B, channel, K, L]
        
        # 展平特征和序列维度：从 [B, channel, K, L] 到 [B, channel, K*L]
        x = x.reshape(B, channel, K * L)

        # 扩散嵌入投影：
        # 1. 线性变换: [B, diffusion_embedding_dim] -> [B, channel]
        # 2. 增加维度: [B, channel] -> [B, channel, 1] (为了广播加法)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        # 将扩散嵌入加到输入上：为每个时间步添加相同的扩散信息
        y = x + diffusion_emb  # 形状: [B, channel, K*L]

        # 分别应用时间和特征维度注意力
        y = self.forward_time(y, base_shape)      # 时间注意力
        y = self.forward_feature(y, base_shape)   # 特征注意力，输出: [B, channel, K*L]
        
        # 中间投影：将通道数翻倍
        y = self.mid_projection(y)  # (B, 2*channel, K*L)

        # 处理条件信息：
        # 获取条件信息的维度
        _, cond_dim, _, _ = cond_info.shape
        # 展平条件信息：从 [B, cond_dim, K, L] 到 [B, cond_dim, K*L]
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        # 条件信息投影：映射到2倍通道数
        cond_info = self.cond_projection(cond_info)  # (B, 2*channel, K*L)
        # 将条件信息加到主路径上
        y = y + cond_info

        # 门控机制：将2*channel拆分为门控和过滤两部分
        # torch.chunk(y, 2, dim=1): 沿通道维度分成2部分
        gate, filter = torch.chunk(y, 2, dim=1)  # 每个形状: [B, channel, K*L]
        # 门控激活：sigmoid(gate) * tanh(filter)
        # - sigmoid(gate): 控制信息流过的比例(0-1)
        # - tanh(filter): 非线性变换，值域(-1,1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # 输出: [B, channel, K*L]
        
        # 输出投影：再次将通道数翻倍
        y = self.output_projection(y)  # 输出: [B, 2*channel, K*L]

        # 分离残差连接和跳跃连接：
        # 将输出沿通道维度分成两部分
        residual, skip = torch.chunk(y, 2, dim=1)  # 每个形状: [B, channel, K*L]
        
        # 恢复原始形状：
        x = x.reshape(base_shape)          # [B, channel, K, L]
        residual = residual.reshape(base_shape)  # [B, channel, K, L]  
        skip = skip.reshape(base_shape)          # [B, channel, K, L]
        
        # 返回：
        # - 残差连接结果: (x + residual) / sqrt(2.0)  [归一化的残差连接]
        # - 跳跃连接: skip [用于最终聚合]
        return (x + residual) / math.sqrt(2.0), skip