"""v1 legacy — CSDI diffusion 模型主体（来自 ../csdi/main_model.py）。

此文件是从原始 CSDI 仓库复制过来的 v1 历史代码，用于参考和对比。
在 v2 流水线中，Module 1 (M1) 使用以下替代：
  - 生产用：methods/dynamics_impute.py（AR-Kalman smoother，无需训练）
  - 完整版：methods/dynamics_csdi.py（在此基础上加了噪声条件化 + 延迟 attention mask，Week-7 WIP）

此文件中的 CSDI_base / CSDI_Forecasting 包含原始的 dataset-specific 数据处理逻辑
（PM25、EEG、PhysioNet），不在 v2 实验中使用。
"""
import numpy as np
import torch
import torch.nn as nn
from csdi.diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        # 调用父类nn.Module的初始化方法
        self.device = device
        # 存储设备信息（CPU/GPU）
        self.target_dim = target_dim
        # 目标维度，通常是特征数量

        self.emb_time_dim = config["model"]["timeemb"]
        # 从配置中获取时间嵌入维度
        self.emb_feature_dim = config["model"]["featureemb"]
        # 从配置中获取特征嵌入维度
        self.is_unconditional = config["model"]["is_unconditional"]
        # 判断是否为无条件生成
        self.target_strategy = config["model"]["target_strategy"]
        # 目标策略（用于掩码生成）

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        # 计算总嵌入维度（时间+特征）
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
            # 对于条件模型，增加掩码维度
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        # 创建特征嵌入层

        config_diff = config["diffusion"]
        # 获取扩散模型配置
        config_diff["side_dim"] = self.emb_total_dim
        # 设置边信息维度

        input_dim = 1 if self.is_unconditional == True else 2
        # 输入维度：无条件为1，有条件为2
        self.diffmodel = diff_CSDI(config_diff, input_dim)
        # 创建扩散模型

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        # 扩散步数
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
            # 二次调度：生成beta序列
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
            # 线性调度：生成beta序列

        self.alpha_hat = 1 - self.beta
        # alpha_hat = 1 - beta
        self.alpha = np.cumprod(self.alpha_hat)
        # alpha的累积乘积
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        # 转换为torch tensor并调整形状为(B,1,1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        # 初始化位置编码矩阵
        position = pos.unsqueeze(2)
        # 增加维度用于广播
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        # 计算除数项
        pe[:, :, 0::2] = torch.sin(position * div_term)
        # 偶数位置使用正弦函数
        pe[:, :, 1::2] = torch.cos(position * div_term)
        # 奇数位置使用余弦函数
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        # 生成随机数并乘以观测掩码
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        # 重塑为2D张量
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            # 随机缺失比例
            num_observed = observed_mask[i].sum().item()
            # 观测到的数量
            num_masked = round(num_observed * sample_ratio)
            # 要掩码的数量
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            # 选择要掩码的位置（设置为-1）
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        # 生成条件掩码（大于0的位置为1）
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
            # 如果没有提供模式掩码，使用观测掩码
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
            # 生成随机掩码

        cond_mask = observed_mask.clone()
        # 克隆观测掩码
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            # 随机选择掩码策略
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
                # 使用随机掩码
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
                # 使用历史样本的掩码
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask
        # 返回观测掩码和测试模式掩码的乘积


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        # 获取批次大小、特征数、序列长度

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        # 生成时间嵌入
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        # 扩展维度以匹配特征数
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        # 生成特征嵌入
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        # 扩展维度以匹配批次和序列长度

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        # 合并时间和特征嵌入
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        # 调整维度顺序

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            # 增加掩码维度
            side_info = torch.cat([side_info, side_mask], dim=1)
            # 将掩码添加到边信息

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        # 初始化损失和
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            # 计算每个时间步的损失
            loss_sum += loss.detach()
            # 累加损失
        return loss_sum / self.num_steps
        # 返回平均损失

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        # 获取输入数据的形状
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
            # 验证阶段使用固定时间步
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
            # 训练阶段随机采样时间步
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        # 获取当前alpha值
        noise = torch.randn_like(observed_data)
        # 生成随机噪声
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        # 前向扩散过程：添加噪声

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        # 准备扩散模型输入

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        # 预测噪声

        target_mask = observed_mask - cond_mask
        # 计算目标区域掩码（需要预测的区域）
        residual = (noise - predicted) * target_mask
        # 计算噪声预测残差
        num_eval = target_mask.sum()
        # 计算目标区域元素数量
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        # 计算MSE损失
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
            # 无条件模型：只使用噪声数据
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            # 条件观测数据
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            # 噪声目标数据
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            # 连接观测数据和噪声目标数据

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        # 获取输入数据的形状

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        # 初始化插补结果张量

        for i in range(n_samples):
            # 生成多个样本
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                # 初始化噪声观测
                noisy_cond_history = []
                # 存储噪声观测历史
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    # 生成随机噪声
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    # 前向扩散过程
                    noisy_cond_history.append(noisy_obs * cond_mask)
                    # 保存条件区域噪声观测

            current_sample = torch.randn_like(observed_data)
            # 从随机噪声开始

            for t in range(self.num_steps - 1, -1, -1):
                # 反向扩散过程（从最后一步到第一步）
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    # 无条件模型：组合条件区域和当前样本
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    # 增加维度
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    # 条件观测数据
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    # 噪声目标数据
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    # 连接观测数据和目标数据
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                # 预测噪声

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                # 系数1
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                # 系数2
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                # 去噪步骤

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    # 生成随机噪声
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    # 计算噪声标准差
                    current_sample += sigma * noise
                    # 添加噪声

            imputed_samples[:, i] = current_sample.detach()
            # 保存当前样本
        return imputed_samples
    def forward(self, batch, is_train=1):
        # 模型的前向传播方法
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        # 处理输入批次数据，提取各个组件
        
        if is_train == 0:
            cond_mask = gt_mask
            # 如果是验证/测试阶段，使用真实的地面真值掩码
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
            # 如果目标策略不是随机，使用历史掩码策略
        else:
            cond_mask = self.get_randmask(observed_mask)
            # 否则使用随机掩码策略

        side_info = self.get_side_info(observed_tp, cond_mask)
        # 生成边信息（时间+特征嵌入）

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        # 选择损失函数：训练用calc_loss，验证用calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)
        # 计算并返回损失

    def evaluate(self, batch, n_samples):
        # 模型评估方法
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)
        # 处理输入批次数据，提取各个组件

        with torch.no_grad():
            # 禁用梯度计算，节省内存和计算资源
            cond_mask = gt_mask
            # 使用地面真值掩码作为条件掩码
            target_mask = observed_mask - cond_mask
            # 计算目标掩码（需要预测的区域）

            side_info = self.get_side_info(observed_tp, cond_mask)
            # 生成边信息

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            # 执行插补，生成n_samples个样本

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
                # 根据cut_length调整目标掩码，避免重复评估
        return samples, observed_data, target_mask, observed_mask, observed_tp
        # 返回插补样本、观测数据、目标掩码、观测掩码和时间点
   


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Lorenz(CSDI_base):
    def __init__(self, config, device, target_dim=15):
        super(CSDI_Lorenz, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

class CSDI_ENSO(CSDI_base):

    def __init__(self, config, device, target_dim=4):
        super(CSDI_ENSO, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

    
class CSDI_EEG(CSDI_base):
    def __init__(self, config, device, target_dim=64):
        super(CSDI_EEG, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # 从batch中提取数据并转移到指定设备
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        # 将数据从 (batch_size, seq_len, channels) 转换为 (batch_size, channels, seq_len)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        # EEG数据不需要截断，所以cut_length设为全0
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        
        # 使用观察到的掩码作为模式掩码
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Weather(CSDI_base):
    def __init__(self, config, device, target_dim=21):
        super(CSDI_Weather, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
