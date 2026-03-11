import numpy as np  # 导入numpy库用于数值计算
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
from diff_models import diff_CSDI  # 从diff_models模块导入diff_CSDI扩散模型


class CSDI_base(nn.Module):  # 定义CSDI基类，继承自nn.Module
    def __init__(self, target_dim, config, device):  # 初始化函数，接收目标维度、配置和设备参数
        super().__init__()  # 调用父类初始化函数
        self.device = device  # 设置计算设备（CPU或GPU）
        self.target_dim = target_dim  # 设置目标数据维度

        self.emb_time_dim = config["model"]["timeemb"]  # 从配置中获取时间嵌入维度
        self.emb_feature_dim = config["model"]["featureemb"]  # 从配置中获取特征嵌入维度
        self.is_unconditional = config["model"]["is_unconditional"]  # 获取是否为无条件模型的标志
        self.target_strategy = config["model"]["target_strategy"]  # 获取目标策略类型

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim  # 计算总嵌入维度（时间+特征）
        if self.is_unconditional == False:  # 如果是条件模型
            self.emb_total_dim += 1  # 为条件掩码增加1个维度
        self.embed_layer = nn.Embedding(  # 创建嵌入层
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim  # 设置嵌入数量和维度
        )

        config_diff = config["diffusion"]  # 获取扩散模型配置
        config_diff["side_dim"] = self.emb_total_dim  # 设置侧信息维度

        input_dim = 1 if self.is_unconditional == True else 2  # 根据是否条件模型设置输入维度
        self.diffmodel = diff_CSDI(config_diff, input_dim)  # 创建扩散模型实例

        # parameters for diffusion models  # 扩散模型参数
        self.num_steps = config_diff["num_steps"]  # 获取扩散步数
        if config_diff["schedule"] == "quad":  # 如果使用二次调度
            self.beta = np.linspace(  # 生成beta序列
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps  # 从开始到结束的平方根线性插值
            ) ** 2  # 再平方得到最终beta值
        elif config_diff["schedule"] == "linear":  # 如果使用线性调度
            self.beta = np.linspace(  # 生成beta序列
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps  # 线性插值
            )

        self.alpha_hat = 1 - self.beta  # 计算alpha_hat（1-beta）
        self.alpha = np.cumprod(self.alpha_hat)  # 计算alpha的累积乘积
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)  # 转换为torch张量并调整维度

    def time_embedding(self, pos, d_model=128):  # 时间位置编码函数
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)  # 初始化位置编码张量
        position = pos.unsqueeze(2)  # 增加一个维度用于广播
        div_term = 1 / torch.pow(  # 计算分母项
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model  # 使用正弦/余弦的频率公式
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin函数
        pe[:, :, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos函数
        return pe  # 返回位置编码

    def get_randmask(self, observed_mask):  # 生成随机掩码函数
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask  # 生成随机值并与观测掩码相乘
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)  # 将掩码展平
        for i in range(len(observed_mask)):  # 遍历每个样本
            sample_ratio = np.random.rand()  # 随机生成缺失比例
            num_observed = observed_mask[i].sum().item()  # 计算观测值数量
            num_masked = round(num_observed * sample_ratio)  # 计算要掩盖的数量
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1  # 将top-k个位置标记为-1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()  # 生成条件掩码（>0的位置）
        return cond_mask  # 返回条件掩码

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):  # 生成历史掩码函数
        if for_pattern_mask is None:  # 如果没有提供模式掩码
            for_pattern_mask = observed_mask  # 使用观测掩码
        if self.target_strategy == "mix":  # 如果策略是混合模式
            rand_mask = self.get_randmask(observed_mask)  # 生成随机掩码

        cond_mask = observed_mask.clone()  # 克隆观测掩码
        for i in range(len(cond_mask)):  # 遍历每个样本
            mask_choice = np.random.rand()  # 随机选择掩码类型
            if self.target_strategy == "mix" and mask_choice > 0.5:  # 如果是混合策略且随机数>0.5
                cond_mask[i] = rand_mask[i]  # 使用随机掩码
            else:  # 否则使用历史掩码（i-1对应另一个样本）
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]  # 与前一个样本的模式掩码相乘
        return cond_mask  # 返回条件掩码

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):  # 获取测试模式掩码函数
        return observed_mask * test_pattern_mask  # 返回观测掩码与测试模式掩码的乘积


    def get_side_info(self, observed_tp, cond_mask):  # 获取侧信息函数
        B, K, L = cond_mask.shape  # 获取批次大小B、特征数K、序列长度L

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # 生成时间嵌入 (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)  # 扩展时间嵌入到特征维度
        feature_embed = self.embed_layer(  # 生成特征嵌入
            torch.arange(self.target_dim).to(self.device)  # 创建特征索引 (K,emb)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # 扩展特征嵌入到批次和时间维度

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # 拼接时间和特征嵌入 (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # 调整维度顺序为 (B,*,K,L)

        if self.is_unconditional == False:  # 如果是条件模型
            side_mask = cond_mask.unsqueeze(1)  # 增加通道维度 (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)  # 将掩码拼接到侧信息

        return side_info  # 返回侧信息

    def calc_loss_valid(  # 计算验证损失函数
        self, observed_data, cond_mask, observed_mask, side_info, is_train  # 接收观测数据、条件掩码、观测掩码、侧信息和训练标志
    ):
        loss_sum = 0  # 初始化损失总和
        for t in range(self.num_steps):  # 遍历所有扩散步
            loss = self.calc_loss(  # 计算当前步的损失
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t  # 传入参数并指定时间步
            )
            loss_sum += loss.detach()  # 累加损失（分离梯度）
        return loss_sum / self.num_steps  # 返回平均损失

    def calc_loss(  # 计算损失函数
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1  # 接收参数，set_t默认为-1
    ):
        B, K, L = observed_data.shape  # 获取数据维度
        if is_train != 1:  # 如果是验证模式
            t = (torch.ones(B) * set_t).long().to(self.device)  # 使用指定的时间步
        else:  # 如果是训练模式
            t = torch.randint(0, self.num_steps, [B]).to(self.device)  # 随机采样时间步
        current_alpha = self.alpha_torch[t]  # 获取当前时间步的alpha值 (B,1,1)
        noise = torch.randn_like(observed_data)  # 生成高斯噪声
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise  # 添加噪声到数据

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)  # 准备扩散模型输入

        predicted = self.diffmodel(total_input, side_info, t)  # 扩散模型预测噪声 (B,K,L)

        target_mask = observed_mask - cond_mask  # 计算目标掩码（需要预测的位置）
        residual = (noise - predicted) * target_mask  # 计算残差并应用目标掩码
        num_eval = target_mask.sum()  # 计算评估点数量
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)  # 计算均方误差损失
        return loss  # 返回损失值

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):  # 设置扩散模型输入函数
        if self.is_unconditional == True:  # 如果是无条件模型
            total_input = noisy_data.unsqueeze(1)  # 直接使用噪声数据 (B,1,K,L)
        else:  # 如果是条件模型
            cond_obs = (cond_mask * observed_data).unsqueeze(1)  # 提取条件观测值
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)  # 提取噪声目标值
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # 拼接为双通道输入 (B,2,K,L)

        return total_input  # 返回模型输入

    def impute(self, observed_data, cond_mask, side_info, n_samples):  # 插补函数
        B, K, L = observed_data.shape  # 获取数据维度

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)  # 初始化插补样本张量

        for i in range(n_samples):  # 对每个样本进行采样
            # generate noisy observation for unconditional model  # 为无条件模型生成噪声观测
            if self.is_unconditional == True:  # 如果是无条件模型
                noisy_obs = observed_data  # 初始化噪声观测
                noisy_cond_history = []  # 初始化噪声条件历史列表
                for t in range(self.num_steps):  # 前向扩散过程
                    noise = torch.randn_like(noisy_obs)  # 生成噪声
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise  # 添加噪声
                    noisy_cond_history.append(noisy_obs * cond_mask)  # 保存加噪的条件部分

            current_sample = torch.randn_like(observed_data)  # 从纯噪声开始

            for t in range(self.num_steps - 1, -1, -1):  # 反向去噪过程（从T-1到0）
                if self.is_unconditional == True:  # 如果是无条件模型
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample  # 组合条件和当前样本
                    diff_input = diff_input.unsqueeze(1)  # 增加通道维度 (B,1,K,L)
                else:  # 如果是条件模型
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)  # 提取条件观测
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)  # 提取噪声目标
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # 拼接输入 (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))  # 预测噪声

                coeff1 = 1 / self.alpha_hat[t] ** 0.5  # 计算去噪系数1
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5  # 计算去噪系数2
                current_sample = coeff1 * (current_sample - coeff2 * predicted)  # 去噪更新

                if t > 0:  # 如果不是最后一步
                    noise = torch.randn_like(current_sample)  # 生成噪声
                    sigma = (  # 计算标准差
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]  # sigma公式
                    ) ** 0.5
                    current_sample += sigma * noise  # 添加随机性

            imputed_samples[:, i] = current_sample.detach()  # 保存插补结果
        return imputed_samples  # 返回所有插补样本

    def forward(self, batch, is_train=1):  # 前向传播函数
        (  # 解包处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            for_pattern_mask,  # 模式掩码
            _,  # 忽略的参数
        ) = self.process_data(batch)  # 处理批次数据
        if is_train == 0:  # 如果是验证模式
            cond_mask = gt_mask  # 使用真实掩码作为条件
        elif self.target_strategy != "random":  # 如果不是随机策略
            cond_mask = self.get_hist_mask(  # 获取历史掩码
                observed_mask, for_pattern_mask=for_pattern_mask  # 传入观测掩码和模式掩码
            )
        else:  # 如果是随机策略
            cond_mask = self.get_randmask(observed_mask)  # 获取随机掩码

        side_info = self.get_side_info(observed_tp, cond_mask)  # 获取侧信息

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid  # 根据训练模式选择损失函数

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)  # 计算并返回损失

    def evaluate(self, batch, n_samples):  # 评估函数
        (  # 解包处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            _,  # 忽略的参数
            cut_length,  # 截断长度
        ) = self.process_data(batch)  # 处理批次数据

        with torch.no_grad():  # 禁用梯度计算
            cond_mask = gt_mask  # 使用真实掩码作为条件
            target_mask = observed_mask - cond_mask  # 计算目标掩码

            side_info = self.get_side_info(observed_tp, cond_mask)  # 获取侧信息

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)  # 进行插补

            for i in range(len(cut_length)):  # 遍历批次中的样本
                target_mask[i, ..., 0 : cut_length[i].item()] = 0  # 将截断部分的目标掩码置零（避免重复评估）
        return samples, observed_data, target_mask, observed_mask, observed_tp  # 返回评估结果


class CSDI_PM25(CSDI_base):  # PM2.5数据的CSDI类
    def __init__(self, config, device, target_dim=36):  # 初始化函数，默认36个特征
        super(CSDI_PM25, self).__init__(target_dim, config, device)  # 调用父类初始化

    def process_data(self, batch):  # 数据处理函数
        observed_data = batch["observed_data"].to(self.device).float()  # 将观测数据移到设备并转为浮点型
        observed_mask = batch["observed_mask"].to(self.device).float()  # 将观测掩码移到设备并转为浮点型
        observed_tp = batch["timepoints"].to(self.device).float()  # 将时间点移到设备并转为浮点型
        gt_mask = batch["gt_mask"].to(self.device).float()  # 将真实掩码移到设备并转为浮点型
        cut_length = batch["cut_length"].to(self.device).long()  # 将截断长度移到设备并转为长整型
        for_pattern_mask = batch["hist_mask"].to(self.device).float()  # 将历史掩码移到设备并转为浮点型

        observed_data = observed_data.permute(0, 2, 1)  # 调整维度顺序：(B,L,K) -> (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)  # 调整掩码维度顺序
        gt_mask = gt_mask.permute(0, 2, 1)  # 调整真实掩码维度顺序
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)  # 调整模式掩码维度顺序

        return (  # 返回处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            for_pattern_mask,  # 模式掩码
            cut_length,  # 截断长度
        )


class CSDI_Lorenz(CSDI_base):  # Lorenz系统数据的CSDI类
    def __init__(self, config, device, target_dim=15):  # 初始化函数，默认15个特征
        super(CSDI_Lorenz, self).__init__(target_dim, config, device)  # 调用父类初始化

    def process_data(self, batch):  # 数据处理函数
        observed_data = batch["observed_data"].to(self.device).float()  # 将观测数据移到设备并转为浮点型
        observed_mask = batch["observed_mask"].to(self.device).float()  # 将观测掩码移到设备并转为浮点型
        observed_tp = batch["timepoints"].to(self.device).float()  # 将时间点移到设备并转为浮点型
        gt_mask = batch["gt_mask"].to(self.device).float()  # 将真实掩码移到设备并转为浮点型

        observed_data = observed_data.permute(0, 2, 1)  # 调整维度顺序：(B,L,K) -> (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)  # 调整掩码维度顺序
        gt_mask = gt_mask.permute(0, 2, 1)  # 调整真实掩码维度顺序

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)  # 创建全零截断长度张量
        for_pattern_mask = observed_mask  # 使用观测掩码作为模式掩码

        return (  # 返回处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            for_pattern_mask,  # 模式掩码
            cut_length,  # 截断长度
        )


class CSDI_Physio(CSDI_base):  # 生理数据的CSDI类
    def __init__(self, config, device, target_dim=35):  # 初始化函数，默认35个特征
        super(CSDI_Physio, self).__init__(target_dim, config, device)  # 调用父类初始化

    def process_data(self, batch):  # 数据处理函数
        observed_data = batch["observed_data"].to(self.device).float()  # 将观测数据移到设备并转为浮点型
        observed_mask = batch["observed_mask"].to(self.device).float()  # 将观测掩码移到设备并转为浮点型
        observed_tp = batch["timepoints"].to(self.device).float()  # 将时间点移到设备并转为浮点型
        gt_mask = batch["gt_mask"].to(self.device).float()  # 将真实掩码移到设备并转为浮点型

        observed_data = observed_data.permute(0, 2, 1)  # 调整维度顺序：(B,L,K) -> (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)  # 调整掩码维度顺序
        gt_mask = gt_mask.permute(0, 2, 1)  # 调整真实掩码维度顺序

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)  # 创建全零截断长度张量
        for_pattern_mask = observed_mask  # 使用观测掩码作为模式掩码

        return (  # 返回处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            for_pattern_mask,  # 模式掩码
            cut_length,  # 截断长度
        )



class CSDI_Forecasting(CSDI_base):  # 预测任务的CSDI类
    def __init__(self, config, device, target_dim):  # 初始化函数
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)  # 调用父类初始化
        self.target_dim_base = target_dim  # 保存基础目标维度
        self.num_sample_features = config["model"]["num_sample_features"]  # 获取采样特征数量

    def process_data(self, batch):  # 数据处理函数
        observed_data = batch["observed_data"].to(self.device).float()  # 将观测数据移到设备并转为浮点型
        observed_mask = batch["observed_mask"].to(self.device).float()  # 将观测掩码移到设备并转为浮点型
        observed_tp = batch["timepoints"].to(self.device).float()  # 将时间点移到设备并转为浮点型
        gt_mask = batch["gt_mask"].to(self.device).float()  # 将真实掩码移到设备并转为浮点型

        observed_data = observed_data.permute(0, 2, 1)  # 调整维度顺序：(B,L,K) -> (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)  # 调整掩码维度顺序
        gt_mask = gt_mask.permute(0, 2, 1)  # 调整真实掩码维度顺序

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)  # 创建全零截断长度张量
        for_pattern_mask = observed_mask  # 使用观测掩码作为模式掩码

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)  # 创建特征ID张量

        return (  # 返回处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            for_pattern_mask,  # 模式掩码
            cut_length,  # 截断长度
            feature_id,  # 特征ID
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):  # 特征采样函数
        size = self.num_sample_features  # 获取采样数量
        self.target_dim = size  # 更新目标维度
        extracted_data = []  # 初始化提取数据列表
        extracted_mask = []  # 初始化提取掩码列表
        extracted_feature_id = []  # 初始化提取特征ID列表
        extracted_gt_mask = []  # 初始化提取真实掩码列表
        
        for k in range(len(observed_data)):  # 遍历批次中的样本
            ind = np.arange(self.target_dim_base)  # 创建特征索引数组
            np.random.shuffle(ind)  # 随机打乱索引
            extracted_data.append(observed_data[k,ind[:size]])  # 提取前size个特征的数据
            extracted_mask.append(observed_mask[k,ind[:size]])  # 提取对应的掩码
            extracted_feature_id.append(feature_id[k,ind[:size]])  # 提取对应的特征ID
            extracted_gt_mask.append(gt_mask[k,ind[:size]])  # 提取对应的真实掩码
        extracted_data = torch.stack(extracted_data,0)  # 堆叠数据张量
        extracted_mask = torch.stack(extracted_mask,0)  # 堆叠掩码张量
        extracted_feature_id = torch.stack(extracted_feature_id,0)  # 堆叠特征ID张量
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)  # 堆叠真实掩码张量
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask  # 返回提取的数据


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):  # 获取侧信息函数（重载）
        B, K, L = cond_mask.shape  # 获取批次大小B、特征数K、序列长度L

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # 生成时间嵌入 (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)  # 扩展时间嵌入到当前目标维度

        if self.target_dim == self.target_dim_base:  # 如果目标维度等于基础维度
            feature_embed = self.embed_layer(  # 生成标准特征嵌入
                torch.arange(self.target_dim).to(self.device)  # 创建特征索引 (K,emb)
            )
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # 扩展到批次和时间维度
        else:  # 如果使用采样特征
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)  # 根据特征ID生成嵌入
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # 拼接时间和特征嵌入 (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # 调整维度顺序为 (B,*,K,L)

        if self.is_unconditional == False:  # 如果是条件模型
            side_mask = cond_mask.unsqueeze(1)  # 增加通道维度 (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)  # 将掩码拼接到侧信息

        return side_info  # 返回侧信息

    def forward(self, batch, is_train=1):  # 前向传播函数（重载）
        (  # 解包处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            _,  # 忽略的参数
            _,  # 忽略的参数
            feature_id,  # 特征ID
        ) = self.process_data(batch)  # 处理批次数据
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):  # 如果是训练且需要采样特征
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)  # 特征采样、调用采样函数
        else:  # 如果不需要采样
            self.target_dim = self.target_dim_base  # 使用基础维度
            feature_id = None  # 特征ID置空

        if is_train == 0:  # 如果是验证模式
            cond_mask = gt_mask  # 使用真实掩码
        else: #test pattern  # 测试模式
            cond_mask = self.get_test_pattern_mask(  # 获取测试模式掩码
                observed_mask, gt_mask  # 传入观测掩码和真实掩码
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)  # 获取侧信息

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid  # 根据训练模式选择损失函数

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)  # 计算并返回损失



    def evaluate(self, batch, n_samples):  # 评估函数（重载）
        (  # 解包处理后的数据
            observed_data,  # 观测数据
            observed_mask,  # 观测掩码
            observed_tp,  # 观测时间点
            gt_mask,  # 真实掩码
            _,  # 忽略的参数
            _,  # 忽略的参数
            feature_id,  # 特征ID
        ) = self.process_data(batch)  # 处理批次数据

        with torch.no_grad():  # 禁用梯度计算
            cond_mask = gt_mask  # 使用真实掩码作为条件
            target_mask = observed_mask * (1-gt_mask)  # 计算目标掩码（预测位置）

            side_info = self.get_side_info(observed_tp, cond_mask)  # 获取侧信息

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)  # 进行插补

        return samples, observed_data, target_mask, observed_mask, observed_tp  # 返回评估结果
