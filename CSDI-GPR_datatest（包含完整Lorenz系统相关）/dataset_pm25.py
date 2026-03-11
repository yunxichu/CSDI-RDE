import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
from torch.utils.data import DataLoader, Dataset  # 从PyTorch导入数据加载器和数据集基类
import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习


class PM25_Dataset(Dataset):  # 定义一个处理PM2.5数据的数据集类，继承自Dataset
    def __init__(self, eval_length=36, target_dim=36, mode="train", validindex=0):  # 初始化方法
        self.eval_length = eval_length  # 设置评估长度（时间步长）
        self.target_dim = target_dim  # 设置目标维度

        path = "./data/pm25/pm25_meanstd.pk"  # 预计算的均值和标准差文件路径
        with open(path, "rb") as f:  # 以二进制读取模式打开文件
            self.train_mean, self.train_std = pickle.load(f)  # 加载均值和标准差
        
        # 根据模式选择不同的月份
        if mode == "train":  # 如果是训练模式
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]  # 训练使用的月份列表
            # 1月、4月、7月、10月被排除在histmask之外（因为这些月份用于创建测试数据集中的缺失模式）
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1]  # 标记哪些月份可用于histmask
            month_list.pop(validindex)  # 移除验证索引对应的月份（用于交叉验证）
            flag_for_histmask.pop(validindex)  # 同时移除对应的histmask标记
        elif mode == "valid":  # 如果是验证模式
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]  # 验证使用的月份列表
            month_list = month_list[validindex : validindex + 1]  # 只选择一个月份作为验证集
        elif mode == "test":  # 如果是测试模式
            month_list = [3, 6, 9, 12]  # 测试使用的月份列表（与训练/验证不同）
        self.month_list = month_list  # 存储选定的月份列表

        # 为批次创建数据
        self.observed_data = []  # 观测值（按月份分开存储）
        self.observed_mask = []  # 观测掩码（按月份分开存储）
        self.gt_mask = []  # 真实值掩码（按月份分开存储）
        self.index_month = []  # 指示月份索引
        self.position_in_month = []  # 指示在月份中的起始位置（长度与index_month相同）
        self.valid_for_histmask = []  # 样本是否用于histmask
        self.use_index = []  # 用于区分训练/验证/测试的索引
        self.cut_length = []  # 从评估目标中排除的长度

        # 读取完整数据（真实值）
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",  # 将datetime列作为索引
            parse_dates=True,  # 解析日期
        )
        # 读取包含缺失值的数据
        df_gt = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
            index_col="datetime",  # 将datetime列作为索引
            parse_dates=True,  # 解析日期
        )
        
        # 处理每个选定的月份
        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]]  # 过滤出当前月份的数据
            current_df_gt = df_gt[df_gt.index.month == month_list[i]]  # 过滤出当前月份的缺失数据
            current_length = len(current_df) - eval_length + 1  # 计算当前月份的有效序列长度

            last_index = len(self.index_month)  # 记录当前索引位置
            self.index_month += np.array([i] * current_length).tolist()  # 添加月份索引
            self.position_in_month += np.arange(current_length).tolist()  # 添加在月份中的位置
            if mode == "train":  # 如果是训练模式
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()  # 添加histmask有效性标记

            # 为观测索引创建掩码（观测值为1）
            c_mask = 1 - current_df.isnull().values  # 创建观测掩码（非空值为1）
            c_gt_mask = 1 - current_df_gt.isnull().values  # 创建真实值掩码
            # 标准化数据：减去均值，除以标准差，并应用掩码
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask

            self.observed_mask.append(c_mask)  # 添加观测掩码
            self.gt_mask.append(c_gt_mask)  # 添加真实值掩码
            self.observed_data.append(c_data)  # 添加标准化后的观测数据

            if mode == "test":  # 如果是测试模式
                n_sample = len(current_df) // eval_length  # 计算样本数量
                # 间隔大小为eval_length（缺失值只被估算一次）
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )  # 创建测试索引
                self.use_index += c_index.tolist()  # 添加使用索引
                self.cut_length += [0] * len(c_index)  # 添加截断长度（初始为0）
                # 避免最后一个时间序列被重复计算
                if len(current_df) % eval_length != 0:  
                    self.use_index += [len(self.index_month) - 1]  # 添加最后一个索引
                    # 计算需要截断的长度
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        # 对于非测试模式，使用所有索引
        if mode != "test":
            self.use_index = np.arange(len(self.index_month))  # 使用所有索引
            self.cut_length = [0] * len(self.use_index)  # 截断长度全部为0

        # 1月、4月、7月、10月的掩码用于创建测试数据中的缺失模式，
        # 因此这些月份从histmask中排除，以避免数据泄露
        if mode == "train":  # 如果是训练模式
            ind = -1  # 初始化索引
            self.index_month_histmask = []  # 初始化histmask的月份索引
            self.position_in_month_histmask = []  # 初始化histmask的位置索引

            # 为每个样本找到有效的histmask索引
            for i in range(len(self.index_month)):
                while True:
                    ind += 1  # 递增索引
                    if ind == len(self.index_month):  # 如果到达末尾
                        ind = 0  # 重置为0
                    if self.valid_for_histmask[ind] == 1:  # 如果索引有效
                        self.index_month_histmask.append(self.index_month[ind])  # 添加月份索引
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]  # 添加位置索引
                        )
                        break  # 退出循环
        else:  # 对于验证和测试模式，使用虚拟值（histmask仅用于训练）
            self.index_month_histmask = self.index_month  # 使用相同的月份索引
            self.position_in_month_histmask = self.position_in_month  # 使用相同的位置索引

    def __getitem__(self, org_index):  # 获取单个样本的方法
        index = self.use_index[org_index]  # 获取实际索引
        c_month = self.index_month[index]  # 获取月份索引
        c_index = self.position_in_month[index]  # 获取在月份中的位置
        hist_month = self.index_month_histmask[index]  # 获取histmask的月份索引
        hist_index = self.position_in_month_histmask[index]  # 获取histmask的位置索引
        
        # 创建样本字典
        s = {
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length  # 获取观测数据片段
            ],
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length  # 获取观测掩码片段
            ],
            "gt_mask": self.gt_mask[c_month][
                c_index : c_index + self.eval_length  # 获取真实值掩码片段
            ],
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length  # 获取历史掩码片段
            ],
            "timepoints": np.arange(self.eval_length),  # 创建时间点数组
            "cut_length": self.cut_length[org_index],  # 获取截断长度
        }

        return s  # 返回样本字典

    def __len__(self):  # 返回数据集长度的方法
        return len(self.use_index)  # 返回使用索引的长度


def get_dataloader(batch_size, device, validindex=0):  # 创建数据加载器的函数
    # 创建训练数据集
    dataset = PM25_Dataset(mode="train", validindex=validindex)
    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True  # 设置批次大小、工作线程数和是否打乱
    )
    # 创建测试数据集
    dataset_test = PM25_Dataset(mode="test", validindex=validindex)
    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False  # 测试集不打乱
    )
    # 创建验证数据集
    dataset_valid = PM25_Dataset(mode="valid", validindex=validindex)
    # 创建验证数据加载器
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False  # 验证集不打乱
    )

    # 将标准化参数转换为PyTorch张量并移动到指定设备
    scaler = torch.from_numpy(dataset.train_std).to(device).float()  # 标准差
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()  # 均值

    # 返回所有数据加载器和标准化参数
    return train_loader, valid_loader, test_loader, scaler, mean_scaler