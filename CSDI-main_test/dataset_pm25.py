# 导入pickle库，用于序列化和反序列化Python对象
import pickle
# 从PyTorch导入数据加载器和数据集基类
from torch.utils.data import DataLoader, Dataset
# 导入pandas库，用于数据处理
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入PyTorch深度学习框架
import torch


# 定义PM2.5数据集类，继承自PyTorch的Dataset基类
class PM25_Dataset(Dataset):
    # 初始化函数，eval_length为评估序列长度，target_dim为目标维度，mode为数据集模式，validindex为验证集索引
    def __init__(self, eval_length=36, target_dim=36, mode="train", validindex=0):
        # 保存评估序列长度
        self.eval_length = eval_length
        # 保存目标维度
        self.target_dim = target_dim

        # 定义训练数据的均值和标准差文件路径
        path = "./data/pm25/pm25_meanstd.pk"
        # 以二进制读模式打开文件
        with open(path, "rb") as f:
            # 加载训练数据的均值和标准差
            self.train_mean, self.train_std = pickle.load(f)
        # 如果是训练模式
        if mode == "train":
            # 定义用于训练的月份列表（1,2,4,5,7,8,10,11月）
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            # 第1、4、7、10月被排除在histmask之外（因为这些月份用于创建测试集的缺失模式）
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1] 
            # 从月份列表中移除用于验证的月份
            month_list.pop(validindex)
            # 从histmask标志列表中移除对应的标志
            flag_for_histmask.pop(validindex)
        # 如果是验证模式
        elif mode == "valid":
            # 定义完整的月份列表
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            # 只保留用于验证的那一个月
            month_list = month_list[validindex : validindex + 1]
        # 如果是测试模式
        elif mode == "test":
            # 使用3、6、9、12月作为测试数据
            month_list = [3, 6, 9, 12]
        # 保存月份列表
        self.month_list = month_list

        # 创建批次数据的各种列表
        self.observed_data = []  # 观测值（按月份分隔）
        self.observed_mask = []  # 观测掩码（按月份分隔）
        self.gt_mask = []  # 真实掩码（按月份分隔）
        self.index_month = []  # 月份索引
        self.position_in_month = []  # 月内起始位置（长度与index_month相同）
        self.valid_for_histmask = []  # 样本是否用于histmask
        self.use_index = []  # 用于分离训练/验证/测试集
        self.cut_length = []  # 从评估目标中排除的长度

        # 读取PM2.5真实数据CSV文件，以datetime列为索引并解析日期
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        # 读取PM2.5缺失数据CSV文件，以datetime列为索引并解析日期
        df_gt = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )
        # 遍历所有月份
        for i in range(len(month_list)):
            # 提取当前月份的数据
            current_df = df[df.index.month == month_list[i]]
            # 提取当前月份的真实掩码数据
            current_df_gt = df_gt[df_gt.index.month == month_list[i]]
            # 计算当前月份可用的样本数量（考虑eval_length的滑动窗口）
            current_length = len(current_df) - eval_length + 1

            # 记录当前index_month列表的长度
            last_index = len(self.index_month)
            # 为当前月份添加月份索引（每个样本都标记为当前月份i）
            self.index_month += np.array([i] * current_length).tolist()
            # 添加月内位置索引（从0到current_length-1）
            self.position_in_month += np.arange(current_length).tolist()
            # 如果是训练模式
            if mode == "train":
                # 为当前月份的所有样本添加histmask有效性标志
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # 创建观测掩码：观测值处为1，缺失值处为0
            c_mask = 1 - current_df.isnull().values
            # 创建真实掩码：真实观测值处为1，缺失值处为0
            c_gt_mask = 1 - current_df_gt.isnull().values
            # 标准化数据：用0填充缺失值，然后进行z-score标准化，最后乘以掩码
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask

            # 添加观测掩码到列表
            self.observed_mask.append(c_mask)
            # 添加真实掩码到列表
            self.gt_mask.append(c_gt_mask)
            # 添加标准化后的观测数据到列表
            self.observed_data.append(c_data)

            # 如果是测试模式
            if mode == "test":
                # 计算样本数量（每个样本长度为eval_length）
                n_sample = len(current_df) // eval_length
                # 创建索引，间隔大小为eval_length（缺失值只插补一次）
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                # 添加索引到使用索引列表
                self.use_index += c_index.tolist()
                # 所有完整样本的切割长度为0
                self.cut_length += [0] * len(c_index)
                # 如果最后有不完整的时间序列（长度不是eval_length的整数倍）
                if len(current_df) % eval_length != 0:  # 避免对最后一个时间序列重复计数
                    # 添加最后一个样本的索引
                    self.use_index += [len(self.index_month) - 1]
                    # 计算需要切割的长度（最后不完整部分）
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        # 如果不是测试模式
        if mode != "test":
            # 使用所有索引
            self.use_index = np.arange(len(self.index_month))
            # 所有样本的切割长度都为0
            self.cut_length = [0] * len(self.use_index)

        # 第1、4、7、10月的掩码用于创建测试数据的缺失模式，
        # 因此这些月份被排除在histmask之外以避免数据泄露
        if mode == "train":
            # 初始化索引为-1
            ind = -1
            # 初始化histmask的月份索引列表
            self.index_month_histmask = []
            # 初始化histmask的月内位置列表
            self.position_in_month_histmask = []

            # 遍历所有样本
            for i in range(len(self.index_month)):
                # 循环查找下一个有效的histmask样本
                while True:
                    # 索引递增
                    ind += 1
                    # 如果索引超出范围，重置为0（循环）
                    if ind == len(self.index_month):
                        ind = 0
                    # 如果当前样本有效用于histmask
                    if self.valid_for_histmask[ind] == 1:
                        # 添加月份索引
                        self.index_month_histmask.append(self.index_month[ind])
                        # 添加月内位置
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        # 跳出循环
                        break
        # 如果不是训练模式
        else:  # 虚拟值（histmask仅用于训练）
            # 使用原始月份索引
            self.index_month_histmask = self.index_month
            # 使用原始月内位置
            self.position_in_month_histmask = self.position_in_month

    # 获取单个样本的方法
    def __getitem__(self, org_index):
        # 根据原始索引获取实际使用的索引
        index = self.use_index[org_index]
        # 获取当前样本的月份索引
        c_month = self.index_month[index]
        # 获取当前样本在月内的位置
        c_index = self.position_in_month[index]
        # 获取histmask对应的月份索引
        hist_month = self.index_month_histmask[index]
        # 获取histmask对应的月内位置
        hist_index = self.position_in_month_histmask[index]
        # 构建样本字典
        s = {
            # 提取长度为eval_length的观测数据
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length
            ],
            # 提取长度为eval_length的观测掩码
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            # 提取长度为eval_length的真实掩码
            "gt_mask": self.gt_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            # 提取长度为eval_length的历史掩码
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length
            ],
            # 生成时间点序列（0到eval_length-1）
            "timepoints": np.arange(self.eval_length),
            # 获取需要切割的长度
            "cut_length": self.cut_length[org_index],
        }

        # 返回样本字典
        return s

    # 返回数据集长度的方法
    def __len__(self):
        # 返回使用索引的数量
        return len(self.use_index)


# 获取数据加载器的函数
def get_dataloader(batch_size, device, validindex=0):
    # 创建训练数据集
    dataset = PM25_Dataset(mode="train", validindex=validindex)
    # 创建训练数据加载器，启用数据打乱
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )
    # 创建测试数据集
    dataset_test = PM25_Dataset(mode="test", validindex=validindex)
    # 创建测试数据加载器，不打乱数据
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False
    )
    # 创建验证数据集
    dataset_valid = PM25_Dataset(mode="valid", validindex=validindex)
    # 创建验证数据加载器，不打乱数据
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )

    # 将标准差转换为PyTorch张量并移至指定设备，数据类型为float
    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    # 将均值转换为PyTorch张量并移至指定设备，数据类型为float
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    # 返回训练、验证、测试数据加载器以及标准化器
    return train_loader, valid_loader, test_loader, scaler, mean_scaler
