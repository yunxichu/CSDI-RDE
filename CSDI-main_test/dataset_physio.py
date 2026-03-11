#dataset_physio.py
import pickle  # 用于对象序列化/反序列化（保存和加载处理后的数据）
import os  # 导入os模块用于文件路径和系统操作
import re  # 导入re模块用于正则表达式（匹配文件名中的ID）
import numpy as np  # 导入numpy用于数值矩阵计算
import pandas as pd  # 导入pandas用于CSV读取和DataFrame操作
from torch.utils.data import DataLoader, Dataset  # 从PyTorch导入数据集和加载器基类

# 35个包含足够多缺失值的生理属性列表
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']

def extract_hour(x):
    """从时间字符串中提取小时数"""
    h, _ = map(int, x.split(":"))  # 分割时间字符串"HH:MM"，取HH并转为整数
    return h  # 返回小时数

def parse_data(x):
    """解析单个时间点的生理数据"""
    # 将DataFrame转换为字典 {参数名: 数值}
    x = x.set_index("Parameter").to_dict()["Value"]  # 设置Parameter列为索引并取Value列
    
    values = []  # 初始化数值列表
    # 按预定义属性顺序提取值
    for attr in attributes:  # 遍历标准属性列表
        if attr in x:  # 如果当前时间点包含该属性
            values.append(x[attr])  # 添加属性值
        else:  # 如果不包含
            values.append(np.nan)  # 填充NaN（缺失值）
    return values  # 返回对齐后的属性值列表

def parse_id(id_, missing_ratio=0.1):
    """处理单个患者的数据文件"""
    # 读取患者数据文件
    data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))  # 根据ID读取TXT/CSV文件
    # 提取时间中的小时
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))  # 将时间列转换为小时整数
    
    # 创建48小时x35属性的矩阵
    observed_values = []  # 初始化观测值容器
    for h in range(48):  # 遍历0到47小时
        # 获取该小时所有记录并解析
        hour_data = data[data["Time"] == h]  # 筛选当前小时的数据行
        observed_values.append(parse_data(hour_data))  # 解析该小时数据并添加到列表
    observed_values = np.array(observed_values)  # 转换为numpy数组 (48, 35)
    
    # 创建掩码标识有效值
    observed_masks = ~np.isnan(observed_values)  # 非NaN的位置标记为True
    
    # 随机选择部分有效值作为缺失（用于模型训练/验证的输入）
    masks = observed_masks.reshape(-1).copy()  # 展平掩码以便进行随机采样
    obs_indices = np.where(masks)[0].tolist()  # 获取所有实际存在数据的索引
    # 随机选择部分索引作为缺失
    miss_indices = np.random.choice(  # 随机抽样
        obs_indices, int(len(obs_indices) * missing_ratio), replace=False  # 计算需要掩盖的数量
    )
    masks[miss_indices] = False  # 将选中的位置设为False（人为制造缺失）
    gt_masks = masks.reshape(observed_masks.shape)  # 恢复形状，这实际上是"Input Mask"
    
    # 数据预处理
    observed_values = np.nan_to_num(observed_values)  # 将NaN替换为0，以便转为Tensor
    observed_masks = observed_masks.astype("float32")  # 转换为float32类型
    gt_masks = gt_masks.astype("float32")  # 转换为float32类型
    
    return observed_values, observed_masks, gt_masks  # 返回数据、原始可用掩码、模拟输入掩码

def get_idlist():
    """获取所有患者ID列表"""
    patient_id = []  # 初始化ID列表
    # 遍历数据目录
    for filename in os.listdir("./data/physio/set-a"):  # 列出目录下所有文件
        match = re.search("\d{6}", filename)  # 正则匹配6位数字ID
        if match:  # 如果匹配到ID
            patient_id.append(match.group())  # 添加到列表
    return np.sort(patient_id)  # 返回排序后的ID列表

class Physio_Dataset(Dataset):
    """生理学数据集类"""
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length  # 设置评估时间长度（默认48小时）
        np.random.seed(seed)  # 设置随机种子以保证复现性
        
        # 初始化数据存储
        self.observed_values = []  # 存放所有患者的观测值
        self.observed_masks = []  # 存放所有患者的原始掩码
        self.gt_masks = []  # 存放所有患者的输入掩码
        
        # 数据缓存文件路径
        path = f"./data/physio_missing{missing_ratio}_seed{seed}.pk"  # 根据缺失率和种子命名缓存
        
        if not os.path.isfile(path):  # 如果缓存文件不存在，则处理原始数据
            idlist = get_idlist()  # 获取所有患者ID
            for id_ in idlist:  # 遍历每个ID
                try:
                    # 解析患者数据
                    observed_values, observed_masks, gt_masks = parse_id(  # 调用解析函数
                        id_, missing_ratio
                    )
                    self.observed_values.append(observed_values)  # 收集数据
                    self.observed_masks.append(observed_masks)  # 收集掩码
                    self.gt_masks.append(gt_masks)  # 收集输入掩码
                except Exception as e:  # 异常处理
                    print(id_, e)  # 打印出错的ID和信息
                    continue
            # 转换为numpy数组
            self.observed_values = np.array(self.observed_values)  # 转为大数组 (N, 48, 35)
            self.observed_masks = np.array(self.observed_masks)  # 转为大数组
            self.gt_masks = np.array(self.gt_masks)  # 转为大数组
            
            # 数据标准化 (Z-score normalization)
            tmp_values = self.observed_values.reshape(-1, 35)  # 展平所有数据
            tmp_masks = self.observed_masks.reshape(-1, 35)  # 展平掩码
            mean = np.zeros(35)  # 初始化均值向量
            std = np.zeros(35)   # 初始化标准差向量
            for k in range(35):  # 对每个属性分别计算
                # 获取该属性所有存在的有效数据（过滤掉NaN/0填充值）
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()  # 计算有效数据的均值
                std[k] = c_data.std()   # 计算有效数据的标准差
            # 标准化数据，并再次应用掩码确保无效区域仍为0
            self.observed_values = (self.observed_values - mean) / std * self.observed_masks
            
            # 保存预处理后的数据到缓存
            with open(path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)
        else:  # 如果缓存存在
            with open(path, "rb") as f:  # 读取缓存
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(f)
        
        # 设置使用的索引列表（用于划分训练/验证/测试）
        self.use_index_list = use_index_list if use_index_list is not None else np.arange(len(self.observed_values))

    def __getitem__(self, org_index):
        """获取单个样本"""
        index = self.use_index_list[org_index]  # 根据子集索引映射到全局索引
        # 构建返回字典
        return {
            "observed_data": self.observed_values[index],  # 标准化后的观测数据
            "observed_mask": self.observed_masks[index],  # 原始数据存在的掩码
            "gt_mask": self.gt_masks[index],  # 经过人工缺失处理的输入掩码
            "timepoints": np.arange(self.eval_length),  # 时间步序列 [0, 1, ..., 47]
        }

    def __len__(self):
        """返回数据集大小"""
        return len(self.use_index_list)  # 返回当前子集的样本数

def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):
    """获取训练、验证和测试的数据加载器"""
    # 初始化数据集以获取完整数据和长度
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))  # 获取所有样本的索引
    
    np.random.seed(seed)  # 设置随机种子
    np.random.shuffle(indlist)  # 随机打乱索引
    
    # 5折交叉验证分割逻辑
    start = int(nfold * 0.2 * len(dataset))  # 当前折的起始位置 (例如第0折: 0.0)
    end = int((nfold + 1) * 0.2 * len(dataset))  # 当前折的结束位置 (例如第0折: 0.2)
    test_index = indlist[start:end]  # 截取测试集索引
    remain_index = np.delete(indlist, np.arange(start, end))  # 剩余索引用于训练和验证
    
    np.random.seed(seed)  # 再次设置种子确保划分一致性
    np.random.shuffle(remain_index)  # 打乱剩余索引
    num_train = int(len(dataset) * 0.7)  # 计算训练集大小 (总量的70%)
    train_index = remain_index[:num_train]  # 前70%作为训练集
    valid_index = remain_index[num_train:]  # 剩余部分作为验证集
    
    # 创建训练集Dataset和DataLoader
    dataset = Physio_Dataset(use_index_list=train_index, missing_ratio=missing_ratio, seed=seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 训练集需要shuffle
    
    # 创建验证集Dataset和DataLoader
    valid_dataset = Physio_Dataset(use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 验证集不需要shuffle
    
    # 创建测试集Dataset和DataLoader
    test_dataset = Physio_Dataset(use_index_list=test_index, missing_ratio=missing_ratio, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集不需要shuffle
    
    return train_loader, valid_loader, test_loader  # 返回三个加载器
