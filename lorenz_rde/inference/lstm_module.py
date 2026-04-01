import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    """
    训练LSTM模型
    X_train: 形状为 (num_samples, sequence_length, input_size)
    y_train: 形状为 (num_samples, output_size)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')
    
    return model


def lstm_predict(model, seq, trainlength=30, steps_ahead=1, target_idx=0):
    """
    使用LSTM模型进行预测
    seq: 输入序列，形状为 (total_length, num_features)
    trainlength: 训练序列长度
    steps_ahead: 预测步长
    target_idx: 目标特征索引
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_steps = len(seq) - trainlength
    predictions = []
    stds = []
    
    with torch.no_grad():
        for step in range(total_steps):
            # 准备训练数据
            traindata = seq[step:step + trainlength, :]
            
            # 准备输入和目标
            X = traindata[:-steps_ahead].reshape(1, -1, traindata.shape[1])
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            # 预测
            pred = model(X_tensor).cpu().numpy()[0, 0]
            predictions.append(pred)
            stds.append(0.0)  # LSTM不直接提供不确定性估计
    
    # 构建结果矩阵
    result = np.zeros((3, total_steps))
    result[0] = predictions  # 预测值
    result[1] = stds        # 标准差（这里设为0）
    result[2] = seq[trainlength:, target_idx] - predictions  # 残差
    
    return result
