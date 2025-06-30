import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import timedelta

# 读取数据
df_train = pd.read_csv('loop_sensor_train.csv')
df_train.columns = ['iu_ac', 't_1h', 'etat_barre', 'q']
df_train['t_1h'] = pd.to_datetime(df_train['t_1h'])

# 归一化（标准化）q列
mean_q = df_train['q'].mean()
std_q = df_train['q'].std()
df_train['q'] = (df_train['q'] - mean_q) / std_q

# 设定采样频率（如1小时）
freq = '1H'

# 新DataFrame用于存放补齐后的所有分组数据
filled_groups = []

for iu_ac, group in df_train.groupby('iu_ac'):
    group = group.sort_values('t_1h')
    # 生成该组完整时间索引
    full_range = pd.date_range(start=group['t_1h'].min(), end=group['t_1h'].max(), freq=freq)
    # 以 t_1h 为索引，reindex到完整时间序列
    group = group.set_index('t_1h').reindex(full_range)
    group['iu_ac'] = iu_ac  # 补齐的行也加上分组号
    # 用前一个时间点的数据填补（ffill）
    group = group.ffill()
    group = group.reset_index().rename(columns={'index': 't_1h'})
    filled_groups.append(group)

# 合并所有分组
df_filled = pd.concat(filled_groups, ignore_index=True)

# 参数
SEQ_LEN = 24  # 用前24个时刻预测下一个
PRED_HOUR = 5  # 预测第5小时后的q
BATCH_SIZE = 256 # 在深度学习中，**batch（批次）**指的是一次送入神经网络进行训练或预测的数据样本的数量。
EPOCHS = 3
LR = 0.0001
# 使用pandas的shift方法向量化地创建滑动窗口，以替代for循环，效率更高
# 首先，确保数据按分组和时间正确排序
df_train_sorted = df_filled.sort_values(['iu_ac', 't_1h'])

# # 创建一个包含所有滞后特征的列表
# # df.groupby('iu_ac')['q'].shift(i) 确保滞后操作在每个分组内独立进行，避免数据跨组泄露
# features = [df_train_sorted.groupby('iu_ac')['q'].shift(i) for i in range(SEQ_LEN, 0, -1)]

# # 将特征列表和目标（原始q值）合并到一个DataFrame中
# df_sequences = pd.concat(features, axis=1)
# df_sequences['y'] = df_train_sorted.groupby('iu_ac')['q'].shift(-PRED_HOUR).values

# # 删除因shift产生的NaN行，这些行是每个序列的开头，没有完整的历史数据
# df_sequences.dropna(inplace=True)

# # 从DataFrame中提取特征X（所有滞后列）和目标y为numpy数组
# X = df_sequences.iloc[:, :-1].values
# y = df_sequences['y'].values

# # 构造成后续代码所需的 (x, y) 元组列表格式
# all_samples = list(zip(X, y))
# print(1)
# # 3. 划分训练集和验证集
# train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)

# # 数据集定义
# class QDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         x, y = self.samples[idx]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)#并把 x 和 y 都转换成 PyTorch 的 float32 张量（tensor），这样神经网络才能处理。

# train_dataset = QDataset(train_samples)
# val_dataset = QDataset(val_samples)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# print(2)
# 定义神经网络Network
class FCNet(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.fc = nn.Sequential(# PyTorch 中用于快速搭建神经网络的一种容器。它可以把多个层（如 Linear、ReLU 等）按顺序组合在一起，自动按顺序执行
            nn.Linear(seq_len, 32), # 线性层1，输入层和隐藏层之间的线性层
            nn.ReLU(),# 使用relu激活
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze()

# # 初始化模型
model = FCNet(SEQ_LEN)# 模型本身，它就是我们设计的神经网络
optimizer = torch.optim.Adam(model.parameters(), lr=LR)# 优化模型中的参数
loss_fn = nn.MSELoss()# 分类问题，使用均方差（？）损失误差
# print(3)
# # 训练
# for epoch in range(EPOCHS):# 外层循环，代表了整个训练数据集的遍历次数
#     model.train()
#     total_loss = 0
#     for x, y in train_loader:# 内层每循环一次，就会进行一次梯度下降算法
#         optimizer.zero_grad()           # 1. 先清零梯度
#         pred = model(x)                 # 2. 前向传播
#         loss = loss_fn(pred, y)         # 3. 计算损失
#         loss.backward()                 # 4. 反向传播
#         optimizer.step()                # 5. 更新参数
#         total_loss += loss.item() * x.size(0)
#     train_loss = total_loss / len(train_dataset)
#     # print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
#     # 验证
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for x, y in val_loader:
#             pred = model(x)
#             loss = loss_fn(pred, y)
#             val_loss += loss.item() * x.size(0)
#     val_loss = val_loss / len(val_dataset)
#     print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# # 保存模型参数
# torch.save(model.state_dict(), 'trained_fcnet.pth')

# 加载参数
model.load_state_dict(torch.load('trained_fcnet.pth'))
model.eval()

# 用于预测
def predict_next_q(q_seq):
    model.eval()
    x = torch.tensor(q_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    return pred.item()

# 读取test集
df_test = pd.read_csv('loop_sensor_test_x.csv')
df_test.columns = ['id', 'iu_ac', 't_1h', 'etat_barre']
df_test['t_1h'] = pd.to_datetime(df_test['t_1h'])
# 预测前，先分组并排序一次
train_grouped = {k: v.sort_values('t_1h') for k, v in df_filled.groupby('iu_ac')}

estimate_qs = []
for row in df_test.itertuples():
    iu_ac = row.iu_ac
    test_time = row.t_1h
    group = train_grouped.get(iu_ac, None)
    if group is not None:
        # 1. 计算输入序列的截止时间（test_time前5小时）
        input_end_time = test_time - timedelta(hours=5)
        # 2. 计算输入序列的起始时间
        input_start_time = input_end_time - timedelta(hours=24)
        # 3. 取这24小时的数据
        q_vals = group[(group['t_1h'] > input_start_time) & (group['t_1h'] <= input_end_time)]['q'].values
        # 4. 如果数据量足够，取最后SEQ_LEN个
        if len(q_vals) >= SEQ_LEN:
            last_seq = q_vals[-SEQ_LEN:]
            pred_q = predict_next_q(last_seq)
        elif len(q_vals) > 0:
            print(f"Not enough data for iu_ac={iu_ac}, test_time={test_time}, got {len(q_vals)} values")
            pred_q = q_vals[-1] # 标准化后平均值就是0
        else:
            pred_q = 0
    else:
        pred_q = 0 # 标准化后平均值就是0
    estimate_qs.append(pred_q)

df_test['estimate_q'] = np.array(estimate_qs) * std_q + mean_q  # 反归一化
df_test[['id', 'estimate_q']].to_csv('nn_test_pred4.csv', index=False)