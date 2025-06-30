import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score

# 加载 train 和 test 数据
df_train = pd.read_csv(r'd:\shenni\交大\程序设计与数据结构3\Python速成\股票\399300.csv')
df_train.columns = ['date','open','pre_Close','high','low','volume','amount','turnover_ratio','OT']
df_train['date'] = pd.to_datetime(df_train['date'])
feature_cols = ['date', 'open', 'pre_Close', 'high', 'low', 'volume', 'amount', 'turnover_ratio', 'OT']
# 归一化（标准化）
mean_dict = {}
std_dict = {}
for col in feature_cols:
    mean_dict[col] = df_train[col].mean()
    std_dict[col] = df_train[col].std()
    df_train[col] = (df_train[col] - mean_dict[col]) / std_dict[col]


# 直接在 DataFrame 上划分
train_val, test = train_test_split(df_train, test_size=0.2, random_state=42)
# train, val = train_test_split(train_val, test_size=0.125, random_state=42)

# 参数
SEQ_LEN = 30  # 用前24个时刻预测下一个
PRED_HOUR = 0  # 预测第0小时后的OT
BATCH_SIZE = 256 # 在深度学习中，**batch（批次）**指的是一次送入神经网络进行训练或预测的数据样本的数量。
EPOCHS = 15 # 训练的轮数
LR = 0.0001
# 使用pandas的shift方法向量化地创建滑动窗口，以替代for循环，效率更高

# 创建一个包含所有滞后特征的列表
# df.shift(i) 确保滞后操作在每个分组内独立进行，避免数据跨组泄露
# 构造多特征滑窗
features = []
for col in feature_cols:
    for i in range(SEQ_LEN, 0, -1):
        features.append(train_val[col].shift(i))
df_sequences = pd.concat(features, axis=1)
df_sequences['y'] = train_val['OT'].shift(-PRED_HOUR).values
# 删除因shift产生的NaN行，这些行是每个序列的开头，没有完整的历史数据
df_sequences.dropna(inplace=True)

# 从DataFrame中提取特征X（所有滞后列）和目标y为numpy数组
X = df_sequences.iloc[:, :-1].values
y = df_sequences['y'].values
all_samples = list(zip(X, y))
# 3. 划分训练集和验证集
train_samples, val_samples = train_test_split(all_samples, test_size=0.125, random_state=42)

print(1)

# 数据集定义
class QDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)#并把 x 和 y 都转换成 PyTorch 的 float32 张量（tensor），这样神经网络才能处理。

train_dataset = QDataset(train_samples)
val_dataset = QDataset(val_samples)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
print(2)
# 定义神经网络Network
class FCNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(# PyTorch 中用于快速搭建神经网络的一种容器。它可以把多个层（如 Linear、ReLU 等）按顺序组合在一起，自动按顺序执行
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze()

# 初始化模型
input_dim = SEQ_LEN * len(feature_cols)
model = FCNet(input_dim)# 模型本身，它就是我们设计的神经网络
optimizer = torch.optim.Adam(model.parameters(), lr=LR)# 优化模型中的参数
loss_fn = nn.MSELoss()# 分类问题，使用均方差（？）损失误差
print(3)
# 训练
for epoch in range(EPOCHS):# 外层循环，代表了整个训练数据集的遍历次数
    model.train()
    total_loss = 0
    for x, y in train_loader:# 内层每循环一次，就会进行一次梯度下降算法
        optimizer.zero_grad()           # 1. 先清零梯度
        pred = model(x)                 # 2. 前向传播
        loss = loss_fn(pred, y)         # 3. 计算损失
        loss.backward()                 # 4. 反向传播
        optimizer.step()                # 5. 更新参数
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_dataset)
    # print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss += loss.item() * x.size(0)
    val_loss = val_loss / len(val_dataset)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 用于预测
def predict_next_OT(q_seq):
    model.eval()
    x = torch.tensor(q_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    return pred.item()

estimate_OTs = []
test = test.reset_index(drop=True)
for idx, row in test.iterrows():
    # 取test_time前的SEQ_LEN条数据（用train_val和test的前面部分）
    all_hist = pd.concat([train_val, test.iloc[:idx]], ignore_index=True)
    if len(all_hist) >= SEQ_LEN:
        # 构造多特征滑窗
        x_seq = []
        for col in feature_cols:
            x_seq.extend(all_hist[col].values[-SEQ_LEN-PRED_HOUR:])
        pred_OT = predict_next_OT(x_seq)
    else:
        pred_OT = 0
    estimate_OTs.append(pred_OT)
    
# 4. 反归一化
estimate_OTs = np.array(estimate_OTs) * std_dict['OT'] + mean_dict['OT']
test['OT_true'] = test['OT'] * std_dict['OT'] + mean_dict['OT']
test['pre_Close'] = test['OT_true'].shift(1)


# 5. 计算指标
#计算MAE
# 假设 predicted_OTs 顺序和 test.index 顺序一致
mae = mean_absolute_error(test['OT_true'].values, estimate_OTs)
print(f"MAE: {mae:.4f}")
# 计算RMSE
rmse = root_mean_squared_error(test['OT_true'].values, estimate_OTs)
print(f"RMSE: {rmse:.4f}")

mask = ~np.isnan(test['pre_Close'].values)
# 真实涨跌方向
real_updown = (test['OT_true'].values[mask] > test['pre_Close'].values[mask]).astype(int)
# 预测涨跌方向
pred_updown = (np.array(estimate_OTs)[mask] > test['pre_Close'].values[mask]).astype(int)
# 计算准确率
acc = accuracy_score(real_updown, pred_updown)
print(f"ACC: {acc:.4f}")

# 真实涨跌幅（实际收益率）
real_return = (test['OT_true'].values[mask] - test['pre_Close'].values[mask]) / test['pre_Close'].values[mask]
# 策略收益：如果预测涨就持有，否则空仓（收益为0）
strategy_return = real_return * pred_updown
# 累计收益率
cumulative_return = np.cumprod(1 + strategy_return)[-1] - 1
print(f"Cumulative Return: {cumulative_return:.4%}")

# 先计算策略每日收益率（已在 strategy_return 里）
# 年化夏普比率 = 日均收益 / 日收益标准差 * sqrt(年交易日数)
# 假设一年有252个交易日
mean_return = np.mean(strategy_return)
std_return = np.std(strategy_return)
sharpe_ratio = mean_return / std_return * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")