import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# 参数
SEQ_LEN = 60
PRED_HOUR = 0
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.0001

# 1. 加载数据
df = pd.read_csv(r'd:\shenni\交大\程序设计与数据结构3\Python速成\股票\399300.csv')
df.columns = ['date','open','pre_close','high','low','volume','amount','turnover_ratio','OT']
df['date'] = pd.to_datetime(df['date'])

feature_cols = ['open', 'pre_close', 'high', 'low', 'volume', 'amount', 'turnover_ratio', 'OT']

# 2. 归一化所有特征（用全体数据的均值和方差，便于后续反归一化）
mean_dict = {}
std_dict = {}
for col in feature_cols:
    mean = df[col].mean()
    std = df[col].std()
    mean_dict[col] = mean
    std_dict[col] = std
    df[col] = (df[col] - mean) / std

# 3. 划分数据集
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

# 4. 构建滑窗样本（只用train_val，防止数据泄漏）
def build_samples(df, feature_cols, seq_len, pred_hour):
    features = []
    for col in feature_cols:
        for i in range(seq_len, 0, -1):
            features.append(df[col].shift(i))
    df_sequences = pd.concat(features, axis=1)
    df_sequences['y'] = df['OT'].shift(-pred_hour).values
    df_sequences.dropna(inplace=True)
    X = df_sequences.iloc[:, :-1].values
    y = df_sequences['y'].values
    return list(zip(X, y))

all_samples = build_samples(train_val, feature_cols, SEQ_LEN, PRED_HOUR)
train_samples, val_samples = train_test_split(all_samples, test_size=0.125, random_state=42)

# 5. PyTorch数据集
class QDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = QDataset(train_samples)
val_dataset = QDataset(val_samples)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 6. 神经网络
class FCNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze()

model = FCNet(SEQ_LEN * len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# 7. 训练
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_dataset)
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

# 8. 用于预测
def predict_next_OT(x_seq):
    model.eval()
    x = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    return pred.item()

# 9. 在test集上滑窗预测
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

# 10. 反归一化
estimate_OTs = np.array(estimate_OTs) * std_dict['OT'] + mean_dict['OT']
test_OT_true = test['OT'].values * std_dict['OT'] + mean_dict['OT']
test_pre_close = test['pre_close'].values * std_dict['pre_close'] + mean_dict['pre_close']

# 11. 计算指标
mae = mean_absolute_error(test_OT_true, estimate_OTs)
rmse = mean_squared_error(test_OT_true, estimate_OTs, squared=False)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# 真实涨跌方向
real_updown = (test_OT_true > test_pre_close).astype(int)
# 预测涨跌方向
pred_updown = (estimate_OTs > test_pre_close).astype(int)
# 计算准确率
acc = accuracy_score(real_updown, pred_updown)
print(f"ACC: {acc:.4f}")

# 真实涨跌幅（实际收益率）
real_return = (test_OT_true - test_pre_close) / test_pre_close
# 策略收益：如果预测涨就持有，否则空仓（收益为0）
strategy_return = real_return * pred_updown
# 累计收益率
cumulative_return = np.cumprod(1 + strategy_return)[-1] - 1
print(f"Cumulative Return: {cumulative_return:.4%}")

# 夏普比率
mean_return = np.mean(strategy_return)
std_return = np.std(strategy_return)
sharpe_ratio = mean_return / std_return * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")