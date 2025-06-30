import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score

# 加载 train 和 test 数据
df_train = pd.read_csv(r'd:\shenni\交大\程序设计与数据结构3\Python速成\股票\399300.csv')
df_train.columns = ['date','open','pre_close','high','low','volume','amount','turnover_ratio','OT']
df_train['date'] = pd.to_datetime(df_train['date'])

# 先划分出 test（20%）
train_val, test = train_test_split(df_train, test_size=0.2, random_state=42)
# 再从剩下的 80% 里划分出 validation（10%/80% = 12.5%）
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

predicted_OTs = []

for row in test.itertuples():
    test_time = row.date
    # 计算时间差
    time_diff = (train['date'] - test_time).abs()
    nearest_index = time_diff.idxmin()
    predicted_OT = train.loc[nearest_index, 'OT']
    predicted_OTs.append(predicted_OT)

#计算MAE
# 假设 predicted_OTs 顺序和 test.index 顺序一致
mae = mean_absolute_error(test['OT'].values, predicted_OTs)
print(f"MAE: {mae:.4f}")
# 计算RMSE
rmse = root_mean_squared_error(test['OT'].values, predicted_OTs)
print(f"RMSE: {rmse:.4f}")

# 真实涨跌方向
real_updown = (test['OT'].values > test['pre_close'].values).astype(int)
# 预测涨跌方向
pred_updown = (np.array(predicted_OTs) > test['pre_close'].values).astype(int)
# 计算准确率
acc = accuracy_score(real_updown, pred_updown)
print(f"ACC: {acc:.4f}")

# 真实涨跌幅（实际收益率）
real_return = (test['OT'].values - test['pre_close'].values) / test['pre_close'].values
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