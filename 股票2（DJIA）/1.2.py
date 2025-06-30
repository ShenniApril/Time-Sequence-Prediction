import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score

# 加载 train 和 test 数据
df_train = pd.read_csv(r'd:\shenni\交大\程序设计与数据结构3\Python速成\股票2\DJIA_merged_data.csv')
df_train.columns = ['Index','Date', 'Label', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7','Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15', 'Top16',
                    'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23', 'Top24', 'Top25', 'Open', 'High','Low','Volume', 'Close', 'Adj Close']
df_train['Date'] = pd.to_datetime(df_train['Date'])

# 先划分出 test（20%）
train_val, test = train_test_split(df_train, test_size=0.2, random_state=42)
# 再从剩下的 80% 里划分出 validation（10%/80% = 12.5%）
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

predicted_Closess = []

for row in test.itertuples():
    test_time = row.Date
    time_diff = (train['Date'] - test_time).abs()
    nearest_indices = time_diff.nsmallest(5).index
    predicted_Closes = train.loc[nearest_indices, 'Close'].mean()
    predicted_Closess.append(predicted_Closes)

#计算MAE
# 假设 predicted_Closess 顺序和 test.index 顺序一致
mae = mean_absolute_error(test['Close'].values, predicted_Closess)
print(f"MAE: {mae:.4f}")
# 计算RMSE
rmse = root_mean_squared_error(test['Close'].values, predicted_Closess)
print(f"RMSE: {rmse:.4f}")

# 先构造前一天的Close
test['Prev_Close'] = test['Close'].shift(1)
# 真实涨跌方向
real_updown = (test['Close'].values > test['Prev_Close'].values).astype(int)
# 预测涨跌方向
pred_updown = (np.array(predicted_Closess) > test['Prev_Close'].values).astype(int)
# 计算准确率
acc = accuracy_score(real_updown, pred_updown)
print(f"ACC: {acc:.4f}")

# 去除含NaN的行
mask = ~np.isnan(test['Prev_Close'].values)
real_updown = (test['Close'].values[mask] > test['Prev_Close'].values[mask]).astype(int)
pred_updown = (np.array(predicted_Closess)[mask] > test['Prev_Close'].values[mask]).astype(int)
real_return = (test['Close'].values[mask] - test['Prev_Close'].values[mask]) / test['Prev_Close'].values[mask]
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