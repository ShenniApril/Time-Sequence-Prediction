import pandas as pd
import numpy as np

# 加载 train 和 test 数据
df_train = pd.read_csv('loop_sensor_train.csv')
df_train.columns = ['iu_ac', 't_1h', 'etat_barre', 'q']
df_train['t_1h'] = pd.to_datetime(df_train['t_1h'])

df_test = pd.read_csv('loop_sensor_test_x.csv')
df_test.columns = ['id','iu_ac', 't_1h', 'etat_barre']
df_test['t_1h'] = pd.to_datetime(df_test['t_1h'])

# 先按 iu_ac 分组，方便后续查找
grouped = {k: v for k, v in df_train.groupby('iu_ac')}

predicted_qs = []

for row in df_test.itertuples():
    iu_ac = row.iu_ac
    test_time = row.t_1h
    if iu_ac in grouped:
        group = grouped[iu_ac]
        # 计算时间差
        time_diff = (group['t_1h'] - test_time).abs()
        nearest_index = time_diff.idxmin()
        predicted_q = group.loc[nearest_index, 'q']
    else:
        predicted_q = np.nan
    predicted_qs.append(predicted_q)

df_test['estimate_q'] = predicted_qs
# 导出为 CSV 文件
df_test.to_csv('loop_sensor_test_pred1.csv', columns=['id', 'estimate_q'], index=False)