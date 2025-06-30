import pandas as pd
import numpy as np

# 加载 train 和 test 数据
df_train = pd.read_csv('loop_sensor_train.csv')
df_train.columns = ['iu_ac', 't_1h', 'etat_barre', 'q']
df_train['t_1h'] = pd.to_datetime(df_train['t_1h'])

df_test = pd.read_csv('loop_sensor_test_x.csv')
df_test.columns = ['id','iu_ac', 't_1h', 'etat_barre']
df_test['t_1h'] = pd.to_datetime(df_test['t_1h'])

predicted_qs = []

grouped_dict = {k: v for k, v in df_train.groupby('iu_ac')}

for row in df_test.itertuples():
    iu_ac = row.iu_ac
    test_time = row.t_1h
    if iu_ac in grouped_dict:
        group = grouped_dict[iu_ac]
        # 只保留小时和分钟与 test_time 相同的行
        same_time = group[(group['t_1h'].dt.hour == test_time.hour) & 
                          (group['t_1h'].dt.minute == test_time.minute)]
        time_diff = (same_time['t_1h'] - test_time).abs()
        nearest_indices = time_diff.nsmallest(5).index
        predicted_q = group.loc[nearest_indices, 'q'].mean()
    else:
        predicted_q = 0
    predicted_qs.append(predicted_q)

df_test['estimate_q'] = predicted_qs
df_test.to_csv('loop_sensor_test_pred3.csv', columns=['id', 'estimate_q'], index=False)