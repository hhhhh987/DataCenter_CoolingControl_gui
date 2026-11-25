'''
数据预处理脚本
'''
import torch
import pandas as pd
import numpy as np

import holidays
from tqdm import tqdm
import config

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def add_time_features(df ,timestamp_col= 'timestamp') -> pd.DataFrame:
    ts = df[timestamp_col]

    # 小时周期编码（24小时）─────────────────────────
    hour = ts.dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # 星期周期编码（7天）───────────────────────────
    dow = ts.dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # 月份周期编码（12个月）────────────────────────
    month = ts.dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # 是否节假日（0/1）─────────────────────────────
    # 使用向量化操作 isin() 替代 apply()，大幅提升计算速度
    years = ts.dt.year.unique().tolist()
    it_holidays = holidays.Italy(years=years)
    # 获取所有节假日的 date 对象列表
    holiday_dates = list(it_holidays.keys())
    # 向量化判断
    df['is_holiday'] = ts.dt.date.isin(holiday_dates).astype(int)

    # ── 5. 是否工作日（0/1）─────────────────────────────
    df['is_workday'] = (
        (ts.dt.dayofweek < 5) &
        (df['is_holiday'] == 0)
    ).astype(int)

    return df

# def build_windows(values, targets, splits, window_size):
#     result = {'train': ([], []), 'val': ([], []), 'test': ([], [])}
#     for i in range(len(values) - window_size):
#         x = values[i : i + window_size]
#         # y = targets[i + window_size]
#         y = targets[i + window_size] - targets[i + window_size - 1]
#         s = splits[i + window_size]
#         result[s][0].append(x)
#         result[s][1].append(y)
#     return result
# ◆ 修改: 返回值从(x, y)变为(x, delta, last_known)
def build_windows(values, targets, splits, window_size):
    result = {'train': ([], [], []), 'val': ([], [], []), 'test': ([], [], [])}  # ◆ 修改: 每组3个列表
    for i in range(len(values) - window_size):
        x = values[i : i + window_size]
        delta = targets[i + window_size] - targets[i + window_size - 1]  # ◆ 修改: y改为差值
        last = targets[i + window_size - 1]                              # ◆ 新增: 最后已知值
        s = splits[i + window_size]
        result[s][0].append(x)
        result[s][1].append(delta)  # ◆ 修改: 存差值
        result[s][2].append(last)   # ◆ 新增: 存last_known
    return result


TARGET_COL   = 'Tot_ict_kW'

FEATURE_COLS = [
    'Tot_ict_kW',
    'temp_C', 'humidity_pct', 'dew_point_C',
    'wind_speed_ms', 'pressure_hPa', 'temp_wet_C',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'is_workday', 'is_holiday',
]

BINARY_COLS  = ['is_workday', 'is_holiday']
SCALE_COLS = [    'Tot_ict_kW',
    'temp_C', 'humidity_pct', 'dew_point_C',
    'wind_speed_ms', 'pressure_hPa', 'temp_wet_C',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
]


def process():
    #1.读取数据文件
    df=pd.read_parquet(config.RAW_DATA_DIR/'merge_dataset.parquet')
    print(len(df),'行数据')
    print(len(df['segment_id'].unique() ), '个segment')

    #2添加时间特征
    df=add_time_features(df)
    print(df.head())
    print(df.columns)
    #此时包含15+timestamp+segment_id
    print(len(df.columns),'个特征值')

    #3划分训练集，验证集，测试集
    total_num=len(df)

    train_end = int( total_num* config.TRAIN_RATIO )
    val_end   = int( total_num * (config.TRAIN_RATIO + config.VAL_RATIO))

    df = df.sort_values(['segment_id', 'timestamp']).reset_index(drop=True)

    df['split'] = 'train'
    df.loc[train_end:val_end - 1, 'split'] = 'val'
    df.loc[val_end:, 'split'] = 'test'

    train_num=len(df[df['split']=='train'])
    val_num=len(df[df['split']=='val'])
    test_num=len(df[df['split']=='test'])
    print('训练集行数：',train_num,end='\t')
    print(df[df['split']=='train']['timestamp'].min(),'->',df[df['split']=='train']['timestamp'].max())
    print('验证集行数：',val_num,end='\t')
    print(df[df['split'] == 'val']['timestamp'].min(), '->', df[df['split'] == 'val']['timestamp'].max())
    print('测试集行数：',test_num,end='\t')
    print(df[df['split'] == 'test']['timestamp'].min(), '->', df[df['split'] == 'test']['timestamp'].max())

    #4数据归一化处理
    scaler=MinMaxScaler()
    scaler.fit(df.loc[ df['split']=='train',SCALE_COLS ])

    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS]).astype(np.float32)

    print(f'  归一化基于 {len(df[df['split']=='train'])} 行训练数据')
    print(f'  归一化: {len(SCALE_COLS)} 列, 跳过: {BINARY_COLS}')




    # 5按片段构造滑窗
    # all_samples = {'train': ([], []), 'val': ([], []), 'test': ([], [])}
    all_samples = {'train': ([], [], []), 'val': ([], [], []), 'test': ([], [], [])}  # ◆ 修改: 3个列表

    for seg_id, seg_df in df.groupby('segment_id'):
        values  = seg_df[FEATURE_COLS].values.astype(np.float32)
        targets = seg_df[TARGET_COL].values.astype(np.float32)
        splits  = seg_df['split'].values

        seg_result = build_windows(values, targets, splits, config.WINDOW_SIZE)


        seg_total = 0
        for s in ['train', 'val', 'test']:
            all_samples[s][0].extend(seg_result[s][0])
            all_samples[s][1].extend(seg_result[s][1])
            all_samples[s][2].extend(seg_result[s][2])  # ◆ 新增
            seg_total += len(seg_result[s][0])

        print(f'  片段{seg_id}: {seg_total} 个样本 '
              f'(train {len(seg_result["train"][0])} / '
              f'val {len(seg_result["val"][0])} / '
              f'test {len(seg_result["test"][0])})')
    # 转为tensor
    data = {}
    for s in ['train', 'val', 'test']:
        # X_list, y_list = all_samples[s]
        X_list, y_list, last_list = all_samples[s]  # ◆ 修改: 解包3个
        if len(X_list) > 0:
            data[f'X_{s}'] = torch.tensor(np.array(X_list))
            data[f'y_{s}'] = torch.tensor(np.array(y_list))
            data[f'last_known_{s}'] = torch.tensor(np.array(last_list))  # ◆ 新增
        else:
            data[f'X_{s}'] = torch.empty(0, config.WINDOW_SIZE, len(FEATURE_COLS))
            data[f'y_{s}'] = torch.empty(0)
            data[f'last_known_{s}'] = torch.empty(0)  # ◆ 新增

    print(f'\n  训练集: X={data["X_train"].shape}, y={data["y_train"].shape}')
    print(f'  验证集: X={data["X_val"].shape},   y={data["y_val"].shape}')
    print(f'  测试集: X={data["X_test"].shape},  y={data["y_test"].shape}')
    # ◆ 新增: 打印差值标签统计
    y_np = data['y_train'].numpy()
    print(f'\n  差值标签统计(训练集): mean={y_np.mean():.6f}, '
          f'std={y_np.std():.6f}, min={y_np.min():.6f}, max={y_np.max():.6f}')

    # 6保存
    print('\nStep6: 保存：：：：：')
    #确保路径存在
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data.update({
        'feature_names': FEATURE_COLS,
        'target_name':   TARGET_COL,
        'window_size':   config.WINDOW_SIZE,
        'scaler_data_min': scaler.data_min_,
        'scaler_data_max': scaler.data_max_,
        'scale_cols':    SCALE_COLS,
    })

    torch.save(data, config.PROCESSED_DATA_DIR / 'dataset2.pt')

    size_mb = (config.PROCESSED_DATA_DIR / 'dataset2.pt').stat().st_size / 1024 / 1024
    print(f'  dataset2.pt  ({size_mb:.1f} MB)')
    print(f'  路径: {config.PROCESSED_DATA_DIR / "dataset2.pt"}')
    print('\n完成 ✓')


if __name__ == '__main__':
    process()