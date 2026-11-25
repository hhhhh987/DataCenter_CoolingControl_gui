"""
extract_schneider_pumps.py
从 schneider_pub 提取 4台水泵的运行状态，合并求和得到运行台数
原始字段：In_marcia_p101 ~ In_marcia_p104 (0/1)
输出：schneider_pumps.parquet 包含 N_pumps_running 列
"""

import os
import numpy as np
import pandas as pd

ROOT = r'D:\PythonProject1'

MONTHS = [
    '21-03', '21-04', '21-05', '21-06','21-07', '21-08', '21-09', '21-10', '21-11', '21-12',
    '22-01', '22-02','22-03','22-04','22-05','22-06','22-07','22-08','22-09'
]

# 4个水泵状态字段
PUMP_METRICS = [
    'PLC_PLC_Q101.In_marcia_p101',
    'PLC_PLC_Q101.In_marcia_p102',
    'PLC_PLC_Q101.In_marcia_p103',
    'PLC_PLC_Q101.In_marcia_p104',
]

OUT_PATH = os.path.join(ROOT, '03CoolingData', 'schneider_pumps.parquet')
FREQ = '1h'


def read_one_pump_one_month(month: str, metric: str) -> pd.DataFrame:
    """读取单台水泵单个月份的运行状态"""
    path = os.path.join(ROOT, f'year_month={month}',
                        'plugin=schneider_pub', f'metric={metric}')

    if not os.path.exists(path):
        return pd.DataFrame()

    files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def resample_pump_status(df: pd.DataFrame, pump_name: str) -> pd.DataFrame:
    """每小时取均值（介于0~1之间，表示该小时内水泵开机的时间比例）"""
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values('timestamp').set_index('timestamp')
    resampled = df['value'].resample(FREQ, origin='start_day').mean().reset_index()
    resampled.columns = ['timestamp', pump_name]
    resampled = resampled.dropna(subset=[pump_name])
    return resampled


if __name__ == '__main__':
    print('开始提取4台水泵运行状态...\n')

    # 每台水泵分别处理
    pump_dfs = {}
    for metric in PUMP_METRICS:
        pump_name = metric.split('.')[-1]  # 取 In_marcia_p101
        print(f'\n=== {pump_name} ===')

        all_months = []
        for month in MONTHS:
            df_raw = read_one_pump_one_month(month, metric)
            if df_raw.empty:
                continue
            df_hourly = resample_pump_status(df_raw, pump_name)
            if not df_hourly.empty:
                all_months.append(df_hourly)

        if all_months:
            pump_df = pd.concat(all_months, ignore_index=True)
            pump_df = pump_df.sort_values('timestamp').reset_index(drop=True)
            pump_dfs[pump_name] = pump_df
            print(f'  {pump_name}: {len(pump_df)} 行')

    # 合并4台水泵，按timestamp做outer join
    df_merged = None
    for pump_name, df in pump_dfs.items():
        if df_merged is None:
            df_merged = df
        else:
            df_merged = df_merged.merge(df, on='timestamp', how='outer')

    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

    # 计算运行台数（4个状态之和，缺失值视为0）
    pump_cols = [c for c in df_merged.columns if c.startswith('In_marcia')]
    df_merged['N_pumps_running'] = df_merged[pump_cols].fillna(0).sum(axis=1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_merged.to_parquet(OUT_PATH, index=False, compression='zstd')

    print(f'\n完成！')
    print(f'输出文件：{OUT_PATH}')
    print(f'总行数：{len(df_merged)}')
    print(f'\nN_pumps_running 分布：')
    print(df_merged['N_pumps_running'].describe().round(2))
    print(f'\n前5行：')
    print(df_merged.head())