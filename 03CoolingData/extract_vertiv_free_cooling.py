
"""
extract_vertiv_Free_Cooling.py
从 vertiv_pub 插件提取 Free_Cooling_Status（自然冷却状态）
6台设备（cdz1~cdz6）分别保留，输出长表格式
输出：vertiv_fan_speed.parquet
"""

import os
import numpy as np
import pandas as pd

# ── 配置 ──────────────────────────────────────────────────
ROOT = r'D:\PythonProject1'

MONTHS = [
    '21-03', '21-04', '21-05', '21-06','21-07', '21-08', '21-09', '21-10', '21-11', '21-12',
    '22-01', '22-02','22-03','22-04','22-05','22-06','22-07','22-08','22-09'
]

FREQ = '1h'

METRIC_NAME = 'Free_Cooling_Status'
OUT_PATH = os.path.join(ROOT, '03CoolingData', 'vertiv_free_cooling.parquet')
VALUE_COL = 'Free_Cooling_Status'
VALUE_MIN = 0
VALUE_MAX = 1      # 状态量，0/1

# ── 功能函数 ──────────────────────────────────────────────

def read_one_month(month: str) -> pd.DataFrame:
    """读取单月数据，包含timestamp/value/device三列"""
    path = os.path.join(ROOT, f'year_month={month}',
                        'plugin=vertiv_pub', f'metric={METRIC_NAME}')

    if not os.path.exists(path):
        print(f'{month}月份的路径不存在')
        return pd.DataFrame()

    files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    if not files:
        print(f'{month}月份路径下无parquet文件')
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_parquet(os.path.join(path, f))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def resample_by_device(df: pd.DataFrame) -> pd.DataFrame:
    """
    按device分组，分别重采样到小时级
    输入：原始数据 [timestamp, value, device]
    输出：小时聚合后的长表 [timestamp, device, Fan_Speed_pct]
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    # 异常值过滤
    df.loc[df['value'] < VALUE_MIN, 'value'] = np.nan
    df.loc[df['value'] > VALUE_MAX, 'value'] = np.nan

    results = []
    for device, group in df.groupby('device'):
        # 单台device的数据重采样到小时
        g = group.sort_values('timestamp').set_index('timestamp')
        resampled = g['value'].resample(FREQ, origin='start_day').mean().reset_index()
        resampled.columns = ['timestamp', VALUE_COL]
        resampled['device'] = device
        resampled = resampled.dropna(subset=[VALUE_COL])
        results.append(resampled)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    # 重新排列列顺序：timestamp, device, value
    out = out[['timestamp', 'device', VALUE_COL]]
    return out


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':

    print(f'开始提取 vertiv {METRIC_NAME} 数据（6台device）...\n')
    all_months = []

    for month in MONTHS:
        print(f'处理 {month} ...', end=' ')

        df_raw = read_one_month(month)
        if df_raw.empty:
            continue

        print(f'原始行数={len(df_raw)}', end=' → ')

        df_hourly = resample_by_device(df_raw)
        if df_hourly.empty:
            print('重采样后为空，跳过')
            continue

        print(f'重采样后={len(df_hourly)}行 (含{df_hourly.device.nunique()}台device)')
        all_months.append(df_hourly)

    if not all_months:
        print('错误：所有月份均无数据')
        exit()

    df_all = pd.concat(all_months, ignore_index=True)
    df_all = df_all.sort_values(['timestamp', 'device']).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_all.to_parquet(OUT_PATH, index=False, compression='zstd')

    # ── 验证输出 ──
    print(f'\n完成！')
    print(f'输出文件：{OUT_PATH}')
    print(f'总行数：{len(df_all)} 条')
    print(f'时间范围：{df_all.timestamp.min()} → {df_all.timestamp.max()}')

    # 每台device的统计
    print(f'\n=== 各device的数据情况 ===')
    device_stats = df_all.groupby('device')[VALUE_COL].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    print(device_stats)

    # 检查每小时是否6台都有数据
    hourly_counts = df_all.groupby('timestamp').size()
    print(f'\n=== 每小时device数量分布 ===')
    print(f'  每小时有6台数据: {(hourly_counts == 6).sum()} 小时')
    print(f'  每小时有<6台数据: {(hourly_counts < 6).sum()} 小时')
    print(f'  每小时device数量的分布:')
    print(hourly_counts.value_counts().sort_index())

    print(f'\n前12行（应该是2台device的前6个小时）：')
    print(df_all.head(12))