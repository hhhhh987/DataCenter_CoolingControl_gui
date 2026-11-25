"""
extract_schneider_flow.py
从 schneider_pub 插件提取 Temp_mandata（供水温度）℃
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

# schneider的metric命名带PLC前缀
FREQ = '1h'
METRIC_NAME = 'PLC_PLC_Q101.Temp_mandata'
OUT_PATH = os.path.join(ROOT, '03CoolingData', 'schneider_t_mandata.parquet')

# 单位换算因子（原始值除以这个数）
UNIT_SCALE = 10.0       # 原始×10
# 异常值过滤阈值（换算后的单位）
VALUE_MIN = 5
VALUE_MAX = 50          # 回水温度可能比供水高
# 重命名列：'T_mandata_C'


# ── 功能函数 ──────────────────────────────────────────────

def read_one_month(month: str) -> pd.DataFrame:
    """读取单个月份的原始数据"""
    path = os.path.join(ROOT, f'year_month={month}',
                        'plugin=schneider_pub', f'metric={METRIC_NAME}')

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


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """单位换算 + 异常过滤 + 重采样为1小时均值"""
    if df.empty:
        return pd.DataFrame()

    # 单位换算（原始×10的整数 → 真实物理值）
    df = df.copy()
    df['value'] = df['value'] / UNIT_SCALE

    # 异常值过滤
    df.loc[df['value'] < VALUE_MIN, 'value'] = np.nan
    df.loc[df['value'] > VALUE_MAX, 'value'] = np.nan

    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')

    resampled_df = df['value'].resample(FREQ, origin='start_day').mean().reset_index()
    resampled_df.columns = ['timestamp', 'T_mandata_C']
    resampled_df = resampled_df.dropna(subset=['T_mandata_C'])

    return resampled_df


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':

    print(f'开始提取 {METRIC_NAME} 数据...\n')
    all_months = []

    for month in MONTHS:
        print(f'处理 {month} ...', end=' ')

        df_raw = read_one_month(month)
        if df_raw.empty:
            continue

        print(f'原始行数={len(df_raw)}', end=' → ')

        df_hourly = resample_to_hourly(df_raw)
        if df_hourly.empty:
            print('重采样后为空，跳过')
            continue

        print(f'重采样后={len(df_hourly)}行')
        all_months.append(df_hourly)

    if not all_months:
        print('错误：所有月份均无数据')
        exit()

    df_all = pd.concat(all_months, ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_all.to_parquet(OUT_PATH, index=False, compression='zstd')

    print(f'\n完成！')
    print(f'输出文件：{OUT_PATH}')
    print(f'总行数：{len(df_all)} 条（每{FREQ}一条）')
    print(f'时间范围：{df_all.timestamp.min()} → {df_all.timestamp.max()}')
    print(f'T_mandata_C范围：{df_all.T_mandata_C.min():.1f} ~ {df_all.T_mandata_C.max():.1f} ℃')
    print(f'\n前5行：')
    print(df_all.head())