"""
extract_tot_ict.py
从 logics_pub 插件提取 Tot_ict（配电侧IT总功耗）
输出：tot_ict.parquet
"""

import os
import numpy as np
import pandas as pd

# ── 配置 ──────────────────────────────────────────────────
ROOT = r'D:\PythonProject1'  # year_month文件夹所在的根目录

# 有logics数据的月份
MONTHS = [
    '21-03', '21-04', '21-05', '21-06','21-07', '21-08', '21-09', '21-10', '21-11', '21-12',
    '22-01', '22-02','22-03','22-04','22-05','22-06','22-07','22-08','22-09'
]

OUT_PATH = os.path.join(ROOT, '01dataFresh', 'tot_ict.parquet')
FREQ = '1h'

# ── 功能函数 ──────────────────────────────────────────────

def read_one_month(month: str) -> pd.DataFrame:
    """
    读取单个月份的 Tot_ict 原始数据。
    返回：带timestamp和value列的DataFrame
    """
    path=os.path.join(ROOT,f'year_month={month}','plugin=logics_pub','metric=Tot_ict')
    
    if not os.path.exists(path):
        print(f'{month}月份的路径不存在')
        return pd.DataFrame()

    files=[f for f in os.listdir(path) if f.endswith('.parquet')]

    if files==[]:
        print(f'{month}月份路径下无parquet文件')
        return pd.DataFrame()

    dfs=[]
    for f in files:
        df=pd.read_parquet(os.path.join(path,f))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    将约1分钟采样的logics数据重采样为1小时均值。
    处理步骤：
    1. 只保留 panel=generals 的行（全局汇总值，排除分项数据）
    2. 按timestamp排序
    3. 以timestamp为索引，每小时取均值
    返回：每小时一行的DataFrame，列为 timestamp 和 Tot_ict_kW
    """
    # Step1：只取 panel=generals（全局汇总）
    if 'panel' in df.columns:
        df = df[df['panel'] == 'generals'].copy()

    if df.empty:
        return pd.DataFrame()

    df.loc[df['value'] <= 100, 'value'] =np.nan

    # Step2：排序并设置索引
    df = df.sort_values('timestamp')

    # Step3：重采样为1小时均值
    df=df.set_index('timestamp')

    resampled_df=df['value'].resample(FREQ,origin='start_day').mean().reset_index()
    resampled_df.columns=['timestamp','Tot_ict_kW']
    resampled_df=resampled_df.dropna(subset=['Tot_ict_kW'])

    # 过滤掉空值（该小时内无数据的时间段）
    resampled_df = resampled_df.dropna(subset=['Tot_ict_kW'])

    return resampled_df


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':

    print('开始提取 Tot_ict 数据...\n')
    all_months = []

    for month in MONTHS:
        print(f'处理 {month} ...', end=' ')

        # 读取原始数据
        df_raw = read_one_month(month)
        if df_raw.empty:
            continue

        print(f'原始行数={len(df_raw)}', end=' → ')

        # 重采样
        df_hourly = resample_to_hourly(df_raw)
        if df_hourly.empty:
            print('重采样后为空，跳过')
            continue

        print(f'重采样后={len(df_hourly)}行')
        all_months.append(df_hourly)

    if not all_months:
        print('错误：所有月份均无数据')
        exit()

    # 合并所有月份
    df_all = pd.concat(all_months, ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_all.to_parquet(OUT_PATH, index=False, compression='zstd')

    # 输出统计
    print(f'\n完成！')
    print(f'输出文件：{OUT_PATH}')
    print(f'总行数：{len(df_all)} 条（每{FREQ}一条）')
    print(f'时间范围：{df_all.timestamp.min()} → {df_all.timestamp.max()}')
    print(f'Tot_ict范围：{df_all.Tot_ict_kW.min():.1f} ~ {df_all.Tot_ict_kW.max():.1f} kW')
    print(f'\n前5行：')
    print(df_all.head())