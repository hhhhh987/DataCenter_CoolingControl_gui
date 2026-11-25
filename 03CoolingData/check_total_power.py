"""
check_total_power.py
验证第一批提取数据的物理合理性：
- 时间戳对齐
- 描述性统计
- PUE计算
- 冷源总功耗与IT功耗相关性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = r'D:\PythonProject1'
TOT_ICT_PATH = os.path.join(ROOT, '01dataFresh', 'tot_ict.parquet')
TOT_CHILLER_PATH = os.path.join(ROOT, '03CoolingData', 'tot_chiller.parquet')
TOT_CDZ_PATH = os.path.join(ROOT, '03CoolingData', 'tot_cdz.parquet')
TOT_QPOMPE_PATH = os.path.join(ROOT, '03CoolingData', 'tot_qpompe.parquet')
OUT_DIR = os.path.join(ROOT, '03CoolingData', 'check_results')


def load_and_merge():
    df_ict = pd.read_parquet(TOT_ICT_PATH)
    df_chiller = pd.read_parquet(TOT_CHILLER_PATH)
    df_cdz = pd.read_parquet(TOT_CDZ_PATH)
    df_qpompe = pd.read_parquet(TOT_QPOMPE_PATH)

    print('各字段原始行数:')
    print(f'  Tot_ict:     {len(df_ict)}')
    print(f'  Tot_chiller: {len(df_chiller)}')
    print(f'  Tot_cdz:     {len(df_cdz)}')
    print(f'  Tot_qpompe:  {len(df_qpompe)}')

    df = df_ict.merge(df_chiller, on='timestamp', how='inner')
    df = df.merge(df_cdz, on='timestamp', how='inner')
    df = df.merge(df_qpompe, on='timestamp', how='inner')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f'\n四字段完整合并后: {len(df)} 行')
    print(f'时间范围: {df.timestamp.min()} → {df.timestamp.max()}')
    return df


def describe(df):
    cols = ['Tot_ict_kW', 'Tot_chiller_kW', 'Tot_cdz_kW', 'Tot_qpompe_kW']
    stats = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print('\n描述性统计 (kW):')
    print(stats.round(2))
    return stats


def compute_pue(df):
    df['P_cooling'] = df['Tot_chiller_kW'] + df['Tot_cdz_kW'] + df['Tot_qpompe_kW']
    df['P_total'] = df['Tot_ict_kW'] + df['P_cooling']
    df['PUE'] = df['P_total'] / df['Tot_ict_kW']

    print(f'\nPUE统计:')
    print(f'  均值:   {df.PUE.mean():.3f}')
    print(f'  中位数: {df.PUE.median():.3f}')
    print(f'  5%~95%: {df.PUE.quantile(0.05):.3f} ~ {df.PUE.quantile(0.95):.3f}')
    print(f'  范围:   {df.PUE.min():.3f} ~ {df.PUE.max():.3f}')

    n_abnormal = ((df.PUE < 1.0) | (df.PUE > 2.5)).sum()
    print(f'  异常PUE (<1.0 或 >2.5): {n_abnormal} 行 ({n_abnormal/len(df)*100:.1f}%)')
    return df


def compute_correlation(df):
    corr = df[['Tot_ict_kW', 'P_cooling', 'Tot_chiller_kW', 'Tot_cdz_kW', 'Tot_qpompe_kW']].corr()
    print('\n相关系数矩阵:')
    print(corr.round(3))
    print(f'\n冷源总功耗与IT功耗的相关系数: {corr.loc["P_cooling", "Tot_ict_kW"]:.3f}')


def plot_all(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df.timestamp, df.Tot_ict_kW, color='tab:blue', linewidth=0.5)
    axes[0].set_ylabel('Tot_ict (kW)')
    axes[1].plot(df.timestamp, df.Tot_chiller_kW, color='tab:red', linewidth=0.5)
    axes[1].set_ylabel('Tot_chiller (kW)')
    axes[2].plot(df.timestamp, df.Tot_cdz_kW, color='tab:green', linewidth=0.5)
    axes[2].set_ylabel('Tot_cdz (kW)')
    axes[3].plot(df.timestamp, df.Tot_qpompe_kW, color='tab:orange', linewidth=0.5)
    axes[3].set_ylabel('Tot_qpompe (kW)')
    axes[3].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'timeseries_four_powers.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(df.timestamp, df.PUE, color='tab:purple', linewidth=0.5)
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='PUE=1.0')
    plt.axhline(1.5, color='gray', linestyle='--', alpha=0.5, label='PUE=1.5')
    plt.ylabel('PUE')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pue_timeseries.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df.Tot_ict_kW, df.P_cooling, s=2, alpha=0.3)
    corr_val = df[['Tot_ict_kW', 'P_cooling']].corr().iloc[0, 1]
    plt.xlabel('Tot_ict (kW)')
    plt.ylabel('P_cooling = chiller + cdz + qpompe (kW)')
    plt.title(f'Cooling Power vs IT Power  |  Correlation = {corr_val:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cooling_vs_it_scatter.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df.PUE, bins=80, edgecolor='black', alpha=0.7)
    plt.axvline(df.PUE.median(), color='red', linestyle='--', label=f'Median={df.PUE.median():.3f}')
    plt.xlabel('PUE')
    plt.ylabel('Count')
    plt.title('PUE Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pue_histogram.png'), dpi=150)
    plt.close()

    print(f'\n图表已保存到: {OUT_DIR}')


if __name__ == '__main__':
    df = load_and_merge()
    describe(df)
    df = compute_pue(df)
    compute_correlation(df)
    plot_all(df)