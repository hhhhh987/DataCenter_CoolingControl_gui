"""
check_liquid_cooling.py
液冷侧数据合并、异常标记、物理量计算与验证
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────
ROOT = r'D:\PythonProject1'
DATA_01 = os.path.join(ROOT, '01dataFresh')
DATA_03 = os.path.join(ROOT, '03CoolingData')
OUT_DIR = os.path.join(DATA_03, 'liquid_check_results')

# 物理常数
CP_WATER = 4.18           # kJ/(kg·°C)，水的比热
RHO_WATER = 1000          # kg/m³，水的密度


# ── Block 1: 数据加载与合并 ──────────────────────────────

def load_and_merge():
    """合并logics(4) + schneider(5) 共9个字段到一张宽表"""
    files = {
        'tot_ict':       os.path.join(DATA_01, 'tot_ict.parquet'),
        'tot_chiller':   os.path.join(DATA_03, 'tot_chiller.parquet'),
        'tot_cdz':       os.path.join(DATA_03, 'tot_cdz.parquet'),
        'tot_qpompe':    os.path.join(DATA_03, 'tot_qpompe.parquet'),
        'portata':       os.path.join(DATA_03, 'schneider_portata.parquet'),
        't_mandata':     os.path.join(DATA_03, 'schneider_t_mandata.parquet'),
        't_ritorno':     os.path.join(DATA_03, 'schneider_t_ritorno.parquet'),
        'inverter':      os.path.join(DATA_03, 'schneider_inverter.parquet'),
        'pumps':         os.path.join(DATA_03, 'schneider_pumps.parquet'),
    }

    print('=== 加载各字段 ===')
    dfs = {}
    for name, path in files.items():
        df = pd.read_parquet(path)
        # pumps文件有多列，只保留timestamp和N_pumps_running
        if name == 'pumps':
            df = df[['timestamp', 'N_pumps_running']]
        dfs[name] = df
        print(f'  {name}: {len(df)} 行, 列: {list(df.columns)}')

    # 按timestamp逐个inner join
    merged = dfs['tot_ict']
    for name in ['tot_chiller', 'tot_cdz', 'tot_qpompe',
                 'portata', 't_mandata', 't_ritorno', 'inverter', 'pumps']:
        merged = merged.merge(dfs[name], on='timestamp', how='inner')

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    print(f'\n合并后行数: {len(merged)}')
    print(f'时间范围: {merged.timestamp.min()} → {merged.timestamp.max()}')
    return merged


# ── Block 2: 异常时段标记 ────────────────────────────────

def mark_anomalies(df):
    """4个判据联合标记异常时段"""
    df['delta_T'] = df['T_ritorno_C'] - df['T_mandata_C']

    cond_low_flow = df['Portata_m3h'] < 10
    cond_high_tman = df['T_mandata_C'] > 25
    cond_low_dt = df['delta_T'] < 1
    cond_high_dt = df['delta_T'] > 15

    df['is_anomaly'] = cond_low_flow | cond_high_tman | cond_low_dt | cond_high_dt

    print('\n=== 异常时段统计 ===')
    print(f'  低流量 (<10):       {cond_low_flow.sum()} ({cond_low_flow.mean()*100:.2f}%)')
    print(f'  高供水温度 (>25):    {cond_high_tman.sum()} ({cond_high_tman.mean()*100:.2f}%)')
    print(f'  低温差 (<1):         {cond_low_dt.sum()} ({cond_low_dt.mean()*100:.2f}%)')
    print(f'  高温差 (>15):        {cond_high_dt.sum()} ({cond_high_dt.mean()*100:.2f}%)')
    print(f'  联合异常总数:        {df.is_anomaly.sum()} ({df.is_anomaly.mean()*100:.2f}%)')
    print(f'  正常时段:           {(~df.is_anomaly).sum()}')

    return df


# ── Block 3: 物理量计算 ──────────────────────────────────

def compute_physics(df):
    """
    计算液冷制冷量与COP
    Q_液冷 [kW] = c_p [kJ/(kg·°C)] × ρ [kg/m³] × V [m³/s] × ΔT [°C]
    流量单位换算：m³/h → m³/s 除以3600
    """
    V_m3s = df['Portata_m3h'] / 3600
    df['Q_liquid_kW'] = CP_WATER * RHO_WATER * V_m3s * df['delta_T']

    # COP_chiller = 制冷量 / 冷水机耗电
    # 排除 Tot_chiller<10 的时段（冷水机停机，COP无意义）
    df['COP_chiller'] = np.where(
        df['Tot_chiller_kW'] > 10,
        df['Q_liquid_kW'] / df['Tot_chiller_kW'],
        np.nan
    )

    # α = 液冷分担的IT热量比例
    df['alpha_liquid'] = df['Q_liquid_kW'] / df['Tot_ict_kW']

    print('\n=== 物理量统计（仅正常时段）===')
    normal = df[~df['is_anomaly']]
    stats = normal[['Q_liquid_kW', 'COP_chiller', 'alpha_liquid']].describe(
        percentiles=[0.05, 0.5, 0.95]).round(3)
    print(stats)

    return df


# ── Block 4: 物理合理性验证 ──────────────────────────────

def validate_physics(df):
    """三层检查：相关性、COP合理性、α合理性"""
    normal = df[~df['is_anomaly']].copy()

    print('\n=== 物理合理性验证 ===')

    # 1. Q_液冷 与 Tot_ict 相关性
    corr_q_ict = normal['Q_liquid_kW'].corr(normal['Tot_ict_kW'])
    print(f'\n[1] Q_liquid 与 Tot_ict 相关系数: {corr_q_ict:.3f}')
    if corr_q_ict > 0.5:
        print('    ✓ 强正相关，物理上合理')
    else:
        print('    ⚠ 相关性偏低，需要进一步分析')

    # 2. COP合理性
    cop_valid = normal['COP_chiller'].dropna()
    cop_mean = cop_valid.mean()
    cop_median = cop_valid.median()
    cop_in_range = ((cop_valid >= 2) & (cop_valid <= 8)).mean() * 100
    print(f'\n[2] COP_chiller 中位数: {cop_median:.2f}, 均值: {cop_mean:.2f}')
    print(f'    COP在[2,8]范围内的比例: {cop_in_range:.1f}%')
    if 3 <= cop_median <= 6:
        print('    ✓ 典型水冷COP范围，合理')
    else:
        print('    ⚠ COP偏离典型范围，可能空调也从冷水机取冷')

    # 3. α合理性（液冷分担比例）
    alpha_valid = normal['alpha_liquid'].dropna()
    alpha_median = alpha_valid.median()
    alpha_in_range = ((alpha_valid >= 0) & (alpha_valid <= 1)).mean() * 100
    print(f'\n[3] α=Q_liquid/Tot_ict 中位数: {alpha_median:.2f}')
    print(f'    5%~95%: {alpha_valid.quantile(0.05):.2f} ~ {alpha_valid.quantile(0.95):.2f}')
    print(f'    α在[0,1]范围内的比例: {alpha_in_range:.1f}%')
    if 0.4 <= alpha_median <= 0.9:
        print('    ✓ 液冷主导（带走IT热量的40~90%），符合M100架构预期')
    elif alpha_median > 1:
        print('    ⚠ α>1 物理不可能，说明流量或温差计算有偏差')
    else:
        print('    ⚠ 液冷分担比例偏低，需要核对')


# ── Block 5: 可视化 ──────────────────────────────────────

def plot_all(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    normal = df[~df['is_anomaly']]

    # 图1: 异常时段时序
    plt.figure(figsize=(14, 3))
    plt.plot(df.timestamp, df.is_anomaly.astype(int),
             linewidth=0.5, color='red')
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Anomaly')
    plt.title(f'Anomaly Distribution Over Time  |  Total: {df.is_anomaly.sum()} hours')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'anomaly_timeline.png'), dpi=150)
    plt.close()

    # 图2: Q_液冷时序
    plt.figure(figsize=(14, 4))
    plt.plot(normal.timestamp, normal.Q_liquid_kW, linewidth=0.5, alpha=0.8)
    plt.ylabel('Q_liquid (kW)')
    plt.title(f'Liquid Cooling Capacity  |  Mean={normal.Q_liquid_kW.mean():.0f}kW')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'q_liquid_timeseries.png'), dpi=150)
    plt.close()

    # 图3: Q_液冷 vs Tot_ict 散点
    plt.figure(figsize=(8, 6))
    plt.scatter(normal.Tot_ict_kW, normal.Q_liquid_kW, s=2, alpha=0.3)
    # y=x 参考线
    vmax = normal.Tot_ict_kW.max()
    plt.plot([0, vmax], [0, vmax], 'r--', alpha=0.5, label='y=x (α=1)')
    plt.plot([0, vmax], [0, vmax*0.7], 'g--', alpha=0.5, label='α=0.7')
    plt.xlabel('Tot_ict (kW)')
    plt.ylabel('Q_liquid (kW)')
    corr = normal['Q_liquid_kW'].corr(normal['Tot_ict_kW'])
    plt.title(f'Q_liquid vs Tot_ict  |  Correlation={corr:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'q_liquid_vs_it.png'), dpi=150)
    plt.close()

    # 图4: COP分布
    cop = normal.COP_chiller.dropna()
    cop = cop[(cop > 0) & (cop < 15)]  # 截掉极端值方便看分布
    plt.figure(figsize=(8, 5))
    plt.hist(cop, bins=80, edgecolor='black', alpha=0.7)
    plt.axvline(cop.median(), color='red', linestyle='--',
                label=f'Median={cop.median():.2f}')
    plt.xlabel('COP_chiller')
    plt.ylabel('Count')
    plt.title('Chiller COP Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cop_distribution.png'), dpi=150)
    plt.close()

    # 图5: 温差 vs 流量 联合分布
    plt.figure(figsize=(8, 6))
    plt.scatter(normal.Portata_m3h, normal.delta_T, s=2, alpha=0.3)
    plt.xlabel('Flow rate (m³/h)')
    plt.ylabel('ΔT (°C)')
    plt.title('Flow vs Temperature Difference')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'flow_vs_dt.png'), dpi=150)
    plt.close()

    # 图6: α分布（液冷分担比例）
    alpha = normal.alpha_liquid.dropna()
    alpha = alpha[(alpha > 0) & (alpha < 2)]
    plt.figure(figsize=(8, 5))
    plt.hist(alpha, bins=80, edgecolor='black', alpha=0.7)
    plt.axvline(alpha.median(), color='red', linestyle='--',
                label=f'Median={alpha.median():.2f}')
    plt.axvline(1.0, color='black', linestyle=':', label='α=1 (上限)')
    plt.xlabel('α = Q_liquid / Tot_ict')
    plt.ylabel('Count')
    plt.title('Liquid Cooling Share Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'alpha_distribution.png'), dpi=150)
    plt.close()

    print(f'\n图表已保存到: {OUT_DIR}')


# ── Block 6: 输出干净数据集 ──────────────────────────────

def save_datasets(df):
    """保存两份数据集：完整版（带异常标记）+ 干净版（剔除异常）"""
    full_path = os.path.join(DATA_03, 'cooling_full.parquet')
    clean_path = os.path.join(DATA_03, 'cooling_clean.parquet')

    df.to_parquet(full_path, index=False, compression='zstd')
    df[~df['is_anomaly']].drop(columns=['is_anomaly']).to_parquet(
        clean_path, index=False, compression='zstd')

    print(f'\n=== 数据集已保存 ===')
    print(f'  完整版（带异常标记）: {full_path}')
    print(f'  干净版（剔除异常）:   {clean_path}')
    print(f'  完整版行数: {len(df)}')
    print(f'  干净版行数: {(~df.is_anomaly).sum()}')


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_and_merge()
    df = mark_anomalies(df)
    df = compute_physics(df)
    validate_physics(df)
    plot_all(df)
    save_datasets(df)