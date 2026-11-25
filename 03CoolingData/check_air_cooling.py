"""
check_air_cooling.py
空调侧验证 + 完整热平衡验证 + 冷水机架构判断
通过数据驱动反演确定空调额定风量V_RATED
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────
ROOT = r'D:\PythonProject1'
DATA_03 = os.path.join(ROOT, '03CoolingData')
DATA_01 = os.path.join(ROOT, '01dataFresh')
OUT_DIR = os.path.join(DATA_03, 'air_check_results')

# 物理常数
CP_AIR = 1.005             # kJ/(kg·°C)
RHO_AIR = 1.2              # kg/m³
# V_RATED_CDZ 不预先赋值，由Block 3.5反推得到

# 反推开机台数的阈值
COMPRESSOR_RUNNING_THRESHOLD = 5   # % 压缩机>5%认为在制冷


# ── Block 1: 加载空调侧数据 ──────────────────────────────

def load_air_data():
    """加载5个vertiv长表数据"""
    files = {
        'fan_speed':    'vertiv_fan_speed.parquet',
        'supply_temp':  'vertiv_supply_temp.parquet',
        'return_temp':  'vertiv_return_temp.parquet',
        'compressor':   'vertiv_compressor.parquet',
        'free_cooling': 'vertiv_free_cooling.parquet',
    }

    dfs = {}
    print('=== 加载空调数据 ===')
    for name, fname in files.items():
        path = os.path.join(DATA_03, fname)
        df = pd.read_parquet(path)
        dfs[name] = df
        print(f'  {name}: {len(df)} 行')

    merged = dfs['fan_speed']
    for name in ['supply_temp', 'return_temp', 'compressor', 'free_cooling']:
        merged = merged.merge(dfs[name], on=['timestamp', 'device'], how='outer')

    merged['Free_Cooling_Status'] = merged['Free_Cooling_Status'].fillna(0)

    print(f'\n空调长表合并: {len(merged)} 行')
    return merged


# ── Block 2: 反推开机台数 ──────────────────────────────────
def estimate_n_cdz(df_long):
    """
    每小时实际制冷台数（修正版）
    判定单台是否在制冷：满足任一条件
    1. 压缩机利用率 > 5%
    2. Free Cooling状态 > 0 且 有制冷量（送风<回风）
    """
    df_long = df_long.copy()
    df_long['delta_T_air'] = df_long['Return_Air_Temp_C'] - df_long['Supply_Air_Temp_C']

    # 条件1：压缩机在工作
    cond_compressor = df_long['Compressor_pct'] > COMPRESSOR_RUNNING_THRESHOLD

    # 条件2：自然冷却启用且产生有效制冷（送风温度低于回风）
    cond_free_cooling = (df_long['Free_Cooling_Status'] > 0.5) & (df_long['delta_T_air'] > 1.0)

    # 任一条件成立 = 在制冷
    df_long['is_running'] = cond_compressor | cond_free_cooling

    n_cdz = df_long.groupby('timestamp')['is_running'].sum().reset_index()
    n_cdz.columns = ['timestamp', 'N_cdz_running']

    print('\n=== N_cdz_running 分布（修正版）===')
    print(n_cdz['N_cdz_running'].describe().round(2))
    print(f'\n各开机台数的小时分布:')
    print(n_cdz['N_cdz_running'].value_counts().sort_index())

    # 对比：纯Compressor判据下的台数
    df_long['is_running_compressor_only'] = cond_compressor
    n_cdz_old = df_long.groupby('timestamp')['is_running_compressor_only'].sum()
    print(f'\n纯Compressor判据 vs 联合判据 对比:')
    print(f'  纯Compressor判据 N=0 小时数: {(n_cdz_old == 0).sum()}')
    print(f'  联合判据 N=0 小时数:        {(n_cdz["N_cdz_running"] == 0).sum()}')
    print(f'  挽回的小时数:               {(n_cdz_old == 0).sum() - (n_cdz["N_cdz_running"] == 0).sum()}')

    return n_cdz

# ── Block 3.5: 数据驱动反推 V_RATED ──────────────────────

def fit_v_rated(df_long_air):
    """
    通过一元无截距线性回归反推空调额定风量
    模型：Y = V_RATED * X
        Y = Tot_ict - Q_液冷  （空调侧应承担的冷量缺口）
        X = 0.000335 * Σ_i (Fan_Speed_i / 100) * ΔT_i
            其中 0.000335 = c_p × ρ / 3600 = 1.005 × 1.2 / 3600
    """
    print('\n=== Block 3.5: V_RATED 反推 ===')

    # 1. 读取已有的液冷数据（Q_液冷 和 Tot_ict 都在cooling_full里）
    liquid_path = os.path.join(DATA_03, 'cooling_full.parquet')
    df_liquid = pd.read_parquet(liquid_path)

    # 2. 计算每台空调每小时的"单位风量制冷因子"
    df_unit = df_long_air.copy()
    df_unit['delta_T_air'] = df_unit['Return_Air_Temp_C'] - df_unit['Supply_Air_Temp_C']
    df_unit['unit_capacity_i'] = (
        (CP_AIR * RHO_AIR / 3600) *
        (df_unit['Fan_Speed_pct'] / 100) *
        df_unit['delta_T_air']
    )

    # 3. 6台聚合：每小时X值
    X_series = df_unit.groupby('timestamp')['unit_capacity_i'].sum().reset_index()
    X_series.columns = ['timestamp', 'X']

    # 4. 合并：算Y = Tot_ict - Q_液冷
    merged = df_liquid[['timestamp', 'Tot_ict_kW', 'Q_liquid_kW', 'is_anomaly']].merge(
        X_series, on='timestamp', how='inner'
    )
    merged['Y'] = merged['Tot_ict_kW'] - merged['Q_liquid_kW']

    # 5. 过滤：异常时段 + 物理不合理的负值
    valid = merged[
        (~merged['is_anomaly']) &
        (merged['Y'] > 0) &
        (merged['X'] > 0)
    ].copy()

    print(f'  全部样本数: {len(merged)}')
    print(f'  有效样本数: {len(valid)} (排除异常和负值)')

    # 6. 一元无截距最小二乘
    X = valid['X'].values
    Y = valid['Y'].values
    V_rated = np.sum(X * Y) / np.sum(X * X)

    # 7. 拟合优度 R²
    Y_pred = V_rated * X
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 8. 残差分析
    residuals = Y - Y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))

    print(f'  反推 V_RATED:  {V_rated:.1f} m³/h')
    print(f'  拟合优度 R²:   {r2:.4f}')
    print(f'  残差RMSE:     {rmse:.2f} kW')

    if 8000 <= V_rated <= 15000:
        print(f'  ✓ V_RATED在典型CRAC范围(8000~15000)')
    else:
        print(f'  ⚠ V_RATED偏离典型范围，需要排查')

    if r2 > 0.8:
        print('  ✓ R²>0.8，热平衡假设在数据上得到验证')
    elif r2 > 0.5:
        print('  ⚠ R²在0.5~0.8，部分热量可能未建模（机房通风、辐射等）')
    else:
        print('  ✗ R²<0.5，物理假设有严重问题')

    # 9. 保存反推过程图
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, s=2, alpha=0.3, label='Data')
    X_line = np.array([X.min(), X.max()])
    plt.plot(X_line, V_rated * X_line, 'r-', label=f'Y = {V_rated:.0f} × X')
    plt.xlabel('X = c_p·ρ/3600 · Σ (Fan_Speed_i/100) · ΔT_i')
    plt.ylabel('Y = Tot_ict - Q_liquid (kW)')
    plt.title(f'V_RATED Inverse Fitting  |  V_RATED={V_rated:.0f}, R²={r2:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'v_rated_fitting.png'), dpi=150)
    plt.close()

    return V_rated, r2


# ── Block 3: 计算空调制冷量（用反推的V_RATED）──────────

def compute_q_air(df_long, V_RATED):
    """
    Q_air_i = c_p * ρ * V_i * ΔT_i
    V_i = (Fan_Speed_i / 100) * V_RATED
    """
    df_long['V_air_m3h'] = (df_long['Fan_Speed_pct'] / 100) * V_RATED
    df_long['delta_T_air'] = df_long['Return_Air_Temp_C'] - df_long['Supply_Air_Temp_C']

    V_m3s = df_long['V_air_m3h'] / 3600
    df_long['Q_air_kW'] = CP_AIR * RHO_AIR * V_m3s * df_long['delta_T_air']

    q_air_total = df_long.groupby('timestamp')['Q_air_kW'].sum().reset_index()
    q_air_total.columns = ['timestamp', 'Q_air_total_kW']

    print('\n=== 单台 Q_air 统计 ===')
    print(df_long['Q_air_kW'].describe(percentiles=[0.05, 0.5, 0.95]).round(2))
    print('\n=== 6台合计 Q_air_total 统计 ===')
    print(q_air_total['Q_air_total_kW'].describe(percentiles=[0.05, 0.5, 0.95]).round(2))

    return q_air_total


# ── Block 4: 合并所有数据 ───────────────────────────────
def merge_all(q_air_total, n_cdz):
    """合并空调侧数据 + 之前的cooling_full数据 + 异常标记（修正版）"""
    liquid_path = os.path.join(DATA_03, 'cooling_full.parquet')
    df_liquid = pd.read_parquet(liquid_path)

    df = df_liquid.merge(q_air_total, on='timestamp', how='inner')
    df = df.merge(n_cdz, on='timestamp', how='inner')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 修正后的异常判定
    # 不再用 N_cdz=0 判定异常，因为可能是数据集中Free Cooling字段缺失导致的
    # 改用：Q_air严重负值 或 极端大值
    df['is_anomaly_air'] = (
            (df['Q_air_total_kW'] < -50) |  # 允许小幅负值（测量噪声）
            (df['Q_air_total_kW'] > 2000)
    )
    df['is_anomaly_full'] = df['is_anomaly'] | df['is_anomaly_air']

    print(f'\n=== 合并后数据（修正版异常判定）===')
    print(f'  总行数: {len(df)}')
    print(f'  液冷异常: {df.is_anomaly.sum()} 小时')
    print(f'  空调异常（修正后）: {df.is_anomaly_air.sum()} 小时')
    print(f'  联合异常: {df.is_anomaly_full.sum()} 小时')
    print(f'  正常时段: {(~df.is_anomaly_full).sum()} 小时')

    return df


# ── Block 5: 热平衡验证 ──────────────────────────────────

def validate_heat_balance(df):
    normal = df[~df['is_anomaly_full']].copy()

    normal['Q_cooling_total'] = normal['Q_liquid_kW'] + normal['Q_air_total_kW']
    normal['balance_ratio'] = normal['Q_cooling_total'] / normal['Tot_ict_kW']
    normal['residual'] = normal['Tot_ict_kW'] - normal['Q_cooling_total']

    print('\n=== 热平衡验证 ===')
    print(f'\n[Q_液冷 + Q_空调] / Tot_ict 比例:')
    print(normal['balance_ratio'].describe(percentiles=[0.05, 0.5, 0.95]).round(3))

    print(f'\n残差 Tot_ict - (Q_液冷 + Q_空调) (kW):')
    print(normal['residual'].describe(percentiles=[0.05, 0.5, 0.95]).round(2))

    ratio_med = normal['balance_ratio'].median()
    if 0.85 <= ratio_med <= 1.15:
        verdict = '✓ 热平衡基本成立（误差<15%）'
    elif ratio_med < 0.85:
        verdict = f'⚠ 总制冷量偏小（{ratio_med:.2f}），存在未建模散热通路'
    else:
        verdict = f'⚠ 总制冷量偏大（{ratio_med:.2f}），V_RATED反推可能有偏'
    print(f'\n结论: {verdict}')

    return normal


# ── Block 6: 冷水机架构判断 ──────────────────────────────

def diagnose_chiller_architecture(df):
    normal = df[~df['is_anomaly_full']].copy()

    normal['COP_corrected'] = np.where(
        normal['Tot_chiller_kW'] > 10,
        (normal['Q_liquid_kW'] + normal['Q_air_total_kW']) / normal['Tot_chiller_kW'],
        np.nan
    )
    normal['COP_liquid_only'] = np.where(
        normal['Tot_chiller_kW'] > 10,
        normal['Q_liquid_kW'] / normal['Tot_chiller_kW'],
        np.nan
    )

    cop_corr = normal['COP_corrected'].dropna()
    cop_only = normal['COP_liquid_only'].dropna()

    print('\n=== 冷水机架构判断 ===')
    print(f'\n[假设A] 冷水机只供液冷:')
    print(f'  COP_liquid_only 中位数: {cop_only.median():.2f}')
    print(f'\n[假设B] 冷水机同时供液冷+空调:')
    print(f'  COP_corrected 中位数: {cop_corr.median():.2f}')

    if 3.5 <= cop_corr.median() <= 6.0:
        arch = 'B (同时供液冷+空调)'
        print(f'\n✓ 架构判定: {arch}')
        print('  修正COP在合理范围(3.5~6.0)，说明冷水机同时给两侧供冷')
    elif cop_corr.median() > 6.0:
        arch = '部分共供 (Free Cooling模式下空调借用冷水)'
        print(f'\n⚠ 架构判定: {arch}')
    else:
        arch = 'A (冷水机只供液冷)'
        print(f'\n✓ 架构判定: {arch}')
        print('  即使加上空调制冷量COP仍偏低，说明空调有独立冷源（直膨式）')

    normal['month'] = normal['timestamp'].dt.month
    summer = normal[normal['month'].isin([6, 7, 8])]
    winter = normal[normal['month'].isin([12, 1, 2])]
    print(f'\n[季节性验证]')
    print(f'  夏季Tot_chiller均值: {summer["Tot_chiller_kW"].mean():.1f} kW')
    print(f'  冬季Tot_chiller均值: {winter["Tot_chiller_kW"].mean():.1f} kW')
    print(f'  夏季Q_air均值: {summer["Q_air_total_kW"].mean():.1f} kW')
    print(f'  冬季Q_air均值: {winter["Q_air_total_kW"].mean():.1f} kW')

    return normal, arch


# ── Block 7: 可视化 ──────────────────────────────────────

def plot_all(df_full, df_normal):
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(df_normal['balance_ratio'], bins=80, edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', label='ratio=1.0')
    plt.axvline(df_normal['balance_ratio'].median(), color='green',
                linestyle='--', label=f'Median={df_normal["balance_ratio"].median():.3f}')
    plt.xlabel('(Q_liquid + Q_air) / Tot_ict')
    plt.ylabel('Count')
    plt.title('Heat Balance Ratio Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'heat_balance_ratio.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(df_normal.timestamp, df_normal.Tot_ict_kW, label='Tot_ict', linewidth=0.5, alpha=0.7)
    plt.plot(df_normal.timestamp, df_normal.Q_liquid_kW, label='Q_liquid', linewidth=0.5, alpha=0.7)
    plt.plot(df_normal.timestamp, df_normal.Q_air_total_kW, label='Q_air', linewidth=0.5, alpha=0.7)
    plt.plot(df_normal.timestamp, df_normal.Q_cooling_total,
             label='Q_liquid+Q_air', linewidth=0.5, alpha=0.7, color='black')
    plt.ylabel('Power / Cooling (kW)')
    plt.title('Heat Balance Time Series')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'heat_balance_timeseries.png'), dpi=150)
    plt.close()

    cop = df_normal.COP_corrected.dropna()
    cop = cop[(cop > 0) & (cop < 15)]
    plt.figure(figsize=(10, 5))
    plt.hist(cop, bins=80, edgecolor='black', alpha=0.7)
    plt.axvline(cop.median(), color='red', linestyle='--', label=f'Median={cop.median():.2f}')
    plt.axvline(4.0, color='green', linestyle=':', label='Typical chiller COP')
    plt.xlabel('COP_corrected = (Q_liquid + Q_air) / Tot_chiller')
    plt.ylabel('Count')
    plt.title('Corrected Chiller COP Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cop_corrected.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    counts = df_normal.N_cdz_running.value_counts().sort_index()
    plt.bar(counts.index, counts.values, edgecolor='black', alpha=0.7)
    plt.xlabel('N_cdz_running')
    plt.ylabel('Hours')
    plt.title('Active CDZ Count Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'n_cdz_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df_normal.Q_air_total_kW, df_normal.Tot_cdz_kW, s=2, alpha=0.3)
    corr = df_normal[['Q_air_total_kW', 'Tot_cdz_kW']].corr().iloc[0, 1]
    plt.xlabel('Q_air_total (kW)')
    plt.ylabel('Tot_cdz (kW)')
    plt.title(f'Q_air vs Tot_cdz  |  Correlation={corr:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'q_air_vs_tot_cdz.png'), dpi=150)
    plt.close()

    df_normal['month'] = df_normal['timestamp'].dt.month
    monthly = df_normal.groupby('month').agg(
        Tot_chiller_mean=('Tot_chiller_kW', 'mean'),
        Q_air_mean=('Q_air_total_kW', 'mean'),
        Q_liquid_mean=('Q_liquid_kW', 'mean'),
    ).reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly.month, monthly.Tot_chiller_mean, 'o-', label='Tot_chiller')
    plt.plot(monthly.month, monthly.Q_liquid_mean, 's-', label='Q_liquid')
    plt.plot(monthly.month, monthly.Q_air_mean, '^-', label='Q_air')
    plt.xlabel('Month')
    plt.ylabel('Mean Power / Cooling (kW)')
    plt.title('Monthly Pattern')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'monthly_pattern.png'), dpi=150)
    plt.close()

    print(f'\n图表已保存到: {OUT_DIR}')


def save_final_datasets(df_full, arch_conclusion, V_rated, r2):
    full_path = os.path.join(DATA_03, 'cooling_all_full.parquet')
    clean_path = os.path.join(DATA_03, 'cooling_all_clean.parquet')

    df_full.to_parquet(full_path, index=False, compression='zstd')
    clean_df = df_full[~df_full['is_anomaly_full']].drop(
        columns=['is_anomaly', 'is_anomaly_air', 'is_anomaly_full']
    )
    clean_df.to_parquet(clean_path, index=False, compression='zstd')

    report_path = os.path.join(DATA_03, 'architecture_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'M100冷却架构判断报告\n')
        f.write(f'{"="*50}\n')
        f.write(f'V_RATED 反推结果: {V_rated:.1f} m³/h\n')
        f.write(f'拟合优度 R²: {r2:.4f}\n')
        f.write(f'数据集行数: {len(df_full)}\n')
        f.write(f'正常时段: {(~df_full.is_anomaly_full).sum()}\n')
        f.write(f'架构判定: {arch_conclusion}\n')

    print(f'\n=== 最终输出 ===')
    print(f'  完整版: {full_path} ({len(df_full)} 行)')
    print(f'  干净版: {clean_path} ({len(clean_df)} 行)')
    print(f'  架构报告: {report_path}')


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':
    # 1. 加载空调长表
    df_long = load_air_data()

    # 2. 反推开机台数
    n_cdz = estimate_n_cdz(df_long)

    # 3. 数据驱动反推 V_RATED（新增）
    V_RATED, R2 = fit_v_rated(df_long)

    # 4. 用反推的V_RATED计算Q_air
    q_air = compute_q_air(df_long, V_RATED)

    # 5. 合并所有数据
    df_full = merge_all(q_air, n_cdz)

    # 6. 热平衡验证
    df_normal = validate_heat_balance(df_full)

    # 7. 架构判断
    df_normal, arch = diagnose_chiller_architecture(df_normal)

    # 8. 可视化 + 输出
    plot_all(df_full, df_normal)
    save_final_datasets(df_full, arch, V_RATED, R2)