"""
model_cdz.py

空调(CDZ)功耗统一建模: P_cdz = f(Q_air, T_amb, N_cdz, N_FC, s_fan)
一个多项式模型覆盖纯DX / 纯FC / 混合三种模式

GA 用法:
  Q_air = physics_formula(N_cdz, N_FC, s_fan, T_amb)  (物理公式算冷量)
  P_cdz = cdz_model.predict(Q_air, T_amb, N_cdz, N_FC, s_fan)  (模型算功耗)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error

ROOT     = r'D:\PythonProject1'
COOL_DIR = os.path.join(ROOT, '03CoolingData')
WX_PATH  = os.path.join(ROOT, '01dataFresh', 'weather.parquet')
MODES_PATH = os.path.join(ROOT, '04EquipmentOptimization', 'cooling_with_modes.parquet')
OUT_DIR  = os.path.join(ROOT, '04EquipmentOptimization', 'cdz_results')
os.makedirs(OUT_DIR, exist_ok=True)

# 物理常数
RHO_AIR = 1.2        # kg/m³
CP_AIR  = 1.005      # kJ/(kg·°C)
V_RATED = 19423.0     # m³/h per unit (等效额定风量)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ── PART 1: 加载数据 ───────────────────────────────────────

def load_data() -> pd.DataFrame:
    print('[PART 1] 加载数据 ...')

    df = pd.read_parquet(MODES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f'  cooling_with_modes: {len(df):,} 行')

    # 合并 T_amb
    wx = pd.read_parquet(WX_PATH)
    wx['timestamp'] = pd.to_datetime(wx['timestamp'], utc=True)
    t_col = next(c for c in wx.columns if c.lower() in ('temp_c', 't_amb'))
    wx = wx[['timestamp', t_col]].rename(columns={t_col: 'temp_C'})
    df = df.merge(wx, on='timestamp', how='inner')

    # 合并 vertiv 回风/送风温度(长表 → 小时均值)
    for fname, out_col in [('vertiv_return_temp.parquet', 'T_return_air'),
                           ('vertiv_supply_temp.parquet', 'T_supply_air')]:
        vt = pd.read_parquet(os.path.join(COOL_DIR, fname))
        vt['timestamp'] = pd.to_datetime(vt['timestamp'], utc=True)
        val_col = [c for c in vt.columns if c not in ('timestamp', 'device', 'panel')][0]
        vt_hourly = vt.groupby('timestamp')[val_col].mean().reset_index()
        vt_hourly.columns = ['timestamp', out_col]
        df = df.merge(vt_hourly, on='timestamp', how='inner')

    # 基础过滤
    df = df[(df['N_cdz_running'] > 0) & (df['Tot_cdz_kW'] > 0)].copy()
    print(f'  合并后: {len(df):,} 行')
    return df


# ── PART 2: 提取物理常数 ──────────────────────────────────

def extract_constants(df: pd.DataFrame) -> dict:
    print('\n[PART 2] 提取物理常数 ...')

    # T_return: 全时段
    t_ret_mean = df['T_return_air'].mean()
    t_ret_std  = df['T_return_air'].std()
    print(f'  T_return: 均值={t_ret_mean:.1f}°C, std={t_ret_std:.1f}°C', end='')
    if t_ret_std < 3.0:
        print('  ✓ 稳定,可当常数')
    else:
        print('  ⚠ 波动较大')

    # T_supply_DX: 仅纯 DX 时段
    dx_mask = df['N_FC_on'] == 0
    t_sup_dx_mean = df.loc[dx_mask, 'T_supply_air'].mean()
    t_sup_dx_std  = df.loc[dx_mask, 'T_supply_air'].std()
    print(f'  T_supply_DX: 均值={t_sup_dx_mean:.1f}°C, std={t_sup_dx_std:.1f}°C '
          f'(纯DX {dx_mask.sum():,}h)')

    # Q_DX_rated: 纯 DX 时段,单台制冷量的 P95
    dx_data = df[dx_mask & (df['Q_air_total_kW'] > 0)].copy()
    dx_data['Q_per_unit'] = dx_data['Q_air_total_kW'] / dx_data['N_cdz_running']
    q_dx_rated = float(np.percentile(dx_data['Q_per_unit'], 95))
    print(f'  Q_DX_rated (P95 per unit): {q_dx_rated:.1f} kW')

    # s_fan 范围(GA 约束用)
    s_fan_min = float(df.loc[df['Fan_Speed_avg'] > 0, 'Fan_Speed_avg'].quantile(0.05))
    s_fan_max = float(df['Fan_Speed_avg'].max())
    print(f'  s_fan 范围: [{s_fan_min:.1f}, {s_fan_max:.1f}]%')

    consts = {
        'T_return_const':    round(float(t_ret_mean), 1),
        'T_return_std':      round(t_ret_std, 2),
        'T_supply_DX_const': round(t_sup_dx_mean, 1),
        'T_supply_DX_std':   round(t_sup_dx_std, 2),
        'Q_DX_rated':        round(q_dx_rated, 1),
        'V_rated':           V_RATED,
        's_fan_min':         round(s_fan_min, 1),
        's_fan_max':         round(s_fan_max, 1),
    }
    return consts


# ── PART 3: 统一模型 P_cdz = f(Q_air, T_amb, N_cdz, N_FC, s_fan) ──

def fit_unified(df: pd.DataFrame) -> dict:
    print('\n[PART 3] 统一模型拟合 ...')

    cols = ['Q_air_total_kW', 'temp_C', 'N_cdz_running', 'N_FC_on', 'Fan_Speed_avg']
    df_fit = df.dropna(subset=cols + ['Tot_cdz_kW']).copy()
    print(f'  拟合样本: {len(df_fit):,}')

    X_raw = df_fit[cols].values
    y = df_fit['Tot_cdz_kW'].values

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_raw)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    names = poly.get_feature_names_out(['Q', 'T', 'N', 'Nfc', 's'])
    print(f'  R2={r2:.4f}, MAE={mae:.2f} kW')

    # 按模式分报告
    for mode, mask_fn in [('纯DX', lambda d: d['N_FC_on'] == 0),
                          ('纯FC', lambda d: d['N_DX'] == 0),
                          ('混合', lambda d: (d['N_FC_on'] > 0) & (d['N_DX'] > 0))]:
        m = mask_fn(df_fit)
        if m.sum() > 0:
            r2_m = r2_score(y[m], y_pred[m])
            mae_m = mean_absolute_error(y[m], y_pred[m])
            print(f'    {mode}: n={m.sum():,}, R2={r2_m:.4f}, MAE={mae_m:.2f} kW')

    return {
        'poly': poly,
        'model': model,
        'r2': r2,
        'mae': mae,
        'n_samples': len(df_fit),
        'feature_names': names,
        'coeffs': model.coef_,
        'input_cols': cols,
    }
# ── PART 4: 诊断图 ────────────────────────────────────────

def diagnostics(df: pd.DataFrame, unified: dict):
    print('\n[PART 4] 诊断图 ...')

    cols = unified['input_cols']
    df_v = df.dropna(subset=cols + ['Tot_cdz_kW']).copy()
    X_poly = unified['poly'].transform(df_v[cols].values)
    P_pred = unified['model'].predict(X_poly)
    P_true = df_v['Tot_cdz_kW'].values

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(P_true, P_pred, s=3, alpha=0.3, color='steelblue')
    lims = [0, max(P_true.max(), P_pred.max()) * 1.05]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('实测 Tot_cdz (kW)')
    ax.set_ylabel('预测 (kW)')
    ax.set_title(f'CDZ 统一模型  R2={unified["r2"]:.3f}, MAE={unified["mae"]:.1f} kW')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'cdz_fit.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ cdz_fit.png')

# ── PART 5: 保存 + 报告 ───────────────────────────────────

def save_and_report(consts: dict, unified: dict):
    print('\n[PART 5] 保存 + 报告 ...')

    save_obj = {
        'constants': consts,
        'unified_model': {
            'poly':       unified['poly'],
            'model':      unified['model'],
            'r2':         unified['r2'],
            'mae':        unified['mae'],
            'input_cols': unified['input_cols'],
        },
        'physics': {
            'V_rated': V_RATED,
            'rho_air': RHO_AIR,
            'Cp_air':  CP_AIR,
        },
    }
    pkl_path = os.path.join(OUT_DIR, 'cdz_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f'  ✓ {pkl_path}')

    lines = []
    lines.append('=' * 55)
    lines.append('CDZ 空调功耗模型报告 (统一模型)')
    lines.append('=' * 55)
    lines.append(f'R2={unified["r2"]:.4f}, MAE={unified["mae"]:.2f} kW')
    lines.append(f'样本: {unified["n_samples"]:,}')
    lines.append(f'输入: {unified["input_cols"]}')
    lines.append('')
    lines.append('-- 物理常数(GA 算 Q_air 用) --')
    lines.append(f'  T_return = {consts["T_return_const"]}°C')
    lines.append(f'  T_supply_DX = {consts["T_supply_DX_const"]}°C')
    lines.append(f'  V_rated = {V_RATED} m³/h/台')
    lines.append(f'  s_fan 范围: [{consts["s_fan_min"]}, {consts["s_fan_max"]}]%')
    lines.append('')
    lines.append('-- 多项式系数 --')
    for n, c in zip(unified['feature_names'], unified['coeffs']):
        lines.append(f'  {n:20s} = {c:+.6f}')
    lines.append('')
    lines.append('-- GA 调用 --')
    lines.append('  Q_air = physics(N_cdz, N_FC, s_fan, T_amb)')
    lines.append('  P_cdz = model.predict(Q_air, T_amb, N_cdz, N_FC, s_fan)')
    lines.append('=' * 55)

    rpt = os.path.join(OUT_DIR, 'cdz_report.txt')
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ {rpt}')
    print('\n'.join(lines))


if __name__ == '__main__':
    df = load_data()
    consts = extract_constants(df)
    unified = fit_unified(df)
    diagnostics(df, unified)
    save_and_report(consts, unified)
    print('\n✓ 全部完成')