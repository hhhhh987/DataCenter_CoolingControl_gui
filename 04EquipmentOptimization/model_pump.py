"""
model_pump.py

水泵功耗建模: P_pump = f(Inverter_pct)
物理基础: Affinity Laws, P ∝ f³
输入: cooling_all_clean.parquet (Inverter_pct, Tot_qpompe_kW)
输出: pump_results/ (pump_fit.png, pump_model.pkl, pump_report.txt)

GA 用法:
  P_pump = model.predict(f_pump)
  约束: f_pump ∈ [f_min, 100]%
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

ROOT     = r'D:\PythonProject1'
COOL_DIR = os.path.join(ROOT, '03CoolingData')
OUT_DIR  = os.path.join(ROOT, '04EquipmentOptimization', 'pump_results')
os.makedirs(OUT_DIR, exist_ok=True)

POLY_DEGREE = 3

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ── PART 1: 加载数据 + 提取 f_min ─────────────────────────

def load_data() -> tuple:
    print('[PART 1] 加载数据 ...')
    df = pd.read_parquet(os.path.join(COOL_DIR, 'cooling_all_clean.parquet'))

    n0 = len(df)
    df = df[['timestamp', 'Inverter_pct', 'Tot_qpompe_kW', 'N_pumps_running', 'Portata_m3h']].dropna()
    df = df[df['Portata_m3h'] > 0].copy()
    df = df[(df['Inverter_pct'] > 0) & (df['Tot_qpompe_kW'] > 0)].copy()
    print(f'  {n0:,} → 过滤后 {len(df):,} (排除停机时段)')

    # f_min: 历史非零运行的 P5(避免传感器异常)
    f_min = max(30.0, float(df['Inverter_pct'].quantile(0.05)))
    f_max = float(df['Inverter_pct'].max())
    n_pumps = int(df['N_pumps_running'].mode().iloc[0])

    print(f'  Inverter_pct: [{df.Inverter_pct.min():.1f}, {f_max:.1f}]%')
    print(f'  Tot_qpompe:   [{df.Tot_qpompe_kW.min():.1f}, {df.Tot_qpompe_kW.max():.1f}] kW')
    print(f'  N_pumps 众数:  {n_pumps} 台(恒定)')
    print(f'  f_min (P5):    {f_min:.1f}%  ← GA 硬约束下限')

    phys = {
        'f_min': f_min,
        'f_max': f_max,
        'N_pumps': n_pumps,
    }
    return df, phys


# ── PART 2: 三次多项式拟合 ─────────────────────────────────

def fit_model(df: pd.DataFrame) -> dict:
    print(f'\n[PART 2] 拟合 P = f(Inverter_pct), degree={POLY_DEGREE} ...')

    # 只拟合变频有效区间(低频段是电机基础损耗,不遵循 Affinity Law)
    F_EFFECTIVE = 30
    mask = (df['Inverter_pct'] >= F_EFFECTIVE) & (df['Portata_m3h'] > 0)
    df_fit = df[mask].copy()
    print(f'  有效样本 f>={F_EFFECTIVE}%: {len(df_fit):,}')

    X_raw = df_fit[['Inverter_pct', 'Portata_m3h']].values
    y = df_fit['Tot_qpompe_kW'].values

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_raw)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    names = poly.get_feature_names_out(['f', 'V'])
    coeffs = model.coef_

    print(f'  R2 = {r2:.4f}, MAE = {mae:.2f} kW')
    for n, c in zip(names, coeffs):
        print(f'    {n:8s} = {c:+.6f}')

    return {
        'poly': poly,
        'model': model,
        'r2': r2,
        'mae': mae,
        'feature_names': names,
        'coeffs': coeffs,
        'n_samples': len(df_fit),
    }


# ── PART 3: 诊断图 ─────────────────────────────────────────

def diagnostics(df: pd.DataFrame, fit: dict, phys: dict):
    print(f'\n[PART 3] 诊断图 ...')

    f_vals = df['Inverter_pct'].values
    V_vals = df['Portata_m3h'].values
    y_true = df['Tot_qpompe_kW'].values
    X_poly = fit['poly'].transform(np.column_stack([f_vals, V_vals]))
    y_pred = fit['model'].predict(X_poly)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=3, alpha=0.3, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('实测 (kW)')
    ax.set_ylabel('预测 (kW)')
    ax.set_title(f'水泵功耗 P=f(Inverter, Portata)  R2={fit["r2"]:.3f}, MAE={fit["mae"]:.2f} kW')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)

    fig.savefig(os.path.join(OUT_DIR, 'pump_fit.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ pump_fit.png')


# ── PART 4: 保存 + 报告 ───────────────────────────────────

def save_and_report(phys: dict, fit: dict):
    print(f'\n[PART 4] 保存 + 报告 ...')

    save_obj = {
        'physics': phys,
        'model': {
            'poly':   fit['poly'],
            'model':  fit['model'],
            'r2':     fit['r2'],
            'mae':    fit['mae'],
            'coeffs': fit['coeffs'],
        },
    }
    pkl_path = os.path.join(OUT_DIR, 'pump_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f'  {pkl_path}')

    lines = []
    lines.append('=' * 50)
    lines.append('水泵功耗模型报告')
    lines.append('=' * 50)
    lines.append(f'样本数: {fit["n_samples"]:,}')
    lines.append(f'运行台数: {phys["N_pumps"]} 台(恒定)')
    lines.append(f'R2 = {fit["r2"]:.4f}')
    lines.append(f'MAE = {fit["mae"]:.2f} kW')
    lines.append('')
    lines.append('-- 物理约束 --')
    lines.append(f'  f_min = {phys["f_min"]:.1f}% (P5, GA 硬约束下限)')
    lines.append(f'  f_max = {phys["f_max"]:.1f}%')
    lines.append('')
    lines.append('-- 多项式系数 --')
    lines.append(f'  P = f(Inverter_pct, Portata_m3h), degree=2')
    for n, c in zip(fit['feature_names'], fit['coeffs']):
        lines.append(f'  {n:8s} = {c:+.6f}')
    lines.append('')
    lines.append('-- GA 调用 --')
    lines.append('  Portata = Q_liquid / (1000 * 4.18 * 6) * 3600')
    lines.append('  X = pump["model"]["poly"].transform([[f_pump, Portata]])')
    lines.append('  P_pump = pump["model"]["model"].predict(X)[0]')
    lines.append(f'  约束: f_pump in [{phys["f_min"]:.1f}, {phys["f_max"]:.1f}]%')
    lines.append('=' * 50)

    report_path = os.path.join(OUT_DIR, 'pump_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  {report_path}')
    print('\n'.join(lines))


# ── main ──────────────────────────────────────────────────

if __name__ == '__main__':
    df, phys = load_data()
    fit = fit_model(df)
    diagnostics(df, fit, phys)
    save_and_report(phys, fit)
    print('\n✓ 全部完成')