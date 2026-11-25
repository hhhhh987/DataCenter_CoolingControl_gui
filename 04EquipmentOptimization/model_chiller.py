"""
model_chiller.py (v3)

从物理极值估计 N_total 和 Q_rated,
标注每小时 N_running,计算 PLR,
拟合 COP = f(PLR, T_amb) 一条曲线供 GA 调用。

GA 用法:
  PLR = Q_liquid / (N_chiller × Q_rated)
  COP = model.predict(PLR, T_amb)
  P_chiller = Q_liquid / COP
  约束: Q_liquid ≤ N_chiller × Q_rated
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ── 配置 ──────────────────────────────────────────────────
ROOT     = r'D:\PythonProject1'
COOL_DIR = os.path.join(ROOT, '03CoolingData')
WX_PATH  = os.path.join(ROOT, '01dataFresh', 'weather.parquet')
OUT_DIR  = os.path.join(ROOT, '04EquipmentOptimization', 'chiller_results')
os.makedirs(OUT_DIR, exist_ok=True)

PEAK_PROMINENCE = 0.0003
POLY_DEGREE = 2

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ── PART 1: 加载数据 ───────────────────────────────────────

def load_data() -> pd.DataFrame:
    print('[PART 1] 加载数据 ...')
    df = pd.read_parquet(os.path.join(COOL_DIR, 'cooling_all_clean.parquet'))
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    wx = pd.read_parquet(WX_PATH)
    wx['timestamp'] = pd.to_datetime(wx['timestamp'], utc=True)
    temp_cols = {}
    for c in wx.columns:
        cl = c.lower()
        if cl in ('temp_c', 't_amb', 'temperature'):
            temp_cols['temp_C'] = c
        elif cl in ('temp_wet_c', 't_wet', 'wet_bulb'):
            temp_cols['temp_wet_C'] = c
    wx_use = wx[['timestamp'] + list(temp_cols.values())].copy()
    wx_use = wx_use.rename(columns={v: k for k, v in temp_cols.items()})
    df = df.merge(wx_use, on='timestamp', how='inner')

    n0 = len(df)
    df = df[(df['Q_liquid_kW'] > 50) & (df['Tot_chiller_kW'] > 30)].copy()
    df['COP_obs'] = df['Q_liquid_kW'] / df['Tot_chiller_kW']
    print(f'  {n0:,} → {len(df):,}')
    return df


# ── PART 2: 物理极值估计 N_total 和 Q_rated ────────────────
# 三条约束过滤:
#   COP_rated ∈ [2.0, 6.0]
#   Q_rated ∈ [100, 500] kW
#   N × Q_rated ≥ Q_max (容量覆盖)
# ──────────────────────────────────────────────────────────

def estimate_capacity(df: pd.DataFrame) -> dict:
    print('\n[PART 2] 物理极值估计 ...')

    # 改为(用实际最大值 + 裕度因子)
    CAPACITY_MARGIN = 1.15
    Q_max = df['Q_liquid_kW'].max() * CAPACITY_MARGIN
    P_max = df['Tot_chiller_kW'].max()

    COP_at_high_load = Q_max / P_max
    print(f'  Q_liquid MAX = {Q_max:.0f} kW')
    print(f'  Tot_chiller MAX = {P_max:.0f} kW')
    print(f'  COP_高负载 = {COP_at_high_load:.2f}')

    print(f'\n  N_total 候选分析:')
    candidates = []
    for n in range(2, 6):
        q_rated = Q_max / n
        p_rated = P_max / n
        cop_rated = q_rated / p_rated  # 等于 COP_at_high_load,但这是满载COP
        # 额定COP通常比满载高 20-30%(最优PLR在70-80%)
        cop_rated_est = cop_rated * 1.25
        valid = (2.0 <= cop_rated_est <= 6.0) and (100 <= q_rated <= 500)
        status = '✓' if valid else '✗'
        print(f'    N={n}: Q_rated={q_rated:.0f} kW, P_rated={p_rated:.0f} kW, '
              f'COP_满载={cop_rated:.2f}, COP_额定≈{cop_rated_est:.2f}  {status}')
        if valid:
            candidates.append({
                'N_total': n, 'Q_rated': q_rated, 'P_rated': p_rated,
                'COP_full_load': cop_rated, 'COP_rated_est': cop_rated_est,
            })

    if not candidates:
        raise RuntimeError('没有满足物理约束的 N_total 候选')

    # 选 COP_额定 最接近典型值 3.0 的(M100 是老旧设备,COP 偏低)
    best = min(candidates, key=lambda c:  abs(c['N_total'] - 4))
    print(f'\n  → 推荐: N_total={best["N_total"]}, '
          f'Q_rated={best["Q_rated"]:.0f} kW, '
          f'COP_额定≈{best["COP_rated_est"]:.2f}')

    best['Q_max'] = Q_max
    best['P_max'] = P_max
    return best


# ── PART 3: 标注每小时 N_running ───────────────────────────
# 方法:
#   1. KDE 找 Tot_chiller 的峰和谷
#   2. 谷值作为分段边界
#   3. 每段的峰位置 / P_rated → 四舍五入 → N_running
#   4. 验证: PLR = Q / (N_running × Q_rated) ∈ [0, ~1.1]
# ──────────────────────────────────────────────────────────

def label_n_running(df: pd.DataFrame, phys: dict) -> pd.DataFrame:
    print('\n[PART 3] 标注 N_running ...')

    P_rated = phys['P_rated']
    Q_rated = phys['Q_rated']
    N_total = phys['N_total']

    # KDE 峰谷
    x = np.linspace(df['Tot_chiller_kW'].min(), df['Tot_chiller_kW'].max(), 1000)
    kde = gaussian_kde(df['Tot_chiller_kW'])
    pdf = kde(x)

    peak_idx, _ = find_peaks(pdf, prominence=PEAK_PROMINENCE)
    peaks = x[peak_idx]

    valley_idx, _ = find_peaks(-pdf, prominence=PEAK_PROMINENCE * 0.5)
    valleys = sorted([x[vi] for vi in valley_idx
                      if any(p < x[vi] for p in peaks) and any(p > x[vi] for p in peaks)])
    valleys = np.array(valleys)

    print(f'  峰: {[f"{p:.0f}" for p in peaks]} kW')
    print(f'  谷(边界): {[f"{v:.0f}" for v in valleys]} kW')

    # 每段赋 N_running
    df = df.copy()
    df['segment'] = np.digitize(df['Tot_chiller_kW'], valleys)

    # 逐小时用需求反推(物理下界: 至少需要几台才能满足制冷需求)
    df['N_running'] = np.clip(
        np.ceil(df['Q_liquid_kW'] / Q_rated).astype(int),
        1, N_total
    )
    df['PLR'] = df['Q_liquid_kW'] / (df['N_running'] * Q_rated)

    for n in sorted(df['N_running'].unique()):
        sub = df[df['N_running'] == n]
        print(f'  N_running={n}: n={len(sub):,}, '
              f'PLR=[{sub.PLR.min():.2f}, {sub.PLR.max():.2f}], '
              f'Tot_chiller=[{sub.Tot_chiller_kW.min():.0f}, {sub.Tot_chiller_kW.max():.0f}] kW')

    print(f'\n  N_running 分布:')
    for n in sorted(df['N_running'].unique()):
        cnt = (df['N_running'] == n).sum()
        print(f'    N={n}: {cnt:,} 小时 ({cnt/len(df)*100:.1f}%)')

    phys['valleys'] = valleys.tolist()
    phys['peaks'] = peaks.tolist()
    phys['N_values'] = sorted(df['N_running'].unique().tolist())

    return df


# ── PART 4: 拟合 COP = f(PLR, T_amb) ─────────────────────
# 全部数据一条曲线,不分组
# 同时试 temp_C / temp_wet_C
# ──────────────────────────────────────────────────────────

def fit_cop_model(df: pd.DataFrame) -> dict:
    print('\n[PART 4] 拟合 COP = f(PLR, T_amb) ...')

    temp_options = []
    if 'temp_C' in df.columns:
        temp_options.append(('temp_C', '干球'))
    if 'temp_wet_C' in df.columns:
        temp_options.append(('temp_wet_C', '湿球'))

    # 过滤极端 COP(物理不合理)
    mask = (df['COP_obs'] > 0.5) & (df['COP_obs'] < 4.0) & (df['PLR'] > 0.05)
    df_fit = df[mask].copy()
    print(f'  拟合样本: {len(df_fit):,} (过滤极端 COP 和 PLR≈0)')

    best_r2 = -np.inf
    best_result = None

    for temp_col, temp_label in temp_options:
        sub = df_fit.dropna(subset=[temp_col])
        X_raw = sub[['PLR', temp_col]].values
        y = sub['COP_obs'].values

        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)
        X_poly = poly.fit_transform(X_raw)

        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        # 反推 P 的精度: P = Q / COP_pred
        P_pred = sub['Q_liquid_kW'].values / np.clip(y_pred, 0.5, 10)
        P_true = sub['Tot_chiller_kW'].values
        r2_power = r2_score(P_true, P_pred)
        mae_power = mean_absolute_error(P_true, P_pred)

        print(f'  {temp_label}: COP R2={r2:.4f}, MAE={mae:.3f}  |  '
              f'功耗 R2={r2_power:.4f}, MAE={mae_power:.2f} kW')

        if r2 > best_r2:
            best_r2 = r2
            best_result = {
                'temp_col':    temp_col,
                'temp_label':  temp_label,
                'poly':        poly,
                'model':       model,
                'r2_cop':      r2,
                'mae_cop':     mae,
                'r2_power':    r2_power,
                'mae_power':   mae_power,
                'n_samples':   len(sub),
                'feature_names': poly.get_feature_names_out(['PLR', 'T']),
                'coeffs':      model.coef_,
            }

    print(f'  → 选择: {best_result["temp_label"]} '
          f'(COP R2={best_result["r2_cop"]:.4f}, 功耗 R2={best_result["r2_power"]:.4f})')
    return best_result, df_fit


# ── PART 5: 诊断图 ─────────────────────────────────────────

def diagnostics(df: pd.DataFrame, df_fit: pd.DataFrame,
                cop_model: dict, phys: dict):
    print('\n[PART 5] 诊断图 ...')

    valleys = phys['valleys']
    peaks = phys['peaks']

    temp_col = cop_model['temp_col']

    # (a) 容量分析: Tot_chiller 直方图 + 峰谷 + N 标注
    fig, ax = plt.subplots(figsize=(12, 5))
    x_kde = np.linspace(df['Tot_chiller_kW'].min(), df['Tot_chiller_kW'].max(), 1000)
    kde = gaussian_kde(df['Tot_chiller_kW'])
    ax.fill_between(x_kde, kde(x_kde), alpha=0.3, color='steelblue')
    ax.plot(x_kde, kde(x_kde), color='navy', lw=2)
    for p in peaks:
        ax.axvline(p, color='red', ls='--', alpha=0.7)
        ax.annotate(f'{p:.0f}kW', xy=(p, kde([p])[0]),
                    fontsize=10, color='red', ha='center',
                    xytext=(0, 10), textcoords='offset points')
    for v in valleys:
        ax.axvline(v, color='green', lw=2, alpha=0.8)
    # 标注每种 N_running 的占比
    for n in sorted(df['N_running'].unique()):
        cnt = (df['N_running'] == n).sum()
        pct = cnt / len(df) * 100
        # 标注在该 N 对应的 Tot_chiller 中位数位置
        med_p = df.loc[df['N_running'] == n, 'Tot_chiller_kW'].median()
        ax.annotate(f'N={n} ({pct:.0f}%)',
                    xy=(med_p, kde([med_p])[0] * 0.5),
                    fontsize=12, fontweight='bold', color='darkblue', ha='center')
    ax.set_xlabel('Tot_chiller (kW)')
    ax.set_ylabel('密度')
    ax.set_title(f'Tot_chiller 分布与 N_running 标注  '
                 f'(N_total={phys["N_total"]}, Q_rated={phys["Q_rated"]:.0f} kW)')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'chiller_capacity.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'  (a) chiller_capacity.png')

    # (b) COP vs PLR 散点(按 T 着色)
    sub = df_fit.dropna(subset=[temp_col])
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(sub['PLR'], sub['COP_obs'], c=sub[temp_col],
                    s=3, alpha=0.4, cmap='coolwarm')
    fig.colorbar(sc, ax=ax, label=f'{cop_model["temp_label"]}温度 (°C)')

    # 画拟合曲线(固定 T 在几个代表值)
    plr_grid = np.linspace(0.1, 1.1, 100)
    for T_val, ls in [(5, '--'), (15, '-'), (25, '--'), (35, ':')]:
        X_grid = np.column_stack([plr_grid, np.full_like(plr_grid, T_val)])
        X_poly = cop_model['poly'].transform(X_grid)
        cop_pred = cop_model['model'].predict(X_poly)
        ax.plot(plr_grid, cop_pred, ls=ls, lw=2, color='black', alpha=0.6,
                label=f'T={T_val}°C')

    ax.set_xlabel('PLR')
    ax.set_ylabel('COP')
    ax.set_title(f'COP vs PLR  R2={cop_model["r2_cop"]:.3f}')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, max(8, sub['COP_obs'].quantile(0.99)))
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'chiller_cop_curve.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'  (b) chiller_cop_curve.png')

    # (c) 功耗拟合质量
    sub = df_fit.dropna(subset=[temp_col])
    X_raw = sub[['PLR', temp_col]].values
    X_poly = cop_model['poly'].transform(X_raw)
    cop_pred = np.clip(cop_model['model'].predict(X_poly), 0.5, 10)
    P_pred = sub['Q_liquid_kW'].values / cop_pred
    P_true = sub['Tot_chiller_kW'].values

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(P_true, P_pred, s=3, alpha=0.3)
    lims = [min(P_true.min(), P_pred.min()), max(P_true.max(), P_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('实测 Tot_chiller (kW)')
    ax.set_ylabel('预测 (kW)')
    ax.set_title(f'功耗预测  R2={cop_model["r2_power"]:.3f}, '
                 f'MAE={cop_model["mae_power"]:.1f} kW')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'chiller_fit_quality.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'  (c) chiller_fit_quality.png')


# ── PART 6: 保存 + 报告 ───────────────────────────────────

def save_and_report(df, phys, cop_model):
    print('\n[PART 6] 保存 + 报告 ...')

    save_obj = {
        'physics': {
            'N_total':   phys['N_total'],
            'Q_rated':   phys['Q_rated'],
            'P_rated':   phys['P_rated'],
            'N_values':  phys['N_values'],
            'valleys':   phys['valleys'],
        },
        'cop_model': {
            'poly':       cop_model['poly'],
            'model':      cop_model['model'],
            'temp_col':   cop_model['temp_col'],
            'r2_cop':     cop_model['r2_cop'],
            'r2_power':   cop_model['r2_power'],
        },
    }
    pkl_path = os.path.join(OUT_DIR, 'chiller_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f'  {pkl_path}')

    lines = []
    lines.append('=' * 60)
    lines.append('冷水机功耗模型报告 (v3)')
    lines.append('=' * 60)
    lines.append(f'数据样本: {len(df):,}')
    lines.append('')

    lines.append('-- 物理参数 --')
    lines.append(f'  N_total = {phys["N_total"]}')
    lines.append(f'  Q_rated = {phys["Q_rated"]:.0f} kW/台')
    lines.append(f'  P_rated = {phys["P_rated"]:.0f} kW/台')
    lines.append(f'  COP_满载 = {phys["COP_full_load"]:.2f}')
    lines.append(f'  COP_额定(估) = {phys["COP_rated_est"]:.2f}')
    lines.append(f'  N_running 取值: {phys["N_values"]}')
    lines.append(f'  分段边界(kW): {[f"{v:.0f}" for v in phys["valleys"]]}')
    lines.append('')

    lines.append('-- COP 模型 --')
    lines.append(f'  形式: COP = f(PLR, {cop_model["temp_label"]})')
    lines.append(f'  COP R2 = {cop_model["r2_cop"]:.4f}')
    lines.append(f'  功耗 R2 = {cop_model["r2_power"]:.4f}')
    lines.append(f'  功耗 MAE = {cop_model["mae_power"]:.2f} kW')
    lines.append(f'  系数:')
    for name, coef in zip(cop_model['feature_names'], cop_model['coeffs']):
        lines.append(f'    {name:12s} = {coef:+.6f}')
    lines.append('')

    lines.append('-- GA 调用 --')
    lines.append('  PLR = Q_liquid / (N_chiller * Q_rated)')
    lines.append('  COP = cop_model.predict(PLR, T_amb)')
    lines.append('  P_chiller = Q_liquid / COP')
    lines.append(f'  约束: Q_liquid <= N_chiller * {phys["Q_rated"]:.0f}')
    lines.append(f'  N_chiller in {phys["N_values"]}')
    lines.append('=' * 60)

    report_path = os.path.join(OUT_DIR, 'chiller_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  {report_path}')
    print('\n'.join(lines))


# ── main ──────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_data()
    phys = estimate_capacity(df)
    df = label_n_running(df, phys)
    cop_model, df_fit = fit_cop_model(df)
    diagnostics(df, df_fit, cop_model, phys)
    save_and_report(df, phys, cop_model)
    print('\n✓ 全部完成')