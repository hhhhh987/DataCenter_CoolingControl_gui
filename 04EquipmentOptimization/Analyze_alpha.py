"""
判定逻辑(五段证据,链式收敛):
  Step 1: α 1D KDE         → 双峰是否真实存在?(验证前提)
  Step 2: 条件 KDE (T_amb) → 双峰能否被 T_amb 解释?(核心判定)
  Step 3: 时序图组          → α 怎么变化?平滑漂移 / 突变阶跃 / 高频抖动?
  Step 4: 模式持续时间      → 两个"模式"是稳定运行状态还是瞬时波动?
  Step 5: T_amb 相关性     → Pearson / Spearman 量化解释力

判定矩阵(自动综合):
  季节驱动: Step 2 单峰 + Step 3 平滑 + Step 4 长持续 + Step 5 强相关
  策略驱动: Step 2 多峰 + Step 3 突变 + Step 4 长持续 + Step 5 弱相关
  噪声主导: Step 4 短持续(<几小时) → 优先级最高,直接否决"模式"概念

输入:
  cooling_all_clean.parquet     (含 α,主输入)
  weather.parquet               (若 cooling_all_clean 缺 T_amb 时备用)

输出:
  alpha_results/
    ├── alpha_1d_kde.png            (Step 1)
    ├── alpha_conditional_kde.png   (Step 2)
    ├── alpha_time_series.png       (Step 3,3 子图: 全时序 + 月均 + 时均)
    ├── alpha_mode_persistence.png  (Step 4,2 子图: 持续时间直方图 + mode 时序)
    ├── alpha_vs_tamb_scatter.png   (Step 5)
    └── alpha_report.txt            (综合结论 + 后续建模建议)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from scipy.signal import find_peaks

# ── 配置 ──────────────────────────────────────────────────
ROOT       = r'D:\PythonProject1'
COOL_DIR   = os.path.join(ROOT, '03CoolingData')
CLEAN_PATH = os.path.join(COOL_DIR, 'cooling_all_clean.parquet')
WX_PATH    = os.path.join(ROOT, '01dataFresh', 'weather.parquet')
OUT_DIR    = os.path.join(ROOT,'04EquipmentOptimization', 'alpha_results')
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1/2: 峰显著性阈值(find_peaks 的 prominence 参数)
PEAK_PROMINENCE = 0.5

# Step 2: T_amb 等分位分箱数
N_TEMP_BINS = 5

# Step 3: "突变"定义——相邻两小时 |Δα| 超过这个值算突变
SUDDEN_JUMP_THRESHOLD = 0.15

# Step 4: 持续时间判断阈值(小时)
SHORT_RUN_THRESHOLD = 6      # <6h 算"短"(噪声特征)
LONG_RUN_THRESHOLD  = 48     # >48h 算"长"(真实运行状态)

# 时序连续性: 两条记录时间差 > 这个,视为断裂(避免跨数据空洞算连续)
CONTINUITY_GAP_HOURS = 1.5

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ── PART 1: 加载数据 ───────────────────────────────────────
# 设计:
#   - cooling_all_clean.parquet 应已含 α(可能叫 alpha / alpha_ratio)
#   - 若没有 T_amb 列,从 weather.parquet 内连接补
#   - 防御性命名:统一 α 列 → 'alpha',温度列 → 'temp_C'
# ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print('加载 cooling_all_clean.parquet ...')
    df = pd.read_parquet(CLEAN_PATH)
    print(f'  行数: {len(df):,}')
    print(f'  列:   {list(df.columns)}')

    # α 列
    alpha_candidates = [c for c in df.columns
                        if c.lower() in ('alpha', 'alpha_ratio',  'alpha_liquid','α')]
    if not alpha_candidates:
        raise ValueError(f'找不到 α 列,实际列: {list(df.columns)}')
    if alpha_candidates[0] != 'alpha':
        df = df.rename(columns={alpha_candidates[0]: 'alpha'})
    print(f'  α 列已就位')

    # T_amb 列
    tamb_candidates = [c for c in df.columns
                       if c.lower() in ('t_amb', 'temp_c', 't_ambient', 'tamb')]
    if tamb_candidates:
        if tamb_candidates[0] != 'temp_C':
            df = df.rename(columns={tamb_candidates[0]: 'temp_C'})
        print(f'  T_amb 列已就位')
    else:
        print(f'  T_amb 不在 cooling_all_clean,从 weather.parquet 内连接补')
        wx = pd.read_parquet(WX_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        wx['timestamp'] = pd.to_datetime(wx['timestamp'], utc=True)
        wx_tcol = next((c for c in wx.columns
                        if c.lower() in ('t_amb', 'temp_c', 'temp', 't_ambient')),
                       None)
        if wx_tcol is None:
            raise ValueError(f'weather 里也没有温度列,实际列: {list(wx.columns)}')
        wx = wx[['timestamp', wx_tcol]].rename(columns={wx_tcol: 'temp_C'})
        n_before = len(df)
        df = df.merge(wx, on='timestamp', how='inner')
        print(f'  合并后: {n_before:,} → {len(df):,}')

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.dropna(subset=['alpha', 'temp_C']).sort_values('timestamp').reset_index(drop=True)

    print(f'\n最终用于分析: {len(df):,} 行')
    print(f'  α 范围:     [{df.alpha.min():.3f}, {df.alpha.max():.3f}]')
    print(f'  T_amb 范围: [{df.temp_C.min():.1f}, {df.temp_C.max():.1f}] °C')
    print(f'  时间范围:   {df.timestamp.min()} ~ {df.timestamp.max()}')
    return df


# ── PART 2: Step 1 — α 1D KDE ─────────────────────────────
# 作用: 确认 α 真的是双峰
# ──────────────────────────────────────────────────────────

def step_1_1d_kde(df: pd.DataFrame) -> dict:
    print('\n[Step 1] α 1D KDE ...')

    alpha = df['alpha'].values
    kde = gaussian_kde(alpha)
    x = np.linspace(alpha.min(), alpha.max(), 500)
    pdf = kde(x)

    peaks_idx, _ = find_peaks(pdf, prominence=PEAK_PROMINENCE)
    peak_locs = x[peaks_idx]
    peak_heights = pdf[peaks_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, pdf, alpha=0.3, color='steelblue')
    ax.plot(x, pdf, color='steelblue', lw=2)
    for loc, h in zip(peak_locs, peak_heights):
        ax.axvline(loc, color='red', ls='--', alpha=0.7)
        ax.annotate(f'α={loc:.3f}',
                    xy=(loc, h), xytext=(loc, h * 1.08),
                    ha='center', fontsize=10, color='red')
    ax.set_xlabel('α (液冷分担比例)')
    ax.set_ylabel('密度')
    ax.set_title(f'α 的 1D KDE   识别峰数 = {len(peak_locs)}')
    ax.grid(alpha=0.3)

    out = os.path.join(OUT_DIR, 'alpha_1d_kde.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    print(f'  ✓ {out}')
    print(f'  峰位置: {[f"{p:.3f}" for p in peak_locs]}')
    return {'peaks': peak_locs.tolist(), 'n_peaks': len(peak_locs)}


# ── PART 3: Step 2 — 条件 KDE on T_amb ─────────────────────
# 作用(核心判定):
#   - T_amb 等分位 5 箱,每箱画 α 的 KDE
#   - 每箱单峰 → 季节驱动
#   - 某箱仍多峰 → 季节解释不了
# ──────────────────────────────────────────────────────────

def step_2_conditional_kde(df: pd.DataFrame) -> dict:
    print('\n[Step 2] 按 T_amb 分箱的条件 KDE ...')

    quantiles = np.linspace(0, 1, N_TEMP_BINS + 1)
    edges = df['temp_C'].quantile(quantiles).values
    labels = [f'[{edges[i]:.1f}, {edges[i+1]:.1f}]°C' for i in range(N_TEMP_BINS)]

    x = np.linspace(df['alpha'].min(), df['alpha'].max(), 500)
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.coolwarm

    peaks_per_bin = []
    for i in range(N_TEMP_BINS):
        if i < N_TEMP_BINS - 1:
            mask = (df['temp_C'] >= edges[i]) & (df['temp_C'] < edges[i + 1])
        else:
            mask = (df['temp_C'] >= edges[i]) & (df['temp_C'] <= edges[i + 1])
        subset = df.loc[mask, 'alpha'].values

        if len(subset) < 30:
            peaks_per_bin.append([])
            print(f'  {labels[i]}: 样本太少({len(subset)}),跳过')
            continue

        kde = gaussian_kde(subset)
        pdf = kde(x)
        color = cmap(i / max(N_TEMP_BINS - 1, 1))
        ax.plot(x, pdf, color=color, lw=2,
                label=f'{labels[i]} (n={len(subset)})')

        peaks_idx, _ = find_peaks(pdf, prominence=PEAK_PROMINENCE)
        peaks_per_bin.append(x[peaks_idx].tolist())

    ax.set_xlabel('α')
    ax.set_ylabel('条件密度 p(α | T_amb)')
    ax.set_title('按 T_amb 等分位分箱的 α 条件 KDE  (蓝=冷,红=暖)')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

    out = os.path.join(OUT_DIR, 'alpha_conditional_kde.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    print(f'  ✓ {out}')
    print(f'  各箱峰数: {[len(p) for p in peaks_per_bin]}')
    for lab, peaks in zip(labels, peaks_per_bin):
        print(f'    {lab}: {[f"{p:.3f}" for p in peaks]}')

    valid_bins = [p for p in peaks_per_bin if p]
    is_seasonal = all(len(p) <= 1 for p in valid_bins) if valid_bins else False
    return {'peaks_per_bin': peaks_per_bin,
            'bin_labels': labels,
            'is_seasonal': is_seasonal}


# ── PART 4: Step 3 — 时序图组 ──────────────────────────────
# 作用: 看 α 怎么变化(三个尺度)
#   (a) 完整时序 → 平滑 / 突变 / 噪声
#   (b) 月均     → 是否有年周期(季节性)
#   (c) 时均     → 是否有日内规律(IT 负荷或温度耦合)
#
# 自动统计:
#   - sudden_jumps: 相邻连续点 |Δα| 超过阈值的次数
#   - diff_std:     相邻 α 变化的标准差(越小越平滑)
#   - monthly_range / hourly_range:    均值范围
# ──────────────────────────────────────────────────────────

def step_3_time_series(df: pd.DataFrame) -> dict:
    print('\n[Step 3] α 时序分析 ...')

    df = df.copy()
    df['month_str']   = df['timestamp'].dt.to_period('M').astype(str)
    df['hour_of_day'] = df['timestamp'].dt.hour

    monthly = df.groupby('month_str')['alpha'].agg(['mean', 'std'])
    hourly  = df.groupby('hour_of_day')['alpha'].agg(['mean', 'std'])

    # 相邻点的时间差和 α 差
    time_gap   = df['timestamp'].diff().dt.total_seconds() / 3600
    alpha_diff = df['alpha'].diff().abs()
    is_continuous = time_gap.fillna(np.inf) <= CONTINUITY_GAP_HOURS

    sudden_jumps = ((alpha_diff > SUDDEN_JUMP_THRESHOLD) & is_continuous).sum()
    diff_std     = alpha_diff[is_continuous].std()
    n_continuous = is_continuous.sum()

    monthly_range = monthly['mean'].max() - monthly['mean'].min()
    hourly_range  = hourly['mean'].max() - hourly['mean'].min()

    print(f'  连续点数: {n_continuous:,}')
    print(f'  突变次数(|Δα|>{SUDDEN_JUMP_THRESHOLD}): {sudden_jumps} '
          f'({sudden_jumps/max(n_continuous,1)*100:.2f}%)')
    print(f'  相邻 α 变化 std: {diff_std:.4f}')
    print(f'  月均范围: [{monthly["mean"].min():.3f}, {monthly["mean"].max():.3f}]  '
          f'(span={monthly_range:.3f})')
    print(f'  时均范围: [{hourly["mean"].min():.3f}, {hourly["mean"].max():.3f}]  '
          f'(span={hourly_range:.3f})')

    # 3 子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # (a) 完整时序
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['alpha'], lw=0.3, alpha=0.6, color='steelblue')
    ax1.set_ylabel('α')
    ax1.set_title(f'(a) α 完整时序  突变={sudden_jumps}, 相邻Δα std={diff_std:.4f}')
    ax1.grid(alpha=0.3)

    # (b) 月均
    ax2 = axes[1]
    ax2.bar(range(len(monthly)), monthly['mean'].values,
            yerr=monthly['std'].values, color='tomato', alpha=0.7,
            error_kw=dict(elinewidth=1, capsize=2))
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('月均 α')
    ax2.set_title(f'(b) 月均 α   年周期 span = {monthly_range:.3f}')
    ax2.grid(alpha=0.3, axis='y')

    # (c) 时均
    ax3 = axes[2]
    ax3.plot(hourly.index, hourly['mean'].values, 'o-', color='darkgreen', lw=2)
    ax3.fill_between(hourly.index,
                     hourly['mean'].values - hourly['std'].values,
                     hourly['mean'].values + hourly['std'].values,
                     color='darkgreen', alpha=0.2)
    ax3.set_xlabel('小时 (0-23)')
    ax3.set_ylabel('小时均 α')
    ax3.set_title(f'(c) 按小时均 α   日内 span = {hourly_range:.3f}')
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'alpha_time_series.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out}')

    return {
        'sudden_jumps':   int(sudden_jumps),
        'jump_pct':       float(sudden_jumps / max(n_continuous, 1) * 100),
        'diff_std':       float(diff_std),
        'monthly_range':  float(monthly_range),
        'hourly_range':   float(hourly_range),
        'n_continuous':   int(n_continuous),
    }


# ── PART 5: Step 4 — 模式持续时间 ──────────────────────────
# 作用:
#   - 把 α 按阈值二值化(threshold = Step 1 两峰的中点)
#   - 统计每段连续同 mode 的持续时间
#   - 时间差 > CONTINUITY_GAP_HOURS 也算断裂(避免跨空洞算连续)
#
# 判定:
#   median < 6h   → 噪声/瞬时波动,不构成真模式
#   median > 48h  → 真实运行状态(季节或策略)
#   中间          → 日内调度
# ──────────────────────────────────────────────────────────

def step_4_mode_persistence(df: pd.DataFrame, peaks: list) -> dict:
    print('\n[Step 4] α 模式持续时间 ...')

    # 阈值: Step 1 的双峰中点;若没识别双峰,fallback 用中位数
    if len(peaks) >= 2:
        threshold = (min(peaks) + max(peaks)) / 2
        print(f'  阈值(Step 1 双峰中点): {threshold:.3f}')
    else:
        threshold = float(df['alpha'].median())
        print(f'  Step 1 未识别多峰,fallback 用中位数: {threshold:.3f}')

    df = df.copy()
    df['mode'] = (df['alpha'] > threshold).astype(int)

    # 断裂判定: 时间 gap 太大 OR mode 切换
    time_gap = df['timestamp'].diff().dt.total_seconds() / 3600
    is_break = (time_gap.fillna(np.inf) > CONTINUITY_GAP_HOURS) | \
               (df['mode'].diff() != 0)
    is_break.iloc[0] = True
    df['run_id'] = is_break.cumsum()

    runs = df.groupby('run_id').agg(
        mode=('mode', 'first'),
        length=('mode', 'count'),
        start_time=('timestamp', 'first'),
    ).reset_index(drop=True)

    # mode 切换次数 = 相邻段 mode 不同的次数(且时间连续)
    runs['next_mode'] = runs['mode'].shift(-1)
    runs['next_start'] = runs['start_time'].shift(-1)
    # 这里简化: 直接数段间 mode 不同的次数
    n_switches = int((runs['mode'].values[:-1] != runs['mode'].values[1:]).sum())

    high_runs = runs.loc[runs['mode'] == 1, 'length']
    low_runs  = runs.loc[runs['mode'] == 0, 'length']

    median_len = float(runs['length'].median())
    mean_len   = float(runs['length'].mean())
    p90_len    = float(runs['length'].quantile(0.9))
    pct_short  = float((runs['length'] < SHORT_RUN_THRESHOLD).mean() * 100)
    pct_long   = float((runs['length'] > LONG_RUN_THRESHOLD).mean() * 100)

    print(f'  段数: {len(runs)} (切换 {n_switches} 次)')
    print(f'  持续时间(h): 中位={median_len:.1f}, 均值={mean_len:.1f}, P90={p90_len:.1f}')
    print(f'  短段(<{SHORT_RUN_THRESHOLD}h): {pct_short:.1f}%')
    print(f'  长段(>{LONG_RUN_THRESHOLD}h): {pct_long:.1f}%')

    # 2 子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 持续时间直方图(对数 x 轴)
    ax1 = axes[0]
    max_len = runs['length'].max()
    bins = np.logspace(0, np.log10(max_len), 30) if max_len > 1 else 20
    ax1.hist(high_runs, bins=bins, alpha=0.6,
             label=f'高 α 模式 (n={len(high_runs)})', color='tomato')
    ax1.hist(low_runs, bins=bins, alpha=0.6,
             label=f'低 α 模式 (n={len(low_runs)})', color='steelblue')
    ax1.axvline(SHORT_RUN_THRESHOLD, ls='--', color='gray',
                label=f'短阈值 ({SHORT_RUN_THRESHOLD}h)')
    ax1.axvline(LONG_RUN_THRESHOLD, ls=':', color='gray',
                label=f'长阈值 ({LONG_RUN_THRESHOLD}h)')
    if max_len > 1:
        ax1.set_xscale('log')
    ax1.set_xlabel('持续时间 (小时, 对数刻度)')
    ax1.set_ylabel('段数')
    ax1.set_title(f'(a) 模式持续时间分布   阈值 α={threshold:.3f}')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # (b) mode 时序(看切换 pattern)
    ax2 = axes[1]
    ax2.plot(df['timestamp'], df['mode'], lw=0.5, alpha=0.7, color='purple')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('mode (0=低 α, 1=高 α)')
    ax2.set_title(f'(b) mode 时序  共 {n_switches} 次切换, {len(runs)} 段')
    ax2.set_yticks([0, 1])
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'alpha_mode_persistence.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out}')

    return {
        'threshold':     float(threshold),
        'n_runs':        len(runs),
        'n_switches':    n_switches,
        'median_length': median_len,
        'mean_length':   mean_len,
        'p90_length':    p90_len,
        'pct_short':     pct_short,
        'pct_long':      pct_long,
    }


# ── PART 6: Step 5 — α vs T_amb 相关性 ────────────────────
# 作用: 量化 T_amb 的解释力
#   Pearson  测线性
#   Spearman 测单调(对非线性更鲁棒)
# ──────────────────────────────────────────────────────────

def step_5_correlation(df: pd.DataFrame) -> dict:
    print('\n[Step 5] α vs T_amb 相关性 ...')

    r_p, p_p = pearsonr(df['temp_C'], df['alpha'])
    r_s, p_s = spearmanr(df['temp_C'], df['alpha'])

    print(f'  Pearson  r = {r_p:.4f}  (p = {p_p:.2e})')
    print(f'  Spearman r = {r_s:.4f}  (p = {p_s:.2e})')

    fig, ax = plt.subplots(figsize=(10, 5))
    hb = ax.hexbin(df['temp_C'], df['alpha'],
                   gridsize=40, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='点数')
    ax.set_xlabel('T_amb (°C)')
    ax.set_ylabel('α')
    ax.set_title(f'α vs T_amb   Pearson r={r_p:.3f}, Spearman r={r_s:.3f}')
    ax.grid(alpha=0.3)

    out = os.path.join(OUT_DIR, 'alpha_vs_tamb_scatter.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out}')

    return {'pearson': r_p, 'spearman': r_s,
            'p_pearson': p_p, 'p_spearman': p_s}


# ── PART 7: 综合判定与报告 ────────────────────────────────
# 决策树(优先级从高到低):
#   1. Step 4 median < 短阈值        → 噪声主导,直接否决"模式"概念
#   2. Step 2 单峰 + Step 5 中等以上 + Step 4 长持续 → 季节驱动
#   3. Step 2 多峰 + Step 5 弱 + Step 4 长持续      → 策略驱动
#   4. 其他                                          → 混合驱动,先 T_amb 单变量
# ──────────────────────────────────────────────────────────

def write_conclusion(results: dict, df: pd.DataFrame):
    r1, r2, r3, r4, r5 = [results[k] for k in ('1', '2', '3', '4', '5')]
    abs_r = abs(r5['pearson'])

    lines = []
    lines.append('=' * 70)
    lines.append('α 双峰来源诊断报告')
    lines.append('=' * 70)
    lines.append(f'样本数: {len(df):,}')
    lines.append(f'时间范围: {df.timestamp.min()} ~ {df.timestamp.max()}')
    lines.append('')

    # Step 1
    lines.append('[Step 1] α 1D KDE')
    lines.append(f'  识别峰数: {r1["n_peaks"]}')
    lines.append(f'  峰位置:   {[round(p, 3) for p in r1["peaks"]]}')
    lines.append('')

    # Step 2
    lines.append('[Step 2] 条件 KDE (按 T_amb 分箱)')
    lines.append(f'  各箱峰数: {[len(p) for p in r2["peaks_per_bin"]]}')
    if r2['is_seasonal']:
        lines.append('  → 各箱均单峰,T_amb 可解释 α 的双峰')
    else:
        lines.append('  → 至少一箱仍多峰,T_amb 无法完全解释')
    lines.append('')

    # Step 3
    lines.append('[Step 3] 时序分析')
    lines.append(f'  突变次数(|Δα|>{SUDDEN_JUMP_THRESHOLD}): {r3["sudden_jumps"]} '
                 f'({r3["jump_pct"]:.2f}%)')
    lines.append(f'  相邻 α 变化 std: {r3["diff_std"]:.4f}')
    lines.append(f'  月均 α span: {r3["monthly_range"]:.3f}')
    lines.append(f'  时均 α span: {r3["hourly_range"]:.3f}')

    if r3['diff_std'] < 0.03 and r3['jump_pct'] < 1.0:
        ts_pattern = '平滑漂移 → 连续变量驱动(支持季节假设)'
    elif r3['jump_pct'] > 5.0:
        ts_pattern = '频繁突变 → 策略切换或控制振荡(支持策略假设)'
    else:
        ts_pattern = '中等波动 → 证据不明朗'
    lines.append(f'  → 时序模式: {ts_pattern}')

    if r3['monthly_range'] > 0.1:
        lines.append(f'  → 月均 span > 0.1,有明显年周期')
    else:
        lines.append(f'  → 月均 span 较小,年周期不明显')
    lines.append('')

    # Step 4
    lines.append('[Step 4] 模式持续时间')
    lines.append(f'  阈值 α = {r4["threshold"]:.3f}  (Step 1 双峰中点)')
    lines.append(f'  段数 {r4["n_runs"]},切换 {r4["n_switches"]} 次')
    lines.append(f'  持续 (h): 中位={r4["median_length"]:.1f}, '
                 f'均值={r4["mean_length"]:.1f}, P90={r4["p90_length"]:.1f}')
    lines.append(f'  短段(<{SHORT_RUN_THRESHOLD}h): {r4["pct_short"]:.1f}%')
    lines.append(f'  长段(>{LONG_RUN_THRESHOLD}h): {r4["pct_long"]:.1f}%')

    if r4['median_length'] < SHORT_RUN_THRESHOLD:
        mode_judge = '持续短 → 噪声/瞬时波动,模式不真实'
    elif r4['median_length'] > LONG_RUN_THRESHOLD:
        mode_judge = '持续长 → 真实运行状态(季节或策略)'
    else:
        mode_judge = '中等持续 → 可能日内调度'
    lines.append(f'  → {mode_judge}')
    lines.append('')

    # Step 5
    lines.append('[Step 5] α vs T_amb 相关性')
    lines.append(f'  Pearson  r = {r5["pearson"]:.4f}  (p = {r5["p_pearson"]:.2e})')
    lines.append(f'  Spearman r = {r5["spearman"]:.4f}  (p = {r5["p_spearman"]:.2e})')
    if abs_r > 0.7:
        c_judge = '强相关 → T_amb 主导'
    elif abs_r > 0.4:
        c_judge = '中等相关 → T_amb 部分驱动'
    else:
        c_judge = '弱相关 → T_amb 难以解释'
    lines.append(f'  → {c_judge}')
    lines.append('')

    # 综合判定
    lines.append('=' * 70)
    lines.append('综合判定')
    lines.append('=' * 70)

    # 优先级 1: 噪声否决
    if r4['median_length'] < SHORT_RUN_THRESHOLD:
        verdict = ('【噪声主导】所谓"双峰"是高频抖动\n'
                   '  - Step 4 中位持续时间 < ' + str(SHORT_RUN_THRESHOLD) + 'h\n'
                   '→ 模式不真实,不需要分组建模,当单分布处理')
    # 优先级 2: 季节驱动
    elif r2['is_seasonal'] and abs_r > 0.4 and r4['median_length'] > LONG_RUN_THRESHOLD:
        verdict = ('【季节驱动】证据一致指向 T_amb\n'
                   '  - Step 2 每箱单峰\n'
                   f'  - Step 5 |Pearson| = {abs_r:.3f} > 0.4\n'
                   '  - Step 4 持续时间长\n'
                   '→ 后续设备建模不需要 mode 变量\n'
                   '→ T_amb 作为输入即可吸收 α 双峰信息')
    # 优先级 3: 策略驱动
    elif (not r2['is_seasonal']) and abs_r < 0.4 and r4['median_length'] > LONG_RUN_THRESHOLD:
        verdict = ('【策略驱动】证据一致指向控制策略\n'
                   '  - Step 2 条件下仍多峰\n'
                   f'  - Step 5 |Pearson| = {abs_r:.3f} < 0.4\n'
                   '  - Step 4 持续时间长\n'
                   '→ 后续设备建模必须加 mode 变量\n'
                   '→ 按 α 阈值二值化,高/低模式分组拟合')
    # 优先级 4: 混合
    else:
        verdict = ('【混合驱动】证据不完全一致\n'
                   '→ 建议先按 T_amb 单变量拟合冷水机模型\n'
                   '→ 若残差呈现 α-相关的系统性偏差,再追加 mode 变量\n'
                   '→ 论文里诚实说明驱动机制的不确定性')

    lines.append(verdict)
    lines.append('=' * 70)

    out = os.path.join(OUT_DIR, 'alpha_report.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\n报告 → {out}\n')
    print('\n'.join(lines))


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_data()

    results = {}
    results['1'] = step_1_1d_kde(df)
    results['2'] = step_2_conditional_kde(df)
    results['3'] = step_3_time_series(df)
    results['4'] = step_4_mode_persistence(df, peaks=results['1']['peaks'])
    results['5'] = step_5_correlation(df)

    write_conclusion(results, df)
    print('\n✓ 全部完成')