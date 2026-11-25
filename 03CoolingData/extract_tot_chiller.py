# -*- coding: utf-8 -*-
"""
================================================================================
extract_servizi_and_pue.py
================================================================================
脚本目的
--------
1.从数据集提取 Tot_servizi(辅助服务总耗电:照明、UPS损耗、BMS 楼宇管理、安防等),按小时 resample，
与现有 Tot_ict / Tot_chiller / Tot_cdz / Tot_qpompe 合并对齐
3. 计算两个能效指标:
     - pPUE_cooling:冷源子系统部分 PUE(本项目优化对象)
     - PUE_true:    真实数据中心 PUE(论文外部 benchmark 对标)
4. 输出诊断图(时序、分布、组成占比、月度变化)和文字统计报告

代码结构(按从上到下顺序)
------------------------
[配置区]    路径常量、绘图样式
[PART 1]   extract_tot_servizi()  - 从 ExaMon 提取并保存 parquet
[PART 2]   load_all_powers()      - 加载所有 Tot_* 字段
[PART 3]   merge_powers()         - 按 timestamp 内连接对齐
[PART 4]   compute_pue()          - 计算 pPUE / PUE_true / 占比
[PART 5]   plot_diagnostics()     - 5 张诊断图
[PART 6]   write_report()         - 文字统计报告
[main]     按顺序调用以上模块

设计要点
--------
- 单脚本完成"提取+计算+诊断"全流程,论文写作时可一键复现
- PUE 严格区分 pPUE 与 true PUE(避免学术不严谨)
- 内连接保证 PUE 计算口径一致(分子分母同步骤)
- 所有中间产物落盘,论文图表/数字可追溯
================================================================================
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 配置区
# ==============================================================================

ROOT = Path(r'D:\PythonProject1')
IT_DIR = ROOT / '01dataFresh'                  # IT 耗电数据目录
COOL_DIR = ROOT / '03CoolingData'              # 冷源数据目录
OUT_DIR = COOL_DIR / 'pue_results'              # 诊断输出目录
OUT_DIR.mkdir(exist_ok=True)

# ★ M100 ExaMon 数据根目录:与现有 extract_tot_chiller.py 保持一致
M100_DATA_ROOT = Path(r'D:\path\to\m100\data')   # ← 改成你本机的实际路径

# 绘图样式(中文 + 高 DPI)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ==============================================================================
# PART 1: 提取 Tot_servizi
# ------------------------------------------------------------------------------
# 设计逻辑:
#   1. 完全套用 extract_tot_chiller.py 的同款流程(单字段提取范式)
#   2. M100 数据按 year_month / plugin / metric 三级分区
#   3. plugin='logics'(聚合面板),metric='Tot_servizi'
#   4. 跨月份 concat → 小时 resample → 单列 parquet
#
# ★★ 如果你现有 extract_*.py 的 ExaMon 读取方式与下面不同,
#    请将"跨月份读取"那段替换为你的同款代码,
#    本函数其余部分(resample / save)可不动。
# ==============================================================================

def extract_tot_servizi() -> pd.DataFrame:
    """
    从 M100 数据集提取 Tot_servizi 字段并保存为 parquet

    返回
    ----
    DataFrame[timestamp, Tot_servizi_kW]
    """
    PLUGIN = 'logics'
    METRIC = 'Tot_servizi'
    OUT_COL = 'Tot_servizi_kW'
    OUT_PATH = COOL_DIR / 'tot_servizi.parquet'

    print(f"\n[PART 1] 提取 {METRIC} ...")

    # ----- 跨月份读取所有 parquet 分片 -----
    dfs = []
    for ym_dir in sorted(M100_DATA_ROOT.glob('year_month=*')):
        metric_dir = ym_dir / f'plugin={PLUGIN}' / f'metric={METRIC}'
        if not metric_dir.exists():
            continue
        for f in metric_dir.glob('*.parquet'):
            dfs.append(pd.read_parquet(f))

    if not dfs:
        raise FileNotFoundError(
            f"找不到 plugin={PLUGIN}/metric={METRIC} 的任何数据。\n"
            f"请确认 M100_DATA_ROOT 路径和 metric 名称是否正确。"
        )

    df = pd.concat(dfs, ignore_index=True)
    print(f"  读入原始记录: {len(df):,} 条")

    # ----- 时间戳解析(ExaMon 一般是 Unix 秒)-----
    # 如果你现有脚本是别的格式(ms / iso),请改这一行
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.set_index('timestamp').sort_index()

    # ----- 小时 resample(与其他 Tot_* 脚本对齐:整点 + 均值)-----
    df_hourly = (
        df['value']
        .resample('1h', origin='start_day')
        .mean()
        .to_frame(name=OUT_COL)
        .reset_index()
    )
    print(f"  小时级数据:  {len(df_hourly):,} 行")
    print(f"  时间范围:    {df_hourly['timestamp'].min()} ~ "
          f"{df_hourly['timestamp'].max()}")
    print(f"  统计:        均值={df_hourly[OUT_COL].mean():.2f} kW, "
          f"中位数={df_hourly[OUT_COL].median():.2f} kW")

    # ----- 保存 -----
    df_hourly.to_parquet(OUT_PATH, index=False)
    print(f"  ✓ 已保存: {OUT_PATH}")

    return df_hourly


# ==============================================================================
# PART 2: 加载所有 Tot_* 字段
# ------------------------------------------------------------------------------
# 设计逻辑:
#   - 集中读取所有功耗 parquet,避免在 main 里散落多个 read_parquet
#   - 返回 dict 而不是 list,便于按 name 索引
# ==============================================================================

def load_all_powers() -> dict:
    """
    加载所有功耗字段,返回字典 {name: DataFrame}
    """
    print(f"\n[PART 2] 加载所有 Tot_* 字段 ...")

    sources = {
        'Tot_ict':     IT_DIR  / 'tot_ict.parquet',
        'Tot_chiller': COOL_DIR / 'tot_chiller.parquet',
        'Tot_cdz':     COOL_DIR / 'tot_cdz.parquet',
        'Tot_qpompe':  COOL_DIR / 'tot_qpompe.parquet',
        'Tot_servizi': COOL_DIR / 'tot_servizi.parquet',
    }

    powers = {}
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"缺失输入: {path}")
        df = pd.read_parquet(path)
        powers[name] = df
        print(f"  {name:12s}: {len(df):,} 行  ({path.name})")

    return powers


# ==============================================================================
# PART 3: 合并对齐
# ------------------------------------------------------------------------------
# 设计逻辑:
#   - 使用内连接(inner join):只保留所有字段都有值的小时
#   - 原因:PUE 计算必须分子分母同口径,缺一个字段就跳过该小时
#   - 缺失小时数 = 全时段 - 内连接结果,在报告里说明
#
#   - 防御性重命名:不同 extract 脚本可能列名不一致(value / Tot_chiller_kW
#     / Power 等),这里强制统一为 <Name>_kW
# ==============================================================================

def merge_powers(powers: dict) -> pd.DataFrame:
    """
    按 timestamp 内连接所有功耗字段

    返回
    ----
    DataFrame[timestamp, Tot_ict_kW, Tot_chiller_kW, Tot_cdz_kW,
              Tot_qpompe_kW, Tot_servizi_kW]
    """
    print(f"\n[PART 3] 合并对齐(内连接)...")

    dfs_normalized = []
    for name, df in powers.items():
        col_kw = f'{name}_kW'
        df = df.copy()
        # 强制重命名:第一列时间,第二列数值
        df.columns = ['timestamp', col_kw]
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        dfs_normalized.append(df.set_index('timestamp'))

    merged = pd.concat(dfs_normalized, axis=1, join='inner').reset_index()

    # 覆盖率统计
    t_min = min(df.index.min() for df in dfs_normalized)
    t_max = max(df.index.max() for df in dfs_normalized)
    full_hours = int((t_max - t_min).total_seconds() / 3600) + 1
    coverage = len(merged) / full_hours * 100

    print(f"  合并后行数:  {len(merged):,}")
    print(f"  时间范围:    {merged['timestamp'].min()} ~ {merged['timestamp'].max()}")
    print(f"  覆盖率:      {coverage:.1f}% (相对于全时段 {full_hours} 小时)")

    return merged


# ==============================================================================
# PART 4: 计算 PUE
# ------------------------------------------------------------------------------
# 公式与含义:
#
#   P_cooling   = Tot_chiller + Tot_cdz + Tot_qpompe   (冷源子系统总耗电)
#   P_facility  = Tot_ict + P_cooling + Tot_servizi    (设施总耗电)
#
#   pPUE_cooling = (Tot_ict + P_cooling) / Tot_ict
#       含义:冷源子系统部分 PUE,本项目 GA 优化对象
#       不含:servizi(优化不可控部分)
#
#   PUE_true    = P_facility / Tot_ict
#       含义:数据中心整体能效(Green Grid / ASHRAE 标准定义)
#       含全部非 IT 消耗
#
#   share_*    = 各分量 / P_facility   (能耗组成占比)
# ==============================================================================

def compute_pue(df: pd.DataFrame) -> pd.DataFrame:
    """
    在合并后的功耗表上计算 PUE 和分项占比

    新增列
    ------
    P_cooling_kW, P_facility_kW
    pPUE_cooling, PUE_true
    share_IT, share_cooling, share_servizi
    """
    print(f"\n[PART 4] 计算 PUE 指标 ...")

    df = df.copy()
    df['P_cooling_kW']  = df['Tot_chiller_kW'] + df['Tot_cdz_kW'] + df['Tot_qpompe_kW']
    df['P_facility_kW'] = df['Tot_ict_kW'] + df['P_cooling_kW'] + df['Tot_servizi_kW']

    df['pPUE_cooling'] = (df['Tot_ict_kW'] + df['P_cooling_kW']) / df['Tot_ict_kW']
    df['PUE_true']     = df['P_facility_kW'] / df['Tot_ict_kW']

    df['share_IT']      = df['Tot_ict_kW']     / df['P_facility_kW']
    df['share_cooling'] = df['P_cooling_kW']   / df['P_facility_kW']
    df['share_servizi'] = df['Tot_servizi_kW'] / df['P_facility_kW']

    # 控制台核心统计
    print(f"  pPUE_cooling: 均值={df['pPUE_cooling'].mean():.3f}, "
          f"中位数={df['pPUE_cooling'].median():.3f}")
    print(f"  PUE_true:     均值={df['PUE_true'].mean():.3f}, "
          f"中位数={df['PUE_true'].median():.3f}")
    print(f"  能耗占比(均值): IT={df['share_IT'].mean()*100:.1f}%, "
          f"Cooling={df['share_cooling'].mean()*100:.1f}%, "
          f"Servizi={df['share_servizi'].mean()*100:.1f}%")

    return df


# ==============================================================================
# PART 5: 诊断图
# ------------------------------------------------------------------------------
# 五张图,各自回答一个问题:
#   图1 pue_timeseries          : pPUE 和 true PUE 随时间怎么变?
#   图2 pue_distribution        : 两个指标的分布形态?有 outlier 吗?
#   图3 power_composition_pie   : 整体能耗各部分占多少?
#   图4 power_composition_stacked: 能耗组成随时间怎么变化?
#   图5 pue_monthly             : 月度 PUE,看季节性
# ==============================================================================

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


def plot_diagnostics(df: pd.DataFrame):
    """生成 5 张诊断图,保存到 pue_results/"""
    print(f"\n[PART 5] 生成诊断图 ...")

    # ---- 图1: PUE 时序 ----
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['timestamp'], df['pPUE_cooling'], lw=0.5, alpha=0.6,
            label='pPUE_cooling (冷源)')
    ax.plot(df['timestamp'], df['PUE_true'],     lw=0.5, alpha=0.6,
            label='PUE_true (整体)')
    ax.set_xlabel('时间')
    ax.set_ylabel('PUE')
    ax.set_title('PUE 时间序列')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / 'pue_timeseries.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 图2: PUE 分布(直方图)----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, color in zip(axes, ['pPUE_cooling', 'PUE_true'],
                              ['steelblue', 'tomato']):
        ax.hist(df[col].dropna(), bins=80, color=color, alpha=0.7)
        m, med = df[col].mean(), df[col].median()
        ax.axvline(m,   color='k', ls='--', label=f'均值={m:.3f}')
        ax.axvline(med, color='k', ls=':',  label=f'中位数={med:.3f}')
        ax.set_xlabel(col)
        ax.set_ylabel('频次')
        ax.set_title(f'{col} 分布')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'pue_distribution.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 图3: 能耗占比饼图 ----
    means = {
        'IT (Tot_ict)':   df['Tot_ict_kW'].mean(),
        'Chiller':        df['Tot_chiller_kW'].mean(),
        'CDZ (空调)':     df['Tot_cdz_kW'].mean(),
        'Qpompe (水泵)':  df['Tot_qpompe_kW'].mean(),
        'Servizi (服务)': df['Tot_servizi_kW'].mean(),
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(means.values(), labels=means.keys(), colors=COLORS,
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f'M100 数据中心平均能耗组成\n'
                 f'(总功率均值 {sum(means.values()):.1f} kW)')
    fig.savefig(OUT_DIR / 'power_composition_pie.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 图4: 能耗组成随时间(stacked area)----
    # 重采样到日级别便于可视化(小时级太密)
    df_daily = df.set_index('timestamp').resample('1D').mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(
        df_daily.index,
        df_daily['Tot_ict_kW'], df_daily['Tot_chiller_kW'],
        df_daily['Tot_cdz_kW'], df_daily['Tot_qpompe_kW'],
        df_daily['Tot_servizi_kW'],
        labels=['IT', 'Chiller', 'CDZ', 'Qpompe', 'Servizi'],
        colors=COLORS, alpha=0.85
    )
    ax.set_xlabel('时间(日)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title('能耗组成随时间(日均)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / 'power_composition_stacked.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 图5: 月度 PUE 箱线图 ----
    df_m = df.copy()
    df_m['year_month'] = df_m['timestamp'].dt.to_period('M').astype(str)
    months = sorted(df_m['year_month'].unique())

    data_ppue = [df_m.loc[df_m['year_month'] == m, 'pPUE_cooling'].dropna()
                 for m in months]
    data_true = [df_m.loc[df_m['year_month'] == m, 'PUE_true'].dropna()
                 for m in months]

    fig, ax = plt.subplots(figsize=(14, 5))
    pos = np.arange(len(months))
    ax.boxplot(data_ppue, positions=pos - 0.2, widths=0.35,
               patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax.boxplot(data_true, positions=pos + 0.2, widths=0.35,
               patch_artist=True,
               boxprops=dict(facecolor='tomato', alpha=0.6))
    ax.set_xticks(pos)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_ylabel('PUE')
    ax.set_title('月度 PUE 分布对比 (蓝=pPUE_cooling, 红=PUE_true)')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'pue_monthly.png', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ 5 张图已保存到 {OUT_DIR}")


# ==============================================================================
# PART 6: 文字报告
# ==============================================================================

def write_report(df: pd.DataFrame):
    """生成统计报告 pue_report.txt(同时打印到控制台)"""
    print(f"\n[PART 6] 生成统计报告 ...")

    def fmt(x): return f"{x:.4f}"

    lines = []
    lines.append("=" * 70)
    lines.append("M100 数据中心 PUE 统计报告")
    lines.append("=" * 70)
    lines.append(f"数据时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    lines.append(f"有效小时数:   {len(df):,}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("【一】功耗均值 (kW)")
    lines.append("-" * 70)
    for col in ['Tot_ict_kW', 'Tot_chiller_kW', 'Tot_cdz_kW',
                'Tot_qpompe_kW', 'Tot_servizi_kW',
                'P_cooling_kW', 'P_facility_kW']:
        lines.append(f"  {col:18s}  均值={df[col].mean():8.2f}  "
                     f"中位数={df[col].median():8.2f}  "
                     f"标准差={df[col].std():8.2f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("【二】PUE 指标")
    lines.append("-" * 70)
    for col in ['pPUE_cooling', 'PUE_true']:
        s = df[col].dropna()
        lines.append(f"  {col}")
        lines.append(f"    均值     = {fmt(s.mean())}")
        lines.append(f"    中位数   = {fmt(s.median())}")
        lines.append(f"    标准差   = {fmt(s.std())}")
        lines.append(f"    最小值   = {fmt(s.min())}")
        lines.append(f"    最大值   = {fmt(s.max())}")
        lines.append(f"    25%分位  = {fmt(s.quantile(0.25))}")
        lines.append(f"    75%分位  = {fmt(s.quantile(0.75))}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("【三】能耗组成(均值占比)")
    lines.append("-" * 70)
    lines.append(f"  IT 占比:    {df['share_IT'].mean()*100:6.2f}%")
    lines.append(f"  冷源占比:   {df['share_cooling'].mean()*100:6.2f}%")
    lines.append(f"  服务占比:   {df['share_servizi'].mean()*100:6.2f}%")
    lines.append("")

    lines.append("-" * 70)
    lines.append("【四】解读")
    lines.append("-" * 70)
    diff = df['PUE_true'].median() - df['pPUE_cooling'].median()
    lines.append("  pPUE_cooling 表征冷源子系统效率,值越接近 1 越好。")
    lines.append("  PUE_true 是数据中心整体能效的行业标准指标。")
    lines.append(f"  PUE_true - pPUE_cooling = {diff:.3f}")
    lines.append("  ↑ 这部分由 Servizi(照明/UPS损耗/BMS/安防等)贡献,")
    lines.append("    本项目优化范围之外。")
    lines.append("")
    lines.append("=" * 70)

    report_path = OUT_DIR / 'pue_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ 报告已保存: {report_path}")
    # 控制台也打印一遍(便于即时查看)
    print('\n'.join(lines[1:]))


# ==============================================================================
# main 编排
# ==============================================================================

def main():
    print("=" * 70)
    print("M100 PUE 提取与计算(Tot_servizi + 全部 PUE 指标)")
    print("=" * 70)

    # PART 1: 提取 Tot_servizi(已提取过则跳过)
    servizi_path = COOL_DIR / 'tot_servizi.parquet'
    if not servizi_path.exists():
        extract_tot_servizi()
    else:
        print(f"\n[PART 1] tot_servizi.parquet 已存在,跳过提取")

    # PART 2-3: 加载并合并
    powers = load_all_powers()
    df = merge_powers(powers)

    # PART 4: 计算
    df = compute_pue(df)

    # 保存合并后的完整数据(供后续脚本复用)
    out_full = OUT_DIR / 'all_powers_with_pue.parquet'
    df.to_parquet(out_full, index=False)
    print(f"\n  ✓ 完整功耗+PUE 表已保存: {out_full}")

    # PART 5-6: 诊断 + 报告
    plot_diagnostics(df)
    write_report(df)

    print("\n" + "=" * 70)
    print("✓ 全部完成")
    print("=" * 70)


if __name__ == '__main__':
    main()