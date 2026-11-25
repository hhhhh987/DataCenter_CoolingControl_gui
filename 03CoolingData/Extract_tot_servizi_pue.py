"""
extract_servizi_and_pue.py

目的:
  1. 提取 Tot_servizi(套用 extract_tot_chiller.py 同款流程)
  2. 合并所有 Tot_* 字段,计算 pPUE_cooling 与 PUE_true
  3. 输出最小诊断:
       - tot_servizi.parquet           (新提取字段)
       - power_composition_pie.png     (论文用饼图)
       - pue_distribution.png          (作者自检:两 PUE 直方图)
       - pue_report.txt                (PUE 数字 + 占比)

适用阶段: 阶段3 - Step 0 (补真 PUE)

代码结构:
  [PART 1] extract_tot_servizi()  - 单字段提取,完全对齐 extract_tot_chiller.py
  [PART 2] compute_pue_table()    - 加载+内连接+计算 PUE
  [PART 3] make_plots()           - 2 张图(饼图 + PUE 直方图)
  [PART 4] write_report()         - 文字报告
  [main]   按序调用
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────
ROOT = r'D:\PythonProject1'

MONTHS = [
    '21-03', '21-04', '21-05', '21-06', '21-07', '21-08', '21-09', '21-10', '21-11', '21-12',
    '22-01', '22-02', '22-03', '22-04', '22-05', '22-06', '22-07', '22-08', '22-09'
]
FREQ = '1h'

IT_PATH      = os.path.join(ROOT, '01dataFresh', 'tot_ict.parquet')
COOL_DIR     = os.path.join(ROOT, '03CoolingData')
SERVIZI_OUT  = os.path.join(COOL_DIR, 'tot_servizi.parquet')
OUT_DIR      = os.path.join(COOL_DIR, 'pue_results')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ── PART 1: 提取 Tot_servizi ───────────────────────────────
# 完全套用 extract_tot_chiller.py 的逻辑,只改三处:
#   metric:   Tot_chiller    → Tot_servizi
#   列名:     Tot_chiller_kW → Tot_servizi_kW
#   输出:     tot_chiller    → tot_servizi
# ──────────────────────────────────────────────────────────

def read_one_month_servizi(month: str) -> pd.DataFrame:
    """读取单个月份的 Tot_servizi 原始数据"""
    path = os.path.join(ROOT, f'year_month={month}',
                        'plugin=logics_pub', 'metric=Tot_servizi')

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


def resample_servizi_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """重采样为1小时均值"""
    if 'panel' in df.columns:
        df = df[df['panel'] == 'generals'].copy()

    if df.empty:
        return pd.DataFrame()

    # 异常过滤:Servizi(辅助服务)正常范围 30~150kW,>1000 为明显异常
    df.loc[df['value'] < 0, 'value'] = np.nan
    df.loc[df['value'] > 1000, 'value'] = np.nan

    df = df.sort_values('timestamp').set_index('timestamp')
    out = df['value'].resample(FREQ, origin='start_day').mean().reset_index()
    out.columns = ['timestamp', 'Tot_servizi_kW']
    out = out.dropna(subset=['Tot_servizi_kW'])
    return out


def extract_tot_servizi() -> pd.DataFrame:
    print('开始提取 Tot_servizi 数据...\n')
    all_months = []

    for month in MONTHS:
        print(f'处理 {month} ...', end=' ')

        df_raw = read_one_month_servizi(month)
        if df_raw.empty:
            continue
        print(f'原始行数={len(df_raw)}', end=' → ')

        df_h = resample_servizi_hourly(df_raw)
        if df_h.empty:
            print('重采样后为空,跳过')
            continue
        print(f'重采样后={len(df_h)}行')
        all_months.append(df_h)

    if not all_months:
        raise RuntimeError('错误:所有月份均无 Tot_servizi 数据')

    df_all = pd.concat(all_months, ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    df_all.to_parquet(SERVIZI_OUT, index=False, compression='zstd')

    print(f'\n完成!')
    print(f'输出文件:{SERVIZI_OUT}')
    print(f'总行数:{len(df_all)} 条(每{FREQ}一条)')
    print(f'时间范围:{df_all.timestamp.min()} → {df_all.timestamp.max()}')
    print(f'Tot_servizi 范围:{df_all.Tot_servizi_kW.min():.1f} ~ {df_all.Tot_servizi_kW.max():.1f} kW')
    print(f'均值:{df_all.Tot_servizi_kW.mean():.1f} kW, 中位数:{df_all.Tot_servizi_kW.median():.1f} kW\n')

    return df_all


# ── PART 2: 加载 + 内连接 + 算 PUE ──────────────────────────
# 设计逻辑:
#   - 内连接:只保留所有字段都有值的小时(PUE 分子分母同口径)
#   - 一次性算 P_cooling / P_facility / pPUE / PUE_true / share_*
# ──────────────────────────────────────────────────────────

def compute_pue_table() -> pd.DataFrame:
    print('加载所有功耗 parquet ...')
    sources = {
        'Tot_ict':     IT_PATH,
        'Tot_chiller': os.path.join(COOL_DIR, 'tot_chiller.parquet'),
        'Tot_cdz':     os.path.join(COOL_DIR, 'tot_cdz.parquet'),
        'Tot_qpompe':  os.path.join(COOL_DIR, 'tot_qpompe.parquet'),
        'Tot_servizi': SERVIZI_OUT,
    }

    dfs = []
    for name, path in sources.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'缺失输入:{path}')
        df = pd.read_parquet(path)
        df.columns = ['timestamp', f'{name}_kW']
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        dfs.append(df.set_index('timestamp'))
        print(f'  {name:12s}: {len(df):>6,} 行')

    df = pd.concat(dfs, axis=1, join='inner').reset_index()
    print(f'内连接后:{len(df):,} 行')
    print(f'时间范围:{df.timestamp.min()} → {df.timestamp.max()}\n')

    # 衍生量
    df['P_cooling_kW']  = df['Tot_chiller_kW'] + df['Tot_cdz_kW'] + df['Tot_qpompe_kW']
    df['P_facility_kW'] = df['Tot_ict_kW'] + df['P_cooling_kW'] + df['Tot_servizi_kW']

    df['pPUE_cooling'] = (df['Tot_ict_kW'] + df['P_cooling_kW']) / df['Tot_ict_kW']
    df['PUE_true']     = df['P_facility_kW'] / df['Tot_ict_kW']

    df['share_IT']      = df['Tot_ict_kW']     / df['P_facility_kW']
    df['share_cooling'] = df['P_cooling_kW']   / df['P_facility_kW']
    df['share_servizi'] = df['Tot_servizi_kW'] / df['P_facility_kW']

    return df


# ── PART 3: 两张图 ─────────────────────────────────────────
# 图1 power_composition_pie.png : 论文用,能耗组成饼图
# 图2 pue_distribution.png      : 作者自检,验证两 PUE 平行偏移
# ──────────────────────────────────────────────────────────

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


def plot_pie(df: pd.DataFrame):
    """图1: 能耗组成饼图 — 论文用"""
    means = {
        'IT (Tot_ict)':   df['Tot_ict_kW'].mean(),
        'Chiller':        df['Tot_chiller_kW'].mean(),
        'CDZ (空调)':     df['Tot_cdz_kW'].mean(),
        'Qpompe (水泵)':  df['Tot_qpompe_kW'].mean(),
        'Servizi (服务)': df['Tot_servizi_kW'].mean(),
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(means.values(), labels=means.keys(), colors=COLORS,
           autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(edgecolor='white', linewidth=1.5))
    ax.set_title(f'M100 数据中心平均能耗组成\n'
                 f'(总功率均值 {sum(means.values()):.1f} kW)')
    out = os.path.join(OUT_DIR, 'power_composition_pie.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out}')


def plot_pue_dist(df: pd.DataFrame):
    """图2: PUE 双指标分布直方图 — 作者自检"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, color in zip(axes,
                              ['pPUE_cooling', 'PUE_true'],
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
    out = os.path.join(OUT_DIR, 'pue_distribution.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out}')


def make_plots(df: pd.DataFrame):
    print('生成诊断图 ...')
    plot_pie(df)
    plot_pue_dist(df)


# ── PART 4: 文字报告 ───────────────────────────────────────

def write_report(df: pd.DataFrame):
    """生成 pue_report.txt 并打印到控制台"""

    def fmt(x): return f'{x:.4f}'

    lines = []
    lines.append('=' * 60)
    lines.append('M100 数据中心 PUE 统计报告')
    lines.append('=' * 60)
    lines.append(f'数据范围:{df.timestamp.min()} ~ {df.timestamp.max()}')
    lines.append(f'有效小时:{len(df):,}')
    lines.append('')

    lines.append('【一】功耗均值 (kW)')
    for col in ['Tot_ict_kW', 'Tot_chiller_kW', 'Tot_cdz_kW',
                'Tot_qpompe_kW', 'Tot_servizi_kW',
                'P_cooling_kW', 'P_facility_kW']:
        lines.append(f'  {col:18s}  均值={df[col].mean():8.2f}  '
                     f'中位数={df[col].median():8.2f}')
    lines.append('')

    lines.append('【二】PUE 指标')
    for col in ['pPUE_cooling', 'PUE_true']:
        s = df[col].dropna()
        lines.append(f'  {col}')
        lines.append(f'    均值={fmt(s.mean())}, 中位数={fmt(s.median())}')
        lines.append(f'    P25 ={fmt(s.quantile(0.25))}, P75={fmt(s.quantile(0.75))}')
    diff = df['PUE_true'].median() - df['pPUE_cooling'].median()
    lines.append(f'  PUE_true - pPUE_cooling = {diff:.3f}   (← Servizi 贡献)')
    lines.append('')

    lines.append('【三】能耗占比 (均值)')
    lines.append(f'  IT      {df.share_IT.mean()*100:6.2f}%')
    lines.append(f'  Cooling {df.share_cooling.mean()*100:6.2f}%')
    lines.append(f'  Servizi {df.share_servizi.mean()*100:6.2f}%')
    lines.append('')
    lines.append('=' * 60)

    out = os.path.join(OUT_DIR, 'pue_report.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\n报告 → {out}\n')
    print('\n'.join(lines))


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':

    # PART 1: 提取(已有则跳过)
    if not os.path.exists(SERVIZI_OUT):
        extract_tot_servizi()
    else:
        print(f'tot_servizi.parquet 已存在,跳过提取\n')

    # PART 2: 算 PUE(in-memory,不落盘,本次任务用完即弃)
    df = compute_pue_table()

    # PART 3-4: 图和报告
    make_plots(df)
    write_report(df)

    print('\n✓ 全部完成')