"""
compare_q101_q102.py
对比 Schneider PLC 的 Q101 和 Q102 两个面板的 Portata_attiva 数据
目的：判断冗余双系统的实际工作模式，决定后续提取策略
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────
ROOT = r'D:\PythonProject1'

MONTHS = [
    '21-03', '21-04', '21-05', '21-06','21-07', '21-08', '21-09', '21-10', '21-11', '21-12',
    '22-01', '22-02','22-03','22-04','22-05','22-06','22-07','22-08','22-09'
]

# 两个面板的metric路径
METRIC_Q101 = 'PLC_PLC_Q101.Portata_attiva'
METRIC_Q102 = 'PLC_PLC_Q102.Portata_attiva'

OUT_DIR = os.path.join(ROOT, '03CoolingData', 'compare_results')
FREQ = '1h'
UNIT_SCALE = 10.0    # m³/h × 10 → m³/h


# ── Block 1: 提取单个面板的小时级数据 ──────────────────────

def extract_one_panel(metric_name: str, col_name: str) -> pd.DataFrame:
    """
    提取一个面板的全部月份数据并重采样到小时级
    返回：[timestamp, col_name] 两列的DataFrame
    """
    print(f'\n=== 提取 {metric_name} ===')
    all_months = []

    for month in MONTHS:
        path = os.path.join(ROOT, f'year_month={month}',
                            'plugin=schneider_pub', f'metric={metric_name}')

        if not os.path.exists(path):
            continue

        files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        if not files:
            continue

        # 读取该月所有parquet
        dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # 单位换算 + 异常过滤
        df['value'] = df['value'] / UNIT_SCALE
        df.loc[(df['value'] < 0) | (df['value'] > 500), 'value'] = np.nan

        # 重采样到小时
        df = df.sort_values('timestamp').set_index('timestamp')
        resampled = df['value'].resample(FREQ, origin='start_day').mean().reset_index()
        resampled.columns = ['timestamp', col_name]
        resampled = resampled.dropna(subset=[col_name])

        all_months.append(resampled)
        print(f'  {month}: {len(resampled)} 行')

    if not all_months:
        return pd.DataFrame()

    result = pd.concat(all_months, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    print(f'  总行数: {len(result)}')
    return result


# ── Block 2: 合并两个面板并分析覆盖情况 ────────────────────

def merge_and_analyze_coverage(df_q101: pd.DataFrame, df_q102: pd.DataFrame):
    """
    outer join合并两个面板，统计三类时段的样本数：
    - 只有Q101有数据
    - 只有Q102有数据
    - 两者都有数据
    """
    merged = df_q101.merge(df_q102, on='timestamp', how='outer')
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    only_q101 = merged['Q101'].notna() & merged['Q102'].isna()
    only_q102 = merged['Q101'].isna() & merged['Q102'].notna()
    both = merged['Q101'].notna() & merged['Q102'].notna()

    print('\n=== 数据覆盖统计 ===')
    print(f'  总时间点（outer join）: {len(merged)}')
    print(f'  只有 Q101 有数据:       {only_q101.sum()} ({only_q101.mean()*100:.1f}%)')
    print(f'  只有 Q102 有数据:       {only_q102.sum()} ({only_q102.mean()*100:.1f}%)')
    print(f'  两者同时有数据:         {both.sum()} ({both.mean()*100:.1f}%)')

    return merged, both


# ── Block 3: 关系判定 ────────────────────────────────────

def diagnose_relationship(merged: pd.DataFrame, both_mask: pd.Series):
    """
    根据"两者同时有数据"的时段，判断Q101和Q102的关系类型
    返回：诊断结论字符串
    """
    if both_mask.sum() < 100:
        print('\n两者同时有数据的样本太少（<100），无法做相关性分析')
        return 'INSUFFICIENT_OVERLAP'

    df_both = merged[both_mask].copy()
    q101 = df_both['Q101']
    q102 = df_both['Q102']

    # 关键统计量
    corr = q101.corr(q102)
    diff = q101 - q102
    diff_abs = diff.abs()
    rel_diff = diff_abs / q101.where(q101 != 0, np.nan) * 100  # 百分比差异
    sum_val = q101 + q102

    print('\n=== 同时有数据时段的统计 ===')
    print(f'  样本数: {len(df_both)}')
    print(f'  Q101均值: {q101.mean():.1f} m³/h, 范围: {q101.min():.1f}~{q101.max():.1f}')
    print(f'  Q102均值: {q102.mean():.1f} m³/h, 范围: {q102.min():.1f}~{q102.max():.1f}')
    print(f'  相关系数: {corr:.4f}')
    print(f'  平均绝对差: {diff_abs.mean():.2f} m³/h')
    print(f'  平均相对差: {rel_diff.mean():.2f}%')
    print(f'  Q101+Q102 均值: {sum_val.mean():.1f} m³/h')

    # 判定逻辑
    print('\n=== 关系判定 ===')

    if corr > 0.99 and rel_diff.mean() < 2:
        conclusion = 'IDENTICAL'
        print('  结论: 【完全一致】两面板数据几乎相同（相关性>0.99, 相对差<2%）')
        print('  含义: 真冗余备份，两面板读取同一组传感器')
        print('  建议: 只提取Q101即可，Q102无需提取')

    elif corr > 0.9 and rel_diff.mean() < 10:
        conclusion = 'SIMILAR'
        print('  结论: 【高度相似】两面板趋势一致但有小差异')
        print('  含义: 同一回路的冗余传感器，存在校准误差')
        print('  建议: 取均值 (Q101+Q102)/2 作为代表值，或只用Q101')

    elif corr > 0.5:
        conclusion = 'INDEPENDENT_BRANCH'
        print('  结论: 【独立分支】两面板正相关但数值差异大')
        print('  含义: 可能各控制液冷回路的一个分支')
        print('  建议: 取 Q101+Q102 作为总流量（两条支路合流）')

    else:
        conclusion = 'UNCLEAR'
        print('  结论: 【关系不明】相关性低，需要进一步分析')
        print('  建议: 查看下面的时序图判断具体情况')

    return conclusion


# ── Block 4: 可视化 ──────────────────────────────────────

def plot_comparison(merged: pd.DataFrame, both_mask: pd.Series):
    """画三张对比图：散点图、时序图、差值直方图"""
    os.makedirs(OUT_DIR, exist_ok=True)

    df_both = merged[both_mask]

    # 图1：散点图，看Q101 vs Q102的对应关系
    plt.figure(figsize=(7, 7))
    plt.scatter(df_both['Q101'], df_both['Q102'], s=2, alpha=0.3)
    # 画 y=x 参考线
    vmin = min(df_both['Q101'].min(), df_both['Q102'].min())
    vmax = max(df_both['Q101'].max(), df_both['Q102'].max())
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, label='y=x')
    plt.xlabel('Q101 Portata (m³/h)')
    plt.ylabel('Q102 Portata (m³/h)')
    plt.title(f'Q101 vs Q102  |  Correlation = {df_both["Q101"].corr(df_both["Q102"]):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scatter_q101_q102.png'), dpi=150)
    plt.close()

    # 图2：时序图，看两条曲线随时间的关系
    plt.figure(figsize=(14, 5))
    plt.plot(merged['timestamp'], merged['Q101'], label='Q101', linewidth=0.5, alpha=0.8)
    plt.plot(merged['timestamp'], merged['Q102'], label='Q102', linewidth=0.5, alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Portata (m³/h)')
    plt.title('Q101 and Q102 Time Series Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'timeseries_q101_q102.png'), dpi=150)
    plt.close()

    # 图3：差值直方图，看Q101-Q102的分布
    plt.figure(figsize=(8, 5))
    diff = df_both['Q101'] - df_both['Q102']
    plt.hist(diff, bins=60, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='diff=0')
    plt.xlabel('Q101 - Q102 (m³/h)')
    plt.ylabel('Count')
    plt.title(f'Difference Distribution  |  Mean={diff.mean():.2f}, Std={diff.std():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'diff_histogram.png'), dpi=150)
    plt.close()

    print(f'\n图表已保存到: {OUT_DIR}')


# ── 主流程 ────────────────────────────────────────────────

if __name__ == '__main__':
    # 第1块：分别提取Q101和Q102
    df_q101 = extract_one_panel(METRIC_Q101, 'Q101')
    df_q102 = extract_one_panel(METRIC_Q102, 'Q102')

    if df_q101.empty:
        print('Q101 数据为空，无法对比')
        exit()
    if df_q102.empty:
        print('Q102 数据为空，说明只有Q101可用，直接用Q101即可')
        exit()

    # 第2块：合并并分析覆盖
    merged, both_mask = merge_and_analyze_coverage(df_q101, df_q102)

    # 第3块：关系判定
    conclusion = diagnose_relationship(merged, both_mask)

    # 第4块：可视化
    plot_comparison(merged, both_mask)

    # 最终建议
    print('\n' + '='*60)
    print('最终建议：')
    print('='*60)
    if conclusion == 'IDENTICAL' or conclusion == 'SIMILAR':
        print('继续按当前脚本（只提取Q101）的方案执行')
        print('修改6个schneider提取脚本：保持METRIC_NAME中的Q101前缀')
    elif conclusion == 'INDEPENDENT_BRANCH':
        print('需要修改6个schneider提取脚本：')
        print('  - 同时提取Q101和Q102两份数据')
        print('  - 流量字段取相加（总流量 = Q101 + Q102）')
        print('  - 温度字段取均值或Q101（温度不能相加）')
    else:
        print('情况复杂，需要看图人工判断')