"""
calc_liquid_cooling_ratio.py
计算液冷覆盖率：液冷制冷量 / IT总功耗
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = r'D:\PythonProject1\01dataFresh'

# 读取已有数据
df_cooling = pd.read_parquet(
    os.path.join(OUT_DIR, 'cooling_load.parquet')
)
df_it = pd.read_parquet(
    os.path.join(OUT_DIR, 'it_power_final.parquet')
)

# 按时间戳合并
df = df_cooling[['timestamp', 'cooling_load_kW']].merge(
    df_it[['timestamp', 'it_power_final_kW']],
    on='timestamp', how='inner'
)

# 过滤异常值
df = df[
    (df['cooling_load_kW'] > 0) &
    (df['it_power_final_kW'] > 100)  # IT功耗太低说明采集异常
]

# 计算液冷覆盖率
df['liquid_ratio'] = df['cooling_load_kW'] / df['it_power_final_kW']

# 过滤明显异常的覆盖率（物理上不可能超过1.5）
df = df[df['liquid_ratio'].between(0.1, 1.5)]

# 统计结果
print('=' * 50)
print('液冷覆盖率统计（液冷制冷量 / IT总功耗）')
print('=' * 50)
print(f'均值：  {df["liquid_ratio"].mean():.2%}')
print(f'中位数：{df["liquid_ratio"].median():.2%}')
print(f'最小值：{df["liquid_ratio"].min():.2%}')
print(f'最大值：{df["liquid_ratio"].max():.2%}')
print(f'样本数：{len(df)} 个时间点')

# 按月统计
df['month'] = df['timestamp'].dt.to_period('M')
monthly = df.groupby('month')['liquid_ratio'].agg(['mean','min','max'])
monthly.columns = ['均值', '最小值', '最大值']
print('\n按月统计：')
print(monthly.round(3).to_string())

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 时间序列
axes[0].plot(df['timestamp'], df['liquid_ratio'],
             linewidth=0.5, alpha=0.7)
axes[0].axhline(y=df['liquid_ratio'].mean(),
                color='r', linestyle='--',
                label=f'均值={df["liquid_ratio"].mean():.2%}')
axes[0].set_ylabel('液冷覆盖率')
axes[0].set_title('液冷覆盖率时间序列')
axes[0].legend()

# 分布直方图
axes[1].hist(df['liquid_ratio'], bins=50, edgecolor='black')
axes[1].axvline(x=df['liquid_ratio'].mean(),
                color='r', linestyle='--',
                label=f'均值={df["liquid_ratio"].mean():.2%}')
axes[1].set_xlabel('液冷覆盖率')
axes[1].set_ylabel('频次')
axes[1].set_title('液冷覆盖率分布')
axes[1].legend()

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'liquid_cooling_ratio.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f'\n图表已保存：{out_path}')