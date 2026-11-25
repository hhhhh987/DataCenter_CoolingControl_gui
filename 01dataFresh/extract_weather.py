"""
extract_weather.py
==================
从 weather_pub 插件提取气象数据
输出：01dataFresh/weather.parquet

输出列（共6列）：
  timestamp        时间戳

  temp_C           室外温度(°C)           直接提取
  humidity_pct     相对湿度(%)            直接提取
  dew_point_C      露点温度(°C)           直接提取
  wind_speed_ms    风速(m/s)              直接提取
  pressure_hPa     气压(hPa)             直接提取
  temp_wet_C       湿球温度(°C)           Stull(2011)公式计算

原始采样：10分钟一条
输出粒度：1小时均值（与其他数据对齐）
"""

import os
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1：配置区
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT    = r'D:\PythonProject1'
OUT_DIR = r'D:\PythonProject1\01dataFresh'
FREQ    = '1h'

# 有weather数据的月份（21-04到22-02）
MONTHS = [
    '21-03', '21-04', '21-05', '21-06','21-07', '21-08', '21-09', '21-10', '21-11', '21-12',
    '22-01', '22-02','22-03','22-04','22-05','22-06','22-07','22-08','22-09'
]

# 需要提取的5个原始指标
# key：数据集中的metric名称
# value：输出DataFrame中的列名
WEATHER_METRICS = {
    'temp'      : 'temp_C',
    'humidity'  : 'humidity_pct',
    'dew_point' : 'dew_point_C',
    'wind_speed': 'wind_speed_ms',
    'pressure'  : 'pressure_hPa',
}

os.makedirs(OUT_DIR, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2：工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def read_parquet_dir(path: str) -> pd.DataFrame:
    """
    读取指定目录下所有parquet文件并合并。

    功能：
      1. 检查路径和文件是否存在
      2. 读取所有parquet文件
      3. 统一timestamp格式（支持毫秒整数和字符串两种格式）

    参数：path  目录路径
    返回：合并后的DataFrame，失败返回空DataFrame
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # 统一timestamp格式
    if df['timestamp'].dtype in ['int64', 'float64']:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    return df


def resample_weather(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    将10分钟采样的weather数据重采样为1小时均值。

    功能：
      1. 只保留timestamp和value两列
      2. 按时间排序后以timestamp为索引
      3. resample('1h').mean() 每小时取均值
      4. 删除空值行（该小时内无数据）

    注意：
      weather插件没有panel/device列
      不需要像logics那样过滤panel
      直接对value列重采样即可

    参数：
      df        原始DataFrame
      col_name  输出列名

    返回：每小时一行的DataFrame（timestamp | col_name）
    """
    if df.empty:
        return pd.DataFrame()

    df = df[['timestamp', 'value']].sort_values('timestamp')
    df = df.set_index('timestamp')

    result =df['value'].resample(FREQ, origin='start_day').mean().reset_index()

    result.columns = ['timestamp', col_name]
    result = result.dropna(subset=[col_name])
    return result


def calc_wet_bulb_temp(temp: pd.Series,
                       humidity: pd.Series) -> pd.Series:
    """
    用Stull(2011)近似公式计算湿球温度。
      Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity
      and Air Temperature. Journal of Applied Meteorology and
      Climatology, 50(11), 2267-2269.
    公式：
      Tw = T × arctan(0.151977 × (RH + 8.313659)^0.5)
         + arctan(T + RH)
         - arctan(RH - 1.676331)
         + 0.00391838 × RH^1.5 × arctan(0.023101 × RH)
         - 4.686035

    用途：
      评估冷却塔散热效率
      冷却塔出水温度理论下限 ≈ 湿球温度
      湿球温度越低，冷却效果越好

    参数：
      temp      干球温度(°C)
      humidity  相对湿度(%)

    返回：湿球温度(°C)
    """
    rh = humidity
    t  = temp

    tw = (
        t * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3：主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
主流程逻辑：

Step1：逐月处理
  对每个月份：
    逐一读取5个指标的parquet文件
    各自重采样为1小时均值
    以temp_C为基准，inner join其他4个指标
    （inner join确保只保留5个指标都有数据的时间点）
    计算湿球温度

Step2：合并所有月份
  将9个月的DataFrame按行拼接
  按timestamp排序

Step3：检查时间连续性
  找出时间断点（连续缺失超过1小时）
  供后续src/process.py处理缺失值参考

Step4：保存输出
"""

print('=' * 60)
print('提取 weather 数据')
print('=' * 60)

all_months = []

# Step1：逐月处理
for month in MONTHS:
    print(f'\n[{month}]')
    month_data = {}
    month_ok = True

    # 逐一提取5个原始指标
    for metric, col_name in WEATHER_METRICS.items():
        path = os.path.join(
            ROOT, f'year_month={month}',
            'plugin=weather_pub', f'metric={metric}'
        )
        df_raw = read_parquet_dir(path)
        df_h   = resample_weather(df_raw, col_name)

        if df_h.empty:
            print(f'  ⚠️  {metric} 无数据')
            month_ok = False
        else:
            print(f'  {col_name}: {len(df_h)} 行')
            month_data[col_name] = df_h

    # 任意一个指标缺失则跳过该月
    if not month_ok or len(month_data) < len(WEATHER_METRICS):
        print(f'  [{month}] 数据不完整，跳过')
        continue

    # 合并该月5个指标（inner join，只保留5个都有数据的时间点）
    df_month = month_data['temp_C']
    for col in ['humidity_pct', 'dew_point_C', 'wind_speed_ms', 'pressure_hPa']:
        df_month = df_month.merge(month_data[col], on='timestamp', how='inner')

    # 计算湿球温度（Stull 2011公式）
    df_month['temp_wet_C'] = calc_wet_bulb_temp(
        df_month['temp_C'],
        df_month['humidity_pct']
    )

    # 打印该月统计
    print(f'  合并后：{len(df_month)} 行')
    print(f'  温度范围：{df_month["temp_C"].min():.1f} ~ '
          f'{df_month["temp_C"].max():.1f} °C')
    print(f'  湿球温度范围：{df_month["temp_wet_C"].min():.1f} ~ '
          f'{df_month["temp_wet_C"].max():.1f} °C')

    all_months.append(df_month)

# Step2：合并所有月份
if not all_months:
    print('\n错误：无任何weather数据')
    exit()

df_all = pd.concat(all_months, ignore_index=True)
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

# Step3：检查时间连续性
time_diff = df_all['timestamp'].diff()
gaps = df_all[time_diff > pd.Timedelta('1h1min')]
print(f'\n时间断点（连续缺失>1小时）：{len(gaps)} 处')
if len(gaps) > 0:
    for _, row in gaps.head(5).iterrows():
        print(f'  断点位置：{row.timestamp}')

# 在第280行之前添加
print('\n各列缺失值数量：')
print(df_all.isna().sum())
# 缺失率统计
for col in df_all.columns:
    if col != 'timestamp':
        missing = df_all[col].isna().sum()
        if missing > 0:
            print(f'  {col} 缺失：{missing} 行（{missing/len(df_all)*100:.1f}%）')

# Step4：保存输出
out_path = os.path.join(OUT_DIR, 'weather.parquet')
df_all.to_parquet(out_path, index=False, compression='zstd')

print(f'\n输出文件：{out_path}')
print(f'总行数：{len(df_all)} 条（每{FREQ}一条）')
print(f'时间范围：{df_all.timestamp.min()} → {df_all.timestamp.max()}')
print(f'\n各列统计：')
print(df_all[[
    'temp_C', 'humidity_pct', 'dew_point_C',
    'wind_speed_ms', 'pressure_hPa', 'temp_wet_C'
]].describe().round(2).to_string())
print(f'\n前5行：')
print(df_all.head().to_string())