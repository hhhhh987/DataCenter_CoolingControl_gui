"""
data_collector.py
数据采集器: 维护168小时滑动窗口 buffer.parquet
冷启动从历史数据初始化, 之后每小时采集IT负载+天气追加1行
用法:
    python data_collector.py init    # 冷启动(仅首次)
    python data_collector.py         # 采集当前小时
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

from weather_client import fetch_weather, calc_derived_weather
from deploy_config import QWEATHER_API_KEY, QWEATHER_LOCATION

# ── 路径与常量 ────────────────────────────────────────

from runtime_paths import DEPLOY_DIR, PROJECT_ROOT
HISTORY_PATH = PROJECT_ROOT / '02predictModel' / 'data' / 'raw' / 'merge_dataset.parquet'
BUFFER_PATH  = DEPLOY_DIR / 'buffer.parquet'
WINDOW       = 168

BUFFER_COLS = [
    'timestamp', 'Tot_ict_kW',
    'temp_C', 'humidity_pct', 'wind_speed_ms',
    'pressure_hPa', 'dew_point_C', 'temp_wet_C',
]


# ── 历史数据加载(全局缓存,避免重复读取) ───────────────

_history_cache = None

def _load_history():
    global _history_cache
    if _history_cache is None:
        df = pd.read_parquet(HISTORY_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        _history_cache = df
        print(f'历史数据加载: {len(df)} 行, '
              f'{df["timestamp"].iloc[0]} → {df["timestamp"].iloc[-1]}')
    return _history_cache


# ── IT负载采集(适配器模式) ────────────────────────────

def fetch_real_data():
    """真实IT负载采集接口 (待接入实际监控系统, 返回dict或None)"""
    return None


def fetch_historical_replay(ts_utc):
    """从历史数据匹配同月同日同时的记录, 返回一行数据dict"""
    df = _load_history()

    month, day, hour = ts_utc.month, ts_utc.day, ts_utc.hour

    mask = ((df['timestamp'].dt.month == month) &
            (df['timestamp'].dt.day == day) &
            (df['timestamp'].dt.hour == hour))
    candidates = df[mask]

    if candidates.empty:
        mask = ((df['timestamp'].dt.month == month) &
                (df['timestamp'].dt.hour == hour))
        candidates = df[mask]
        print(f'  精确匹配无结果, 放宽到同月同时: {len(candidates)} 条')

    if candidates.empty:
        raise RuntimeError(f'历史数据中无法匹配: {ts_utc}')

    # 新代码: 只返回IT负载
    row = candidates.iloc[-1]
    return {'Tot_ict_kW': float(row['Tot_ict_kW'])}

def fetch_it_load(ts_utc):
    """适配器: 优先真实数据, 无则历史回放"""
    real = fetch_real_data()
    if real is not None:
        return real
    return fetch_historical_replay(ts_utc)


# ── 冷启动 ────────────────────────────────────────────

def init_buffer():
    """从历史数据找锚点, 向前连续取168行初始化buffer"""
    df = _load_history()

    now_utc = datetime.now(timezone.utc)
    now_hour = now_utc.replace(minute=0, second=0, microsecond=0)
    month, day, hour = now_utc.month, now_utc.day, now_utc.hour

    # 逐步放宽匹配: 同月同日同时 → 同月同日 → 同月
    for label, m in [
        ('同月同日同时', (df['timestamp'].dt.month == month) &
                        (df['timestamp'].dt.day == day) &
                        (df['timestamp'].dt.hour == hour)),
        ('同月同日',     (df['timestamp'].dt.month == month) &
                        (df['timestamp'].dt.day == day)),
        ('同月',         df['timestamp'].dt.month == month),
    ]:
        candidates = df[m]
        if not candidates.empty:
            print(f'  锚点匹配策略: {label}, 找到 {len(candidates)} 条')
            break

    if candidates.empty:
        raise RuntimeError(f'历史数据中无法找到锚点: {now_utc}')

    anchor_idx = candidates.index[-1]
    start_idx = max(0, anchor_idx - WINDOW + 1)
    chunk = df.iloc[start_idx:anchor_idx + 1].copy()
    print(f'  连续切片: idx [{start_idx}:{anchor_idx + 1}], 共 {len(chunk)} 行')

    # 选取buffer列, 缺失列补NaN
    for col in BUFFER_COLS:
        if col not in chunk.columns:
            chunk[col] = np.nan
    chunk = chunk[BUFFER_COLS].copy()

    # 补算派生天气(露点/湿球)
    if chunk['dew_point_C'].isna().any() or chunk['temp_wet_C'].isna().any():
        chunk = calc_derived_weather(chunk)

    # NaN兜底
    chunk = chunk.ffill().bfill()
    # 覆写时间戳为当前时间序列, 保证时间特征与真实运行一致
    chunk['timestamp'] = pd.date_range(
        end=now_hour, periods=len(chunk), freq='h', tz='UTC')

    chunk.to_parquet(BUFFER_PATH, index=False)
    print(f'冷启动完成: {len(chunk)} 行')
    print(f'  时间范围: {chunk["timestamp"].iloc[0]} → {chunk["timestamp"].iloc[-1]}')
    print(f'  保存路径: {BUFFER_PATH}')
    return chunk


# ── 每小时更新 ────────────────────────────────────────

def update_buffer(new_row_dict):
    """更新buffer: 同整点覆盖最后一行, 新整点则追加并丢弃最旧, 保持168行"""
    buffer = pd.read_parquet(BUFFER_PATH)
    new_row = pd.DataFrame([new_row_dict])[BUFFER_COLS]

    # 同整点覆盖: 新行时间戳与buffer最后一行相同则替换, 不堆积
    new_ts = pd.Timestamp(new_row_dict['timestamp'])
    last_ts = pd.Timestamp(buffer['timestamp'].iloc[-1])
    if new_ts == last_ts:
        buffer = pd.concat([buffer.iloc[:-1], new_row], ignore_index=True)
    else:
        buffer = pd.concat([buffer, new_row], ignore_index=True)

    buffer.to_parquet(BUFFER_PATH, index=False)
    return buffer


def collect_one_hour():
    """采集当前小时: IT负载 + 天气 → 合并1行 → 更新buffer"""
    if not BUFFER_PATH.exists():
        print('buffer不存在, 执行冷启动...')
        init_buffer()

    now_utc = datetime.now(timezone.utc)
    bj_tz = timezone(timedelta(hours=8))
    now_bj = now_utc.astimezone(bj_tz)

    # 1. IT负载
    it_data = fetch_it_load(now_utc)

    # 2. 实时天气
    try:
        weather = fetch_weather(QWEATHER_API_KEY, QWEATHER_LOCATION)
    except Exception as e:
        print(f'  天气API失败, 将由NaN兜底使用前一小时值: {e}')
        weather = {}

    now_hour = now_utc.replace(minute=0, second=0, microsecond=0)
    # 3. 组装新行(IT + 天气独立拼接)
    new_row = {'timestamp': pd.Timestamp(now_hour)}
    new_row.update(it_data)
    for key in ['temp_C', 'humidity_pct', 'wind_speed_ms',
                'pressure_hPa', 'dew_point_C', 'temp_wet_C']:
        new_row[key] = weather.get(key, np.nan)

    # 4. NaN最终兜底: 用buffer最后一行的值填充
    buffer = pd.read_parquet(BUFFER_PATH)
    for col in BUFFER_COLS:
        if col != 'timestamp' and pd.isna(new_row.get(col)):
            new_row[col] = float(buffer[col].iloc[-1])
            print(f'  {col} 值缺失, 使用前一小时值兜底')

    # 5. 更新buffer
    buffer = update_buffer(new_row)

    print(f'采集完成: {now_bj.strftime("%Y-%m-%d %H:%M")} (北京时间)')
    print(f'  IT负载: {new_row["Tot_ict_kW"]:.1f} kW')
    print(f'  温度: {new_row["temp_C"]:.1f}°C  湿度: {new_row["humidity_pct"]:.0f}%')
    print(f'  buffer行数: {len(buffer)}')
    return new_row


# ── 入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'init':
        init_buffer()
    else:
        collect_one_hour()