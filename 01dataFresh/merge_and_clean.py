
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR       = Path(r'D:\PythonProject1\01dataFresh')
TOT_ICT_PATH  = BASE_DIR / 'tot_ict.parquet'
WEATHER_PATH   = BASE_DIR / 'weather.parquet'
OUTPUT_PATH    = BASE_DIR / 'merge_dataset.parquet'

#  参数配置
ICT_ANOMALY_THRESHOLD  = 100    # kW
ICT_MAX_INTERP_GAP     = 1     # 小时
WEATHER_MAX_INTERP_GAP = 3     # 小时
MIN_SEGMENT_HOURS      = 168   # 小时

def selective_interpolate(series: pd.Series, max_gap: int) -> pd.Series:
    if not series.isna().any():
        return series.copy()

    is_nan = series.isna()
    filled = series.interpolate(method='linear', limit_area='inside')

    gap_groups = (~is_nan).cumsum()
    for _, grp in is_nan[is_nan].groupby(gap_groups[is_nan]):
        if len(grp) > max_gap:
            filled.loc[grp.index] = np.nan

    return filled


def split_segments(df: pd.DataFrame, min_hours: int) -> list:
    has_nan = df.drop(columns=['timestamp']).isna().any(axis=1)
    group_id = has_nan.cumsum()

    segments = []
    for _, seg in df[~has_nan].groupby(group_id[~has_nan]):
        if len(seg) >= min_hours:
            segments.append(seg.reset_index(drop=True))
    return segments


def main():
    # ── Step1: 读取 ──────────────────────────────────────
    print('Step1: 读取数据：：：：：')

    df_ict = pd.read_parquet(TOT_ICT_PATH)
    df_weather = pd.read_parquet(WEATHER_PATH)

    df_ict['timestamp'] = pd.to_datetime(
        df_ict['timestamp'], unit='ms',utc=True
    ).dt.floor('1h')
    df_weather['timestamp'] = pd.to_datetime(
        df_weather['timestamp'], unit='ms',utc=True
    ).dt.floor('1h')

    df_ict = (
        df_ict[['timestamp', 'Tot_ict_kW']]
        .groupby('timestamp', as_index=False)['Tot_ict_kW']
        .mean()
    )

    weather_cols = [
        'temp_C', 'humidity_pct', 'dew_point_C',
        'wind_speed_ms', 'pressure_hPa', 'temp_wet_C'
    ]
    df_weather = (
        df_weather[['timestamp'] + weather_cols]
        .groupby('timestamp', as_index=False)
        .mean()
    )

    print(f'  Tot_ict : {len(df_ict)} 行, '     , f'{df_ict["timestamp"].min()} → {df_ict["timestamp"].max()}')
    print(f'  Weather : {len(df_weather)} 行, ' , f'{df_weather["timestamp"].min()} → {df_weather["timestamp"].max()}')

    # ── Step2: 构建完整小时索引并合并 ────────────────────
    print('Step2: 合并数据：：：：：')

    start = max(df_ict['timestamp'].min(), df_weather['timestamp'].min())
    end   = min(df_ict['timestamp'].max(), df_weather['timestamp'].max())

    full_index = pd.DataFrame( {'timestamp': pd.date_range(start, end, freq='1h', tz='UTC')} )
    df = full_index.merge(df_ict, on='timestamp', how='left')
    df = df.merge(df_weather, on='timestamp', how='left')

    print(f'  重叠区间: {start} → {end}')
    print(f'  总行数  : {len(df)}')

    # ── Step3: Tot_ict 异常值过滤 ────────────────────────
    #提取时已过滤<100kw设置为nan

    # ── Step4: 选择性线性插值 ────────────────────────────
    print('Step4: 选择性插值：：：：：')

    #ict插值
    n_ict_before = df['Tot_ict_kW'].isna().sum()
    df['Tot_ict_kW'] = selective_interpolate(
        df['Tot_ict_kW'], ICT_MAX_INTERP_GAP
    )
    n_ict_after = df['Tot_ict_kW'].isna().sum()
    print(f'  Tot_ict : {n_ict_before} NaN → {n_ict_after} NaN '
          f'(插值修复 {n_ict_before - n_ict_after} 行)')

    #weather插值
    n_w_before = df[weather_cols].isna().any(axis=1).sum()
    for col in weather_cols:
        df[col] = selective_interpolate(df[col], WEATHER_MAX_INTERP_GAP)
    n_w_after = df[weather_cols].isna().any(axis=1).sum()
    print(f'  气象    : {n_w_before} 行缺失 → {n_w_after} 行缺失 '
          f'(插值修复 {n_w_before - n_w_after} 行)')

    remaining_nan = df.drop(columns=['timestamp']).isna().any(axis=1).sum()
    print(f'  剩余无效行: {remaining_nan} (将作为切断点)')

    # ── Step5: 切割有效片段 ──────────────────────────────
    print('Step5: 切割有效片段：：：：：')

    segments = split_segments(df, MIN_SEGMENT_HOURS)

    result_parts = []
    total_hours = 0
    for i, seg in enumerate(segments):
        seg = seg.copy()
        seg['segment_id'] = i
        result_parts.append(seg)
        hours = len(seg)
        total_hours += hours
        t0 = seg['timestamp'].iloc[0]
        t1 = seg['timestamp'].iloc[-1]
        print(f'  片段{i + 1}: {t0} → {t1}  ({hours} 小时)')

    discarded = len(df) - remaining_nan - total_hours
    print(f'\n  有效片段: {len(segments)} 个')
    print(f'  有效小时: {total_hours}')
    print(f'  丢弃(短片段): {discarded} 小时')
    est_samples = total_hours - len(segments) * MIN_SEGMENT_HOURS
    print(f'  预计可构造样本: ~{est_samples} 个')

    # ── Step6: 保存 ──────────────────────────────────────
    print('\n')
    print('Step6: 保存：：：：：')

    result = pd.concat(result_parts, ignore_index=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print(f'  文件: {OUTPUT_PATH}')
    print(f'  行数: {len(result)}')
    print(f'  列  : {result.columns.tolist()}')
    print('\n完成 ✓')


if __name__ == '__main__':
    main()