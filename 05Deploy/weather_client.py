"""
和风天气API客户端 + 气象派生量计算(露点/湿球)
供 predict.py 和 run_hourly.py 共同使用
"""

import numpy as np
import pandas as pd
import requests
import deploy_config

API_HOST = 'https://m64up5kf87.re.qweatherapi.com'
DEFAULT_API_KEY = deploy_config.QWEATHER_API_KEY
DEFAULT_LOCATION = deploy_config.QWEATHER_LOCATION


# ── 底层公式(标量/ndarray 通用) ─────────────────────────

def _magnus_dew_point(T, RH):
    """Magnus公式: 从温度+相对湿度算露点"""
    gamma = np.log(RH / 100.0) + 17.67 * T / (243.5 + T)
    return 243.5 * gamma / (17.67 - gamma)


def _stull_wet_bulb(T, RH):
    """Stull 2011公式: 从温度+相对湿度算湿球温度"""
    return (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )


# ── DataFrame 批量接口(predict.py 用) ──────────────────

def calc_derived_weather(df):
    """补算 dew_point_C 和 temp_wet_C: 有值保留, 空值公式补"""
    T = df['temp_C'].values
    RH = df['humidity_pct'].values

    dew_calc = _magnus_dew_point(T, RH)
    if 'dew_point_C' not in df.columns:
        df['dew_point_C'] = dew_calc
    else:
        df['dew_point_C'] = df['dew_point_C'].fillna(
            pd.Series(dew_calc, index=df.index))

    wet_calc = _stull_wet_bulb(T, RH)
    if 'temp_wet_C' not in df.columns:
        df['temp_wet_C'] = wet_calc
    else:
        df['temp_wet_C'] = df['temp_wet_C'].fillna(
            pd.Series(wet_calc, index=df.index))

    return df


# ── 和风天气API接口(run_hourly.py 用) ─────────────────

def fetch_weather(api_key, location):
    """
    调用和风实时天气 /v7/weather/now
    location: LocationID 或 '经度,纬度' (如 '116.41,39.92')
    返回: dict{temp_C, humidity_pct, wind_speed_ms, pressure_hPa,
               dew_point_C, temp_wet_C, obs_time}
    """
    resp = requests.get(
        f'{API_HOST}/v7/weather/now',
        params={'location': location, 'key': api_key, 'unit': 'm', 'lang': 'zh'},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get('code') != '200':
        raise RuntimeError(f'和风API错误: code={data.get("code")}')

    now = data['now']

    T = float(now['temp'])
    RH = float(now['humidity'])

    dew_raw = now.get('dew')
    dew_val = float(dew_raw) if dew_raw not in (None, '', 'null') else None

    return {
        'temp_C':        T,
        'humidity_pct':  RH,
        'wind_speed_ms': round(float(now['windSpeed']) / 3.6, 2),
        'pressure_hPa':  float(now['pressure']),
        'dew_point_C':   dew_val if dew_val is not None
                         else round(float(_magnus_dew_point(T, RH)), 1),
        'temp_wet_C':    round(float(_stull_wet_bulb(T, RH)), 1),
        'obs_time':      now.get('obsTime'),
    }


# ── 测试 ───────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_KEY
    loc = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LOCATION
    result = fetch_weather(key, loc)
    print('实时天气:')
    for k, v in result.items():
        print(f'  {k}: {v}')