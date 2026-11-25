"""
state_reader.py
前端状态读取器(纯展示, 不参与预测/优化)
- get_reference_temps: 从历史数据匹配供回水温度(参考状态)
- get_device_detail:   反算设备物理量(COP/PLR/功耗等)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runtime_paths import DEPLOY_DIR, PROJECT_ROOT
OPT_DIR = PROJECT_ROOT / '04EquipmentOptimization'

sys.path.insert(0, str(OPT_DIR))

# 含供回水温度的历史数据(按需修改路径)
COOLING_PATH = PROJECT_ROOT / '03CoolingData' / 'cooling_all_clean.parquet'
TEMP_COLS = ['T_mandata_C', 'T_ritorno_C', 'delta_T']


# ── 参考温度(历史同期匹配) ───────────────────────────

_cooling_cache = None

def _load_cooling():
    global _cooling_cache
    if _cooling_cache is None:
        df = pd.read_parquet(COOLING_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        _cooling_cache = df
    return _cooling_cache


def get_reference_temps(timestamp):
    """
    按当前时刻匹配历史同月/日/时的供回水温度
    返回: {T_mandata, T_ritorno, delta_T} 或全 None(无匹配/无文件)
    """
    empty = {'T_mandata': None, 'T_ritorno': None, 'delta_T': None}

    if not COOLING_PATH.exists():
        return empty

    try:
        df = _load_cooling()
    except Exception:
        return empty

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')

    # 逐步放宽: 同月同日同时 → 同月同时
    mask = ((df['timestamp'].dt.month == ts.month) &
            (df['timestamp'].dt.day == ts.day) &
            (df['timestamp'].dt.hour == ts.hour))
    candidates = df[mask]
    if candidates.empty:
        mask = ((df['timestamp'].dt.month == ts.month) &
                (df['timestamp'].dt.hour == ts.hour))
        candidates = df[mask]
    if candidates.empty:
        return empty

    row = candidates.iloc[-1]
    return {
        'T_mandata':  round(float(row['T_mandata_C']), 2),
        'T_ritorno':  round(float(row['T_ritorno_C']), 2),
        'delta_T':    round(float(row['delta_T']), 2),
    }


# ── 设备详情(反算物理量) ─────────────────────────────

def get_device_detail(log_row, models):
    """
    从日志行 + 设备模型反算 COP/PLR/各设备功耗
    log_row: hourly_log.csv 的一行(dict 或 Series)
    models:  ga_optimizer.load_models() 的返回值
    返回: evaluate_detail 的完整物理量字典
    """
    import ga_optimizer

    individual = [
        float(log_row['N_chiller']),
        float(log_row['f_pump']),
        float(log_row['N_cdz']),
        float(log_row['N_FC']),
        float(log_row['s_fan']),
    ]
    context = {
        'Tot_ict_Q50':    float(log_row['Q50_kW']),
        'Tot_ict_Q90':    float(log_row['Q90_kW']),
        'T_amb':          float(log_row['temp_C']),
        'price':          float(log_row['price_yuan']),
        'prev_N_chiller': None,
    }
    return ga_optimizer.evaluate_detail(individual, context, models)


# ── 测试 ───────────────────────────────────────────────

if __name__ == '__main__':
    from datetime import datetime, timezone
    temps = get_reference_temps(datetime.now(timezone.utc))
    print('参考温度:')
    for k, v in temps.items():
        print(f'  {k}: {v}')