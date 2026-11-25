"""
每小时定时任务入口: 采集 → 预测 → 优化 → 记录
串联 data_collector / predict / ga_optimizer 三个模块
"""

import sys
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

from runtime_paths import DEPLOY_DIR, PROJECT_ROOT
PREDICT_DIR = PROJECT_ROOT / '02predictModel' / 'src'
OPT_DIR     = PROJECT_ROOT / '04EquipmentOptimization'

sys.path.insert(0, str(PREDICT_DIR))
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(DEPLOY_DIR))
import data_collector
from data_collector import BUFFER_PATH
import predict as predict_module
import ga_optimizer
from electricity_pricing import get_price

LOG_PATH = DEPLOY_DIR / 'hourly_log.csv'

LOG_FIELDS = [
    'timestamp', 'Q10_kW', 'Q50_kW', 'Q90_kW',
    'N_chiller', 'f_pump', 'N_cdz', 'N_FC', 's_fan',
    'cost_yuan', 'cost_baseline', 'saving_yuan', 'saving_pct',
    'price_yuan', 'temp_C',
    # 实时天气(GUI 环境面板展示用, 来自 data_collector 采集的 new_row)
    'humidity_pct', 'wind_speed_ms', 'pressure_hPa',
    'dew_point_C', 'temp_wet_C',
]

# 保守基线: 冷水机全开 + 空调全开 + 满频满速
BASELINE_IND = [4, 100.0, 6, 0, 100.0]

# 设备模型只加载一次(模块级缓存)
_models = None

def get_models():
    global _models
    if _models is None:
        _models = ga_optimizer.load_models()
    return _models


def compute_baseline(context, models):
    """全开策略电费(不含切换惩罚)"""
    ctx = dict(context)
    ctx['prev_N_chiller'] = None
    return ga_optimizer.fitness(BASELINE_IND, ctx, models)[0]


def read_prev_n_chiller():
    """从日志最后一行读取上一小时冷水机台数, 无记录返回 None"""
    if not LOG_PATH.exists():
        return None
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df) == 0:
            return None
        return int(df['N_chiller'].iloc[-1])
    except Exception:
        return None


def write_log(row):
    """写日志: 同整点覆盖最后一行(只保留最新), 新整点则追加"""
    new_hour = pd.Timestamp(row['timestamp']).floor('h')

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        if len(df) > 0:
            last_hour = pd.Timestamp(df['timestamp'].iloc[-1]).floor('h')
            if new_hour == last_hour:
                df = df.iloc[:-1]   # 同整点: 丢弃旧的最后一行
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(LOG_PATH, index=False, encoding='utf-8')
    else:
        pd.DataFrame([row]).to_csv(LOG_PATH, index=False, encoding='utf-8')


def run_once():
    now_utc = datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(timezone(timedelta(hours=8)))
    print(f'\n{"="*50}')
    print(f'每小时任务启动: {now_bj.strftime("%Y-%m-%d %H:%M")} (北京时间)')
    print(f'{"="*50}')

    # ── 第1步: 采集 ──
    print('\n[1/4] 数据采集 ...')
    new_row = data_collector.collect_one_hour()
    T_amb = new_row['temp_C']

    # ── 第2步: 预测 ──
    print('\n[2/4] 负载预测 ...')
    pred = predict_module.predict(BUFFER_PATH, mode='single')
    Q10, Q50, Q90 = pred['Q10'], pred['Q50'], pred['Q90']

    # ── 第3步: 优化 ──
    print('\n[3/4] GA 优化 ...')
    models = get_models()
    context = {
        'Tot_ict_Q50':    Q50,
        'Tot_ict_Q90':    Q90,
        'T_amb':          T_amb,
        'price':          get_price(now_bj),
        'prev_N_chiller': read_prev_n_chiller(),
    }
    best_ind, best_cost, _ = ga_optimizer.optimize_hour(context, models)

    N_chiller = int(round(best_ind[0]))
    f_pump    = best_ind[1]
    N_cdz     = int(round(best_ind[2]))
    N_FC      = int(round(best_ind[3]))
    s_fan     = best_ind[4]

    print(f'  最优解: N_chiller={N_chiller}, f_pump={f_pump:.1f}%, '
          f'N_cdz={N_cdz}, N_FC={N_FC}, s_fan={s_fan:.1f}%')
    print(f'  预计电费: {best_cost:.2f} 元/h')

    # 保守基线 + 省电
    cost_baseline = compute_baseline(context, models)
    saving_yuan = cost_baseline - best_cost
    saving_pct = saving_yuan / cost_baseline * 100 if cost_baseline > 0 else 0
    print(f'  保守基线: {cost_baseline:.2f} 元/h')
    print(f'  预计省电: {saving_yuan:.2f} 元/h ({saving_pct:.1f}%)')

    # ── 第4步: 记录 ──
    print('\n[4/4] 写入日志 ...')
    write_log({
        'timestamp':     now_utc.isoformat(),
        'Q10_kW':        round(Q10, 1),
        'Q50_kW':        round(Q50, 1),
        'Q90_kW':        round(Q90, 1),
        'N_chiller':     N_chiller,
        'f_pump':        round(f_pump, 1),
        'N_cdz':         N_cdz,
        'N_FC':          N_FC,
        's_fan':         round(s_fan, 1),
        'cost_yuan':     round(best_cost, 2),
        'cost_baseline': round(cost_baseline, 2),
        'saving_yuan':   round(saving_yuan, 2),
        'saving_pct':    round(saving_pct, 1),
        'price_yuan':    round(context['price'], 4),
        'temp_C':        round(T_amb, 1),
        # 实时天气五项(new_row 在 data_collector 里已做 NaN 兜底, 必有值)
        'humidity_pct':  round(float(new_row['humidity_pct']), 0),
        'wind_speed_ms': round(float(new_row['wind_speed_ms']), 2),
        'pressure_hPa':  round(float(new_row['pressure_hPa']), 1),
        'dew_point_C':   round(float(new_row['dew_point_C']), 1),
        'temp_wet_C':    round(float(new_row['temp_wet_C']), 1),
    })
    print(f'  已记录: {LOG_PATH}')
    print(f'\n任务完成 ✓')


if __name__ == '__main__':
    try:
        run_once()
    except Exception as e:
        import traceback
        print(f'\n任务失败: {e}')
        traceback.print_exc()