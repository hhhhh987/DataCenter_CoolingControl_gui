"""
electricity_pricing.py
上海大工业用电(10kV)分时电价 — 2026年1月
输入: 时间戳 → 输出: 电价(元/kWh)
"""

import pandas as pd
from chinese_calendar import is_workday, is_holiday

# 大工业 10kV 两部制,1月 (元/kWh)
PRICES = {
    '尖峰': 1.5151,
    '高峰': 1.2287,
    '平时': 0.7196,
    '低谷': 0.3377,
    '深谷': 0.2105,
}

# 1月深谷仅限法定节假日(备注4: 2-6月和9-11月的周末才有深谷,1月周末无深谷)
DEEP_VALLEY_WEEKEND_MONTHS = {2, 3, 4, 5, 6, 9, 10, 11}


def get_day_type(dt) -> str:
    """
    判断日类型(chinese_calendar 自动处理调休)
    调休上班日 → 'workday'
    调休休息日 → 'weekend'
    法定节假日 → 'holiday'
    """
    d = dt.date() if hasattr(dt, 'date') else dt
    if is_holiday(d):
        return 'holiday'
    if is_workday(d):
        return 'workday'
    return 'weekend'


def get_period(hour: int, day_type: str, month: int = 1) -> str:
    """
    小时 + 日类型 → 时段名称

    1月工作日: 19-21 尖峰, 8-11/18-19 高峰, 6-8/11-18/21-22 平时, 22-6 低谷
    1月周末:   无尖峰, 19-21 变高峰, 22-6 低谷
    节假日:    无尖峰, 0-6/22-24 深谷(替代低谷)
    深谷周末:  2-6月和9-11月的周末, 0-6/22-24 也是深谷
    """
    is_night = (hour >= 22) or (hour < 6)

    # 深谷判定: 节假日 或 特定月份的周末
    if is_night:
        if day_type == 'holiday':
            return '深谷'
        if day_type == 'weekend' and month in DEEP_VALLEY_WEEKEND_MONTHS:
            return '深谷'
        if day_type == 'weekend' or day_type == 'workday':
            return '低谷'

    # 白天时段(6-22时)
    if day_type == 'workday' and month in (1, 12) and 19 <= hour < 21:
        return '尖峰'

    if 8 <= hour < 11 or 18 <= hour < 21:
        return '高峰'

    return '平时'


def get_price(timestamp) -> float:
    """单个时间戳 → 电价(元/kWh)"""
    if hasattr(timestamp, 'hour'):
        hour = timestamp.hour
        month = timestamp.month
    else:
        timestamp = pd.Timestamp(timestamp)
        hour = timestamp.hour
        month = timestamp.month

    day_type = get_day_type(timestamp)
    period = get_period(hour, day_type, month)
    return PRICES[period]


def add_prices(df, timestamp_col='timestamp') -> pd.DataFrame:
    """批量添加电价列(GA 回测用)"""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df['price_yuan'] = [get_price(t) for t in ts]
    df['period'] = [get_period(t.hour, get_day_type(t), t.month) for t in ts]
    return df


# ── 验证 ──────────────────────────────────────────────────

if __name__ == '__main__':
    print('上海大工业 10kV 分时电价(1月)')
    print(f'电价等级: {PRICES}\n')

    # 打印一周示例(2026-01-05 周一 ~ 01-11 周日)
    dates = pd.date_range('2022-01-03', periods=7*24, freq='h')
    for d in dates:
        day_type = get_day_type(d)
        period = get_period(d.hour, day_type, d.month)
        price = PRICES[period]
        if d.hour == 0:
            print(f'\n{d.strftime("%Y-%m-%d %A")}  [{day_type}]')
        if d.hour in (0, 6, 8, 11, 18, 19, 21, 22):
            print(f'  {d.hour:2d}:00  {period}  {price:.4f} 元/kWh')

    # 元旦验证(应有深谷)
    print(f'\n元旦验证:')
    new_year = pd.Timestamp('2026-01-01 02:00')
    print(f'  {new_year}: {get_day_type(new_year)} → {get_period(2, get_day_type(new_year), 1)} '
          f'→ {get_price(new_year):.4f} 元')

    from datetime import date
    from chinese_calendar import is_workday, is_holiday

    for d in [date(2026, 1, 5), date(2026, 1, 10), date(2026, 1, 11), date(2026, 1, 1)]:
        print(f'{d} {d.strftime("%A"):10s} is_workday={is_workday(d)}, is_holiday={is_holiday(d)}')