"""
输出的时间特征（共8个）：
  hour_sin      小时正弦编码
  hour_cos      小时余弦编码
  dow_sin       星期正弦编码
  dow_cos       星期余弦编码
  month_sin     月份正弦编码
  month_cos     月份余弦编码
  is_workday    是否工作日(0/1)
                = 周一至周五 且 非节假日
  is_holiday    是否法定节假日(0/1)
                = 意大利法定节假日
"""

import numpy as np
import pandas as pd
import holidays


def add_time_features(df ,timestamp_col= 'timestamp') -> pd.DataFrame:
    """
    从timestamp列提取时间特征。

    返回：
      新增时间特征列的DataFrame

    注意：
      timestamp必须是pandas datetime类型
      建议统一使用UTC时间
      hour_sin/cos基于UTC小时计算
      意大利数据中心位于UTC+1/UTC+2
      如需转换为本地时间，在调用前先转换timestamp
    """

    ts = df[timestamp_col]

    # ── 1. 小时周期编码（24小时）─────────────────────────
    # 捕捉一天内的负载变化规律
    # 数据中心白天负载高，凌晨低
    # sin/cos编码保证23点和0点的连续性
    hour = ts.dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # ── 2. 星期周期编码（7天）───────────────────────────
    # 捕捉工作日和周末的负载差异
    # dayofweek: 0=周一, 1=周二, ..., 6=周日
    # sin/cos编码保证周日(6)和周一(0)的连续性
    dow = ts.dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # ── 3. 月份周期编码（12个月）────────────────────────
    # 捕捉季节性规律
    # 夏季高温 → 冷负荷大；冬季低温 → 冷负荷小
    # sin/cos编码保证12月和1月的连续性
    month = ts.dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # ── 4. 是否节假日（0/1）─────────────────────────────
    # 意大利法定节假日：元旦、复活节、劳动节、圣诞节等
    # 节假日负载接近周末水平，与dow_sin/cos互补
    # dow_sin/cos无法区分"圣诞节周三"和"普通周三"
    # is_holiday专门捕捉节假日对正常规律的打断
    #
    # 迁移到其他地区时只需修改此处：
    #   新加坡：holidays.Singapore(years=years)
    #   中国：  holidays.China(years=years)
    #           中国调休建议额外使用 chinesecalendar 库

    # 使用向量化操作 isin() 替代 apply()，大幅提升计算速度
    years = ts.dt.year.unique().tolist()
    it_holidays = holidays.Italy(years=years)

    # 获取所有节假日的 date 对象列表
    holiday_dates = list(it_holidays.keys())

    # 向量化判断
    df['is_holiday'] = ts.dt.date.isin(holiday_dates).astype(int)

    # ── 5. 是否工作日（0/1）─────────────────────────────
    # 定义：周一至周五（dow < 5）且 非节假日（is_holiday=0）
    # 与dow_sin/cos的区别：
    #   dow_sin/cos 表达一周内的连续位置（渐变）
    #   is_workday  表达工作/休息的硬边界（跳变）
    # 与is_holiday的区别：
    #   is_holiday 只标记法定假日（少数特殊日）
    #   is_workday 标记所有非工作状态（周末+假日）
    # 三者组合提供完整的工作制度信息，互补不冗余
    df['is_workday'] = (
        (ts.dt.dayofweek < 5) &
        (df['is_holiday'] == 0)
    ).astype(int)

    return df