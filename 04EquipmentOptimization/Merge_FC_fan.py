"""
merge_fc_fan.py
合并 vertiv FC 状态 / 风扇转速 / 压缩机负载到主数据集,供空调建模使用。
输入: vertiv_*.parquet (长表) + cooling_all_clean.parquet
输出: 04EquipmentOptimization/cooling_with_modes.parquet
"""

import os
import numpy as np
import pandas as pd

ROOT     = r'D:\PythonProject1'
COOL_DIR = os.path.join(ROOT, '03CoolingData')
OUT_DIR  = os.path.join(ROOT, '04EquipmentOptimization')
os.makedirs(OUT_DIR, exist_ok=True)

FC_PATH   = os.path.join(COOL_DIR, 'vertiv_free_cooling.parquet')
FAN_PATH  = os.path.join(COOL_DIR, 'vertiv_fan_speed.parquet')
COMP_PATH = os.path.join(COOL_DIR, 'vertiv_compressor.parquet')
CLEAN_PATH = os.path.join(COOL_DIR, 'cooling_all_clean.parquet')
OUT_PATH  = os.path.join(OUT_DIR, 'cooling_with_modes.parquet')


# ── PART 1: 读取 vertiv 长表并按小时聚合 ──────────────────

def load_vertiv() -> pd.DataFrame:
    print('[PART 1] 读取 vertiv 数据 ...')

    agg_frames = []

    # --- Free Cooling: 每小时有几台 FC 开启 ---
    fc = pd.read_parquet(FC_PATH)
    fc['timestamp'] = pd.to_datetime(fc['timestamp'], utc=True)
    print(f'  free_cooling: {len(fc):,} 行, 列={list(fc.columns)}')

    # 识别值列(可能叫 value / Free_Cooling_Status 等)
    val_col = [c for c in fc.columns if c not in ('timestamp', 'device', 'panel')][0]
    fc_hourly = (fc.groupby('timestamp')[val_col]
                 .agg(N_FC_on='sum', FC_mean='mean')
                 .reset_index())
    fc_hourly['N_FC_on'] = fc_hourly['N_FC_on'].round().astype(int)
    agg_frames.append(fc_hourly[['timestamp', 'N_FC_on']])
    print(f'  → N_FC_on 聚合完成: {len(fc_hourly):,} 小时')

    # --- Fan Speed: 运行机组的平均转速 ---
    fan = pd.read_parquet(FAN_PATH)
    fan['timestamp'] = pd.to_datetime(fan['timestamp'], utc=True)
    print(f'  fan_speed: {len(fan):,} 行')

    val_col = [c for c in fan.columns if c not in ('timestamp', 'device', 'panel')][0]
    fan_hourly = (fan.groupby('timestamp')[val_col]
                  .agg(Fan_Speed_avg='mean', Fan_Speed_max='max')
                  .reset_index())
    agg_frames.append(fan_hourly[['timestamp', 'Fan_Speed_avg', 'Fan_Speed_max']])
    print(f'  → Fan_Speed 聚合完成: {len(fan_hourly):,} 小时')

    # --- Compressor: 平均压缩机负载(交叉验证用) ---
    comp = pd.read_parquet(COMP_PATH)
    comp['timestamp'] = pd.to_datetime(comp['timestamp'], utc=True)
    print(f'  compressor: {len(comp):,} 行')

    val_col = [c for c in comp.columns if c not in ('timestamp', 'device', 'panel')][0]
    comp_hourly = (comp.groupby('timestamp')[val_col]
                   .agg(Compressor_avg='mean')
                   .reset_index())
    agg_frames.append(comp_hourly[['timestamp', 'Compressor_avg']])
    print(f'  → Compressor 聚合完成: {len(comp_hourly):,} 小时')

    # --- 合并三个聚合表 ---
    merged = agg_frames[0]
    for af in agg_frames[1:]:
        merged = merged.merge(af, on='timestamp', how='outer')

    print(f'  vertiv 聚合表: {len(merged):,} 小时')
    return merged


# ── PART 2: 和 cooling_all_clean 内连接 ───────────────────

def merge_with_clean(vertiv_agg: pd.DataFrame) -> pd.DataFrame:
    print('\n[PART 2] 合并到 cooling_all_clean ...')

    df = pd.read_parquet(CLEAN_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    n_before = len(df)

    df = df.merge(vertiv_agg, on='timestamp', how='inner')

    # 衍生列
    df['N_DX'] = (df['N_cdz_running'] - df['N_FC_on']).clip(lower=0)
    df['FC_active'] = (df['N_FC_on'] > 0).astype(int)

    print(f'  {n_before:,} → 内连接后 {len(df):,}')
    print(f'  新增列: N_FC_on, Fan_Speed_avg, Fan_Speed_max, Compressor_avg, N_DX, FC_active')
    return df


# ── PART 3: 交叉验证 ──────────────────────────────────────

def validate(df: pd.DataFrame):
    print('\n[PART 3] FC vs Compressor 交叉验证 ...')

    fc_on = df[df['FC_active'] == 1]
    fc_off = df[df['FC_active'] == 0]

    print(f'  FC_active=1: {len(fc_on):,} 小时 ({len(fc_on)/len(df)*100:.1f}%)')
    print(f'  FC_active=0: {len(fc_off):,} 小时 ({len(fc_off)/len(df)*100:.1f}%)')

    if len(fc_on) > 0:
        print(f'  FC 开启时 Compressor 均值: {fc_on["Compressor_avg"].mean():.1f}%')
    if len(fc_off) > 0:
        print(f'  FC 关闭时 Compressor 均值: {fc_off["Compressor_avg"].mean():.1f}%')

    # FC 开启时压缩机应该低(部分机组 FC,部分 DX,所以不一定为 0)
    if len(fc_on) > 0 and len(fc_off) > 0:
        diff = fc_off['Compressor_avg'].mean() - fc_on['Compressor_avg'].mean()
        if diff > 5:
            print(f'  ✓ FC 开启时压缩机负载低 {diff:.1f} 个百分点,物理一致')
        else:
            print(f'  ⚠ FC 开/关时压缩机差异仅 {diff:.1f}%,可能 FC 标识不准')

    # N_FC_on 分布
    print(f'\n  N_FC_on 分布:')
    for n in sorted(df['N_FC_on'].unique()):
        cnt = (df['N_FC_on'] == n).sum()
        print(f'    N_FC={n}: {cnt:,} ({cnt/len(df)*100:.1f}%)')

    # 模式统计
    print(f'\n  运行模式统计:')
    print(f'    纯 DX (N_FC=0):        {(df["N_FC_on"]==0).sum():,}')
    print(f'    纯 FC (N_DX=0):        {(df["N_DX"]==0).sum():,}')
    print(f'    混合 (DX+FC 并存):     {((df["N_FC_on"]>0)&(df["N_DX"]>0)).sum():,}')


# ── PART 4: 保存 ──────────────────────────────────────────

def save(df: pd.DataFrame):
    print(f'\n[PART 4] 保存 ...')
    df.to_parquet(OUT_PATH, index=False, compression='zstd')
    print(f'  {OUT_PATH}')
    print(f'  行数: {len(df):,}')
    print(f'  列数: {len(df.columns)}')
    print(f'  新增列: N_FC_on, Fan_Speed_avg, Fan_Speed_max, Compressor_avg, N_DX, FC_active')


# ── main ──────────────────────────────────────────────────

if __name__ == '__main__':
    vertiv_agg = load_vertiv()
    df = merge_with_clean(vertiv_agg)
    validate(df)
    save(df)
    print('\n✓ 全部完成')