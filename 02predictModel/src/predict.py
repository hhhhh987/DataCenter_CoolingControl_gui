"""
预测脚本
给定最近 168h 数据 → 预测下一小时 Q10/Q50/Q90 (kW)
single 模式: 预测最后一个时刻(部署用)
batch  模式: 滑窗逐小时预测(回测用)
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from process import add_time_features, FEATURE_COLS, SCALE_COLS, TARGET_COL
from model import CoolingPredictModel
import config

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / '05Deploy'))
from weather_client import calc_derived_weather


def inverse_transform(values, data_min, data_max):
    return values * (data_max - data_min) + data_min


def predict(input_path, mode='single'):
    # 1 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2 加载模型
    model = CoolingPredictModel().to(device)
    model.load_state_dict(
        torch.load(config.MODELS_DIR / 'quantile_model.pth',
                   map_location=device, weights_only=True)
    )
    model.eval()

    # 3 加载 scaler 参数
    meta = torch.load(config.PROCESSED_DATA_DIR / 'dataset.pt', weights_only=False)
    scaler_min = meta['scaler_data_min']
    scaler_max = meta['scaler_data_max']
    target_min = float(scaler_min[0])
    target_max = float(scaler_max[0])

    # 4 读取数据 + 补算天气 + 时间特征 + 归一化
    df = pd.read_parquet(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f'读入: {len(df):,} 行, {df.timestamp.min()} ~ {df.timestamp.max()}')

    df = calc_derived_weather(df)
    df = add_time_features(df)

    data_range = scaler_max - scaler_min
    data_range[data_range == 0] = 1.0
    for i, col in enumerate(SCALE_COLS):
        df[col] = (df[col].values - scaler_min[i]) / data_range[i]

    values = df[FEATURE_COLS].values.astype(np.float32)
    targets = df[TARGET_COL].values.astype(np.float32)
    timestamps = df['timestamp'].values

    # 5 构建窗口
    W = config.WINDOW_SIZE

    if mode == 'single':
        if len(values) < W:
            raise ValueError(f'数据不足: 需要 {W} 行, 实际 {len(values)} 行')
        windows = values[-W:][np.newaxis, :]       # [1, 168, 15]
        last_known = np.array([targets[-1]])         # 窗口最后一个归一化值
        ts_out = np.array([timestamps[-1]])          # 对应时间戳
        print(f'single 模式: 预测 {ts_out[0]} 之后一小时')

    elif mode == 'batch':
        windows, last_known, ts_out = [], [], []
        for i in range(len(values) - W):
            windows.append(values[i: i + W])
            last_known.append(targets[i + W - 1])
            ts_out.append(timestamps[i + W])
        windows = np.array(windows)
        last_known = np.array(last_known)
        ts_out = np.array(ts_out)
        print(f'batch 模式: {len(windows):,} 个滑窗')

    else:
        raise ValueError(f'mode 必须是 single 或 batch, 收到: {mode}')

    # 6 推理
    X = torch.tensor(windows, dtype=torch.float32)
    all_preds = []
    batch_size = 256

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start: start + batch_size].to(device)
            output = model(batch)
            all_preds.append(output.cpu().numpy())

    preds_delta = np.concatenate(all_preds)  # [N, 3]

    # 7 反归一化: delta + last_known → 绝对值(归一化) → kW
    preds_kw = np.zeros_like(preds_delta)
    for i in range(config.NUM_QUANTILES):
        abs_norm = preds_delta[:, i] + last_known
        preds_kw[:, i] = inverse_transform(abs_norm, target_min, target_max)

    # 8 输出结果
    if mode == 'single':
        Q10, Q50, Q90 = preds_kw[0]
        print(f'\n预测结果:')
        print(f'  Q10 = {Q10:.1f} kW')
        print(f'  Q50 = {Q50:.1f} kW')
        print(f'  Q90 = {Q90:.1f} kW')
        print(f'  区间宽度 = {Q90 - Q10:.1f} kW')
        return {'Q10': Q10, 'Q50': Q50, 'Q90': Q90}

    else:
        result = pd.DataFrame({
            'timestamp': ts_out,
            'Q10_kW': preds_kw[:, 0],
            'Q50_kW': preds_kw[:, 1],
            'Q90_kW': preds_kw[:, 2],
        })
        out_path = config.PROCESSED_DATA_DIR / 'predictions.parquet'
        result.to_parquet(out_path, index=False)
        print(f'\n预测完成: {len(result):,} 小时')
        print(f'  Q50 均值: {result.Q50_kW.mean():.1f} kW')
        print(f'  区间宽度均值: {(result.Q90_kW - result.Q10_kW).mean():.1f} kW')
        print(f'  已保存: {out_path}')
        return result


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'single'
    predict(config.RAW_DATA_DIR / 'merge_dataset.parquet', mode=mode)