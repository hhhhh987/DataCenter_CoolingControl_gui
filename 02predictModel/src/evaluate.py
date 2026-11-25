'''
测试集评估脚本
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from dataset import get_dataloader
from model import CoolingPredictModel


def inverse_transform(values, data_min, data_max):
    return values * (data_max - data_min) + data_min


def evaluate():
    #1确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #2加载数据集
    test_loader = get_dataloader('test')
    meta = torch.load(config.PROCESSED_DATA_DIR / 'dataset.pt', weights_only=False)
    #3加载模型
    model = CoolingPredictModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'quantile_model.pth', weights_only=True ))

    #4预测评估
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            output = model(X)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.numpy())

    preds_delta = np.concatenate(all_preds)       # shape: [N, 3]          # ← CHANGED: 从[N]变为[N,3]
    targets_delta = np.concatenate(all_targets)    # shape: [N]

    #5加载scaler参数
    target_min = meta['scaler_data_min'][0]
    target_max = meta['scaler_data_max'][0]
    last_known_test = meta['last_known_test'].numpy()  # ◆ 新增: 加载last_known

    # #反归一化
    # ← CHANGED: 三个分位数分别反归一化
    preds_kw = np.zeros_like(preds_delta)
    for i in range(config.NUM_QUANTILES):
        preds_abs_norm = preds_delta[:, i] + last_known_test
        preds_kw[:, i] = inverse_transform(preds_abs_norm, target_min, target_max)

    targets_abs_norm = targets_delta + last_known_test
    targets_kw = inverse_transform(targets_abs_norm, target_min, target_max)

    # ← NEW: 提取三个分位数的kW值
    q10_kw = preds_kw[:, 0]
    q50_kw = preds_kw[:, 1]
    q90_kw = preds_kw[:, 2]

    #6计算预测指标
    # ====== Q50点预测指标（与v1对比用）======
    mae = mean_absolute_error(targets_kw, q50_kw)  # ← CHANGED: 用q50
    rmse = np.sqrt(mean_squared_error(targets_kw, q50_kw))
    mape = np.mean(np.abs((targets_kw - q50_kw) / targets_kw)) * 100
    r2 = r2_score(targets_kw, q50_kw)

    print(f'测试集样本数: {len(targets_delta)}')
    print(f'\n── Q50 点预测指标（与v1对比）──')
    print(f'MAE:  {mae:.2f} kW')
    print(f'RMSE: {rmse:.2f} kW')
    print(f'MAPE: {mape:.2f}%')
    print(f'R²:   {r2:.4f}')

    # ====== 分位数专属指标 ======                                          # ← NEW: 整个区块
    coverage = np.mean((targets_kw >= q10_kw) & (targets_kw <= q90_kw)) * 100
    interval_width = np.mean(q90_kw - q10_kw)

    below_q10 = np.mean(targets_kw < q10_kw) * 100
    above_q90 = np.mean(targets_kw > q90_kw) * 100

    print(f'\n── 分位数区间指标 ──')
    print(f'[Q10, Q90] 覆盖率: {coverage:.1f}%  (目标: 80%)')
    print(f'平均区间宽度: {interval_width:.2f} kW')
    print(f'低于Q10比例: {below_q10:.1f}%  (目标: 10%)')
    print(f'高于Q90比例: {above_q90:.1f}%  (目标: 10%)')

    #7预测曲线
    #图1: 预测曲线 + 置信区间                             # ← CHANGED
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    x_axis = np.arange(len(targets_kw))
    plt.fill_between(x_axis, q10_kw, q90_kw, alpha=0.3, label='Q10-Q90 band')
    plt.plot(targets_kw, label='Actual', alpha=0.8, linewidth=0.8)
    plt.plot(q50_kw, label='Q50 Predicted', alpha=0.8, linewidth=0.8)
    plt.xlabel('Sample')
    plt.ylabel('Tot_ict (kW)')
    plt.title(f'Quantile Prediction  |  Coverage={coverage:.1f}%  Width={interval_width:.1f}kW  R²={r2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.LOGS_DIR / 'pred_curve.png', dpi=150)              # ← CHANGED: 文件名精简
    plt.close()

    #图2 误差分布
    errors = q50_kw - targets_kw                                           # ← CHANGED: 用q50
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error (kW)')
    plt.ylabel('Count')
    plt.title(f'Q50 Error Distribution  |  Mean={errors.mean():.2f}kW  Std={errors.std():.2f}kW')
    plt.tight_layout()
    plt.savefig(config.LOGS_DIR / 'error_dist.png', dpi=150)              # ← CHANGED: 文件名精简
    plt.close()

    #图3: 区间宽度变化                                 # ← NEW: 整个图
    widths = q90_kw - q10_kw
    plt.figure(figsize=(14, 4))
    plt.plot(widths, alpha=0.8, linewidth=0.8)
    plt.xlabel('Sample')
    plt.ylabel('Interval Width (kW)')
    plt.title(f'Prediction Interval Width  |  Mean={interval_width:.1f}kW  Std={widths.std():.1f}kW')
    plt.tight_layout()
    plt.savefig(config.LOGS_DIR / 'interval_width.png', dpi=150)
    plt.close()

    print(f'\n图表已保存: {config.LOGS_DIR}')


    # ── 排除极端点后的指标 ──
    threshold = 500
    normal_mask = targets_kw > threshold
    n_excluded = (~normal_mask).sum()

    if n_excluded > 0:
        q50_n = q50_kw[normal_mask]                                        # ← CHANGED
        targets_n = targets_kw[normal_mask]

        mae_n = mean_absolute_error(targets_n, q50_n)
        rmse_n = np.sqrt(mean_squared_error(targets_n, q50_n))
        mape_n = np.mean(np.abs((targets_n - q50_n) / targets_n)) * 100
        r2_n = r2_score(targets_n, q50_n)

        # ← NEW: 排除极端点后的覆盖率
        q10_n = q10_kw[normal_mask]
        q90_n = q90_kw[normal_mask]
        coverage_n = np.mean((targets_n >= q10_n) & (targets_n <= q90_n)) * 100

        print(f'\n排除 {n_excluded} 个异常点 (Tot_ict < {threshold}kW) 后:')
        print(f'MAE:  {mae_n:.2f} kW')
        print(f'RMSE: {rmse_n:.2f} kW')
        print(f'MAPE: {mape_n:.2f}%')
        print(f'R²:   {r2_n:.4f}')
        print(f'覆盖率: {coverage_n:.1f}%')

    # ====== 验证集评估 ======
    val_loader = get_dataloader('val')
    last_known_val = meta['last_known_val'].numpy()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            output = model(X)
            val_preds.append(output.cpu().numpy())
            val_targets.append(y.numpy())

    val_p_delta = np.concatenate(val_preds)        # [N, 3]               # ← CHANGED
    val_t_delta = np.concatenate(val_targets)

    # ← CHANGED: 验证集也做三分位数反归一化
    val_p_kw = np.zeros_like(val_p_delta)
    for i in range(config.NUM_QUANTILES):
        val_p_kw[:, i] = inverse_transform(
            val_p_delta[:, i] + last_known_val, target_min, target_max
        )
    val_t_kw = inverse_transform(val_t_delta + last_known_val, target_min, target_max)

    val_q50 = val_p_kw[:, 1]
    val_coverage = np.mean(
        (val_t_kw >= val_p_kw[:, 0]) & (val_t_kw <= val_p_kw[:, 2])
    ) * 100

    print(f'\n── 验证集 ──')
    print(f'MAE:  {mean_absolute_error(val_t_kw, val_q50):.2f} kW')
    print(f'R²:   {r2_score(val_t_kw, val_q50):.4f}')
    print(f'覆盖率: {val_coverage:.1f}%')


if __name__ == '__main__':
    evaluate()