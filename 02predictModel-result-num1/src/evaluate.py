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
    meta = torch.load(config.PROCESSED_DATA_DIR / 'dataset2.pt', weights_only=False)

    #3加载模型
    model = CoolingPredictModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model4.pth', weights_only=True))

    #4预测评估
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            output = model(X)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.numpy())

    preds_delta = np.concatenate(all_preds)  # ◆ 修改: 变量名改为preds_delta
    targets_delta = np.concatenate(all_targets)  # ◆ 修改: 变量名改为targets_delta

    #5加载scaler参数
    target_min = meta['scaler_data_min'][0]
    target_max = meta['scaler_data_max'][0]
    last_known_test = meta['last_known_test'].numpy()  # ◆ 新增: 加载last_known

    # #反归一化
    # preds_kw = inverse_transform(preds, target_min, target_max)
    # targets_kw = inverse_transform(targets, target_min, target_max)
    # ◆ 修改: 反归一化逻辑——差值+last_known还原为归一化空间的绝对值，再反归一化到真实kW
    preds_abs_norm = preds_delta + last_known_test
    targets_abs_norm = targets_delta + last_known_test

    preds_kw = inverse_transform(preds_abs_norm, target_min, target_max)
    targets_kw = inverse_transform(targets_abs_norm, target_min, target_max)

    #6计算预测指标
    mae = mean_absolute_error(targets_kw, preds_kw)
    rmse = np.sqrt(mean_squared_error(targets_kw, preds_kw))
    mape = np.mean(np.abs((targets_kw - preds_kw) / targets_kw)) * 100
    r2 = r2_score(targets_kw, preds_kw)

    print(f'测试集样本数: {len(preds_delta)}')
    print(f'MAE:  {mae:.2f} kW')
    print(f'RMSE: {rmse:.2f} kW')
    print(f'MAPE: {mape:.2f}%')
    print(f'R²:   {r2:.4f}')

    #7预测曲线
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(targets_kw, label='Actual', alpha=0.8)
    plt.plot(preds_kw, label='Predicted', alpha=0.8)
    plt.xlabel('Sample')
    plt.ylabel('Tot_ict (kW)')
    plt.title(f'Prediction vs Actual  |  MAE={mae:.2f}kW  RMSE={rmse:.2f}kW  R²={r2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.LOGS_DIR / 'pred_curveMSE.png', dpi=150)
    plt.close()

    # ── 误差分布 ──
    errors = preds_kw - targets_kw
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error (kW)')
    plt.ylabel('Count')
    plt.title(f'Error Distribution  |  Mean={errors.mean():.2f}kW  Std={errors.std():.2f}kW')
    plt.tight_layout()
    plt.savefig(config.LOGS_DIR / 'error_distributionMSE.png', dpi=150)
    plt.close()

    print(f'\n图表已保存: {config.LOGS_DIR}')

    # ── 排除极端点后的指标 ──
    threshold = 700  # kW, 低于此值视为停机/异常
    normal_mask = targets_kw > threshold
    n_excluded = (~normal_mask).sum()

    if n_excluded > 0:
        preds_normal = preds_kw[normal_mask]
        targets_normal = targets_kw[normal_mask]

        mae_n = mean_absolute_error(targets_normal, preds_normal)
        rmse_n = np.sqrt(mean_squared_error(targets_normal, preds_normal))
        mape_n = np.mean(np.abs((targets_normal - preds_normal) / targets_normal)) * 100
        r2_n = r2_score(targets_normal, preds_normal)

        print(f'\n排除 {n_excluded} 个异常点 (Tot_ict < {threshold}kW) 后:')
        print(f'MAE:  {mae_n:.2f} kW')
        print(f'RMSE: {rmse_n:.2f} kW')
        print(f'MAPE: {mape_n:.2f}%')
        print(f'R²:   {r2_n:.4f}')

    # ◆ 新增: 验证集评估
    val_loader = get_dataloader('val')
    last_known_val = meta['last_known_val'].numpy()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            output = model(X)
            val_preds.append(output.cpu().numpy())
            val_targets.append(y.numpy())

    val_p_delta = np.concatenate(val_preds)
    val_t_delta = np.concatenate(val_targets)
    val_p_kw = inverse_transform(val_p_delta + last_known_val, target_min, target_max)
    val_t_kw = inverse_transform(val_t_delta + last_known_val, target_min, target_max)
    print(f'\n验证集 MAE: {mean_absolute_error(val_t_kw, val_p_kw):.2f} kW')
    print(f'验证集 R²:  {r2_score(val_t_kw, val_p_kw):.4f}')


if __name__ == '__main__':
    evaluate()