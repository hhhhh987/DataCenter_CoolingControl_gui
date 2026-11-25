'''
模型训练脚本
'''

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import CoolingPredictModel


def train():
    #1确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    #2加载数据集
    train_loader = get_dataloader('train')
    val_loader = get_dataloader('val')
    print(f'训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次')
    print(f'验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次')

    #3加载模型
    model = CoolingPredictModel().to(device)

    #4损失函数
    def quantile_loss(preds, targets, quantiles=config.QUANTILES):
        losses = []
        for i, q in enumerate(quantiles):
            errors = targets - preds[:, i]
            losses.append(torch.max(q * errors, (q - 1) * errors).mean())
        return sum(losses)
    #5优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5
    )

    #6日志记录
    writer = SummaryWriter(log_dir=config.LOGS_DIR / 'tensorboard_logs')
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    #7按轮训练
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, config.EPOCHS + 1):
        #训练阶段
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch}/{config.EPOCHS} 训练'):
            X, y = X.to(device), y.to(device)

            #前向传播
            output = model(X)
            loss = quantile_loss(output, y)

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        #每轮损失
        train_loss /= len(train_loader)

        #验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                #前向传播，无反向传播
                output = model(X)
                loss = quantile_loss(output, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 【新增】1. 更新学习率调度器
        # 必须在计算完 val_loss 后立即调用，告诉调度器当前的 Loss 是多少
        scheduler.step(val_loss)

        #记录
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        print(f'  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}', end='')

        #Early Stopping早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODELS_DIR / 'quantile_model.pth')
            print(' 保存最优模型')
        else:
            patience_counter += 1
            print(f'  (patience {patience_counter}/{config.PATIENCE})')
            if patience_counter >= config.PATIENCE:
                print(f'\n验证集loss连续{config.PATIENCE}轮未下降，提前停止')
                break

    writer.close()

    model.load_state_dict(torch.load(config.MODELS_DIR / 'quantile_model.pth', weights_only=True))
    print(f'\n训练完成，最优验证loss: {best_val_loss:.6f}')
    print(f'模型已保存: {config.MODELS_DIR / "quantile_model.pth"}')


if __name__ == '__main__':
    train()