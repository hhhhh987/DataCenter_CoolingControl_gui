'''
自定义DataSet
'''

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class CoolingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(split='train'):
    data = torch.load(
        config.PROCESSED_DATA_DIR / 'dataset2.pt',
        weights_only=False
    )
    X = data[f'X_{split}']
    y = data[f'y_{split}']
    dataset = CoolingDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == 'train')
    )
    return dataloader


if __name__ == '__main__':
    for s in ['train', 'val', 'test']:
        dl = get_dataloader(split=s)
        X_batch, y_batch = next(iter(dl))
        print(f'{s:5s}: {len(dl.dataset)} 样本, '
              f'{len(dl)} 批次, '
              f'X={X_batch.shape}, y={y_batch.shape}')