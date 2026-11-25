'''
模型结构定义
'''

import torch
import torch.nn as nn
import config


class Attention(nn.Module):
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_size]
        query = lstm_output[:, -1:, :]
        scores = torch.bmm(query, lstm_output.transpose(1, 2))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, lstm_output)
        return context.squeeze(1)


class CoolingPredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT,
            bidirectional=False
        )
        self.attention = Attention()
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

    def forward(self, x):
        # x: [batch, 168, 15]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, 168, hidden_size]
        context = self.attention(lstm_out)
        # context: [batch, hidden_size]
        output = self.linear(context)
        # output: [batch, 1]
        return output.squeeze(1)


if __name__ == '__main__':
    model = CoolingPredictModel()
    x = torch.randn(config.BATCH_SIZE, config.WINDOW_SIZE, config.INPUT_SIZE)
    y = model(x)
    print(f'输入: {x.shape}')
    print(f'输出: {y.shape}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
