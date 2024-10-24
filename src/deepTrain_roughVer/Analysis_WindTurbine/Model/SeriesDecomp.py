import torch
from torch import nn as nn
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 만약 x가 2차원이면 3차원으로 변환
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=3):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x): # x: [batch, sequence, num_features]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return moving_mean, res
        
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=[7, 12, 14, 24, 48]):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x): # x: [batch, sequence, num_features]
        x = x.float()
        print(f"Input shape: {x.shape}")
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            print(f"Moving avg shape: {moving_avg.shape}")
            moving_mean.append(moving_avg.unsqueeze(-1))
        print(f"Moving avg after shape: {moving_avg.shape}")
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return moving_mean, res 
