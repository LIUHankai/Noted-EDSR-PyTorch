import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
#对RGB分量进行均值处理类
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):#传入均值和方差

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1) #处理权重变量
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std #处理偏置变量
        for p in self.parameters(): #设置参数梯度更新模式
            p.requires_grad = False
#定义基本的神经网络层类
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):
        #传入基本参数，包括卷积类型，输入尺寸，输出尺寸，卷积核大小，激活函数等

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)] #添加卷积层
        if bn:#添加正则化层
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:#添加激活函数
            m.append(act)

        super(BasicBlock, self).__init__(*m)
#定义残差类
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        #传入基本参数，包括卷积类型，特征尺寸，卷积核大小，激活函数等
        super(ResBlock, self).__init__()
        m = []
        for i in range(2): #添加2层卷积层
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: #正则化层
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0: #激活函数
                m.append(act)

        self.body = nn.Sequential(*m) #模型主干
        self.res_scale = res_scale #残差尺寸
    #参数的前向传播
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)#数据首先经过模型主干，乘以尺寸后获得残差模块
        res += x #残差模块输出与传入数据的叠加

        return res #返回经过一个残差模块的网络输出值
#定义上采样类
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        #传入基本参数，包括卷积类型，特征尺寸，卷积核大小，激活函数等

        m = []
        if (scale & (scale - 1)) == 0: #尺寸为2*n的上采样处理
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias)) #添加卷积层
                m.append(nn.PixelShuffle(2)) #上采样
                if bn: #正则化层
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu': #激活函数
                    m.append(nn.ReLU(True))
                elif act == 'prelu': #激活函数
                    m.append(nn.PReLU(n_feats))

        elif scale == 3: #尺寸为3的上采样处理
            m.append(conv(n_feats, 9 * n_feats, 3, bias)) #添加卷积层
            m.append(nn.PixelShuffle(3)) #上采样
            if bn: #正则化层
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':#激活函数
                m.append(nn.ReLU(True))
            elif act == 'prelu': #激活函数
                m.append(nn.PReLU(n_feats))
        else: #程序报错提示
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

