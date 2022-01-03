from model import common #从model文件夹中导入common

import torch.nn as nn #导入torch的nn模块，重命名为nn

url = {
    'r16f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr_baseline-a00cab12.pt',
    'r80f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr-4a78bedf.pt'
}
#传入参数设置并实例化MDSR神经网络
def make_model(args, parent=False):
    return MDSR(args)
#定义神经网络模型MDSR类（ Multi-scale Deep Super-Resolution）
class MDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MDSR, self).__init__()
        n_resblocks = args.n_resblocks #残差模块的数量
        n_feats = args.n_feats #特征数目
        kernel_size = 3 #卷积核数目
        act = nn.ReLU(True) #激活函数
        self.scale_idx = 0 #索引
        self.url = url['r{}f{}'.format(n_resblocks, n_feats)] #根据配置命名模型名称
        self.sub_mean = common.MeanShift(args.rgb_range) #差值计算层，实例化MeanShift类
        self.add_mean = common.MeanShift(args.rgb_range, sign=1) #求和计算，实例化MeanShift类
        #定义神经网络模型的头部
        m_head = [conv(args.n_colors, n_feats, kernel_size)] #添加卷积模块

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ]) #为多尺度的数据分别添加残差模块

        # 定义神经网络模型的主干，残差模块
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))#卷积模块

        self.upsample = nn.ModuleList([
            common.Upsampler(conv, s, n_feats, act=False) for s in args.scale
        ]) #为多尺度数据添加不同的上采样模块
        # 定义神经网络的尾部，卷积模块
        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head) #添加神经网络的头部
        self.body = nn.Sequential(*m_body) #添加神经网络的主体
        self.tail = nn.Sequential(*m_tail) #添加神经网络的尾部
    # 参数的前向传播
    def forward(self, x):
        x = self.sub_mean(x) #对传入的图像数据进行平均差值处理
        x = self.head(x) #经过网络头部
        x = self.pre_process[self.scale_idx](x) #对多种尺度的数据添加残差

        res = self.body(x) #传入神经网络模型主体
        res += x #数据叠加

        x = self.upsample[self.scale_idx](res) #对不同尺度的数据进行上采样
        x = self.tail(x) #传入尾部
        x = self.add_mean(x) #对传入的图像数据进行平均求和计算

        return x #返回神经网络的输出结果
    #设置多尺度的大小
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

