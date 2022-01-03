from model import common #从model文件夹中导入common

import torch.nn as nn #导入torch的nn模块，重命名为nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}
#传入参数设置并实例化EDSR神经网络模型
def make_model(args, parent=False):
    return EDSR(args)
#定义神经网络模型EDSR类（Enhanced Deep Super-Resolution）
class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks #残差模块的数量
        n_feats = args.n_feats #特征数目
        kernel_size = 3  #卷积核数目
        scale = args.scale[0] #尺寸参数
        act = nn.ReLU(True) #激活函数
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)#根据配置命名模型名称
        if url_name in url: #判断是否已存在训练好的模型
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range) #差值计算层，实例化MeanShift类
        self.add_mean = common.MeanShift(args.rgb_range, sign=1) #求和计算，实例化MeanShift类

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)] #卷积模块

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ] #神经网络模型的主干，残差模块
        m_body.append(conv(n_feats, n_feats, kernel_size)) #添加卷积层

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ] #神经网络的尾部，添加上采样模块和卷积模块

        self.head = nn.Sequential(*m_head) #添加神经网络的头部
        self.body = nn.Sequential(*m_body) #添加神经网络的主体
        self.tail = nn.Sequential(*m_tail) #添加神经网络的尾部
    #参数的前向传播
    def forward(self, x):
        x = self.sub_mean(x) #对传入的图像数据进行平均差值处理
        x = self.head(x) #经过网络头部

        res = self.body(x) #传入网络主体
        res += x #数据叠加

        x = self.tail(res) #残差输出的结果传入尾部
        x = self.add_mean(x) #求和处理

        return x #返回神经网络的输出结果
#加载模型数据
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict() #复制已有的模型数据
        for name, param in state_dict.items():#处理导入的模型数据
            if name in own_state: #为变量逐个赋值
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception: #维度不匹配时，报错
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1: #不存在变量名时，报错
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

