import os # 导入操作系统接口模块
import math # 导入math模块，用于数学计算
from decimal import Decimal # 导入decimal模块，用于十进制浮点运算

import utility # 导入包含多种通用函数的utility.py文件
import imageio # 导入imageio库，用于读写图像数据
import torch # 导入pytorch框架
import torch.nn.utils as utils # 该模块包含诸多对参数和向量的操作
from tqdm import tqdm # 导入进度条库
#定义Trainer类
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp): # 参数包含：多种参数args，数据loader，训练模型my_model，代价函数my_loss，检查点ckp
        self.args = args # 硬件、模型、训练、测试等参数
        self.scale = args.scale # 超分辨率倍数

        self.ckp = ckp # 检查点checkpoint
        self.loader_train = loader.loader_train # 训练集
        self.loader_test = loader.loader_test # 测试集
        self.model = my_model # 训练模型
        self.loss = my_loss # 代价函数
        self.optimizer = utility.make_optimizer(args, self.model) # 生成优化器和调度器

        if self.args.load != '': # 若日志文件名非空（即若填写了日志文件名）
            self.optimizer.load(ckp.dir, epoch=len(ckp.log)) # 导入已有日志

        self.error_last = 1e8 # 设定代价函数阈值
    #训练函数
    def train(self):
        self.loss.step() # 更新代价函数
        epoch = self.optimizer.get_last_epoch() + 1 # 更新训练轮数
        lr = self.optimizer.get_lr() # 设置学习率learning rate

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)) # 记录训练轮数和学习率
        )
        self.loss.start_log() #记录loss
        self.model.train()  #训练神经网络模型

        timer_data, timer_model = utility.timer(), utility.timer() # 创建计时器
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train): #加载训练数据
            lr, hr = self.prepare(lr, hr) #准备待训练数据
            timer_data.hold() #时间戳
            timer_model.tic() #时间戳

            self.optimizer.zero_grad() # 梯度归零，即前一步的损失清零
            sr = self.model(lr, 0) #神经网络输出超分辨率数据
            loss = self.loss(sr, hr) #根据原始分辨率与网络输出结果计算loss
            loss.backward() # 反向传播计算每个参数的梯度
            if self.args.gclip > 0: #修剪模型参数的梯度
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step() # 执行梯度下降进行参数更新

            timer_model.hold() #时间戳

            if (batch + 1) % self.args.print_every == 0: #定期记录训练结果
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release())) #写入日志文件训练进度

            timer_data.tic() #时间戳

        self.loss.end_log(len(self.loader_train)) ##训练结束的lossjieg
        self.error_last = self.loss.log[-1, -1] #计算误差
        self.optimizer.schedule() #调度器
    #测试函数
    def test(self):
        torch.set_grad_enabled(False) #测试时，参数梯度不传递

        epoch = self.optimizer.get_last_epoch()#获取当前回合数
        self.ckp.write_log('\nEvaluation:')#写入日志文件标题
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )#写入日志文件
        self.model.eval() # 结束训练，固定权重、偏置，进入测试模式

        timer_test = utility.timer()#开始测试计时
        #if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):#加载测试模型
            for idx_scale, scale in enumerate(self.scale):#数据尺寸
                d.dataset.set_scale(idx_scale)#格式化数据
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)#准备测试数据
                    sr = self.model(lr, idx_scale)#模型处理测试数据
                    sr = utility.quantize(sr, self.args.rgb_range)#格式化图像数据

                    save_list = [sr]#保存测试结果
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )#记入日志文件
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    #if self.args.save_results:
                    #    self.ckp.save_results(d, filename[0], save_list, scale)

                    postfix = ('SR', 'LR', 'HR')
                    for v, p in zip(save_list, postfix):
                        normalized = v[0].mul(255 / self.args.rgb_range)
                        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                        imageio.imwrite(('../experiment/test/results-{}/{}_x{}_{}.png'.format(d.dataset.name,filename[0],scale, p)), tensor_cpu.numpy())

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )#记录测试信息，包括数据集名称，精度，尺寸，回合数等

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))#测试时间写入日志
        self.ckp.write_log('Saving...')#保存信息写入日志

        #if self.args.save_results:
        #    self.ckp.end_background()

        if not self.args.test_only:#训练模式下
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))#记录训练信息

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True#记录时间
        )

        torch.set_grad_enabled(True)#梯度反向传播求导设置
    # 批量准备训练数据
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda') #选择GPU/CPU运算
        def _prepare(tensor):#根据硬件，转换数据格式
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args] #批量提取数据
    # 终止训练/测试
    def terminate(self):
        if self.args.test_only:#仅测试
            self.test()#测试模型
            return True
        else:#训练直至达到回合数
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs