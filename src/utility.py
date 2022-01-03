import os #导入系统编程的操作模块
import math #导入内置数学类函数库，完成基本的数学运算
import time #导入处理时间数据的标准库
import datetime #导入处理时间日期的标准库
from multiprocessing import Process #导入进程模块，实现对资源的调用、内存的管理、网络接口的调用
from multiprocessing import Queue #导入队列模块，实现信息在多个线程间信息的安全交换

import matplotlib #导入绘图工具库
matplotlib.use('Agg') #配置matplotlib的后端，使用‘Agg’渲染器，该语句的效果会使得pycharm不显示图像。
import matplotlib.pyplot as plt #导入matplotlib中的pyplot模块，并将其重命名为plt

import numpy as np #导入用于基础数字计算的标准库numpy,并将其重命名为np.
import imageio #导入图像处理模块，用于读取和写入各种图像数据

import torch #导入torch的框架
torch.cuda.empty_cache() #清除没用的临时变量
import torch.optim as optim #导入torch中的优化器模块，并将其重命名为optim
import torch.optim.lr_scheduler as lrs #导入torch中的学习率调整模块，并将其重命名为lrs

#定义timer类
class timer():
    def __init__(self): #定义类的基本属性
        self.acc = 0 #定义timer类的acc属性，用于表示累计的时间
        self.tic() #定义timer类的tic()方法

    def tic(self): #该方法用于获取当前时间
        self.t0 = time.time() #调用time.time()，获取当前时间，并为初始时间变量t0赋值。

    def toc(self, restart=False): #toc方法，导入参数restart
        diff = time.time() - self.t0 #获取当前时间与初始时间t0的差值，计算时间差值
        if restart: self.t0 = time.time() #参数为真，重置初始时间变量t0
        return diff #返回时间差值

    def hold(self): #计算累计时间
        self.acc += self.toc() #对acc变量叠加时间差值

    def release(self): #释放累计时间，并返回累计时间
        ret = self.acc #参数赋值
        self.acc = 0 #重置acc

        return ret #返回累计时间

    def reset(self): #重置累计时间变量acc
        self.acc = 0 #累计时间变量acc置0

#定义bg_target方法
def bg_target(queue):
    while True: #输入变量为队列时
        if not queue.empty(): #当队列非空
            filename, tensor = queue.get() #获取队列的值，分别对应图像名字和图像的数据
            if filename is None: break #队列名字None,跳出当前循环
            imageio.imwrite(filename, tensor.numpy()) #根据队列中的信息写入图像

#定义checkpoint类
class checkpoint():
    def __init__(self, args): #定义基本属性，并根据实例化参数复制
        self.args = args #定义arg属性，根据实例化参数赋值
        self.ok = True #初始化OK属性
        self.log = torch.Tensor() #定义log属性的类别为torch.Tensor()类
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') #获取当前时间，以形式化的方式记录

        if not args.load: #判断load属性的值,当load属性值不存在时，判断save属性
            if not args.save: #判断save属性的值
                args.save = now #当save属性值不存在时，为其赋值
            self.dir = os.path.join('..', 'experiment', args.save) #当save属性值存在时，从导入的参数中获取‘保存地址’
        else: #当load属性值存在时
            self.dir = os.path.join('..', 'experiment', args.load) #从导入的参数中获取‘加载地址’
            if os.path.exists(self.dir): #判断是否已经存在加载地址
                self.log = torch.load(self.get_path('psnr_log.pt')) #根据加载地址，加载模型，存入log变量
                print('Continue from epoch {}...'.format(len(self.log))) #输出log变量长度
            else:
                args.load = '' #load属性置空

        if args.reset: #当传入的参数reset属性为真时，重置
            os.system('rm -rf ' + self.dir) #调研os对操作系统的方法，直接删除当前目录下的所有文件及目录
            args.load = '' #load属性置空

        os.makedirs(self.dir, exist_ok=True) #创建文件夹
        os.makedirs(self.get_path('model'), exist_ok=True) #创建存放模型的文件夹
        for d in args.data_test: #根据传入的数据地址，逐个创建文件夹
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True) #创建存放数据的文件夹

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w' #设置文件打开方式，如果文件夹存在，模式为‘a':追加，否则为'w',新建
        self.log_file = open(self.get_path('log.txt'), open_type) #打开日志文件
        with open(self.get_path('config.txt'), open_type) as f: #打开配置文件
            f.write(now + '\n\n') #为配置文件写入时间
            for arg in vars(args): #为配置文件逐个记录参数
                f.write('{}: {}\n'.format(arg, getattr(args, arg))) #为配置文件写入参数名称及对应的数值
            f.write('\n') #换行

        self.n_processes = 8 #并行程序数目=8

    def get_path(self, *subdir): #获取地址
        return os.path.join(self.dir, *subdir) #对地址属性进行格式化处理，并返回相应的地址

    def save(self, trainer, epoch, is_best=False): #保存
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best) #获取地址，保存模型，记录回合信息
        trainer.loss.save(self.dir)#根据地址保存loss
        trainer.loss.plot_loss(self.dir, epoch) #根据地址和回合数绘制loss曲线

        self.plot_psnr(epoch) #绘制PSNR的训练轨迹图
        trainer.optimizer.save(self.dir) #保存优化器模型
        torch.save(self.log, self.get_path('psnr_log.pt')) #保存log信息

    def add_log(self, log):#增加日志信息
        self.log = torch.cat([self.log, log]) #在原有的日志信息上叠加

    def write_log(self, log, refresh=False): #写日志文件
        print(log) #打印当前日志信息
        self.log_file.write(log + '\n') #写入一行日志
        if refresh: #重新开启日志
            self.log_file.close() #关闭日志
            self.log_file = open(self.get_path('log.txt'), 'a') #打开日志文件

    def done(self): #关闭日志文件
        self.log_file.close()

    def plot_psnr(self, epoch):#绘制PSNR曲线
        axis = np.linspace(1, epoch, epoch) #创建序列【1，epoch】间隔为1
        for idx_data, d in enumerate(self.args.data_test): #获取测试集的数据和地址（名称）
            label = 'SR on {}'.format(d) #根据地址命名图像
            fig = plt.figure() #画布
            plt.title(label) #设置画布名称
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                ) #根据日志文件中的数据绘制PSNR曲线
            plt.legend() #图例
            plt.xlabel('Epochs') #设置横轴名称
            plt.ylabel('PSNR') #设置纵轴名称
            plt.grid(True) #设置网格化显示
            plt.savefig(self.get_path('test_{}.pdf'.format(d))) #保存图像
            plt.close(fig) #关闭图像
    '''    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()'''

    def begin_background(self):
        self.queue = Queue()

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
#对图像数据格式化处理
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

#计算peak signal-to-noise ratio（PSNR）
def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

# redefine lambda x: x.requires_grad
def return_x(x):
    return x.requires_grad

#redefine lambda x: int(x)
def return_y(y):
    return int(y)

#设置优化器，根据参数和待优化目标
def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(return_x, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':#设置SGD类型的优化器相关参数
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':#设置ADAM类型优化器相关参数
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':#设置RMSprop类型优化器相关参数
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler设置调度器
    milestones = list(map(return_y, args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    #定制优化器类
    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
        #注册调度器
        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)
        #保存优化器
        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))
        #加载优化器
        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()
        #获取优化器地址
        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')
        #调度器
        def schedule(self):
            self.scheduler.step()
        #获取lr
        def get_lr(self):
            return self.scheduler.get_lr()[0]
        #获取回合数
        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)#实例化优化器
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)#注册调度器
    return optimizer