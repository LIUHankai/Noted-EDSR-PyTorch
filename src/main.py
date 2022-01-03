import torch # 导入pytorch框架
import utility # 导入包含多种通用函数的utility.py文件
import data # 导入关于数据处理的data文件夹
import model # 导入存储训练模型的model文件夹
import loss # 导入关于代价函数的loss文件夹
from option import args # 从option.py中导入硬件、模型、训练、测试等参数
from trainer import Trainer #从trainer.py中导入Trainer类，用于训练

torch.manual_seed(args.seed) # 为CPU设置种子用于生成随机数，使其生成确定的初始化参数
checkpoint = utility.checkpoint(args) # 生成checkpoint类的对象，用于存储某一时刻的模型架构、权重、训练配置、优化器状态等信息

def main():
    global model # 将model文件夹设置为全局变量
    if args.data_test == ['video']: # 对视频文件的超分辨率处理
        from videotester import VideoTester # 从videotester.py中导入VideoTester类，用于测试视频文件
        model = model.Model(args, checkpoint) # 由参数和检查点信息生成深度学习模型
        t = VideoTester(args, model, checkpoint) # 构造视频测试器
        t.test() # 执行测试
    else: # 对图片文件的超分辨率处理
        if checkpoint.ok: # 确认检查点状态
            loader = data.Data(args) # 导入数据
            _model = model.Model(args, checkpoint) # 由参数和检查点信息生成深度学习模型
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None # 设置代价函数；若仅测试，则不设置代价函数
            t = Trainer(args, loader, _model, _loss, checkpoint) # 构造训练器
            while not t.terminate(): # 若训练器未被终止
                t.train() # 执行训练
                t.test() # 执行测试

            checkpoint.done() # 停止记录

if __name__ == '__main__':
    main()
