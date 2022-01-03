* 本项目对[1]的代码框架进行了整理，并对main.py, trainer.py, utility.py文件, 及model相关代码中的_init_.py, common.py, edsr.py, mdsr.py文件进行了详细的中文注释。

* 环境需求等请参照[1]。

* 请于[2]下载模型并放于../models文件夹。

* 进入../src目录，在终端执行以下命令以进行一次超分辨率处理：

python main.py --data_test Demo --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../models/EDSR_x2.pt --test_only --n_threads 0 --chop

* 结果可见于..\experiment\test\results-Demo目录。

[1] sanghyun-son, EDSR-PyTorch, (2018), GitHub repository, https://github.com/sanghyun-son/EDSR-PyTorch.
[2] https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar.