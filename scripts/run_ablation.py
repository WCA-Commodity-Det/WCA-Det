import os
import inspect
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.modules as nn_modules
import ultralytics.nn.tasks as tasks


# ==========================================
# 1. 注册核心模块 (EMA & SimAM)
# ==========================================
class EMA(nn.Module):
    def __init__(self, channels=None, factor=8, *args, **kwargs):
        super(EMA, self).__init__()
        if channels is None:
            frame = inspect.currentframe().f_back
            while frame:
                if 'ch' in frame.f_locals and 'f' in frame.f_locals:
                    ch = frame.f_locals['ch']
                    f = frame.f_locals['f']
                    channels = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
                    break
                frame = frame.f_back
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


setattr(tasks, 'EMA', EMA)
setattr(nn_modules, 'EMA', EMA)
if hasattr(tasks, 'base_modules') and EMA not in tasks.base_modules:
    tasks.base_modules = tasks.base_modules + (EMA,)


def simam_forward(x, e_lambda=1e-4):
    b, c, h, w = x.size()
    n = w * h - 1
    x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
    y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + e_lambda)) + 0.5
    return x * torch.sigmoid(y)


original_c3k2_forward = nn_modules.C3k2.forward


def new_c3k2_forward(self, x):
    return simam_forward(original_c3k2_forward(self, x))


# ==========================================
# 2. 动态生成网络架构的工厂函数
# ==========================================
def create_yaml(mode="baseline"):
    # 根据不同模式，选择使用官方原版组件还是我们的创新组件
    backbone_layer_10 = "- [-1, 2, C2PSA, [1024]]" if mode != "ema_only" else "- [-1, 2, EMA, []]"

    if mode == "simam_only":
        head_c3k2 = "C3k2_SimAM"
        # 临时挂载 SimAM
        nn_modules.C3k2.forward = new_c3k2_forward
    else:
        head_c3k2 = "C3k2"
        # 恢复官方 C3k2
        nn_modules.C3k2.forward = original_c3k2_forward

    yaml_code = f"""
nc: 12
scales:
  m: [0.75, 0.50, 1024] 

backbone:
  - [-1, 1, Conv, [64, 3, 2]] 
  - [-1, 1, Conv, [128, 3, 2]] 
  - [-1, 2, C3k2, [256, False, 0.25]] 
  - [-1, 1, Conv, [256, 3, 2]] 
  - [-1, 2, C3k2, [512, False, 0.25]] 
  - [-1, 1, Conv, [512, 3, 2]] 
  - [-1, 2, C3k2, [512, True]] 
  - [-1, 1, Conv, [1024, 3, 2]] 
  - [-1, 2, C3k2, [1024, True]] 
  - [-1, 1, SPPF, [1024, 5]] 
  {backbone_layer_10} # 动态切换 C2PSA 或 EMA

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] 
  - [[-1, 6], 1, Concat, [1]] 
  - [-1, 2, {head_c3k2}, [512, False]] 

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] 
  - [[-1, 4], 1, Concat, [1]] 
  - [-1, 2, {head_c3k2}, [256, False]] 

  - [-1, 1, Conv, [256, 3, 2]] 
  - [[-1, 13], 1, Concat, [1]] 
  - [-1, 2, {head_c3k2}, [512, False]] 

  - [-1, 1, Conv, [512, 3, 2]] 
  - [[-1, 10], 1, Concat, [1]] 
  - [-1, 2, {head_c3k2}, [1024, True]] 

  - [[16, 19, 22], 1, Detect, [nc]] 
"""
    yaml_path = f"yolo11-{mode}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_code.strip())
    return yaml_path


# ==========================================
# 3. 自动化排队训练引擎
# ==========================================
def main():
    # 我们要跑的三个实验
    experiments = {
        "baseline": "yolo11m_01_Baseline",
        "ema_only": "yolo11m_02_EMA_Only",
        "simam_only": "yolo11m_03_SimAM_Only"
    }

    print("🚀 自动化消融实验已启动！今晚由 5060Ti 为您接管战场...")

    for mode, run_name in experiments.items():
        print(f"\n==============================================")
        print(f"👉 正在开始训练: {run_name}")
        print(f"==============================================")

        yaml_path = create_yaml(mode)
        model = YOLO(yaml_path, task='detect')
        model.load("yolo11m.pt")

        # 保持与满血版一模一样的超参数和数据集，保证变量唯一！
        model.train(
            data="D:/spsb/data_merged/dataset_merged.yaml",
            epochs=200,
            imgsz=640,
            batch=8,
            device="0",
            workers=4,
            project="runs/ablation",  # 单独建一个文件夹放消融实验结果，免得乱
            name=run_name,
            mixup=0.15,
            copy_paste=0.15,
            hsv_v=0.4,
            auto_augment="randaugment",
            cos_lr=True,
            close_mosaic=15
        )

    print("\n🎉 恭喜！所有消融实验已顺利跑完！")


if __name__ == '__main__':
    main()