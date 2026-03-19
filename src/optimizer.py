import torch
from torch.optim import Adam, AdamW
from src.arguments import TrainingArguments

# def get_optimizer(model, training_args: TrainingArguments, type="adam"):
#     # 分结构设置学习率
#     param_groups = [
#         {'params': [p for n, p in model.named_parameters() if 'visual' in n and 'lora' in n], 'lr': 3e-4},
#         {'params': [p for n, p in model.named_parameters() if 'model.layers' in n and 'lora' in n], 'lr': 1e-5}
#     ]
#     optimizer = AdamW(param_groups, weight_decay=0.01)
#     return optimizer

# def get_optimizer(model, training_args: TrainingArguments, type="adam"):
#     # 目前lr最佳实验方案
#     # if type == "adam":
#     #     optimizer = Adam(
#     #         model.parameters(),
#     #         lr=training_args.learning_rate,
#     #         weight_decay=training_args.weight_decay,
#     #         eps=1e-8
#     #     )
#     # else:
#     #     raise ValueError(f"Unsupported optimizer type: {type}")
#     max_lr = 1e-4        # 最大学习率
#     weight_decay = 1e-4  # 权重衰减
#     optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
#     return optimizer

# def get_optimizer(model, training_args: TrainingArguments, type="adam"):
#     max_lr = 5e-4        # 最大学习率
#     weight_decay = 5e-4  # 权重衰减
#     optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
#     return optimizer


from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

def create_custom_optimizer_and_scheduler(model, total_steps):
    # 1. 优化器配置 (保持原有参数)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # 初始学习率（实际由调度器控制）
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    # 2. 多阶段学习率调度逻辑
    def lr_lambda(current_step):
        # 阶段划分
        warmup_steps = int(0.1 * total_steps)  # 前10% steps用于warmup
        plateau_steps = 30000  # 平台期结束步数
        decay_steps = 20000    # 首次衰减结束步数

        # Warmup阶段（线性增长）
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 首次余弦衰减阶段（warmup_steps → decay_steps）
        # 修改点1：以total_steps为终点计算衰减（而非decay_steps）
        elif current_step < decay_steps:
            progress = (current_step-warmup_steps) / total_steps  # 全局进度
            return 0.5 * (1 + math.cos(math.pi * progress))  # 余弦下降至0.5*lr_max
        
        # 学习率平台期（decay_steps → plateau_steps）
        elif current_step < plateau_steps:
            # 修改点2：平台期学习率=首次余弦衰减结束时的值（decay_steps/total_steps处）
            progress_at_decay_end = (decay_steps-warmup_steps) / total_steps
            lr_plateau = 0.5 * (1 + math.cos(math.pi * progress_at_decay_end))
            return lr_plateau
        
        # 二次余弦衰减阶段（plateau_steps → total_steps）
        else:
            # 关键修改：确保起始点与平台期结束值完全一致
            # 计算平台期结束时的学习率
            # lr_plateau_end = 0.5 * (1 + math.cos(math.pi * 
            #                     ((plateau_steps - warmup_steps) / (decay_steps - warmup_steps))))
            # 计算当前相对于平台期结束的进度
            lr_plateau_end = 0.794
            progress = (current_step - plateau_steps) / (total_steps - plateau_steps)
            
            # 从平台期结束值开始衰减
            return max(0.001, lr_plateau_end * (1 + math.cos(math.pi * progress)) / 2)

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler