from torch.optim.lr_scheduler import LambdaLR
from src.arguments import TrainingArguments
import math

def create_scheduler(optimizer, num_warmup_steps, num_training_steps, base_lr, max_lr, decay_steps, decay_factor):
    """
    乘数因子*优化器给定的初始化lr=current_lr
    """
    def lr_lambda(current_step):
        """
        返回乘数因子
        """
        if current_step < num_warmup_steps:
            # 线性从 base_lr 增加到 max_lr
            return base_lr / max_lr + (1 - base_lr / max_lr) * (current_step / num_warmup_steps)
        elif current_step >= decay_steps:
            #  余弦退火阶段：从 max_lr 开始衰减
            progress = (current_step - decay_steps) / (num_training_steps - decay_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # 比值范围 [0, 1]
        else:
            # 保持最大学习率
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

def custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    freeze_after_steps=30000
):
    assert num_warmup_steps < freeze_after_steps <= num_training_steps, "参数范围不合法"
    
    base_lr = optimizer.param_groups[0]["lr"]  # 初始学习率
    
    def lr_lambda(current_step):
        # Warmup阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 冻结阶段：固定为第3万步时的学习率
        elif current_step >= freeze_after_steps:
            progress = float(freeze_after_steps - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)