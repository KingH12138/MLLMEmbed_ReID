from transformers import TrainerCallback
import matplotlib.pyplot as plt
import os, json
import torch

class LossLoggerCallback(TrainerCallback):
    def __init__(self, steps_per_record=1):
        self.losses = []
        self.learning_rates = []
        self.steps = []
        self.steps_per_record = steps_per_record
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        logging_steps决定多少steps调用一次on_log函数
        """
        if state.global_step % self.steps_per_record == 0 and "loss" in logs:
            self.losses.append((state.global_step, logs["loss"]))
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # 绘制曲线
        steps, losses = zip(*self.losses)
        plt.figure(figsize=(10, 5))
        # plt.ylim(0,10)
        plt.plot(steps, losses, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curve (per {self.steps_per_record} steps)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{args.output_dir}/loss_curve.png")
        plt.close()
        
    def on_step_end(self, args, state, control, **kwargs):
        # 获取当前学习率
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            self.steps.append(state.global_step)
            # 每50步绘制一次曲线
            if state.global_step % 50 == 0:
                self.plot_learning_rate(args)
    
    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束时绘制最终的学习率曲线
        self.plot_learning_rate(args)

    def plot_learning_rate(self,args):
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.learning_rates, 'b-', label='Learning Rate')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{args.output_dir}/lr_curve.png")
        plt.close()


# class LossLoggerCallback(TrainerCallback):
#     def __init__(self, steps_per_record=1000):
#         self.losses = []  # 存储 (step, loss)
#         self.learning_rates = []  # 存储 (step, lr) 或 (step, lr_from_logs)
#         self.steps_for_lr = []  # 存储记录学习率时的 step
#         self.steps_per_record = steps_per_record
#         # 判断是否是主进程，避免所有进程都执行记录和绘图
#         self.is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         """利用 Trainer 的日志系统记录 loss 和 learning_rate"""
#         if not self.is_main_process:
#             return

#         if state.global_step % self.steps_per_record == 0:
#             # 记录损失
#             if "loss" in logs:
#                 self.losses.append((state.global_step, logs["loss"]))
            
#             # 记录学习率 (优先从 logs 中获取)
#             current_lr = logs.get("learning_rate")
#             # 如果 logs 里没有，可以尝试其他方式，但 logs 是首选
#             # 例如：current_lr = args.learning_rate # 但这通常是初始值
#             if current_lr is not None:
#                 self.learning_rates.append(current_lr)
#                 self.steps_for_lr.append(state.global_step)

#     def on_epoch_end(self, args, state, control, **kwargs):
#         """每个 epoch 结束时绘制损失曲线"""
#         if not self.is_main_process or not self.losses:
#             return

#         steps, losses = zip(*self.losses)
#         plt.figure(figsize=(10, 5))
#         plt.plot(steps, losses, label="Training Loss")
#         plt.xlabel("Steps")
#         plt.ylabel("Loss")
#         plt.title(f"Training Loss Curve (per {self.steps_per_record} steps)")
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(f"{args.output_dir}/loss_curve_epoch_{state.epoch}.png")  # 按epoch保存
#         plt.close()

#     def on_train_end(self, args, state, control, **kwargs):
#         """训练结束时绘制最终的学习率曲线和损失曲线"""
#         if not self.is_main_process:
#             return

#         # 绘制最终的学习率曲线
#         if self.steps_for_lr and self.learning_rates:
#             plt.figure(figsize=(10, 5))
#             plt.plot(self.steps_for_lr, self.learning_rates, 'b-', label='Learning Rate')
#             plt.xlabel('Training Steps')
#             plt.ylabel('Learning Rate')
#             plt.title('Learning Rate Schedule')
#             plt.grid(True)
#             plt.legend()
#             plt.savefig(f"{args.output_dir}/lr_curve_final.png")
#             plt.close()

#         # （可选）绘制最终的损失曲线
#         if self.losses:
#             steps, losses = zip(*self.losses)
#             plt.figure(figsize=(10, 5))
#             plt.plot(steps, losses, label="Training Loss")
#             plt.xlabel("Steps")
#             plt.ylabel("Loss")
#             plt.title("Final Training Loss Curve")
#             plt.grid(True)
#             plt.legend()
#             plt.savefig(f"{args.output_dir}/loss_curve_final.png")
#             plt.close()

        
class MemoryCleanCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # 每次保存后清理未使用的显存
        
    def on_evaluate_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # 清空无用缓存
