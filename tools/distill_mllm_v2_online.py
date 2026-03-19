import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm, os, json

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser
import torch.distributed as dist
from src.model_utils import load_processor, get_backbone_name
from src.model import MLLMReIDModel, SmallMultimodalReID, IRRA_CLIP
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from src.sampler import DistributedBalancedMultiModalRandomIdentitySampler
from src.dataset import DistillMLLMReIDTrainDataset
from src.collator import MLLMDistillDataCollator
from transformers import CLIPProcessor
from src.loss import compute_sdm
from torch.amp import autocast, GradScaler

import numpy as np
import matplotlib.pyplot as plt

def weighted_mse_loss(input, target, weight_mask=None):
    """
    计算按特征维度加权的MSE损失
    input: 学生模型输出特征 [batch_size, d]
    target: 教师模型输出特征 [batch_size, d]
    weight_mask: 维度权重向量 [d, ]
    """
    # 1. 计算逐元素的差值
    diff = input - target
    # 2. 对差值进行加权（核心步骤）：每个维度乘以对应的权重
    if weight_mask:
        weighted_diff = diff * weight_mask.unsqueeze(0) # 广播：[batch_size, d] * [1, d] -> [batch_size, d]
    else:
        weighted_diff = diff
    # 3. 计算加权后的平方误差，并求所有元素的平均值
    loss = (weighted_diff ** 2).mean()
    # 计算没有加权的平方误差，用于正常记录
    unweighted_loss = (diff ** 2).mean()
    return loss, unweighted_loss

def get_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    return model_args, data_args, training_args

class DistillationTrainer:
    def __init__(self, device, teacher_model, student_model, training_args, data_args, data_loader_length, world_size):
        self.device = device
        
        # 初始化模型
        self.teacher_model = teacher_model.eval().to(device,dtype=torch.bfloat16)
        self.student_model = student_model.train().to(device)
        # 优化器与混合精度
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=training_args.learning_rate)
        self.world_size = world_size
        # 损失函数
        # self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.total_steps = data_loader_length * training_args.num_train_epochs
        self.warmup_steps = int(0.1 * self.total_steps) # 0.1的ratio
        self.cosine_steps = self.total_steps - self.warmup_steps

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda step: min(step / self.warmup_steps, 1.0)
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.cosine_steps,
                    eta_min=1e-6
                )
            ],
            milestones=[self.warmup_steps]
        )
        # 状态保存器
        self.training_state_dicts = []
        self.json_log_path = os.path.join(training_args.output_dir, "training_log.json")
        # 验证一下lr
        if self.device.index == 0:
            lrs = []
            # 保存原始状态以便恢复
            original_state = self.scheduler.state_dict()
            
            # 模拟训练步骤
            for _ in range(self.total_steps):
                self.scheduler.step()  # 关键：更新调度器状态
                current_lr = self.scheduler.get_last_lr()[0]  # 获取当前学习率（第一个参数组）
                lrs.append(current_lr)
            
            # 恢复原始状态（避免影响实际训练）
            self.scheduler.load_state_dict(original_state)
            
            # 绘制并保存曲线
            plt.figure()
            plt.plot(lrs, label='Learning Rate Schedule')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Curve')
            plt.grid(True)
            plt.savefig("lr_schedule.png")
            plt.close()
    
    def cal_loss(self,student_feats, teacher_feats):
        # 普通mse loss
        # mse_loss = self.mse_loss(student_feats, teacher_feats)
        # 启用svd_dim_mask
        mse_loss, ori_mse_loss = weighted_mse_loss(student_feats, teacher_feats)
        # 余弦相似度损失
        ori_mse_loss = ori_mse_loss.detach()
        cosine_loss = self.cosine_loss(
            student_feats, teacher_feats, 
            torch.ones(student_feats.size(0)).to(self.device)
        )
        record_cosine_loss = cosine_loss.detach()
        loss = 0.7 * mse_loss + 0.3 * cosine_loss
        ori_loss = 0.7*ori_mse_loss + 0.3*record_cosine_loss
        return {"mse_loss":mse_loss,"ori_mse_loss":ori_mse_loss,
                "cosine_loss":cosine_loss,
                "all_loss":loss, "ori_all_loss":ori_loss}
    
    def train_epoch(self, dataloader, epoch):
        self.student_model.train()
        print(f"Epoch {epoch} starts.")
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
            # 数据移动到当前GPU
            batch['rgb_pid'] = batch['rgb_pid'].to(self.device)
            batch['ir_pid'] = batch['ir_pid'].to(self.device)
            batch['sketch_pid'] = batch['sketch_pid'].to(self.device)
            batch['text_pid'] = batch['text_pid'].to(self.device)
            
            batch['rgb_camid'] = batch['rgb_camid'].to(self.device)
            batch['ir_camid'] = batch['ir_camid'].to(self.device)
            batch['sketch_camid'] = batch['sketch_camid'].to(self.device)
            batch['text_camid'] = batch['text_camid'].to(self.device)
            
            with torch.no_grad():
                # 获取教师特征
                teacher_feats_rgb = self.teacher_model.encode_message(
                    batch['vlm_inputs_rgb'].to(self.device, dtype=torch.bfloat16)
                )['feat']
                teacher_feats_ir = self.teacher_model.encode_message(
                    batch['vlm_inputs_ir'].to(self.device, dtype=torch.bfloat16)
                )['feat']
                teacher_feats_sketch = self.teacher_model.encode_message(
                    batch['vlm_inputs_sketch'].to(self.device, dtype=torch.bfloat16)
                )['feat']
                teacher_feats_text = self.teacher_model.encode_message(
                    batch['vlm_inputs_text'].to(self.device, dtype=torch.bfloat16)
                )['feat']
                # teacher_outputs = {
                #     'rgb': teacher_feats_rgb,
                #     'ir': teacher_feats_ir,
                #     'sketch': teacher_feats_sketch,
                #     'text': teacher_feats_text
                # }
            
            # 学生模型前向
            self.optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float32):
                student_outputs, _ = self.student_model(
                    rgb_images_inputs=batch['rgb_images'].to(self.device),
                    ir_images_inputs=batch['ir_images'].to(self.device),
                    sketch_images_inputs=batch['sketch_images'].to(self.device),
                    text_inputs=batch['text_inputs'].to(self.device)
                )
                # 计算不同模态的损失
                rgb_loss_dict = self.cal_loss(student_outputs['rgb'],teacher_feats_rgb)
                ir_loss_dict = self.cal_loss(student_outputs['ir'],teacher_feats_ir)
                sketch_loss_dict = self.cal_loss(student_outputs['sketch'],teacher_feats_sketch)
                text_loss_dict = self.cal_loss(student_outputs['text'],teacher_feats_text)
                
                loss = rgb_loss_dict['all_loss'] + ir_loss_dict['all_loss'] + sketch_loss_dict['all_loss'] + text_loss_dict['all_loss']
                ori_loss = rgb_loss_dict['ori_all_loss'] + ir_loss_dict['ori_all_loss'] + sketch_loss_dict['ori_all_loss'] + text_loss_dict['ori_all_loss']
                ########################################################################################################
                # 计算需要记录的损失
                mse_loss = rgb_loss_dict['mse_loss'] + ir_loss_dict['mse_loss'] + sketch_loss_dict['mse_loss'] + text_loss_dict['mse_loss']
                ori_mse_loss = rgb_loss_dict['ori_mse_loss'] + ir_loss_dict['ori_mse_loss'] + sketch_loss_dict['ori_mse_loss'] + text_loss_dict['ori_mse_loss']
                cosine_loss = rgb_loss_dict['cosine_loss'] + ir_loss_dict['cosine_loss'] + sketch_loss_dict['cosine_loss'] + text_loss_dict['cosine_loss']
                
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # 求和
                global_loss = loss.item() /self.world_size  # 计算均值
                
                dist.all_reduce(ori_loss, op=dist.ReduceOp.SUM)  # 求和
                global_ori_loss = ori_loss.item()/self.world_size
                
                # 计算log的蒸馏损失
                dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)  # 求和
                global_mse_loss = mse_loss.item()/self.world_size
                
                dist.all_reduce(ori_mse_loss, op=dist.ReduceOp.SUM)  # 求和
                global_ori_mse_loss = ori_mse_loss.item()/self.world_size
                
                dist.all_reduce(cosine_loss, op=dist.ReduceOp.SUM)  # 求和
                global_cosine_loss = cosine_loss.item()/self.world_size
                
                # 修改后的存储逻辑（确保JSON可读性）
                if self.device.index == 0:
                    # 转换张量为Python标量
                    state_dict = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'mse_loss': float(global_mse_loss),  # 确保JSON可序列化
                        "ori_mse_loss": float(global_ori_mse_loss),
                        'cosine_loss': float(global_cosine_loss),
                        'total_loss': float(global_loss),
                        "ori_total_loss": float(global_ori_loss),
                        'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                        'world_size': world_size  # 记录GPU数量
                    }
                    self.training_state_dicts.append(state_dict)
                        
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()  # 学习率调整应在参数更新之后
            
            if batch_idx%25==0:
                if self.device.index==0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {global_loss:.4f}, LR: {current_lr:.2e}")

        # 保存当前epoch的模型state_dict,每个epoch结束保存一次
        if self.device.index==0:
            torch.save({
                'student_model_state_dict': self.student_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch
            }, os.path.join(training_args.output_dir, f"student_model_epoch{epoch}.pth"))
            # 使用JSON保存（非二进制）
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_state_dicts, f, ensure_ascii=False, indent=4)

        
def main(rank, world_size, model_args, data_args, training_args):
    # 设置当前设备
    device = torch.device(f'cuda:{rank}')
    
    # 初始化数据集和数据加载器
    train_dataset = DistillMLLMReIDTrainDataset(data_args, model_args)
    sampler = DistributedBalancedMultiModalRandomIdentitySampler(
        train_dataset, training_args.per_device_train_batch_size,
        num_instances=16, num_modality=4,
        rank=rank, world_size=world_size,
        shuffle=True
    )
    teacher_model = MLLMReIDModel.load(model_args, training_args, data_args)
    teacher_model = teacher_model.to(device, dtype=torch.bfloat16).eval()
    
    base_model_name = model_args.student_base_model_name
    # 使用DDP包装学生模型
    # student_model = SmallMultimodalReID(base_model_name, base_model_name, True).to(device)
    student_model = IRRA_CLIP().to(device).train()
    student_model = DDP(student_model, device_ids=[rank], find_unused_parameters=True)
    
    teacher_processor = load_processor(model_args, data_args)
    model_backbone = get_backbone_name(hf_config=teacher_model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    processor = load_processor(model_args, data_args)
    setattr(teacher_processor, 'processor', processor)
    
    student_processor = CLIPProcessor.from_pretrained(base_model_name)
    collator = MLLMDistillDataCollator(data_args, model_args, teacher_processor, student_processor)
    dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False  # sampler已经处理了shuffle
    )
    print("dataloader创建完毕，准备开始训练.")
    trainer = DistillationTrainer(device, teacher_model, student_model, training_args, data_args, len(dataloader), world_size)
    
    for epoch in range(training_args.num_train_epochs):
        sampler.set_epoch(epoch)  # 添加此行
        trainer.train_epoch(dataloader, epoch)
    print("Training done.")

if __name__ == "__main__":
    # 在主进程中解析参数
    model_args, data_args, training_args = get_args()
    # 获取本地rank和world_size
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    os.makedirs(training_args.output_dir, exist_ok=True)
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    # 运行主函数
    main(local_rank, world_size, model_args, data_args, training_args)