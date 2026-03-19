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
from src.dataset import DistillMLLMReIDTrainDataset, OfflineDistillMLLMReIDTrainDataset
from src.collator import MLLMDistillDataCollator, OfflineMLLMDistillDataCollator
from transformers import CLIPProcessor
from src.loss import compute_sdm
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# def standard_mse_loss(input, target,temperature=3.0):
#     """标准MSE损失"""
#     input = input/temperature
#     target = target/temperature
#     return F.mse_loss(input, target)

def get_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    return model_args, data_args, training_args

class DistillationTrainer:
    def __init__(self, device, student_model, training_args, data_args, data_loader_length, world_size):
        self.device = device
        self.dict2detach_fn = lambda d:{k:v.detach().cpu().tolist() for k,v in d.items()}
        # 初始化模型
        self.student_model = student_model.train().to(device)
        # 优化器与混合精度
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=training_args.learning_rate)
        self.world_size = world_size
        # 损失函数
        # self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.loss_temp = 3.0
        
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
    
    def cal_loss(self, scores, student_feats, teacher_feats, pids, camids, modality, logits_scale):
        # 余弦相似度损失
        cosine_loss = self.cosine_loss(
            student_feats/self.loss_temp, teacher_feats/self.loss_temp, 
            torch.ones(student_feats.size(0)).to(self.device)
        )
        loss = cosine_loss
        return {
            "cosine_loss":cosine_loss,
            "all_loss":loss
        }
    
    def train_epoch(self, dataloader, epoch):
        self.student_model.train()
        print(f"Epoch {epoch} starts.")
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
            self.optimizer.zero_grad()
            # 数据移动到当前GPU
            batch['rgb_pid'] = batch['rgb_pid'].to(self.device)
            batch['ir_pid'] = batch['ir_pid'].to(self.device)
            batch['sketch_pid'] = batch['sketch_pid'].to(self.device)
            batch['text_pid'] = batch['text_pid'].to(self.device)
            
            batch['rgb_camid'] = batch['rgb_camid'].to(self.device)
            batch['ir_camid'] = batch['ir_camid'].to(self.device)
            batch['sketch_camid'] = batch['sketch_camid'].to(self.device)
            batch['text_camid'] = batch['text_camid'].to(self.device)
            
            teacher_feats_rgb = batch['vlm_inputs_rgb'].to(self.device)
            teacher_feats_ir = batch['vlm_inputs_ir'].to(self.device)
            teacher_feats_sketch = batch['vlm_inputs_sketch'].to(self.device)
            teacher_feats_text = batch['vlm_inputs_text'].to(self.device)
            
            # 学生模型前向
            
            with autocast(device_type='cuda', dtype=torch.float32):
                student_outputs, student_scores = self.student_model(
                    rgb_images_inputs=batch['rgb_images'].to(self.device),
                    ir_images_inputs=batch['ir_images'].to(self.device),
                    sketch_images_inputs=batch['sketch_images'].to(self.device),
                    text_inputs=batch['text_inputs'].to(self.device)
                )
                # 计算不同模态的损失
                rgb_n = len(student_outputs['rgb'])
                ir_n = len(student_outputs['ir'])
                sketch_n = len(student_outputs['sketch'])
                text_n = len(student_outputs['text'])
                feats = torch.cat([student_outputs['rgb'],student_outputs['ir'],student_outputs['sketch'],student_outputs['text']],dim=0)
                teacher_feats = torch.cat([teacher_feats_rgb, teacher_feats_ir, teacher_feats_sketch, teacher_feats_text],dim=0)
                scores = torch.cat([student_scores['rgb'],student_scores['ir'],student_scores['sketch'],student_scores['text']],dim=0)
                pids = torch.cat([batch['rgb_pid'],batch['ir_pid'],batch['sketch_pid'],batch['text_pid']],dim=0)
                camids = torch.cat([batch['rgb_camid'],batch['ir_camid'],batch['sketch_camid'],batch['text_camid']],dim=0)
                modalities = ['RGB']*rgb_n + ['IR']*ir_n + ['sketch']*sketch_n + ['text']*text_n
                # 计算不同模态的损失
                loss_dict = self.cal_loss(scores, feats, teacher_feats, pids, camids, modalities, self.student_model.module.logit_scale)
                loss = loss_dict['all_loss']
                record_loss = loss.detach().item()
                ########################################################################################################

                # 修改后的存储逻辑（确保JSON可读性）
                if self.device.index == 0:
                    # 转换张量为Python标量
                    state_dict = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                        'world_size': world_size,
                        **self.dict2detach_fn(loss_dict.copy())
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
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {record_loss:.4f}, LR: {current_lr:.2e}")

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
    train_dataset = OfflineDistillMLLMReIDTrainDataset(data_args, model_args)
    sampler = DistributedBalancedMultiModalRandomIdentitySampler(
        train_dataset, training_args.per_device_train_batch_size,
        num_instances=16, num_modality=4,
        rank=rank, world_size=world_size,
        shuffle=True
    )

    
    base_model_name = model_args.student_base_model_name
    # 使用DDP包装学生模型
    # student_model = SmallMultimodalReID(base_model_name, base_model_name, True).to(device)
    student_model = IRRA_CLIP().to(device)
    student_model = DDP(student_model, device_ids=[rank], find_unused_parameters=True)
    
    teacher_processor = load_processor(model_args, data_args)

    processor = load_processor(model_args, data_args)
    setattr(teacher_processor, 'processor', processor)
    
    student_processor = CLIPProcessor.from_pretrained(base_model_name)
    collator = OfflineMLLMDistillDataCollator(data_args, model_args, teacher_processor, student_processor)
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
    trainer = DistillationTrainer(device, student_model, training_args, data_args, len(dataloader), world_size)
    
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