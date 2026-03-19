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
from src.sampler import DistributedBalancedMultiModalRandomIdentitySampler
from src.dataset import DistillMLLMReIDTrainDataset, OfflineDistillMLLMReIDTrainDataset
from src.collator import MLLMDistillDataCollator, OfflineMLLMDistillDataCollator
from transformers import CLIPProcessor
from src.loss import compute_sdm
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def standard_mse_loss(input, target,temperature=3.0):
    """标准MSE损失"""
    input = input/temperature
    target = target/temperature
    return F.mse_loss(input, target)

def svd_projection_loss(input, target, k=50, temperature=3.0):
    input = input/temperature
    target = target/temperature
    input_fp32 = input.to(torch.float32)
    target_fp32 = target.to(torch.float32)

    # input_fp32 = F.normalize(input_fp32, dim=1)
    # target_fp32 = F.normalize(target_fp32, dim=1)
    
    with torch.no_grad():
        _, _, Vt = torch.linalg.svd(target_fp32, full_matrices=False)
        # total_energy = torch.sum(S)
        # k_energy = torch.sum(S[:k])
        # energy_ratio = k_energy / total_energy
        # print(f"前{k}个奇异值能量占比: {energy_ratio:.4f}")
        V_k = Vt[:k, :].t()
    
    target_proj = torch.matmul(target_fp32, V_k)
    input_proj = torch.matmul(input_fp32, V_k)
    
    return F.mse_loss(input_proj, target_proj)

def feature_correlation_loss(input, target, temperature=3.0):
    """修正后的特征相关性损失（基于相关系数矩阵）"""
    # 特征中心化
    input = input/temperature
    target = target/temperature
    input_centered = input - input.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    
    # 计算协方差矩阵
    batch_size = input.size(0)
    cov_input = torch.matmul(input_centered.t(), input_centered) / (batch_size - 1) # (bs,d)->(d,bs)*(bs,d)->(d,d)
    cov_target = torch.matmul(target_centered.t(), target_centered) / (batch_size - 1)
    
    # 转换为相关系数矩阵
    std_input = torch.sqrt(torch.diag(cov_input)).unsqueeze(1)
    std_target = torch.sqrt(torch.diag(cov_target)).unsqueeze(1)
    
    corr_input = cov_input / (std_input @ std_input.t() + 1e-8)
    corr_target = cov_target / (std_target @ std_target.t() + 1e-8)
    
    return F.mse_loss(corr_input, corr_target)

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
        # 三种mse-loss
        normal_mse_loss = standard_mse_loss(student_feats, teacher_feats)
        fc_loss = feature_correlation_loss(student_feats, teacher_feats)
        svd_proj_loss = svd_projection_loss(student_feats, teacher_feats)
        cosine_loss = self.cosine_loss(
            student_feats, teacher_feats, 
            torch.ones(student_feats.size(0)).to(self.device)
        )
        loss = 0.15*normal_mse_loss + 0.25*fc_loss + 0.3*svd_proj_loss + 0.3 * cosine_loss
        return {
            "standard_mse_loss":normal_mse_loss,
            "fc_loss":fc_loss,
            "svd_proj_loss":svd_proj_loss,
            "cosine_loss":cosine_loss,
            "all_loss":loss,
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
                ########################################################################################################
                # 计算需要记录的损失
                stand_mse_loss = rgb_loss_dict['standard_mse_loss'].detach() + ir_loss_dict['standard_mse_loss'].detach() + sketch_loss_dict['standard_mse_loss'].detach() + text_loss_dict['standard_mse_loss'].detach()
                fc_mse_loss = rgb_loss_dict['fc_loss'].detach() + ir_loss_dict['fc_loss'].detach() + sketch_loss_dict['fc_loss'].detach() + text_loss_dict['fc_loss'].detach()
                svd_mse_loss = rgb_loss_dict['svd_proj_loss'].detach() + ir_loss_dict['svd_proj_loss'].detach() + sketch_loss_dict['svd_proj_loss'].detach() + text_loss_dict['svd_proj_loss'].detach()

                cosine_loss = rgb_loss_dict['cosine_loss'].detach() + ir_loss_dict['cosine_loss'].detach() + sketch_loss_dict['cosine_loss'].detach() + text_loss_dict['cosine_loss'].detach()
                
                
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # 求和
                global_loss = loss.detach().item() /self.world_size  # 计算均值
                
                dist.all_reduce(stand_mse_loss, op=dist.ReduceOp.SUM)  # 求和
                global_stand_mse_loss = stand_mse_loss.item()/self.world_size

                dist.all_reduce(fc_mse_loss, op=dist.ReduceOp.SUM)  # 求和
                global_fc_mse_loss = fc_mse_loss.item()/self.world_size

                dist.all_reduce(svd_mse_loss, op=dist.ReduceOp.SUM)  # 求和
                global_svd_mse_loss = svd_mse_loss.item()/self.world_size
                
                dist.all_reduce(cosine_loss, op=dist.ReduceOp.SUM)  # 求和
                global_cosine_loss = cosine_loss.item()/self.world_size
                
                # 修改后的存储逻辑（确保JSON可读性）
                if self.device.index == 0:
                    # 转换张量为Python标量
                    state_dict = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'stand_mse_los': float(global_stand_mse_loss),
                        'fc_mse_loss': float(global_fc_mse_loss),
                        'svd_mse_loss': float(global_svd_mse_loss),
                        'cosine_loss': float(global_cosine_loss),
                        'total_loss': float(global_loss),
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
    train_dataset = OfflineDistillMLLMReIDTrainDataset(data_args, model_args)
    sampler = DistributedBalancedMultiModalRandomIdentitySampler(
        train_dataset, training_args.per_device_train_batch_size,
        num_instances=16, num_modality=4,
        rank=rank, world_size=world_size,
        shuffle=True
    )
    teacher_model = MLLMReIDModel.load(model_args, training_args, data_args)
    teacher_model = teacher_model.to(device).eval()
    teacher_model = teacher_model.to(dtype=torch.bfloat16)
    
    base_model_name = model_args.student_base_model_name
    # 使用DDP包装学生模型
    # student_model = SmallMultimodalReID(base_model_name, base_model_name, True).to(device)
    student_model = IRRA_CLIP().to(device)
    student_model = DDP(student_model, device_ids=[rank], find_unused_parameters=True)
    
    teacher_processor = load_processor(model_args, data_args)
    model_backbone = get_backbone_name(hf_config=teacher_model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
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