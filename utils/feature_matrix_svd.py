"""
做大模型推理，然后输出ReID Tokens，后形成(n,d)的特征矩阵，做SVD分析
"""
import sys
import torch
import numpy as np
from transformers import (
    HfArgumentParser,
)
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.dataset import MLLMReIDTestDataset, AIOReIDTestDataset, AIOReIDValidDataset
from src.collator import CustomDataCollator, AIODataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MLLMReIDModel
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name
from utils.reid_eval_tools import eval_func, euclidean_distance
from torch.utils.data import DataLoader, RandomSampler

if __name__ == '__main__':
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    # print([r"{}".format(i) for i in sys.argv])
    # 1. 获取模型参数和任务参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if training_args.test_output_dir is None:
        work_dir = os.path.join(training_args.test_output_dir,f"{data_args.dataset_name}_{timestamp}")
    else:
        work_dir = training_args.test_output_dir
    os.makedirs(work_dir, exist_ok=True)
    # 2. 获取模型和处理器
    model = MLLMReIDModel.load(model_args, training_args, data_args)
    model = model.to('cuda', dtype=torch.bfloat16).eval()
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)
    
    test_dataset = AIOReIDTestDataset(data_args, model_args)
    print("length of applied dataset:", len(test_dataset))
    # 随机取一部分数据做测试(1000个)    
    collator = AIODataCollator(data_args, model_args, processor)
    sample_num = 4000   # 1000个样本对
    sampler = RandomSampler(
        test_dataset, 
        replacement=False,      # 如果 sample_num > len(my_dataset) 或希望有放回采样，设为 True
        num_samples=sample_num # 指定要采样的数量
    )
    dataloader = DataLoader(
        test_dataset,       
        batch_size=training_args.test_batchsize, 
        sampler=sampler,        
        collate_fn=collator,
        num_workers=training_args.test_workers
    )
    feat_save_path = os.path.join(work_dir,"featlabel.pth")
    
    try:
        output_data = torch.load(feat_save_path)
    except:
        output_data = {
            'image_paths':[],
            'feats':[],
            'pids':[],
            'modality':[],
            'camids':[]
        }
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                feat = model.encode_message(batch['vlm_inputs'].to("cuda", dtype=torch.bfloat16))['feat']
                output_data['image_paths'].extend(batch['image_path'])
                output_data['feats'].extend(feat)
                output_data['pids'].extend(batch['pid'])
                output_data['modality'].extend(batch['modality'])
                output_data['camids'].extend(batch['camid'])
        torch.save(output_data, feat_save_path)
        print("Feature extracting finished.")
    # 这里简单看一下modality的分布
    import torch
    import matplotlib.pyplot as plt
    import os

    # 确保模型和数据处理在GPU上进行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 直接从output_data中获取特征并转移到GPU
    # 假设output_data['feats']是一个包含多个特征张量的列表
    feat_matrix = torch.stack(output_data['feats'], dim=0).float()  # (n, d)
    feat_matrix = feat_matrix.to(device)  # 转移到GPU

    print(type(feat_matrix), feat_matrix.shape)

    # 对ReID token特征矩阵进行标准化（在GPU上）
    # 手动计算均值和标准差进行标准化，结果均值为0，标准差为1
    mean = feat_matrix.mean(dim=0)
    std = feat_matrix.std(dim=0, unbiased=True)
    # 避免除以零，将零标准差替换为1
    std = torch.where(std == 0, torch.ones_like(std), std)
    features_scaled = (feat_matrix - mean) / std

    # 使用PyTorch的SVD分解[2,3](@ref)
    # torch.svd返回U, S, V（注意：V是右奇异向量矩阵，不是V的转置）[2](@ref)
    U, S, V = torch.svd(features_scaled, some=True)  # some=True 等价于 full_matrices=False[2](@ref)

    # 计算累积解释方差比例
    # 注意：SVD后的奇异值S已经按降序排列[2](@ref)
    explained_variance = (S ** 2) / (features_scaled.shape[0] - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
    
    # 将结果移回CPU用于绘图（matplotlib需要在CPU上工作）
    explained_variance_ratio_cpu = explained_variance_ratio.cpu()
    cumulative_variance_ratio_cpu = cumulative_variance_ratio.cpu()
    S_cpu = S.cpu()

    # 绘制奇异值重要性图（碎石图）
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(S_cpu) + 1), explained_variance_ratio_cpu.numpy(), 'bo-', linewidth=2, markersize=5)
    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot (Variance per Component)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(S_cpu) + 1), cumulative_variance_ratio_cpu.numpy(), 'ro-', linewidth=2, markersize=5)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'svd_analysis.png'))
    plt.close()  # 防止内存泄漏
    # 获取原始D个维度重要性排列序列
    weights = explained_variance_ratio
    feature_importance = (torch.abs(V) * weights).sum(dim=1) # [d, k] * [k] -> 广播后求和得到 [d]
    # 3. 按重要性得分从高到低排序，获取索引序列
    importance_ordered_indices = torch.argsort(feature_importance, descending=True)

    # 4. (可选) 将结果移回CPU并转换为NumPy数组或列表，方便后续处理
    importance_ordered_indices_cpu = importance_ordered_indices.cpu().numpy()

    print("原始特征维度按重要性从高到低排列的索引序列:")
    print(importance_ordered_indices_cpu)
    np.savetxt(os.path.join(work_dir, 'feature_importance_indices.txt'), importance_ordered_indices_cpu, fmt='%d')
    