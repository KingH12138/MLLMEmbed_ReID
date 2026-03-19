"""
写一个可视化脚本，微调后的Qwen2VL模型在行人重识别任务中的注意力可视化
"""
import logging
import sys
import torch
import json
import numpy as np
from transformers import (
    HfArgumentParser,
)
import os
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("/mnt/82_store/xty/jhb/codes/MLLMEmbed_ReID")

from src.dataset import MLLMReIDTestDataset, AIOReIDTestDataset, AIOReIDValidDataset
from src.collator import CustomDataCollator, AIODataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MLLMReIDModel
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name
from utils.reid_eval_tools import eval_func, euclidean_distance

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from matplotlib.colors import LinearSegmentedColormap

def visualize_attention(attention_matrix, original_image, cmap_name='jet', alpha=0.5):
    """
    可视化注意力分布在行人图像上
    
    参数:
    attention_matrix: 二维numpy数组，已归一化的注意力权重矩阵，形状为(h_patches, w_patches)
    original_image: 原始行人图像，可以是文件路径或numpy数组
    cmap_name: 颜色映射名称，默认为'jet'
    alpha: 热力图透明度，默认为0.5
    
    返回:
    fig: matplotlib图像对象
    """
    
    # 读取图像（如果是文件路径）
    if isinstance(original_image, str):
        img = cv2.imread(original_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = original_image.copy()
    
    # 获取图像尺寸和注意力矩阵形状
    h_img, w_img = img.shape[:2]
    h_patches, w_patches = attention_matrix.shape
    
    # 调整注意力矩阵大小以匹配图像尺寸
    attention_resized = cv2.resize(attention_matrix, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    
    # 创建自定义颜色映射
    if cmap_name == 'custom_heat':
        # 创建从蓝到红的热力图颜色映射
        colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # 蓝→青→黄→红
        cmap = LinearSegmentedColormap.from_list('custom_heat', colors, N=256)
    else:
        cmap = plt.get_cmap(cmap_name)
    
    # 应用颜色映射到注意力矩阵
    heatmap = cmap(attention_resized)[:, :, :3]  # 忽略alpha通道
    
    # 创建可视化图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制原始图像
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 绘制热力图
    heatmap_display = ax2.imshow(attention_resized, cmap=cmap)
    ax2.set_title('Attention Heatmap', fontsize=14)
    ax2.axis('off')
    plt.colorbar(heatmap_display, ax=ax2, shrink=0.8)
    
    # 绘制叠加图
    ax3.imshow(img)
    ax3.imshow(heatmap, alpha=alpha)
    ax3.set_title('Image with Attention Overlay', fontsize=14)
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


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
    model = model.to('cuda', dtype=torch.bfloat16)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)
    
    test_dataset = AIOReIDTestDataset(data_args, model_args)
    
    collator = AIODataCollator(data_args, model_args, processor)
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=training_args.test_batchsize, shuffle=False, num_workers=training_args.test_workers,
        collate_fn=collator
    )
    
    # 用于提取出的特征
    output_data = {
        'image_paths':[],
        'feats':[],
        'pids':[],
        'modality':[],
        'camids':[]
    }
    
    save_iter = 10
    
    model.eval()    # 记得手动切换为eager模式(/mnt/82_store/xty/jhb/codes/MLLMEmbed_ReID/src/model.py 280-285行)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader)):
            output_dict = model.encode_message(batch['vlm_inputs'].to("cuda", dtype=torch.bfloat16), 
                                        output_attentions=True)
            attention_mask = output_dict['attention_mask'].cpu()
            bs = len(attention_mask)
            # 随机取batch中的一个或者多个样本进行可视化
            sample_idx = int(np.random.randint(0, bs, size=1)[0])
            # 取出该样本的image_path, modality, pid
            image_path = batch['image_path'][sample_idx]
            modality = batch['modality'][sample_idx]
            pid = batch['pid'][sample_idx].item()
            bpe_str = [processor.tokenizer.decode(i) for i in batch['vlm_inputs']['input_ids'][sample_idx]]
            attention_mask = attention_mask[sample_idx]
            lastlayer_output_attentions = output_dict['attentions'][-1][sample_idx]  # (num_heads, seq_len, seq_len)
            num_heads,_,_ = lastlayer_output_attentions.shape
            save_dir = f"visualization_outputs/{modality}_{pid}"
            os.makedirs(save_dir, exist_ok=True)
            print(image_path)
            print("Saving to ", save_dir)
            
            # 如果modality不是text
            if modality != 'text':
                position_ids, mrope_position_deltas = model.encoder.get_rope_index(
                    input_ids=batch['vlm_inputs']['input_ids'],
                    image_grid_thw=batch['vlm_inputs']['image_grid_thw'], # 图像网格的 (时间, 高度, 宽度) 信息，图像通常时间维为1
                    attention_mask=batch['vlm_inputs']['attention_mask'],
                )
                visual_token_indices = torch.tensor([True if i=="<|image_pad|>" else False for i in bpe_str])
                num_visual_tokens = visual_token_indices.sum().item()
                height_coords = position_ids[1, sample_idx, visual_token_indices] # 高度方向上的位置ID
                width_coords = position_ids[2, sample_idx, visual_token_indices] # 宽度方向上的位置ID
                # 获取被铺平的图像token在原始图像网格中的位置
                height_coords-=height_coords.min()
                width_coords-=width_coords.min()
                
                for head in range(num_heads):
                    # 首先生成attention_map2d
                    attention_map2d = torch.zeros((height_coords.max()+1, width_coords.max()+1), dtype=torch.float32)
                    for i in range(num_visual_tokens):
                        attention_map2d[height_coords[i], width_coords[i]] = lastlayer_output_attentions[head, -1, visual_token_indices][i].cpu().float()
                    # 然后根据attention_map2d和image_path生成attention_map2d叠加图
                    fig = visualize_attention(attention_map2d.numpy(), image_path, cmap_name='custom_heat', alpha=0.5)
                    fig.suptitle(f'Attention Visualization - Head {head+1}', fontsize=16)
                    plt.savefig(f'{save_dir}/attention_head{head+1}_sample.png')
                        
            # 绘制一个attention heatmap，num_heads为12，那么我就绘制成4行3列
            fig, axes = plt.subplots(12, 1, figsize=(30, 10))
            for head in range(num_heads):
                ax = axes[head]
                last_token_attention = lastlayer_output_attentions[head][-1].cpu().float().unsqueeze(0)  # 取最后一个token的注意力分布
                sns.heatmap(last_token_attention, ax=ax, cmap='viridis',
                                annot=False,
                                cbar=False)  
                ax.set_title(f'Head {head+1}')
                # 将x轴和y轴的刻度标签设置为bpe_str
                ax.set_yticks([])
                ax.set_xticks(np.arange(len(bpe_str))+0.5)
                if head<num_heads-1:
                    ax.set_xticks([])
                else:
                    ax.set_xticklabels(bpe_str, rotation=90, fontsize=8)
            plt.subplots_adjust(hspace=0.6, wspace=0.6)
            # 把image_path, modality, pid也显示在图上
            plt.figtext(0.5, 0.01, f'Image Path: {image_path} | Modality: {modality} | PID: {pid}', wrap=True, horizontalalignment='center', fontsize=12)
            plt.savefig(f'{save_dir}/attention_head_sample.png')
            plt.close()
            break