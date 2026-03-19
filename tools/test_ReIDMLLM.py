"""
写一个测试脚本，测试微调后的Qwen2VL-2B-ReID(sfted_rstpreid)模型在market、msmt17等数据集上的性能
"""
import logging
import sys
import torch
import wandb
import json
import numpy as np
from transformers import (
    HfArgumentParser,
)
import os
from datetime import datetime

from torch.utils.data import DataLoader

from tqdm import tqdm
from src.dataset import MLLMReIDTestDataset
from src.collator import CustomDataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MLLMReIDModel
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name

if __name__ == '__main__':
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    # print([r"{}".format(i) for i in sys.argv])
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
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
    
    test_dataset = MLLMReIDTestDataset(data_args, model_args)
    
    collator = CustomDataCollator(data_args, model_args, processor)
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=training_args.test_batchsize, shuffle=False, num_workers=training_args.test_workers,
        collate_fn=collator
    )
    
    output_data = {
        "query":{
            'feats':[],
            'pids':[],
            'camids':[]
        },
        "gallery":{
            'feats':[],
            'pids':[],
            'camids':[]
        }
    }
    save_iter = 10

    for batch_idx, batch in tqdm(enumerate(test_dataloader)):
        feat = model.encode_message(batch['vlm_inputs'].to("cuda", dtype=torch.bfloat16))['feat']
        if len(set(batch['type']))==1:  # 优化一下，如果一个batch里面的type都相同
            output_data[batch['type'][0]]['feats'].extend(feat.detach().cpu())
            output_data[batch['type'][0]]['pids'].extend(batch['pid'])
            output_data[batch['type'][0]]['camids'].extend(batch['camid'])
        else:
            for i in range(len(batch)):
                output_data[batch['type'][i]]['feats'].append(feat[i].detach().cpu())
                output_data[batch['type'][i]]['pids'].append(batch['pid'][i])
                output_data[batch['type'][i]]['camids'].append(batch['camid'][i])
        if not (batch_idx+1)%save_iter:
            torch.save(output_data, os.path.join(work_dir,"featlabel.pth"))
        
    print("Feature extracting finished.")

            
    
    
    
    