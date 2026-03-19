"""
collator的347行可以改一下。
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
import sys
sys.path.append("/data/jhb_data/codes/MLLMEmbed_ReID")
from tqdm import tqdm
from src.dataset import MLLMReIDTestDataset, AIOReIDTestDataset, AIOReIDValidDataset
from src.collator import CustomDataCollator, AIODataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MLLMReIDModel
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name
from utils.reid_eval_tools import eval_func, euclidean_distance

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
    # if not os.path.exists(os.path.join(work_dir,"featlabel.pth")):
    # 用于提取出的特征
    output_data = {
        'image_paths':[],
        'feats':[],
        'pids':[],
        'modality':[],
        'camids':[]
    }
    
    save_iter = 10
    if not os.path.exists(os.path.join(work_dir,"featlabel.pth")):
        model.eval()
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_dataloader)):
                feat = model.bottleneck(model.encode_message(batch['vlm_inputs'].to("cuda", dtype=torch.bfloat16))['feat'])
                output_data['image_paths'].extend(batch['image_path'])
                output_data['feats'].extend(feat)
                output_data['pids'].extend(batch['pid'])
                output_data['modality'].extend(batch['modality'])
                if batch['camid'] is not None:
                    output_data['camids'].extend(batch['camid'])
                batch_count+=1
                if not (batch_idx+1)%save_iter:
                    torch.save(output_data, os.path.join(work_dir,"featlabel.pth"))
        torch.save(output_data, os.path.join(work_dir,"featlabel.pth"))
        print("Feature extracting finished.")
    else:
        print("Feature file already exists, skip feature extraction.")
    from utils.test_util import display_aio_results_from_test_log
    display_aio_results_from_test_log(os.path.join(work_dir,"featlabel.pth"),10,save_rank_index=None,worst_query_num=0)