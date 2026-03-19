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
from src.model import MLLMReIDModel
from tqdm import tqdm
from src.utils import print_master
from src.dataset import DistillMLLMReIDTestDataset
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import CLIPProcessor
from src.collator import StudentTrainDataCollator
from src.model_utils import load_processor, get_backbone_name
from src.model import SmallMultimodalReID, IRRA_CLIP
import torch.nn.functional as F
import torch

def cross_modality_test(output_data, q_mod, g_mod, mode='student'):
    if mode=="student":
        q_feats_m = output_data[q_mod]['student_feat'].cpu()
        g_feats_m = output_data[g_mod]['student_feat'].cpu()
    elif mode=="teacher":
        q_feats_m = output_data[q_mod]['teacher_feat'].cpu()
        g_feats_m = output_data[g_mod]['teacher_feat'].cpu()
    else:
        raise(f"There is no {mode} mode.")
    q_camids_m = output_data[q_mod]['camid'].cpu()
    g_camids_m = output_data[g_mod]['camid'].cpu()
    q_pids_m = output_data[q_mod]['pid'].cpu()
    g_pids_m = output_data[g_mod]['pid'].cpu()
    print(f"测试{q_mod}->{g_mod},测试信息如下:")
    print(f"特征数目:{len(q_feats_m)}->{len(g_feats_m)}")
    print(f"camid数目:{len(q_camids_m)}->{len(g_camids_m)}")
    print(f"pid数目:{len(q_pids_m)}->{len(g_pids_m)}")
    
    from utils.reid_eval_tools import eval_func_with_query_ap, euclidean_distance
    
    q_feats_m = torch.nn.functional.normalize(q_feats_m, dim=1, p=2)
    g_feats_m = torch.nn.functional.normalize(g_feats_m, dim=1, p=2)
    distmat = euclidean_distance(q_feats_m, g_feats_m)
    
    # 计算CMC和mAP
    if q_mod == 'text' and g_mod == 'rgb':
        cmc, mAP, mINP, _ = eval_func_with_query_ap(distmat, q_pids_m.numpy(), g_pids_m.numpy(), q_camids_m.numpy(), g_camids_m.numpy(), set=0)
    else:
        cmc, mAP, mINP, _ = eval_func_with_query_ap(distmat, q_pids_m.numpy(), g_pids_m.numpy(), q_camids_m.numpy(), g_camids_m.numpy(), set=2)
    
    print(f"模态对 {q_mod}->{g_mod} 的评估结果:")
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    print("mINP: {:.1%}".format(mINP))
    return {"rankacc":cmc,"mAP":mAP,"mINP":mINP}


def main(model_args, data_args, training_args):
    # 1. 设置device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2. 创建数据集和dataloader
    test_dataset = DistillMLLMReIDTestDataset(data_args, model_args)

    
    base_model_path = model_args.student_base_model_name

    student_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    collator = StudentTrainDataCollator(data_args, model_args, student_processor)
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=training_args.test_batchsize, 
        num_workers=training_args.test_workers,
        collate_fn=collator, 
        shuffle=False
    )
    
    # 3. 加载模型
    # student_model = SmallMultimodalReID(base_model_path,base_model_path,True).to(device).eval()
    student_model = IRRA_CLIP().to(device).eval()
    results_dicts = []
    ################################################################################
    epoch=49
    model_base_path = "/hongbojiang/workdirs/mllmreid/distill_stu_finetune_svd_warmbackbone(onlysvd)/"
    ################################################################################
    for i in range(1, epoch):
        results_dict = {"epoch":i}
        student_test_ckpt_path = model_base_path + f"student_model_epoch{i}.pth"
        dataset_name = data_args.dataset_meta.split("/")[-1].split('.')[0]
        work_dir = os.path.join(model_base_path, f"test_output/{dataset_name}_{i}")

        load_state_dict = torch.load(student_test_ckpt_path)['student_model_state_dict']
        print(f"从{student_test_ckpt_path}中加载学生模型的参数.")
        load_state_dict = {k.removeprefix("module."):v for k,v in load_state_dict.items()}
        student_model.load_state_dict(load_state_dict)
        # 4. 特征提取
        output_data = {
            'rgb': {'teacher_feat': [], 'student_feat': [], 'pid': [], 'camid': []},
            'ir': {'teacher_feat': [], 'student_feat': [], 'pid': [], 'camid': []},
            'sketch': {'teacher_feat': [], 'student_feat': [], 'pid': [], 'camid': []},
            'text': {'teacher_feat': [], 'student_feat': [], 'pid': [], 'camid': []},
        }
        
        featlabel_path = os.path.join(work_dir, "featlabel.pth")
        if not os.path.exists(featlabel_path):
            print(f"Can't find {featlabel_path} and prepare to generate it.")
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(test_dataloader)):
                    student_outputs, _ = student_model(
                        rgb_images_inputs=batch['rgb_images'].to(device),
                        ir_images_inputs=batch['ir_images'].to(device),
                        sketch_images_inputs=batch['sketch_images'].to(device),
                        text_inputs=batch['text_inputs'].to(device)
                    )
                    for modality in student_outputs:
                        output_data[modality]['student_feat'].append(student_outputs[modality])
                        output_data[modality]['pid'].append(batch[f'{modality}_pid'].to(device))
                        output_data[modality]['camid'].append(batch[f'{modality}_camid'].to(device))
            
            # 5. 所有模态的特征在bs维度上合并一下
            for modality in ['rgb', 'ir', 'sketch', 'text']:
                output_data[modality]['student_feat'] = torch.cat(output_data[modality]['student_feat'], dim=0)
                output_data[modality]['pid'] = torch.cat(output_data[modality]['pid'], dim=0)
                output_data[modality]['camid'] = torch.cat(output_data[modality]['camid'], dim=0)
            
            # 保存结果
            print("测试完成准备保存结果。")
            os.makedirs(work_dir, exist_ok=True)
            torch.save(output_data, featlabel_path)
            print("特征提取和保存完成。")

        output_data = torch.load(featlabel_path)
        # 进行跨模态测试
        modality_pairs = [('ir', 'rgb'), ('text', 'rgb'), ('sketch', 'rgb')]
        # mse_loss = calculate_mse_loss(output_data)
        # print("#"*20,"mse_loss","#"*20)
        # for mod, dist in mse_loss.items():
        #     print(f"{mod}: {dist:.4f}")
        # print("#"*20,"cosine_loss","#"*20)
        # cosine_loss = calculate_cosine_embedding_loss(output_data)
        # for mod, dist in cosine_loss.items():
        #     print(f"{mod}: {dist:.4f}")
        for q_mod, g_mod in modality_pairs:
            print("#"*20,"student model test output:","#"*20)
            result = cross_modality_test(output_data, q_mod, g_mod, "student")
            result = {k:v.tolist() for k,v in result.items()}
            results_dict["{}->{}".format(q_mod, g_mod)] = result
        results_dicts.append(results_dict)
        with open(os.path.join(model_base_path,f"test_output/test_{epoch},json"),'w') as f:
            json.dump(results_dicts, f)

if __name__ == '__main__':
    # 1. 获取参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 2. 运行主函数
    main(model_args, data_args, training_args)