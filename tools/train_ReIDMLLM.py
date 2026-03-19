# Adapted from Tevatron code
import logging
import sys
import torch
import wandb,math
import json
from transformers import (
    HfArgumentParser,
)
import os
import sys
sys.path.append("/data/jhb_data/codes/MLLMEmbed_ReID")
from dataclasses import asdict
from src.dataset import TrainTextImageDataset, MLLMReIDTrainDataset, AIOReIDTrainDataset
from src.collator import TrainTextImageDataCollator, CustomDataCollator, AIODataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MLLMReIDModel
from src.trainer import CustomTrainer
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name
from src.trackback import LossLoggerCallback, MemoryCleanCallback
logger = logging.getLogger(__name__)
import torch.distributed as dist

def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    # print([r"{}".format(i) for i in sys.argv])
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    # 训练参数会保存为training_args.bin
    os.makedirs(training_args.output_dir,exist_ok=True)
    with open(f"{training_args.output_dir}/data_args.json", "w") as f:
        json.dump(asdict(data_args), f, indent=4)
    with open(f"{training_args.output_dir}/model_args.json", "w") as f:
        json.dump(asdict(model_args), f, indent=4)
    with open(f"{training_args.output_dir}/training_args.json", "w") as f:
        json.dump(asdict(training_args), f, indent=4)
    
    model = MLLMReIDModel.build(model_args, training_args, data_args)
    # ###########################################
    total_params = sum(p.numel() for p in model.parameters())  # 总参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数
    ratio = trainable_params / total_params * 100  # 可训练参数占比
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"可训练参数占比: {ratio:.2f}%")
    # ###########################################
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)
    
    train_dataset = AIOReIDTrainDataset(data_args, model_args)
    collator = AIODataCollator(data_args, model_args, processor)
    
    # ####################################################################################
    # from src.sampler import CMReIDDistibutedsampler
    # train_sampler = CMReIDDistibutedsampler(
    #     train_dataset, training_args.per_device_train_batch_size, data_args.num_instance_per_identifty, 4,
    #     rank=training_args.local_rank, world_size=training_args.world_size
    # )
    # # 创建一个dataloader
    # from torch.utils.data import DataLoader
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=training_args.per_device_train_batch_size,
    #     sampler=train_sampler,
    #     collate_fn=collator,
    #     num_workers=training_args.dataloader_num_workers,
    #     pin_memory=training_args.dataloader_pin_memory,
    # )
    # # 然后依次检查：
    # # 每个rank的样本数目是否尽量一致
    # print_rank(f"每个rank的样本数目：{len(train_dataloader)*training_args.per_device_train_batch_size}")
    # # 每个rank的每个batch是否严格遵循了“4模态样本数目均衡的要求”
    # for batch in train_dataloader:
    #     modalities = batch['modality']
    #     # 检查是否满足4个模态的样本数目一样的要求
    #     modality_counts = {}
    #     for modality in modalities:
    #         modality_counts[modality] = modality_counts.get(modality, 0) + 1
    #     if not all([i==4 for i in list(modality_counts.values())]):
    #         raise ValueError(f"Batch中模态分布不均衡: {modality_counts}")
    # ####################################################################################
    
    # model.train()
    trainer = CustomTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        callbacks=[LossLoggerCallback(), MemoryCleanCallback()],
        optimizers=(None, None),
        # optimizers=(optimizer, scheduler),
        num_modality=data_args.num_modality,
        instance_per_pid=data_args.num_instance_per_identifty
    )
    train_dataset.trainer = trainer
    if training_args.resume:
        trainer.train(resume_from_checkpoint=training_args.resume)
    else:
        trainer.train()
    trainer.save_model(training_args.output_dir)
    
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
