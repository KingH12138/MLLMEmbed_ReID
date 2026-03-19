from __future__ import annotations
from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    model_backbone: str = field(
        default=None,
        metadata={"help": "backbone name"}
    )
    processor_name: str = field(
        default=None, metadata={"help": "processor_name, huggingface model name or path"}
    )
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    pooling: str = field(
        default='last',
        metadata={"help": "pooling method for encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "temperature for softmax"}
    )
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    lora_target_modules: str = field(
        default="qkv,proj,q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "lora target modules"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "number of crops used in image encoder"}
    )
    dist_train: bool = field(
        default=True, metadata={"help": "whether to use distributed training"}
    )
    checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path"}
    )
    student_test_ckpt_path: str = field(
        default=None, metadata={"help": "a local model path(.pth)"}
    )
    student_base_model_name: str = field(
        default=None, metadata={"help": "a local model path(.pth)"}
    )
    stu_resume: str = field(
        default=None, metadata={"help": "a local model path(.pth)"}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_meta: str = field(
        default=None, metadata={"help": "数据集元数据文件路径，存储相关信息。"}
    )
    sampler_name: str = field(
        default='multimodalty', metadata={"help": "softmax, softmax_triplet,multimodalty"}
    )
    max_len: int = field(
        default=150, metadata={"help": "The maximum total input sequence length after tokenization. "
                                        "Use with caution, since it may truncate text prompts due to large image lengths."},
    )
    # reid
    data_root: str = field(
        default='/data/jhb_data/datasets/st_reid', metadata={"help": "数据集根目录"}
    )
    resize: tuple[int] = field(
        default=(336,196), metadata={"help": "统一的resize大小(h,w)"}
    )
    num_classes: int = field(
        default=-1, metadata={"help": "pid数目,market为751,msmt17为1041 .etc"}
    )
    padding: int = field(
        default=10, metadata={"help": "padding size"}
    )
    flip_prob: float = field(
        default=0.5, metadata={"help": "随机翻转概率"}
    )
    pixel_mean: tuple[float] = field(
        default=(0.485, 0.456, 0.406), metadata={"help": "像素均值"}
    )
    pixel_std: tuple[float] = field(
        default=(0.229, 0.224, 0.225), metadata={"help": "像素标准差"}
    )
    erasing_prob: float = field(
        default=0.5, metadata={"help": "随机擦除概率"}
    )
    num_instance_per_identifty: int = field(
        default=8, metadata={"help": "每个身份的实例数"}
    )
    num_modality: int = field(
        default=4, metadata={"help": "模态数"}
    )

    
@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    project_name: str = field(
        default=None, metadata={"help": "project name"}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "logging steps"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    # ReID config
    metric_loss_type: str = field(
        default='triplet', metadata={"help": "metric loss type"}
    )
    triplet_loss_margin: float = field(
        default=0.3, metadata={"help": "triplet loss margin"}
    )
    id_loss_weight: float = field(
        default=0.25, metadata={"help": "triplet loss weight(according to paper:When Large Vision-Language Models Meet Person Re-Identification)"}
    )
    triplet_loss_weight: float = field(
        default=1, metadata={"help": "triplet loss weight"}
    )
    label_smooth: bool = field(
        default=True, metadata={"help": "交叉熵损失计算平滑"}
    )
    resume: str = field(
        default=None, metadata={"help": "是否需要断点续训？如果需要，请输入检查点路径"}
    )
    # test
    test_output_dir: str = field(
        default="./test_logs", metadata={"help": "测试log输出路径"}
    )
    test_batchsize: int = field(
        default=128, metadata={"help": "batch size of test dataloader"}
    )
    test_workers: int = field(
        default=4, metadata={"help": "number of workers"}
    )
