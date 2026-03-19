import contextlib
import functools
import shutil
import sys
import time
import json

from packaging import version
from accelerate import skip_first_batches, DistributedType
from transformers.trainer import Trainer, TRAINING_ARGS_NAME, TRAINER_STATE_NAME
import torch.distributed as dist
from typing import Optional
import os
import torch
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.collator import split_vlm_inputs, get_dense_rep, split_and_process_vlm_inputs
from src.model_utils import process_vlm_inputs_fns
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from itertools import repeat

from transformers.training_args import OptimizerNames, ParallelMode
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_utils import (
    TrainOutput,
    has_length,
    speed_metrics,
)

from transformers.trainer_pt_utils import (
    get_model_param_count,
)

from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_torch_xla_available,
    logging, is_sagemaker_mp_enabled,
)

if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

logger = logging.get_logger(__name__)


from src.sampler import BalancedMultiModalRandomIdentitySampler, DistributedBalancedMultiModalRandomIdentitySampler
from torch.utils.data import DataLoader, BatchSampler
from utils.reid_eval_tools import eval_func, euclidean_distance
import numpy as np
from filelock import FileLock
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


from transformers.utils import is_peft_available
from peft import PeftModel
import importlib
from src.utils import print_rank

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel
            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

class CustomTrainer(Trainer):
    def __init__(self, instance_per_pid=8, num_modality=4, *args, **kwargs):
        super(CustomTrainer,self).__init__(*args, **kwargs)
        self.instance_per_pid = instance_per_pid
        self.num_modality = num_modality
        print_rank("当前rank为:{}, worldsize为{}.".format(self.args.local_rank, self.args.world_size))
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_dict = model(vlm_inputs=inputs['vlm_inputs'],pid=inputs['pid'],camid=inputs['camid'], modality=inputs['modality'])
        saved_loss_dict = {k:v.cpu().tolist() for k,v in loss_dict.items()}
        self.log(saved_loss_dict)
        output_json_path = os.path.join(self.args.output_dir, "loss_logs.json")
        with open(output_json_path, "a") as f:
            json.dump({"step": self.state.global_step, **saved_loss_dict}, f)
            f.write("\n")  # 每行一个JSON对象
        return loss_dict['all_loss']
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            # dataloader_params["sampler"] = BalancedMultiModalRandomIdentitySampler(train_dataset, self._train_batch_size, self.instance_per_pid, self.num_modality)
            dataloader_params['sampler'] = DistributedBalancedMultiModalRandomIdentitySampler(train_dataset, self._train_batch_size, self.instance_per_pid, self.num_modality, rank=self.args.local_rank, world_size=self.args.world_size)
        
        # sampler_len = dataloader_params['sampler'].__len__()
        # sampler_iter_len = len(list(dataloader_params['sampler'].__iter__()))
        # assert sampler_iter_len==sampler_len, "AIO采样器理论长度{}和实际迭代函数输出的长度{}不一样.".format(sampler_len, sampler_iter_len)
        # print_rank("AIO采样器长度为:{}".format(sampler_len))
        dl = DataLoader(train_dataset, **dataloader_params)
        print_rank(f"数据加载器长度:{len(dl)}")
        return self.accelerator.prepare(dl)
    
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        与父类方法唯一的区别就是判断加载的state_dict的key是不是没有encoder前缀，没有就给其加上。
        """
        if model is None:
            model = self.model
        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not supported."
                        )
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    new_state_dict = {}
                    for k,v in state_dict.items():
                        new_state_dict['encoder.' + k] = v
                    state_dict = new_state_dict
                    del new_state_dict
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                new_state_dict = {}
                for k,v in state_dict.items():
                    new_state_dict['encoder.' + k] = v
                state_dict = new_state_dict
                del new_state_dict
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model.encoder):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model.encoder, "active_adapter") or hasattr(model.encoder, "active_adapters")) and hasattr(
                model.encoder, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model.encoder, "active_adapters"):
                        active_adapters = model.encoder.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.encoder.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.encoder.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.encoder.set_adapter(active_adapter)
                    else:
                        model.encoder.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model.encoder, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        # 加载classifier
        model.load_classifier(resume_from_checkpoint)
        
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        wrong_dict = {}
        encoder_dict = {}
        classifier_dict = {}
        for k in state_dict.keys():
            if k.startswith(prefix):
                encoder_dict[k]=state_dict[k]
            elif k.startswith("classifier"):
                classifier_dict[k]=state_dict[k]
            else:
                wrong_dict[k]=state_dict[k]
        assert all(k.startswith(prefix) for k in encoder_dict.keys())
        encoder_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=encoder_dict, safe_serialization=self.args.save_safetensors
        )
        torch.save(classifier_dict, f'{output_dir}/classifier.pt')

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))