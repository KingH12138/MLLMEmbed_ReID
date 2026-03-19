from typing import Dict, Optional
import torch,os
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments, DataArguments
from src.model_utils import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, backbone2model
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.loss import make_loss


def get_lora_target_modules_method1(model, base_target_modules):
    # 定义视觉和语言模型的层间隔
    visual_interval = 4*4  # 视觉模型每4层选1层
    language_interval = 4*7  # 语言模型每4层选1层
    
    # 获取所有候选模块
    all_modules = []
    for name, _ in model.named_modules():
        # 检查是否是目标模块类型
        if any(target in name for target in base_target_modules):
            all_modules.append(name)

    # 筛选视觉模型模块 (visual.blocks)
    visual_blocks = [m for m in all_modules if 'visual.blocks' in m]
    selected_visual = []
    for i in range(0, len(visual_blocks), visual_interval):
        # 获取当前间隔的模块
        selected_visual.extend(visual_blocks[i:i+4])
    # 手动补全最后一层visual
    selected_visual.extend(visual_blocks[31*4:31*4+4])  # 假设visual.blocks.31是最后一层
    # 筛选语言模型模块 (model.layers)
    language_layers = [m for m in all_modules if 'model.layers' in m]
    selected_language = []
    for i in range(0, len(language_layers), language_interval):
        # 获取当前间隔的模块
        selected_language.extend(language_layers[i:i+7])
    # 手动补全最后一层model
    selected_language.extend(language_layers[27*7:27*7+7])  # 假设model.layers.27是最后一层
    # 合并选中的模块和其他必要模块（如merger, embed等）
    other_modules = [m for m in all_modules if 'visual.blocks' not in m and 'model.layers' not in m]
    final_target_modules = selected_visual + selected_language + other_modules
    
    return final_target_modules

def get_lora_target_modules_method2(base_model):
    """
    分层选择LoRA目标模块：
    1. qkv/proj/mlp.0/mlp.2/q_proj/k_proj/v_proj/o_proj：所有层均纳入
    2. fc1/fc2/gate_proj/up_proj/down_proj：仅倒数3层纳入
    """
    full_targets = ['qkv', 'proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    last3_targets = ['fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']
    special_targets = ['mlp.0', 'mlp.2']  # 特殊处理的模块名
    final_modules = []
    
    # 遍历所有模块名
    for name, _ in base_model.named_modules():
        # 规则1：全层包含的模块
        if any(suffix in name.split(".") for suffix in full_targets):
            final_modules.append(name)
        
        # 规则2：倒数3层包含的模块
        elif any(suffix in name.split(".") for suffix in last3_targets):
            # 提取层编号（适用于视觉和语言模型的层次结构）
            parts = name.split('.')
            for part in parts:
                if part.isdigit():  # 检测层编号
                    layer_num = int(part)
                    # 判断是否为倒数2层（视觉模型32层，语言模型28层）
                    if (layer_num >= 30 and 'visual.blocks' in name) or (layer_num >= 26 and 'model.layers' in name):
                        final_modules.append(name)
                    break
        elif any(name.endswith(suffix) for suffix in special_targets):
            # 特殊处理的模块名
            final_modules.append(name)
    # 去重并返回
    final_modules = list(set(final_modules))
    return final_modules

def get_lora_target_modules_method3(base_model):
    """
    分层选择LoRA目标模块：
    1. qkv/proj/mlp.0/mlp.2/q_proj/k_proj/v_proj/o_proj：所有层均纳入
    2. fc1/fc2/gate_proj/up_proj/down_proj：仅倒数3层纳入
    """
    full_targets = ['qkv', 'proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    last3_targets = ['fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']
    special_targets = ['mlp.0', 'mlp.2']  # 特殊处理的模块名
    final_modules = []
    
    # 遍历所有模块名
    for name, _ in base_model.named_modules():
        # 规则1：全层包含的模块
        if any(suffix in name.split(".") for suffix in full_targets):
            final_modules.append(name)
        
        # 规则2：倒数3层包含的模块
        elif any(suffix in name.split(".") for suffix in last3_targets):
            # 提取层编号（适用于视觉和语言模型的层次结构）
            parts = name.split('.')
            for part in parts:
                if part.isdigit():  # 检测层编号
                    layer_num = int(part)
                    # 判断是否为倒数4层（视觉模型32层，语言模型28层
                    if (layer_num >= 28 and 'visual.blocks' in name) or (layer_num >= 24 and 'model.layers' in name):
                        final_modules.append(name)
                    # 整除4为0但不是最后4层
                    if (layer_num%2==0 and 'visual.blocks' in name) or (layer_num%2==0 and 'model.layers' in name):
                        final_modules.append(name)
                    break
        elif any(name.endswith(suffix) for suffix in special_targets):
            # 特殊处理的模块名
            final_modules.append(name)
    # 去重并返回
    final_modules = list(set(final_modules))
    return final_modules

class MLLMReIDModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 num_classes: int,
                 id_loss_fn,
                 match_loss_fn,
                 triplet_loss_fn,
                 pooling: str = 'cls',
                 in_planes = 1536,
                 normalize: bool = False,
                 temperature: float = 1.0,
                 temperature_CL: float = 0.05
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.id_loss = id_loss_fn
        self.triplet_loss = triplet_loss_fn
        self.match_loss = match_loss_fn
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        # ID Loss + triplet loss
        self.in_planes = in_planes   # 根据模型结构修改
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # 关键：不学习 bias
        self.bottleneck.apply(self._init_weights)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.logit_scale = torch.ones([]) * (1 / temperature_CL) 
        self._keys_to_ignore_on_save = ['classifier.weight', 'bottleneck.weight', 'bottleneck.bias']
    
    # 权重初始化函数
    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def _pooling(self, last_hidden_state, attention_mask):
        """_summary_

        Args:
            last_hidden_state (_type_): (bs, seq_len, dim):示Transformer最后一层的隐藏状态（每个token的向量表示）。
            attention_mask (_type_): (bs, seq_len):标记哪些位置是有效token（1）或填充token（0）。
        Returns:
            _type_: _description_
        """
        if self.pooling == 'last' or self.pooling == 'eos':
            # 检查attention_mask的最后一列是否全为1（即所有序列的最后一个位置都是有效token）。
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:    # 若成立，说明序列是​​左填充​​（如BERT的输入格式，填充在左侧），此时直接取最后一个位置的隐藏状态即可。
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:   # 若不成立，说明填充在右侧或混合填充，需通过attention_mask定位实际序列的末尾。
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    
    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments=None, data_args: DataArguments=None,**kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
        # 根据training_args和model_args生成对应的loss_fn
        triplet_loss_fn, id_loss_fn, match_loss_fn = make_loss(training_args, data_args)
        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            # 使用自定义的lora-target-modules-get函数
            # lora_target_modules = get_lora_target_modules_method1(base_model, model_args.lora_target_modules.split(','))
            # lora_target_modules = get_lora_target_modules_method2(base_model)
            # lora_target_modules = get_lora_target_modules_method3(base_model)
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                # target_modules=lora_target_modules,
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            lora_model.print_trainable_parameters()
            # 打印出所有又lora的层：
            print_master("#"*25+"LoRA Display"+"#"*25)
            for name, module in lora_model.named_modules():
                if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                    print_master(f"LoRA层: {name}")  # 输出如"model.layers.0.self_attn.q_proj"
            print_master("#"*25+"LoRA Display"+"#"*25)
            model = cls(
                encoder=lora_model,
                id_loss_fn=id_loss_fn,
                triplet_loss_fn=triplet_loss_fn,
                match_loss_fn=match_loss_fn,  # sdm-loss
                num_classes=data_args.num_classes,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                id_loss_fn=id_loss_fn,
                triplet_loss_fn=triplet_loss_fn,
                match_loss_fn=match_loss_fn,  # sdm-loss
                num_classes=data_args.num_classes,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, training_args: TrainingArguments=None, data_args: DataArguments=None, **kwargs):
        # Loading the base model
        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')

        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL}:
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                config=config
            )
        elif model_args.model_backbone == PHI3V:
            # Loading the base model
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                checkpoint_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
        # 根据training_args和model_args生成对应的loss_fn
        triplet_loss_fn, id_loss_fn, match_loss_fn = make_loss(training_args, data_args)
        # Building the model on top of the base
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)

            merged_model = lora_model.merge_and_unload()
            model = cls(
                encoder=merged_model,
                id_loss_fn=id_loss_fn,
                triplet_loss_fn=triplet_loss_fn,
                match_loss_fn=match_loss_fn,  # sdm-loss
                num_classes=data_args.num_classes,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
            print_master("Loading Lora adaptor successfully.")
        else:
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                config=config
            )
            print_master("Loading base_model from checkpoint_path:{}.".format(checkpoint_path))
            model = cls(
                encoder=base_model,
                id_loss_fn=id_loss_fn,
                triplet_loss_fn=triplet_loss_fn,
                match_loss_fn=match_loss_fn,  # sdm-loss
                num_classes=data_args.num_classes,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
            print_master("Not lora-ft,loading full model.")
        # 加载classifier
        if model_args.checkpoint_path:
            classifier_path = f"{model_args.checkpoint_path}/classifier.pt"
            if os.path.exists(classifier_path):
                classifier_dict = torch.load(classifier_path)
                classifier_dict = {k.replace("classifier.", ""):v for k,v in classifier_dict.items()}
                try:
                    model.classifier.load_state_dict(classifier_dict)
                except:
                    print_master("There are something wrong when you attempt to load classifer's weight. If you are testing the reid model and you can ignore it.")
                print_master("Loaded classifier weights successfully.")
            else:
                print_master("No classifier weights found, using random initialization.")
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), f"{output_dir}/classifier.pt")
    
    def freeze_unfreeze_vit(self, flag: bool = True):
        """
        冻结/解冻(flag=True/False)vit的参数
        """
        for param in self.encoder.visual.parameters():
            if not flag:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if not flag:
            print_master("unFreeze vit parameters.")
        else:
            print_master("Freeze vit parameters.")
        
    def freeze_unfreeze_llm(self, flag: bool = True):
        """
        冻结/解冻(flag=True/False)llm的参数
        """
        for param in self.encoder.model.parameters():
            if not flag:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if not flag:
            print_master("unFreeze llm parameters.")
        else:
            print_master("Freeze llm parameters.")
    
    def freeze_unfreeze_classifier(self, flag: bool = True):
        """
        冻结/解冻(flag=True/False)classifier的参数
        """
        for param in self.classifier.parameters():
            if not flag:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if not flag:
            print_master("unFreeze classifier parameters.")
        else:
            print_master("Freeze classifier parameters.")
    
    def load_classifier(self, checkpoint_path: str):
        if checkpoint_path:
            classifier_path = f"{checkpoint_path}/classifier.pt"
            if os.path.exists(classifier_path):
                classifier_dict = torch.load(classifier_path)
                classifier_dict = {k.replace("classifier.", ""):v for k,v in classifier_dict.items()}
                self.classifier.load_state_dict(classifier_dict)
                print_master("Loaded classifier weights successfully.")
            else:
                print_master("No classifier weights found, using random initialization.")
    
    def encode_message(self, inputs, output_last_hidden_states=False, output_position_ids=False, output_attentions=False):
        outputs = self.encoder(**inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
        hidden_states = outputs.hidden_states[-1] # 取最后一层
        feat = self._pooling(hidden_states, inputs['attention_mask'])
        output_dict = {}
        output_dict['feat'] = feat
        output_dict['attention_mask'] = inputs['attention_mask']
        if output_last_hidden_states:
            output_dict['last_hidden_state'] = hidden_states
        if output_position_ids:
            output_dict['position_ids'] = outputs.position_ids
        if output_attentions:
            output_dict['attentions'] = outputs.attentions
        return output_dict
        
    # def forward(self, vlm_inputs, pid=None, camid=None, modality=None, *args, **kwargs):
    #     raw_feats = self.encode_message(vlm_inputs)['feat']
        
    #     # 聚合每个GPU上的标签和特征
    #     # if self.is_ddp:
    #     # feats = self._dist_gather_tensor(feats)
    #     # pid = self._dist_gather_tensor(pid)
    #     # camid = self._dist_gather_tensor(camid)

    #     feats = self.bottleneck(raw_feats)  # 加一层1D-BN让输出分布更加稳定
    #     score = self.classifier(feats)
    #     id_loss_dict = self.id_loss(score, pid) # id loss
    #     triplet_loss_dict = self.triplet_loss(raw_feats, pid)
    #     match_loss_dict = self.match_loss(raw_feats,pid,camid,modality,self.logit_scale)    # 匹配loss
    #     loss_dict = {**id_loss_dict, **match_loss_dict}
        
    #     # loss_dict['all_loss'] = id_loss_dict['id_loss'] + triplet_loss_dict['triplet_loss'] +\
    #     #     match_loss_dict['text_rgb'] + match_loss_dict['text_ir'] + match_loss_dict['text_sketch'] + \
    #     #     match_loss_dict['rgb_ir'] + match_loss_dict['rgb_sketch'] + \
    #     #     match_loss_dict['ir_sketch'] 
        
    #     loss_dict['all_loss'] = id_loss_dict['id_loss'] + triplet_loss_dict['triplet_loss'] + sum(match_loss_dict.values())
    
    #     # if self.is_ddp:
    #     #     loss_dict['all_loss'] = loss_dict['all_loss'] * self.world_size
        
    #     return loss_dict
    
    def forward(self, vlm_inputs, pid=None, camid=None, modality=None, *args, **kwargs):
        # 1. 提取原始特征
        raw_feats = self.encode_message(vlm_inputs)['feat']
        
        # 2. BNNeck 逻辑
        bn_feats = self.bottleneck(raw_feats)  
        score = self.classifier(bn_feats)
        
        # 3. 计算各项 Loss
        id_loss_dict = self.id_loss(score, pid)
        # 建议 Triplet 使用归一化后的特征以增强鲁棒性
        triplet_loss_dict = self.triplet_loss(raw_feats, pid) 
        match_loss_dict = self.match_loss(raw_feats, pid, camid, modality, self.logit_scale)
        
        # 4. 损失加权（防止 Match Loss 淹没 ID Loss）
        # 假设 match_loss 包含 6 个对齐项，取平均是一个稳妥的做法
        avg_match_loss = sum(match_loss_dict.values()) / len(match_loss_dict) if match_loss_dict else 0
        
        loss_dict = {**id_loss_dict, **match_loss_dict, **triplet_loss_dict}
        loss_dict['all_loss'] = id_loss_dict['id_loss'] + triplet_loss_dict['triplet_loss'] + avg_match_loss
        
        return loss_dict

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    

from transformers import CLIPVisionModel, CLIPTextModel, CLIPVisionConfig,CLIPTextConfig
import torch.nn.functional as F

class SmallMultimodalReID(nn.Module):
    def __init__(self, vision_model_name, text_model_name, load_pretrained_param=True, num_classes=17278):
        super().__init__()
        self.temperature_CL = 0.07
        self.logit_scale = torch.ones([]) * (1 / self.temperature_CL) 
        print(f"从{vision_model_name} {text_model_name}加载学生模型.")
        
        # 视觉编码器
        if load_pretrained_param:   # 加载预训练权重
            self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)
        else:
            vision_config = CLIPVisionConfig.from_pretrained(vision_model_name)
            text_config = CLIPTextConfig.from_pretrained(text_model_name)
            self.vision_encoder = CLIPVisionModel(vision_config)
            self.text_encoder = CLIPTextModel(text_config)
        
        # 输出层(与教师模型输出维度对齐)
        # self.visual_output_proj = nn.Linear(1024, 1536)
        self.rgb_output_proj = nn.Linear(1024, 1536)
        self.ir_output_proj = nn.Linear(1024, 1536)
        self.sketch_output_proj = nn.Linear(1024, 1536)
        self.text_output_proj = nn.Linear(768, 1536)
        
        
        self.classifier = nn.Linear(1536, num_classes, bias=False)

        # self.visual_norm = nn.LayerNorm(1536)
        # self.text_norm = nn.LayerNorm(1536)

        self.classifier = nn.Linear(1536, num_classes, bias=False)
    
    def encode_images(self, images_inputs, modality, normalize=True):
        # 处理三种图像模态
        image_features = self.vision_encoder(images_inputs).pooler_output  # 取[CLS] token
        # 输出投影
        if modality=="RGB":
            image_proj = self.rgb_output_proj(image_features)
        elif modality=="IR":
            image_proj = self.ir_output_proj(image_features)
        elif modality=="sketch":
            image_proj = self.sketch_output_proj(image_features)
        else:
            ValueError(f"modality:{modality}输入有问题")
        if normalize:
            # image_proj = self.visual_norm(image_proj)
            image_proj = F.normalize(image_proj, p=2, dim=1)
        return image_proj
    
    def encode_texts(self, text_inputs, normalize=True):
        # 处理文本模态
        text_features = self.text_encoder(**text_inputs).pooler_output
        # 输出投影
        text_proj = self.text_output_proj(text_features)
        if normalize:
            # text_proj = self.text_norm(text_proj)
            text_proj = F.normalize(text_proj, p=2, dim=1)
        return text_proj
    
    def forward(self, rgb_images_inputs, ir_images_inputs, sketch_images_inputs, text_inputs):
        # 训练过程encode 4种模态信息
        rgb_features = self.encode_images(rgb_images_inputs,'RGB') if rgb_images_inputs is not None else None
        ir_features = self.encode_images(ir_images_inputs,"IR") if ir_images_inputs is not None else None
        sketch_features = self.encode_images(sketch_images_inputs,"sketch") if sketch_images_inputs is not None else None
        text_features = self.encode_texts(text_inputs) if text_inputs is not None else None
        
        rgb_scores = self.classifier(rgb_features)
        ir_scores = self.classifier(ir_features)
        sketch_scores = self.classifier(sketch_features)
        text_scores = self.classifier(text_features)
        
        feat_dict = {
            "rgb": rgb_features,
            "ir": ir_features,
            "sketch": sketch_features,
            "text": text_features
        }

        score_dict = {
            "rgb": rgb_scores,
            "ir": ir_scores,
            "sketch": sketch_scores,
            "text": text_scores
        }

        return feat_dict, score_dict
    
from .clip_model import build_CLIP_from_openai_pretrained

class IRRA_CLIP(nn.Module):
    def __init__(self,teacher_dim=1536, num_classes=17278):
        super().__init__()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained('ViT-L/14', (280,140), 14)
        self.embed_dim = base_cfg['embed_dim']
        temperature_CL = 0.07
        self.logit_scale = torch.ones([]) * (1 / temperature_CL)
        # 视觉统一proj
        # self.visual_output_proj = nn.Linear(self.embed_dim, teacher_dim)
        # 分别proj
        self.rgb_output_proj = nn.Linear(self.embed_dim, teacher_dim)
        self.ir_output_proj = nn.Linear(self.embed_dim, teacher_dim)
        self.sketch_output_proj = nn.Linear(self.embed_dim, teacher_dim)
        self.text_output_proj = nn.Linear(self.embed_dim, teacher_dim)
        # 各自更强的proj
        # self.rgb_output_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dim, teacher_dim),
        # )
        # self.ir_output_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dim, teacher_dim),
        # )
        # self.sketch_output_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dim, teacher_dim),
        # )
        # self.text_output_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dim, teacher_dim),
        # )

        self.classifier = nn.Linear(1536, num_classes, bias=False)

    def encode_image(self, image, modality):
        x = self.base_model.encode_image(image)
        x = x[:, 0, :].float()
        # 输出投影
        if modality=="RGB":
            x = self.rgb_output_proj(x)
        elif modality=="IR":
            x = self.ir_output_proj(x)
        elif modality=="sketch":
            x = self.sketch_output_proj(x)
        else:
            ValueError(f"modality:{modality}输入有问题")
        x = F.normalize(x, p=2, dim=1)
        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        # 增加到老师模型向量空间的映射
        x = self.text_output_proj(x)
        # x = self.text_norm(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, rgb_images_inputs, ir_images_inputs, sketch_images_inputs, text_inputs):
        # 训练过程encode 4种模态信息
        rgb_features = self.encode_image(rgb_images_inputs,'RGB') if rgb_images_inputs is not None else None
        ir_features = self.encode_image(ir_images_inputs,"IR") if ir_images_inputs is not None else None
        sketch_features = self.encode_image(sketch_images_inputs,"sketch") if sketch_images_inputs is not None else None
        text_features = self.encode_text(text_inputs) if text_inputs is not None else None
        
        rgb_scores = self.classifier(rgb_features)
        ir_scores = self.classifier(ir_features)
        sketch_scores = self.classifier(sketch_features)
        text_scores = self.classifier(text_features)
        
        feat_dict = {
            "rgb": rgb_features,
            "ir": ir_features,
            "sketch": sketch_features,
            "text": text_features
        }

        score_dict = {
            "rgb": rgb_scores,
            "ir": ir_scores,
            "sketch": sketch_scores,
            "text": text_scores
        }

        return feat_dict, score_dict