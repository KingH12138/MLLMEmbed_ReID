import torch

def load_checkpoint(model, checkpoint_path):
    """安全加载检查点的函数"""
    # 加载保存的检查点
    if isinstance(checkpoint_path, str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = checkpoint_path
    # 如果文件直接保存了 state_dict，则直接使用；如果保存的是完整模型，可能需要通过键（如'state_dict'）来获取
    pretrained_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
    
    model_dict = model.state_dict()

    # 1. 筛选出键名和形状都匹配的参数
    filtered_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_pretrained_dict[k] = v
        else:
            print(f"跳过参数 '{k}' (原因: 键不存在或形状不匹配。期望: {model_dict.get(k, 'None')}, 获取: {v.shape if k in model_dict else 'N/A'})")

    # 2. 更新当前模型的参数字典
    model_dict.update(filtered_pretrained_dict)
    
    # 3. 非严格模式加载，允许缺失键（使用初始化值）和意外键
    load_result = model.load_state_dict(model_dict, strict=False)
    
    # 4. 打印加载摘要
    print("=== 参数加载摘要 ===")
    print(f"成功加载参数: {len(filtered_pretrained_dict)}")
    print(f"缺失的键 (使用初始化参数): {len(load_result.missing_keys)}")
    if load_result.missing_keys:
        print("详情:", load_result.missing_keys)
    print(f"意外的键 (被忽略): {len(load_result.unexpected_keys)}")
    if load_result.unexpected_keys:
        print("详情:", load_result.unexpected_keys)
    
    return model