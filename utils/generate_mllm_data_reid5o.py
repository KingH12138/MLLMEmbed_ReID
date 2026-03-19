"""
20250423
根据现有的ReID数据集生成mllm数据集，保存格式如下：
dataset:
-images
-data.csv
"""
import os,json,random
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/data/jhb_data/codes/ReID5o_ORBench/ReID5o/datasets")
sys.path.append("/data/jhb_data/codes/ReID5o_ORBench/ReID5o")
from orbench import ORBench

def aio_to_train(output_path):
    split = "train"  
    reid_prompt = "Generate the image's embedding, focusing on age, gender, clothing, and biometric features."
    reid_text_prompt = "Generate this text's embedding, focusing on age, gender, clothing, and biometric features."
    # 导入ReID5o数据集类
    ds = ORBench(root="/data/jhb_data/datasets/")
    train_ds = ds.train # pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption
    data = []
    for ft in train_ds:
        # RGB
        rgb_prompt = "The modality of this item:{}. {}".format(
            'rgb', reid_prompt
        )
        data.append([ft[2], ft[0], rgb_prompt, "RGB", None, 1, ft[1], split])
        # IR
        ir_prompt = "The modality of this item:{}. {}".format(
            'ir', reid_prompt
        )
        data.append([ft[3], ft[0], ir_prompt, "IR", None, 1, ft[1], split])
        # Sketch
        sketch_prompt = "The modality of this item:{}. {}".format(
            'sketch', reid_prompt
        )
        data.append([ft[5], ft[0], sketch_prompt, "sketch", None, 1, ft[1], split])
        # Text
        text_prompt = "The modality of this item:{}. The caption is '{}'. {}".format(
            'text', ft[-1], reid_text_prompt
        )
        data.append([None, ft[0], text_prompt, "text", ft[-1], 0, ft[1], split])
        
    df = pd.DataFrame(data, columns=["image_path","pid","text","modality","ori_caption", "num_images","camid","split"])
    df.to_csv(output_path,  index=False, encoding='utf-8')
    return data

def aio_to_test(output_path):
    split = "test"  
    reid_prompt = "Generate the image's embedding, focusing on age, gender, clothing, and biometric features."
    reid_text_prompt = "Generate this text's embedding, focusing on age, gender, clothing, and biometric features."
    # 导入ReID5o数据集类
    ds = ORBench(root="/data/jhb_data/datasets/")
    test_ds = ds.test # pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption
    
    data = []
    g_num = len(test_ds['gallery_pids'])
    for i in range(g_num):
        # RGB
        rgb_prompt = "The modality of this item:{}. {}".format(
            'rgb', reid_prompt
        )
        data.append([test_ds['gallery_paths'][i], test_ds['gallery_pids'][i], rgb_prompt, "RGB", None, 1, None, split])

    # IR
    for item in test_ds['queries']['NIR']:
        # IR
        ir_prompt = "The modality of this item:{}. {}".format(
            'ir', reid_prompt
        )
        data.append([item[1], item[0], ir_prompt, "IR", None, 1, None, split])
    # Sketch
    for item in test_ds['queries']['SK']:
        sketch_prompt = "The modality of this item:{}. {}".format(
            'sketch', reid_prompt
        )
        data.append([item[1], item[0], sketch_prompt, "sketch", None, 1, None, split])
    # Text
    for item in test_ds['queries']['TEXT']:
        text_prompt = "The modality of this item:{}. The caption is '{}'. {}".format(
            'text', item[1], reid_text_prompt
        )
        data.append([None, item[0], text_prompt, "text", item[1], 0, None, split])

    df = pd.DataFrame(data, columns=["image_path","pid","text","modality","ori_caption", "num_images","camid","split"])
    df.to_csv(output_path,  index=False, encoding='utf-8')
    return data

if __name__ == "__main__":
    # 这里合并三个数据集
    # data1 = aio_to_train('/data/jhb_data/datasets/ORBench/reid5o_aio_train.csv')
    # print(len(data1), len(data1)/4)
    data2 = aio_to_test('/data/jhb_data/datasets/ORBench/reid5o_aio_test.csv')
    print(len(data2))