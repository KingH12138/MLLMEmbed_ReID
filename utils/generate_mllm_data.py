"""
20250423
根据现有的ReID数据集生成mllm数据集，保存格式如下：
dataset:
-images
-data.csv
"""
import sys, os, csv
sys.path.append(os.path.abspath("./"))

# from reid_datasets.market1501 import Market1501
# from reid_datasets.msmt17 import MSMT17
# from reid_datasets.dukemtmcreid import DukeMTMCreID

import os,json,random
import pandas as pd

# def dukemtmc_to_mllm(reid_data_dir,output_dir):
#     dataset = DukeMTMCreID(reid_data_dir)
#     # 获取训练集
#     train_list = dataset.train  # ele:absolute_path,pid,camid,1
#     query_list = dataset.query
#     gallery_list = dataset.gallery
#     # 获取最大的pid和camid
#     max_pid = max([x[1] for x in train_list])
#     max_camid = max([x[2] for x in train_list])
#     # 将其转为train_list转为包含absolute_path,pid,camid三个字段的csv文件
#     with open(output_dir + "/dukemtmcreid_train.csv", "w") as f:
#         f.write("image_path,pid,camid,type\n")
#         for x in train_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},train\n")
#         for x in query_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},query\n")
#         for x in gallery_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},gallery\n")
#     return max_pid, max_camid

# def market1501_to_mllm(reid_data_dir,output_dir):
#     dataset = Market1501(reid_data_dir)
#     # 获取训练集
#     train_list = dataset.train  # ele:absolute_path,pid,camid,1
#     query_list = dataset.query
#     gallery_list = dataset.gallery
#     # 获取最大的pid和camid
#     max_pid = max([x[1] for x in train_list])
#     max_camid = max([x[2] for x in train_list])
#     # 将其转为train_list转为包含absolute_path,pid,camid三个字段的csv文件
#     with open(output_dir + "/market1501_train.csv", "w") as f:
#         f.write("image_path,pid,camid,type\n")
#         for x in train_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},train\n")
#         for x in query_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},query\n")
#         for x in gallery_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},gallery\n")
#     return max_pid, max_camid

# def msmt17_to_mllm(reid_data_dir,output_dir):
#     dataset = MSMT17(reid_data_dir)
#     # 获取训练集
#     train_list = dataset.train  # ele:absolute_path,pid,camid,1
#     query_list = dataset.query
#     gallery_list = dataset.gallery
#     # 获取最大的pid和camid
#     max_pid = max([x[1] for x in train_list])
#     max_camid = max([x[2] for x in train_list])
#     # 将其转为train_list转为包含absolute_path,pid,camid三个字段的csv文件
#     with open(output_dir + "/msmt17_train.csv", "w") as f:
#         f.write("image_path,pid,camid,type\n")
#         for x in train_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},train\n")
#         for x in query_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},query\n")
#         for x in gallery_list:
#             f.write(f"{x[0]},{x[1]},{x[2]},gallery\n")
#     return max_pid, max_camid


import tqdm
import pandas as pd

def aio_to_mllm(aio_dataset_dir,json_name,output_path, two_caption=False):  
    with open(os.path.join(aio_dataset_dir, json_name), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    reid_prompt = "Generate the image's embedding, focusing on age, gender, clothing, and biometric features."
    reid_text_prompt = "Generate this text's embedding, focusing on age, gender, clothing, and biometric features."
    rgb_dir = os.path.join(aio_dataset_dir, 'imgs')
    ir_dir = os.path.join(aio_dataset_dir, 'imgs-IR')
    if aio_dataset_dir.endswith('CUHK-PEDES'):
        sketch_dir = os.path.join(aio_dataset_dir, 'imgs-sketch2')  # cuhk是img-sketch2, icfg是img-sketch
    else:
        sketch_dir = os.path.join(aio_dataset_dir, 'imgs-sketch')
    data = []
    real2label = {}
    start_id = 0
    image_id = 0    # 每个数据的4个模态共享一个camid用于后续的检索
    for item in tqdm.tqdm(json_data):  # 遍历每个数据的4个模态
        if aio_dataset_dir.endswith("CUHK-PEDES"):
            id = item['id']-1
        else:
            id = item['id']
        if 'file_path' not in item:
            img_rel_path = item['img_path']
        else:
            img_rel_path = item['file_path']
        caption0 = item['captions'][0]
        if two_caption:
            caption1 = item['captions'][1]
        rgb_img_path = os.path.join(rgb_dir, img_rel_path)
        ir_img_path = os.path.join(ir_dir, img_rel_path)
        sketch_img_path = os.path.join(sketch_dir, img_rel_path)
        if two_caption:
            if not (os.path.exists(rgb_img_path) and os.path.exists(ir_img_path) and os.path.exists(sketch_img_path) and caption0 is not None and caption1 is not None):continue  # 如果不是四个模态齐全就跳过
        else:
            if not (os.path.exists(rgb_img_path) and os.path.exists(ir_img_path) and os.path.exists(sketch_img_path) and caption0 is not None):continue  # 如果不是四个模态齐全就跳过
        rgb_prompt = "The modality of this item:{}. {}".format(
            'rgb', reid_prompt
        )
        ir_prompt = "The modality of this item:{}. {}".format(
            'ir', reid_prompt
        )
        sketch_prompt = "The modality of this item:{}. {}".format(
            'sketch', reid_prompt
        )
        text_prompt0 = "The modality of this item:{}. The caption is '{}'. {}".format(
            'text', caption0, reid_text_prompt
        )
        if two_caption:
            text_prompt1 = "The modality of this item:{}. The caption is '{}'. {}".format(
                'text', caption1, reid_text_prompt
            )
        
        if id not in real2label:
            real2label[id] = start_id
            start_id+=1
        
        data.append([
            rgb_img_path, real2label[id], rgb_prompt, 'RGB', None, 1, image_id, item['split']
        ])
        data.append([
            ir_img_path, real2label[id], ir_prompt, 'IR', None, 1, image_id, item['split']
        ])
        data.append([
            sketch_img_path, real2label[id], sketch_prompt, 'sketch', None, 1, image_id, item['split']
        ])
        data.append([
                None, real2label[id], text_prompt0, 'text', caption0, 0, image_id, item['split']
            ])
        # 两个caption
        if two_caption:
            data.append([
                None, real2label[id], text_prompt1, 'text', caption1, 0, image_id, item['split']
            ])
        image_id+=1
    
    df = pd.DataFrame(data, columns=["image_path","pid","text","modality","ori_caption", "num_images","camid","split"])
    df.to_csv(output_path,  index=False, encoding='utf-8')
    return data
        
if __name__ == "__main__":
    # 这里合并三个数据集
    data0 = aio_to_mllm('/Youtu-VITA/hongbojiang/datasets/aio_reid/CUHK-PEDES','reid_raw.json','/Youtu-VITA/hongbojiang/datasets/aio_reid/cuhkpedes_aio_train.csv')
    data1 = aio_to_mllm('/Youtu-VITA/hongbojiang/datasets/aio_reid/ICFG-PEDES','ICFG-PEDES.json','/Youtu-VITA/hongbojiang/datasets/aio_reid/icfgpedes_aio_train.csv')
    data2 = aio_to_mllm('/Youtu-VITA/hongbojiang/datasets/aio_reid/RSTPReid','data_captions.json','/Youtu-VITA/hongbojiang/datasets/aio_reid/rstpreid_aio_train.csv')
    print("生成完毕")