"""
20250423
根据现有的ReID数据集生成mllm数据集，保存格式如下：
dataset:
-images
-data.csv
"""
sys.path.append(os.path.abspath("./"))
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
    data0 = aio_to_mllm('/hongbojiang/datasets/aio_reid/CUHK-PEDES','reid_raw.json','/hongbojiang/datasets/aio_reid/cuhkpedes_aio_train.csv')
    data1 = aio_to_mllm('/hongbojiang/datasets/aio_reid/ICFG-PEDES','ICFG-PEDES.json','/hongbojiang/datasets/aio_reid/icfgpedes_aio_train.csv')
    data2 = aio_to_mllm('/hongbojiang/datasets/aio_reid/RSTPReid','data_captions.json','/hongbojiang/datasets/aio_reid/rstpreid_aio_train.csv')
    print("生成完毕")
