from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os, re, json, random
import pandas as pd
from torch.jit import isinstance

from src.model_utils import PHI3V, vlm_image_tokens
from src.utils import print_master, print_rank

# from datasets import Dataset  # @ruimeng, still buggy
from torch.utils.data import Dataset


def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image


class TrainTextImageDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        for subset in data_args.subset_name:
            subset_data = load_dataset(self.data_args.dataset_name, subset, split=data_args.split_name)
            train_data.append(subset_data[0])
        self.train_data = concatenate_datasets(train_data)

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        if 'neg_text' in self.train_data.column_names:
            neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
        else:
            # neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
            # 20250421-debug
            neg_texts, neg_image_paths = '',''
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            neg_texts = [neg_texts]
            neg_image_paths = [neg_image_paths]
        _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images = [], [], [], [], [], []
        backbone = self.model_args.model_backbone
        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            _neg_texts.append(neg_text)
            _neg_images.append(neg_image)

        return {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images,
                "neg_text": _neg_texts, "neg_image": _neg_images}


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])

        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data


class FlickrDataset(Dataset):
    def __init__(self, modality, model_backbone):
        self.model_backbone = model_backbone
        self.modality = modality
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()
        else:
            self.eval_data, self.image_names = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def get_image_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = "<|image_1|> Find an image caption describing the given image."
        # inst = "<|image_1|> Represent the given image for image caption retrieval."
        # t2i
        # inst = "<|image_1|> Represent the given image."

        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = ""
        # t2i
        # inst = "Retrieve an image that matches the given caption: "
        # inst = "Find me an everyday image that matches the given caption."  # MSCOCO t2i
        for row in self.raw_data:
            for caption in row["caption"]:
                # eval_data.append((caption, None))
                eval_data.append((inst + caption, None))
                image_names.append(row["filename"])
        return eval_data, image_names


from src.arguments import DataArguments, ModelArguments
from torchvision import transforms
import time

class MLLMReIDTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        self.data_args = data_args
        self.model_args = model_args
        print_rank(f"Loading {data_args.dataset_name}. Using meta file:{data_args.dataset_meta}")
        self.data = pd.read_csv(data_args.dataset_meta, encoding='utf-8')
        self.train_data = self.data[self.data['type']=="train"]
        self.column_names = ['image_path','pid','camid','type']    # 这里需要与csv文件中的列名一致
        # assert self.check_id_consistency(self.data), "ReID数据集中存在数据集ID和标签ID不一致情况，请检查。"
        self.train_transform = transforms.Compose([
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(data_args.flip_prob),
            transforms.RandomVerticalFlip(data_args.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
            transforms.RandomErasing(p=data_args.erasing_prob),
        ])
        
    # def check_id_consistency(self,df:pd.DataFrame):        
    #     # 提取image_name并解析数据集ID
    #     df['image_name'] = df['image_path'].apply(lambda x: os.path.basename(x))
        
    #     # 解析数据集中的pid和camid
    #     df['dataset_pid'] = df['image_name'].apply(lambda x: int(x.split('_')[0]))
    #     def extract_camid(image_name):
    #         # 使用正则表达式匹配c后面的数字（直到下划线或s）
    #         match = re.search(r'c(\d+)[_s]', image_name)
    #         if match:
    #             return int(match.group(1))
    #         return None  # 或根据需求返回默认值
    #     df['dataset_camid'] = df['image_name'].apply(extract_camid)
        
    #     # 检查pid一致性
    #     pid_check = df.groupby('dataset_pid')['pid'].nunique()
    #     inconsistent_pids = pid_check[pid_check > 1]
        
    #     # 检查camid一致性
    #     # 对于每个dataset_pid和dataset_camid组合，检查对应的camid是否唯一
    #     camid_check = df.groupby(['dataset_pid', 'dataset_camid'])['camid'].nunique()
    #     inconsistent_camids = camid_check[camid_check > 1]
        
    #     # 打印结果
    #     if not inconsistent_pids.empty:
    #         print("发现不一致的pid标签:")
    #         for pid in inconsistent_pids.index:
    #             print(f"数据集pid {pid} 对应多个标签pid: {df[df['dataset_pid'] == pid]['pid'].unique()}")
        
    #     if not inconsistent_camids.empty:
    #         print("\n发现不一致的camid标签:")
    #         for (pid, camid), _ in inconsistent_camids.items():
    #             print(f"数据集pid {pid} 的camid {camid} 对应多个标签camid: {df[(df['dataset_pid'] == pid) & (df['dataset_camid'] == camid)]['camid'].unique()}")
        
    #     return inconsistent_pids.empty and inconsistent_camids.empty
        
    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.train_transform(image)
        return image
        

    def __getitem__(self, data_idx):
        image_path, pid, camid, data_type = self.train_data.iloc[data_idx][self.column_names]
        image = self._get_image(image_path) # 读取图片并初步resize，其他信息不变
        if isinstance(data_idx, int):
            image = image
            pid = pid
            camid = camid
            data_type = data_type
        return {"image": image, "pid": pid, "camid": camid, "type":data_type}
    
    
class MLLMReIDTestDataset(MLLMReIDTrainDataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        super().__init__(data_args, model_args)
        self.test_data = self.data[self.data['type'].isin(['query', 'gallery'])]
        self.val_transform = transforms.Compose([
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
        ])
        
    def __len__(self):
        return len(self.test_data)

    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.val_transform(image)
        return image

    def __getitem__(self, data_idx):
        image_path, pid, camid, data_type = self.test_data.iloc[data_idx][self.column_names]
        image = self._get_image(image_path) # 读取图片并初步resize，其他信息不变
        return {"image": image, "pid": pid, "camid": camid, "type":data_type}
    
from prettytable import PrettyTable
class AIOReIDTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        self.data_args = data_args
        self.model_args = model_args
        print_rank(f"Loading {data_args.dataset_name}. Using meta file:{data_args.dataset_meta}")
        self.data = pd.read_csv(data_args.dataset_meta, encoding='utf-8')
        self.train_data = self.data[self.data['split']=="train"]
        self.column_names = ['image_path','pid','text','modality', "ori_caption",'num_images', 'camid','split']    # 这里需要与csv文件中的列名一致
        self.train_transform = transforms.Compose([
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(data_args.flip_prob),
            transforms.RandomVerticalFlip(data_args.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
            transforms.RandomErasing(p=data_args.erasing_prob),
        ])
        self.show_subset()

    def __len__(self):
        return len(self.train_data)

    def show_subset(self):
        # 1. 按 split 分组统计
        split_counts = self.data['split'].value_counts().to_dict()

        # 2. 按 split 和 modality 分组统计
        split_modality_counts = self.data.groupby(['split', 'modality']).size().unstack(fill_value=0)

        # 3. 创建 PrettyTable 对象
        table = PrettyTable()
        table.field_names = ["Split", "Total Samples", "Modality Counts (详细分布)"]

        # 4. 填充表格数据
        for split in ['train', 'val', 'test']:
            total = split_counts.get(split, 0)
            modality_details = []
            
            # 获取当前 split 的 modality 分布
            if split in split_modality_counts.index:
                for modality, count in split_modality_counts.loc[split].items():
                    modality_details.append(f"{modality}: {count}")
            
            # 添加到表格行
            table.add_row([
                split,
                total,
                "\n".join(modality_details) if modality_details else "N/A"
            ])

        # 5. 设置表格样式
        table.align["Split"] = "l"
        table.align["Total Samples"] = "r"
        table.align["Modality Counts (详细分布)"] = "l"
        table.border = True
        table.header = True
        table.hrules = 1  # 所有行添加分隔线

        # 6. 打印表格
        print("数据集统计信息：")
        print(table)
    
    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = Image.open(img_path)
        image = self.train_transform(image)
        
        return image

    def __getitem__(self, data_idx):
        image_path, pid, text, modality, _, num_images, camid, split  = self.train_data.iloc[data_idx][self.column_names]
        if isinstance(image_path, str):
            image = self._get_image(image_path)
        else:
            image = None
        return {"image_path":image_path, "image": image, "pid": pid,
                "text":text, 'modality':modality, 'num_images':num_images, 'camid':camid, "split":split}
    
    
class AIOReIDValidDataset(AIOReIDTrainDataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        super().__init__(data_args, model_args)
        val_data = self.data[self.data['split']=="val"]
        if len(val_data)!=0:
            self.val_data = val_data
        else:
            self.val_data = self.data[self.data['split']=="test"]   # 没有验证集就用测试集替代
        self.val_transform = transforms.Compose([
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
        ])
        self.show_subset()

    def __len__(self):
        return len(self.val_data)

    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.val_transform(image)
        return image

    def __getitem__(self, data_idx):
        image_path, pid, text, modality, num_images, camid, split  = self.val_data.iloc[data_idx][self.column_names]
        if isinstance(image_path, str):
            image = self._get_image(image_path)
        else:
            image = None
        return {"image_path":image_path, "image": image, "pid": pid,
                "text":text, 'modality':modality, 'num_images':num_images, 'camid':camid, "split":split}
    
class AIOReIDTestDataset(AIOReIDTrainDataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        super().__init__(data_args, model_args)
        self.test_data = self.data[self.data['split']=="test"]
        self.test_transform = transforms.Compose([
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
        ])
        self.show_subset()

    def __len__(self):
        return len(self.test_data)

    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.test_transform(image)
        return image

    def __getitem__(self, data_idx):
        image_path,pid,text,modality,ori_caption,num_images,camid,split  = self.test_data.iloc[data_idx][self.column_names]
        if isinstance(image_path, str):
            image = self._get_image(image_path)
        else:
            image = None
        return {"image_path":image_path, "image": image, "pid": pid,
                "text":text, 'modality':modality, 'num_images':num_images, 'camid':camid, "split":split}
        
        
def pad_to_square(img):
    """将图像填充为正方形（短边补0）"""
    # 用的学术model是clip，输入size固定，为了resize防止拉伸变形。
    w, h = img.size
    max_side = max(w, h)
    padding = (
        (max_side - w) // 2,  # 左侧填充
        (max_side - h) // 2,   # 顶部填充
        (max_side - w + 1) // 2,  # 右侧填充
        (max_side - h + 1) // 2   # 底部填充
    )
    return transforms.functional.pad(img, padding, fill=0)  # fill=0表示黑色填充


class DistillMLLMReIDTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        self.data_args = data_args
        self.model_args = model_args
        print_rank(f"Loading {data_args.dataset_name}. Using meta file:{data_args.dataset_meta}")
        self.data = pd.read_csv(data_args.dataset_meta, encoding='utf-8')
        self.train_data = self.data[self.data['split']=="train"]
        self.column_names = ['image_path','pid','text','modality', 'ori_caption','num_images', 'camid','split']    # 这里需要与csv文件中的列名一致
        self.train_transform = transforms.Compose([
            # transforms.Lambda(pad_to_square),  # 自定义填充函数
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(data_args.flip_prob),
            transforms.RandomVerticalFlip(data_args.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
            transforms.RandomErasing(p=data_args.erasing_prob),
        ])
        self.show_subset()

    def __len__(self):
        return len(self.train_data)

    def show_subset(self):
        # 1. 按 split 分组统计
        split_counts = self.data['split'].value_counts().to_dict()

        # 2. 按 split 和 modality 分组统计
        split_modality_counts = self.data.groupby(['split', 'modality']).size().unstack(fill_value=0)

        # 3. 创建 PrettyTable 对象
        table = PrettyTable()
        table.field_names = ["Split", "Total Samples", "Modality Counts (详细分布)"]

        # 4. 填充表格数据
        for split in ['train', 'val', 'test']:
            total = split_counts.get(split, 0)
            modality_details = []
            
            # 获取当前 split 的 modality 分布
            if split in split_modality_counts.index:
                for modality, count in split_modality_counts.loc[split].items():
                    modality_details.append(f"{modality}: {count}")
            
            # 添加到表格行
            table.add_row([
                split,
                total,
                "\n".join(modality_details) if modality_details else "N/A"
            ])

        # 5. 设置表格样式
        table.align["Split"] = "l"
        table.align["Total Samples"] = "r"
        table.align["Modality Counts (详细分布)"] = "l"
        table.border = True
        table.header = True
        table.hrules = 1  # 所有行添加分隔线

        # 6. 打印表格
        print("数据集统计信息：")
        print(table)
    
    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.train_transform(image)
        return image

    def __getitem__(self, data_idx):
        image_path, pid, text, modality, ori_caption, num_images, camid, split  = self.train_data.iloc[data_idx][self.column_names]
        if isinstance(image_path, str):
            image = self._get_image(image_path)
        else:
            image = None
        return {"image_path":image_path, "image": image, "pid": pid,
                "text":text, 'modality':modality, 'ori_caption':ori_caption,'num_images':num_images, 'camid':camid, "split":split}


class DistillMLLMReIDTestDataset(DistillMLLMReIDTrainDataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        super().__init__(data_args, model_args)
        self.test_data = self.data[self.data['split']=="test"]
        self.test_transform = transforms.Compose([
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
        ])
        self.show_subset()

    def __len__(self):
        return len(self.test_data)

    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.test_transform(image)
        return image

    def __getitem__(self, data_idx):
        image_path, pid, text, modality, ori_caption, num_images, camid, split  = self.test_data.iloc[data_idx][self.column_names]
        if isinstance(image_path, str):
            image = self._get_image(image_path)
        else:
            image = None
        return {"image_path":image_path, "image": image, "pid": pid,
                "text":text, 'modality':modality, 'ori_caption':ori_caption,'num_images':num_images, 'camid':camid, "split":split}

import torch

class OfflineDistillMLLMReIDTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        self.data_args = data_args
        self.model_args = model_args
        print_rank(f"Loading {data_args.dataset_name}. Using meta file:{data_args.dataset_meta}")
        self.data = torch.load("/hongbojiang/workdirs/mllmreid/teacher_feat_extract/teacher_feat_info.pth", weights_only=False)
        self.train_data = [i for i in self.data if i['split']=='train']
        self.column_names = self.train_data[0].keys()
        print("样本属性:",self.column_names)
        self.train_transform = transforms.Compose([
            # transforms.Lambda(pad_to_square),  # 自定义填充函数(resize成矩形就不建议设置)
            transforms.Resize(data_args.resize, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(data_args.flip_prob),
            transforms.RandomVerticalFlip(data_args.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_args.pixel_mean, std=data_args.pixel_std),
            transforms.RandomErasing(p=data_args.erasing_prob),
        ])
        self.show_subset()

    def __len__(self):
        return len(self.train_data)

    def show_subset(self):
        print(f"离线数据集样本数:{len(self.train_data)}，前5条样本展示:\n{self.train_data[:5]}")
    
    def _get_image(self, img_path, max_retries=50):
        if not img_path:
            return None
        image = None
        for i in range(max_retries):
            try:
                image = Image.open(img_path)
                break
            except (BrokenPipeError, IOError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 * (2 ** i))  # 指数退避策略
        image = self.train_transform(image)
        return image

    def __getitem__(self, data_idx):
        sample_dict = self.train_data[data_idx]
        image_path = sample_dict['image_path']
        if isinstance(image_path, str):
            image = self._get_image(image_path)
        else:
            image = None
        sample_dict['image'] = image
        return sample_dict