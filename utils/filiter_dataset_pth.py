import torch
import pandas as pd
from tqdm import tqdm

data = torch.load("/hongbojiang/workdirs/mllmreid/teacher_feat_extract/teacher_feat_info.pth", weights_only=False)
rstp_df = pd.read_csv("/hongbojiang/datasets/aio_reid/rstpreid_train_regular.csv", encoding='utf-8')
rstp_pidcamid = list(zip(rstp_df['pid'], rstp_df['camid']))

cuhk_df = pd.read_csv("/hongbojiang/datasets/aio_reid/cuhkpedes_train_regular.csv", encoding='utf-8')
cuhk_pidcamid = list(zip(cuhk_df['pid'], cuhk_df['camid']))

icfg_df = pd.read_csv("/hongbojiang/datasets/aio_reid/icfgpedes_train_regular.csv", encoding='utf-8')
icfg_pidcamid = list(zip(icfg_df['pid'], icfg_df['camid']))

rstp_data = []
cuhk_data = []
icfg_data = []
for sample in tqdm(data):
    if sample['split']=='train' and (sample['pid'],sample['camid']) in rstp_pidcamid:
        rstp_data.append(sample)
    elif sample['split']=='train' and (sample['pid'],sample['camid']) in cuhk_pidcamid:
        cuhk_data.append(sample)
    elif sample['split']=='train' and (sample['pid'],sample['camid']) in icfg_pidcamid:
        icfg_data.append(sample)
    else:
        ValueError("pid camid异常")
torch.save(rstp_data, "/hongbojiang/workdirs/mllmreid/teacher_feat_extract/teacher_feat_info_rstp.pth")
torch.save(cuhk_data, "/hongbojiang/workdirs/mllmreid/teacher_feat_extract/teacher_feat_info_cuhk.pth")
torch.save(icfg_data, "/hongbojiang/workdirs/mllmreid/teacher_feat_extract/teacher_feat_info_icfg.pth")
