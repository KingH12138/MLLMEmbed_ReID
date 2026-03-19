import pandas as pd

rstpreid_df = pd.read_csv('/hongbojiang/datasets/aio_reid/rstpreid_aio.csv')
cuhkpedes_df = pd.read_csv('/hongbojiang/datasets/aio_reid/cuhkpedes_aio.csv')
icfgpedes_df = pd.read_csv('/hongbojiang/datasets/aio_reid/icfgpedes_aio.csv')

mode = "train"

rstpreid_df = rstpreid_df[rstpreid_df['split']==mode]
cuhkpedes_df = cuhkpedes_df[cuhkpedes_df['split']==mode]
icfgpedes_df = icfgpedes_df[icfgpedes_df['split']==mode]

# 看有多少个样本
print(len(rstpreid_df),len(cuhkpedes_df),len(icfgpedes_df))

def func0(df):
    """
    输出每个模态的样本数目
    """
    modality_names = ['text','RGB','IR','sketch']
    return [len(df[df['modality']==i]) for i in modality_names]

func0(rstpreid_df),func0(cuhkpedes_df),func0(icfgpedes_df)

print(rstpreid_df['pid'].nunique(),cuhkpedes_df['pid'].nunique(),icfgpedes_df['pid'].nunique())

print(rstpreid_df['camid'].nunique(),cuhkpedes_df['camid'].nunique(),icfgpedes_df['camid'].nunique())


def check_modality_per_camid(df, modality_col='modality', camid_col='camid', expected_modalities=4):
    """
    检查每个camid是否包含预期的模态数量（默认4种）。
    
    参数:
        df: 输入的DataFrame
        modality_col: 模态列名（默认'modality'）
        camid_col: 摄像头ID列名（默认'camid'）
        expected_modalities: 预期每个camid对应的模态数量（默认4）
    
    返回:
        dict: 包含验证结果和错误报告的字典
    """
    # 按camid分组并统计唯一模态数量
    modality_counts = df.groupby(camid_col)[modality_col].nunique()
    
    # 找出不符合条件的camid
    invalid_camids = modality_counts[modality_counts != expected_modalities]
    
    # 构建结果报告
    result = {
        'is_valid': invalid_camids.empty,
        'expected_modalities': expected_modalities,
        'invalid_camids': invalid_camids.to_dict(),
        'total_camids': len(modality_counts),
        'valid_camids': len(modality_counts) - len(invalid_camids)
    }
    
    return result

check_modality_per_camid(rstpreid_df),check_modality_per_camid(cuhkpedes_df),check_modality_per_camid(icfgpedes_df)

rstpreid_df = pd.read_csv('/hongbojiang/datasets/aio_reid/rstpreid_aio.csv')
cuhkpedes_df = pd.read_csv('/hongbojiang/datasets/aio_reid/cuhkpedes_aio.csv')
icfgpedes_df = pd.read_csv('/hongbojiang/datasets/aio_reid/icfgpedes_aio.csv')

rstpreid_df = rstpreid_df[rstpreid_df['split']==mode]
cuhkpedes_df = cuhkpedes_df[cuhkpedes_df['split']==mode]
icfgpedes_df = icfgpedes_df[icfgpedes_df['split']==mode]

print(rstpreid_df.columns)

cuhkpedes_pid_offset = rstpreid_df['pid'].nunique()
icfgpedes_pid_offset = rstpreid_df['pid'].nunique() + cuhkpedes_df['pid'].nunique()
cuhkpedes_df['pid'] += cuhkpedes_pid_offset
icfgpedes_df['pid'] += icfgpedes_pid_offset

cuhkpedes_camid_offset = rstpreid_df['camid'].nunique()
icfgpedes_camid_offset = rstpreid_df['camid'].nunique() + cuhkpedes_df['camid'].nunique()
cuhkpedes_df['camid'] += cuhkpedes_camid_offset
icfgpedes_df['camid'] += icfgpedes_camid_offset



merged_df = pd.concat([rstpreid_df, cuhkpedes_df, icfgpedes_df], ignore_index=True)
merged_df.to_csv(f"/hongbojiang/datasets/aio_reid/aio_{mode}.csv",  index=False, encoding='utf-8')

rstpreid_df.to_csv(f"/hongbojiang/datasets/aio_reid/rstpreid_{mode}_regular.csv",  index=False, encoding='utf-8')
cuhkpedes_df.to_csv(f"/hongbojiang/datasets/aio_reid/cuhkpedes_{mode}_regular.csv",  index=False, encoding='utf-8')
icfgpedes_df.to_csv(f"/hongbojiang/datasets/aio_reid/icfgpedes_{mode}_regular.csv",  index=False, encoding='utf-8')
