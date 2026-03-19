"""
写一个测试脚本，测试微调后的Qwen2VL-2B-ReID(sfted_rstpreid)模型在market、msmt17等数据集上的性能
"""
import sys
sys.path.append('/home/tan/jhb/VLM2Vec')

import torch

import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from utils.reid_eval_tools import eval_func, euclidean_distance, eval_func_TBRlike, eval_func_with_query_ap

def get_lowest_rank1_pids(distmat, q_pids, g_pids, q_camids, g_camids, n=5):
    """
    返回Rank-1准确率最低的前n个PID及其准确率
    参数:
        distmat: 距离矩阵 (num_q x num_g)
        q_pids: 查询集PID列表 (num_q,)
        g_pids: 图库集PID列表 (num_g,)
        q_camids: 查询集摄像头ID列表 (num_q,)
        g_camids: 图库集摄像头ID列表 (num_g,)
        n: 需要返回的PID数量
    返回:
        list: [(pid, rank1_acc), ...] 按准确率升序排列
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # 统计每个PID的Rank-1准确率
    pid_rank1 = {}
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # 排除同摄像头同PID的图库样本
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # 计算当前查询的Rank-1结果
        rank1_result = matches[q_idx][keep][0]  # 取排序后的第一个结果
        if q_pid not in pid_rank1:
            pid_rank1[q_pid] = {'correct': 0, 'total': 0}
        pid_rank1[q_pid]['correct'] += rank1_result
        pid_rank1[q_pid]['total'] += 1
    
    # 计算每个PID的Rank-1准确率
    pid_acc = [
        (pid, counts['correct'] / counts['total']) 
        for pid, counts in pid_rank1.items()
    ]
    
    # 按准确率升序排序并返回前n个
    pid_acc_sorted = sorted(pid_acc, key=lambda x: x[1])
    return pid_acc_sorted[:n]

def display_results_from_test_log(test_log_path,rankn):
    info_dict = torch.load(test_log_path)
    feats = info_dict['query']['feats']
    gallery_feats = info_dict['gallery']['feats']
    q_pids = info_dict['query']['pids']
    g_pids = info_dict['gallery']['pids']
    q_camids = info_dict['query']['camids']
    g_camids = info_dict['gallery']['camids']
    
    feats = torch.cat([i.unsqueeze(0) for i in feats], dim=0)
    gallery_feats = torch.cat([i.unsqueeze(0) for i in gallery_feats], dim=0)
    
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)
    distmat = euclidean_distance(feats, gallery_feats)
    lowest_pids = get_lowest_rank1_pids(
        distmat, q_pids, g_pids, q_camids, g_camids, n=rankn
    )
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    print("mAP: {:.3%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
    print("Rank-1准确率最低的PID列表:")
    for pid, acc in lowest_pids:
        print(f"PID {pid}: Rank-1 Acc = {acc:.3%}")
    return lowest_pids

def select_one_from_every_pid(labels):
    unique_labels = torch.unique(labels)  # 获取唯一标签 [0, 1, 2, 3]
    mask = torch.zeros_like(labels, dtype=torch.bool)  # 初始化全False的mask

    for label in unique_labels:
        # 找到当前标签的所有位置
        indices = (labels == label).nonzero().squeeze()
        # 随机选一个位置，并设置mask为True
        if indices.ndim == 0:
            mask[indices.item()] = True
        else:
            selected_idx = torch.randint(0, len(indices), (1,))
            mask[indices[selected_idx]] = True    
    return mask

from src.utils import print_master
def display_aio_results_from_test_log(test_log_path, rankn=50, save_rank_index=True,worst_query_num=0):
    print_master(f"从{test_log_path}获取测试结果...")
    output_data = torch.load(test_log_path, map_location='cpu')
    feats = output_data['feats']  # list of tensors
    num_sample = len(feats)
    print_master("读取到的样本数：{}.".format(num_sample))
    camids = torch.tensor(output_data['camids'],dtype=torch.long)  # list of tensors
    pids = torch.tensor(output_data['pids'], dtype=torch.long)    # list of tensors
    modality = output_data['modality']  # list of strings
    print_master(np.unique(np.array(modality)))
    
    # 将列表中的tensor拼接成一个大的tensor
    feats = torch.cat([i.unsqueeze(0) for i in feats], dim=0)
    pids = torch.tensor(pids)  # 假设pids中的tensor都是1D的
    
    del output_data

    # 筛选各模态特征和标签
    modality_pairs = [
        ('IR', 'RGB'),
        ('text', 'RGB'), ('sketch', 'RGB'),
        # ('text', 'RGB')
    ]

    results = {}
    for q_mod, g_mod in modality_pairs:
        # 筛选query和gallery的模态
        q_mask = torch.tensor([m == q_mod for m in modality], dtype=torch.bool)
        g_mask = torch.tensor([m == g_mod for m in modality], dtype=torch.bool)
        
        q_feats_m = feats[q_mask]
        g_feats_m = feats[g_mask]
        q_pids_m = pids[q_mask]
        g_pids_m = pids[g_mask]
        if len(camids)!=0:
            q_camids_m = camids[q_mask]
            g_camids_m = camids[g_mask]
        
        if len(q_pids_m) == 0 or len(g_pids_m) == 0:
            print_master(f"跳过模态对 {q_mod}->{g_mod}，因为没有有效样本")
            continue
        print_master(f"query:{len(q_pids_m)} gallery:{len(g_pids_m)}")
        def has_common_elements(tensor1, tensor2):
            set1 = set(tensor1.tolist())
            set2 = set(tensor2.tolist())
            return len(set1 & set2)
        
        if not has_common_elements(q_pids_m, g_pids_m):
            continue  # 如果没有共同PID则跳过
        # 计算距离矩阵
        q_feats_m = torch.nn.functional.normalize(q_feats_m, dim=1, p=2)
        g_feats_m = torch.nn.functional.normalize(g_feats_m, dim=1, p=2)
        distmat = euclidean_distance(q_feats_m, g_feats_m)
        indices = np.argsort(distmat, axis=1)  # shape: (num_q, num_g)
        # 计算CMC和mAP
        if len(camids)!=0:
            if q_mod=='text' and g_mod=='RGB':
                cmc, mAP, mINP, query_ap = eval_func_with_query_ap(distmat, q_pids_m.numpy(), g_pids_m.numpy(), q_camids_m.numpy(), g_camids_m.numpy(), set=0)
            else:
                cmc, mAP, mINP, query_ap = eval_func_with_query_ap(distmat, q_pids_m.numpy(), g_pids_m.numpy(), q_camids_m.numpy(), g_camids_m.numpy(), set=1)
        else:
            cmc, mAP, mINP, query_ap = eval_func_with_query_ap(distmat, q_pids_m.numpy(), g_pids_m.numpy(), None, None, set=0)
        print_master(f"模态对 {q_mod}->{g_mod} 的评估结果:")
        print_master("mAP: {:.4%}".format(mAP))
        for r in [1, 5, 10]:
            print_master("CMC curve, Rank-{:<3}:{:.4%}".format(r, cmc[r - 1]))
        print_master("mINP: {:.4%}".format(mINP))
        if worst_query_num > 0:
            # 按AP值排序，找出AP最低的query
            worst_indices = np.argsort(query_ap)[:worst_query_num]
            worst_aps = query_ap[worst_indices]
            worst_pids = q_pids_m.numpy()[worst_indices]
            worst_camids = q_camids_m.numpy()[worst_indices]
            
            print_master(f"\n表现最差的{worst_query_num}个query (按AP排序):")
            for i, (idx, ap, pid,camid) in enumerate(zip(worst_indices, worst_aps, worst_pids,worst_camids)):
                print_master(f"{i+1}. Query索引: {idx}, PID: {pid}, Camid:{camid} AP: {ap:.4f}")
                
                # 输出该query的前rankn个检索结果
                topn_indices = indices[idx][:rankn]
                topn_pids = g_pids_m.numpy()[topn_indices]
                topn_camids = g_camids_m.numpy()[topn_indices]
                
                print_master(f"   前{rankn}个检索结果的PID: {topn_pids}")
                print_master(f"   前{rankn}个检索结果的CamID: {topn_camids}")
                print_master(f"   前{rankn}个检索结果是否包含正确PID: {pid in topn_pids}")
        
        if not save_rank_index:
            results[f"{q_mod}->{g_mod}"] = {"mAP":mAP, "cmc":cmc, "num_sample":num_sample}
        else:
            results[f"{q_mod}->{g_mod}"] = {"mAP":mAP, "cmc":cmc, "num_sample":num_sample, 
                                          'rank_index': indices, 'query_ap': query_ap,
                                          'worst_pids':worst_pids,'worst_camids':worst_camids,'worst_index':worst_indices}
    results['num_sample'] = num_sample
    return results

def evaluate_single_modality_reid(test_log_path, rankn=50):
    """评估同模态检索效率（自动从同pid同camid中抽取query）"""
    output_data = torch.load(test_log_path, map_location='cpu')
    feats = output_data['feats']  # list of tensors
    num_sample = len(feats)
    print("读取到的样本数：", num_sample)
    camids = torch.tensor(output_data['camids'], dtype=torch.long)
    pids = torch.tensor(output_data['pids'], dtype=torch.long)
    modality = output_data['modality']  # list of strings
    print("模态类型：", np.unique(np.array(modality)))
    
    # 合并特征和标签
    feats = torch.cat([i.unsqueeze(0) for i in feats], dim=0)
    del output_data

    # 仅评估同模态（示例为text->text）
    modality_pairs = [('text', 'text')]  # 可替换为其他同模态如('RGB','RGB')

    results = {}
    for q_mod, g_mod in modality_pairs:
        # 筛选当前模态的所有样本
        mask = torch.tensor([m == q_mod for m in modality], dtype=torch.bool)
        feats_m = feats[mask]
        pids_m = pids[mask]
        camids_m = camids[mask]

        # 构建(pid, camid)到索引的映射字典
        from collections import defaultdict
        pid_camid_dict = defaultdict(list)
        for idx, (pid, camid) in enumerate(zip(pids_m, camids_m)):
            pid_camid_dict[(pid.item(), camid.item())].append(idx)

        # 随机选择query（每个(pid,camid)组合选1个），其余作为gallery
        query_indices = []
        gallery_indices = []
        for indices in pid_camid_dict.values():
            if len(indices) < 2:
                continue  # 至少需要1个query和1个gallery
            # 随机打乱并选择第一个作为query
            shuffled = torch.randperm(len(indices)).tolist()
            query_indices.append(indices[shuffled[0]])
            gallery_indices.extend(np.array(indices)[np.array(shuffled[1:])].tolist())
        
        if not query_indices:
            print(f"跳过模态对 {q_mod}->{g_mod}，无有效的(pid,camid)组合")
            continue

        # 提取query和gallery数据
        q_feats = feats_m[query_indices]
        g_feats = feats_m[gallery_indices]
        q_pids = pids_m[query_indices]
        g_pids = pids_m[gallery_indices]
        q_camids = camids_m[query_indices]
        g_camids = camids_m[gallery_indices]

        print(f"模态对 {q_mod}->{g_mod} 的样本分布:")
        print(f"  Query数量: {len(q_pids)} (来自 {len(pid_camid_dict)} 个(pid,camid)组合)")
        print(f"  Gallery数量: {len(g_pids)} (平均每个组合 {len(g_pids)/len(pid_camid_dict):.1f} 个样本)")

        # 计算距离矩阵
        q_feats = torch.nn.functional.normalize(q_feats, p=2, dim=1)
        g_feats = torch.nn.functional.normalize(g_feats, p=2, dim=1)
        distmat = euclidean_distance(q_feats, g_feats)

        # 评估（同模态默认使用set=2协议）
        cmc, mAP, mINP = eval_func(distmat, q_pids.numpy(), g_pids.numpy(), q_camids.numpy(), g_camids.numpy(), set=2)

        # 打印结果
        print("\n评估结果:")
        print(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            print(f"Rank-{r}: {cmc[r-1]:.1%}")
        print(f"mINP: {mINP:.1%}")

        results[f"{q_mod}->{g_mod}"] = {
            "mAP": mAP,
            "cmc": cmc[:rankn],
            "mINP": mINP,
            "query_num": len(q_pids),
            "gallery_num": len(g_pids),
            "pid_camid_pairs": len(pid_camid_dict)
        }
    
    results['num_sample'] = num_sample
    return results

def tsne_plot(feats: np.asarray, pids: np.asarray, save_dir: str, pid_list: list = None):
    """
    功能：基于PCA和t-SNE的特征可视化，支持筛选指定PID的数据
    参数：
        feats: 特征数组 (N x D)
        pids: 行人ID数组 (N,)
        save_dir: 图片保存路径
        pid_list: 需分析的PID列表（None表示分析全部数据）
    """
    # 1. 数据筛选
    if pid_list is not None:
        mask = np.isin(pids, pid_list)
        print(pids.unique())
        feats = feats[mask]
        pids = pids[mask]
        print(f"筛选后数据量: {len(pids)}个样本，{len(np.unique(pids))}个PID")

    # 2. 降维（PCA + t-SNE）
    pca = PCA(n_components=0.95, svd_solver='full')  # 动态设置维度
    pca_result = pca.fit_transform(feats)
    
    tsne = TSNE(n_components=2, perplexity=35, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)

    # 3. 可视化
    plt.figure(figsize=(10, 8))
    
    # 主散点图（保留原始颜色区分）
    ax = sns.scatterplot(
        x=tsne_result[:, 0], 
        y=tsne_result[:, 1], 
        hue=pids, 
        palette='viridis',
        legend='full',
        s=50,
        alpha=0.7  # 半透明以便观察红圈
    )
    
    # 用红色圆圈标记相同PID的点（参考搜索结果的标记方法[1,6](@ref)）
    for pid in np.unique(pids):
        pid_mask = (pids == pid)
        pid_points = tsne_result[pid_mask]
        # 计算同类点的中心位置
        center = np.mean(pid_points, axis=0)
        # 绘制红圈（半径设为同类点最大距离的1.2倍）
        radius = 1.2 * np.max(np.linalg.norm(pid_points - center, axis=1))
        circle = plt.Circle(center, radius, color='red', fill=False, linestyle='--', linewidth=1.5)
        ax.add_patch(circle)
        # 在中心位置标记PID文本
        plt.text(center[0], center[1], str(pid), color='red', fontsize=8, ha='center')

    plt.title(f"t-SNE Visualization (PID count: {len(np.unique(pids))})")
    plt.savefig(f"{save_dir}/tsne_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def display_aio_results_from_test_log_reid5o(test_log_path, rankn=50, save_rank_index=True,worst_query_num=0):
    print_master(f"从{test_log_path}获取测试结果...")
    output_data = torch.load(test_log_path, map_location='cpu')
    feats = output_data['feats']  # list of tensors
    num_sample = len(feats)
    print_master("读取到的样本数：{}.".format(num_sample))
    # camids = torch.tensor(output_data['camids'],dtype=torch.long)  # list of tensors
    pids = torch.tensor(output_data['pids'], dtype=torch.long)    # list of tensors
    modality = output_data['modality']  # list of strings
    print_master(np.unique(np.array(modality)))
    
    # 将列表中的tensor拼接成一个大的tensor
    feats = torch.cat([i.unsqueeze(0) for i in feats], dim=0)
    pids = torch.tensor(pids)  # 假设pids中的tensor都是1D的
    
    del output_data

    # 筛选各模态特征和标签
    modality_pairs = [
        ('IR', 'RGB'),
        ('text', 'RGB'), ('sketch', 'RGB'),
        # ('text', 'RGB')
    ]

    results = {}
    for q_mod, g_mod in modality_pairs:
        # 筛选query和gallery的模态
        q_mask = torch.tensor([m == q_mod for m in modality], dtype=torch.bool)
        g_mask = torch.tensor([m == g_mod for m in modality], dtype=torch.bool)
        
        q_feats_m = feats[q_mask]
        g_feats_m = feats[g_mask]
        q_pids_m = pids[q_mask]
        g_pids_m = pids[g_mask]
        # q_camids_m = camids[q_mask]
        # g_camids_m = camids[g_mask]
        
        if len(q_pids_m) == 0 or len(g_pids_m) == 0:
            print_master(f"跳过模态对 {q_mod}->{g_mod}，因为没有有效样本")
            continue
        print_master(f"query:{len(q_pids_m)} gallery:{len(g_pids_m)}")
        def has_common_elements(tensor1, tensor2):
            set1 = set(tensor1.tolist())
            set2 = set(tensor2.tolist())
            return len(set1 & set2)
        
        if not has_common_elements(q_pids_m, g_pids_m):
            continue  # 如果没有共同PID则跳过
        # 计算距离矩阵
        q_feats_m = torch.nn.functional.normalize(q_feats_m, dim=1, p=2)
        g_feats_m = torch.nn.functional.normalize(g_feats_m, dim=1, p=2)
        # 计算CMC和mAP
        table = eval_func_TBRlike(q_feats_m, g_feats_m, q_pids_m, g_pids_m, q_mod, g_mod, viusalize=True)
    results['num_sample'] = num_sample
    return results

if __name__ == "__main__":
    test_log_path = "/shared/VauAI/lijie/proj/MLLM_REID/work_dirs/mllmreid/aio_lorasft_lora5(lora4base)/checkpoint-78126/cuhkpedes_testset/featlabel.pth"
    display_aio_results_from_test_log(test_log_path,10,save_rank_index=None,worst_query_num=0)
    
    # test_log_path = "/mnt/43_store/xty/jhb_data/work_dirs/mllmreid/aio_lorasft_lora4(more_training_params)/checkpoint-80000/icfgpedes_testset/featlabel.pth"
    # display_aio_results_from_test_log(test_log_path,10,save_rank_index=None,worst_query_num=0)
    
    # test_log_path = "/shared/VauAI/lijie/proj/MLLM_REID/work_dirs/mllmreid/aio_lorasft_lora5(lora4base)/checkpoint-78126/rstpreid_testset/featlabel.pth"
    # display_aio_results_from_test_log(test_log_path,10,save_rank_index=None,worst_query_num=0)
    # evaluate_single_modality_reid(test_log_path, 10)
    # info_dict = torch.load(test_log_path)
    # # 加载数据
    # train_feats = info_dict['train_feats'].float()
    # train_pids = info_dict['train_pids']
    # train_camids = info_dict['train_camids']
    # tsne_plot(
    #     feats=train_feats,
    #     pids=train_pids,
    #     save_dir="/home/tan/jhb/VLM2Vec/logs/market_2025-05-24_13-20-02",
    #     pid_list=[8,9,13]  # 传入None则分析全部数据
    # )
    pass