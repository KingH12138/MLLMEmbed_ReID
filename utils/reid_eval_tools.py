import torch
import numpy as np
from prettytable import PrettyTable
from torch.nn import functional as F

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().float().numpy() # 转为float32类型，避免后续计算时出现精度问题

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    # inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    # 修改mINP计算部分，处理空张量
    inp = []
    for i, match_row in enumerate(matches):
        nonzero_indices = match_row.nonzero()
        if len(nonzero_indices) > 0:
            last_match = nonzero_indices[-1].item()  # 获取标量值
            inp.append(torch.tensor(
                [tmp_cmc[i][last_match] / (last_match + 1.0)]
            ))
        else:
            inp.append(torch.tensor([0.0]))  # 无匹配时赋0或torch.nan

    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def eval_func_TBRlike(qfeats, gfeats, qpids, gpids, modalityA, modalityB, max_rank=50,viusalize=False):
    qfeats = F.normalize(qfeats, p=2, dim=1) # text features
    gfeats = F.normalize(gfeats, p=2, dim=1) # image features

    similarity = qfeats @ gfeats.t()

    t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qpids, g_pids=gpids, max_rank=max_rank, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.tolist(), t2i_mINP.tolist()
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
    table.add_row(['{}2{}'.format(modalityA, modalityB), t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

    table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
    if viusalize:
        print(table)
    
    return table


def eval_func(distmat, q_pids, g_pids, q_camids=None, g_camids=None, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        if q_camids is not None and g_camids is not None:
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]  # select one row
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
        else:
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]  # select one row
            remove = (g_pids[order] == q_pid)
            keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def get_rankn_gallery_indices(distmat, q_idx, rankn,indices=None):
    if indices is None:  # 若未预计算排序索引
        indices = np.argsort(distmat, axis=1)  # 按距离排序的索引矩阵
    rank5_indices = indices[q_idx, :rankn]  # 取当前查询的前5个gallery索引
    return rank5_indices

def eval_func_with_query_ap(distmat, q_pids, g_pids, q_camids, g_camids, set=0, max_rank=50):
    """修改后的eval_func，返回每个query的AP值"""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    query_ap = np.zeros(num_q)  # 存储每个query的AP值
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        
        if q_camids is not None and g_camids is not None:
            q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        if set == 2:
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep]
        else:
            orig_cmc = matches[q_idx]

        if not np.any(orig_cmc):
            # 如果query identity没有出现在gallery中，AP设为0
            query_ap[q_idx] = 0
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        query_ap[q_idx] = AP  # 存储当前query的AP值

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP, query_ap

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, set=0, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        if set == 2:
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
        else:   # 不进行根据camid和pid的过滤
            orig_cmc = matches[q_idx]

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP