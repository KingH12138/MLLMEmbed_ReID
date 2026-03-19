from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

# def hard_example_mining(dist_mat, labels, return_inds=False):
#     """For each anchor, find the hardest positive and negative sample.
#     Args:
#       dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#       labels: pytorch LongTensor, with shape [N]
#       return_inds: whether to return the indices. Save time if `False`(?)
#     Returns:
#       dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#       dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#       p_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#       n_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     NOTE: Only consider the case in which all labels have same num of samples,
#       thus we can cope with all anchors in parallel.
#     """

#     assert len(dist_mat.size()) == 2
#     assert dist_mat.size(0) == dist_mat.size(1)
#     N = dist_mat.size(0)

#     # shape [N, N]
#     is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#     is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

#     # `dist_ap` means distance(anchor, positive)
#     # both `dist_ap` and `relative_p_inds` with shape [N, 1]
#     dist_ap, relative_p_inds = torch.max(
#         dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
#     # print(dist_mat[is_pos].shape)
#     # `dist_an` means distance(anchor, negative)
#     # both `dist_an` and `relative_n_inds` with shape [N, 1]
#     dist_an, relative_n_inds = torch.min(
#         dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#     # shape [N]
#     dist_ap = dist_ap.squeeze(1)
#     dist_an = dist_an.squeeze(1)

#     if return_inds:
#         # shape [N, N]
#         ind = (labels.new().resize_as_(labels)
#                .copy_(torch.arange(0, N).long())
#                .unsqueeze(0).expand(N, N))
#         # shape [N, 1]
#         p_inds = torch.gather(
#             ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
#         n_inds = torch.gather(
#             ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
#         # shape [N]
#         p_inds = p_inds.squeeze(1)
#         n_inds = n_inds.squeeze(1)
#         return dist_ap, dist_an, p_inds, n_inds

#     return dist_ap, dist_an

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)    # 当前所有进程加起来的batchsize长度

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # 修改后的实现：避免直接view可能导致的问题
    dist_ap = torch.zeros(N, dtype=dist_mat.dtype, device=dist_mat.device)
    dist_an = torch.zeros(N, dtype=dist_mat.dtype, device=dist_mat.device)
    
    if return_inds:
        p_inds = torch.zeros(N, dtype=torch.long, device=dist_mat.device)
        n_inds = torch.zeros(N, dtype=torch.long, device=dist_mat.device)
    
    # 正样本处理
    max_val = torch.tensor(float('-inf'), device=dist_mat.device)
    dist_ap = torch.where(is_pos, dist_mat, max_val).max(dim=1)[0]

    # 负样本处理
    min_val = torch.tensor(float('inf'), device=dist_mat.device)
    dist_an = torch.where(is_neg, dist_mat, min_val).min(dim=1)[0]

    if return_inds:
        p_inds = torch.where(is_pos, dist_mat, max_val).argmax(dim=1)
        n_inds = torch.where(is_neg, dist_mat, min_val).argmin(dim=1)
        return dist_ap, dist_an, p_inds, n_inds
    
    return dist_ap, dist_an

class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean') -> Tensor:
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    pid = pid.unsqueeze(1) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    if image_id != None or len(image_id)!=0:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_sdm_v2(image_fetures, text_fetures, image_pids, text_pids, logit_scale, image_camids=None, text_camids=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching (跨模态对齐)
    image_fetures: [N, D]  → 如 rgb 特征
    text_fetures:  [M, D]  → 如 text 特征
    image_pids:    [N]     → rgb 的 pid
    text_pids:     [M]     → text 的 pid
    image_camids:  [N]     → 可选，用于 soft label
    text_camids:   [M]     → 可选，用于 soft label
    """
    # 构建跨模态的标签矩阵：[N, M]
    image_pids = image_pids.unsqueeze(1)  # [N, 1]
    text_pids = text_pids.unsqueeze(0)    # [1, M]
    pid_match = (image_pids == text_pids).float()  # [N, M] → 正样本位置为 1

    # 可选：加 camera_id 作为 soft label
    if image_camids is not None and text_camids is not None:
        image_camids = image_camids.unsqueeze(1)  # [N, 1]
        text_camids = text_camids.unsqueeze(0)    # [1, M]
        cam_match = (image_camids == text_camids).float()  # [N, M]
        # 混合 label: 0.3 * pid_match + 0.7 * cam_match
        labels = (pid_match - cam_match) * factor + cam_match
    else:
        labels = pid_match

    # 归一化特征
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    # 计算相似度矩阵
    t2i_cosine_theta = text_norm @ image_norm.t()  # [M, N]
    i2t_cosine_theta = t2i_cosine_theta.t()        # [N, M]

    # 计算 logits
    text_proj_image = logit_scale * t2i_cosine_theta  # text → image
    image_proj_text = logit_scale * i2t_cosine_theta  # image → text

    # 归一化 label 分布
    labels_distribute = labels / (labels.sum(dim=1, keepdim=True) + epsilon)  # [N, M]

    # 计算 i2t loss (image → text)
    i2t_pred = F.softmax(image_proj_text, dim=1)  # [N, M]
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))  # [N, M]

    # 计算 t2i loss (text → image)
    t2i_pred = F.softmax(text_proj_image, dim=1)  # [M, N]
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute.t() + epsilon))  # [M, N]

    # 对每个样本求和，再求平均
    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

# 根据training_args和model_args生成对应的loss_fn
# 生成triplet loss(数据集应该使用合适的采样器)
from src.arguments import DataArguments, TrainingArguments
def make_loss(training_args: TrainingArguments, data_args: DataArguments):
    if "triplet" in training_args.metric_loss_type:
        from src.loss import TripletLoss
        triplet = TripletLoss(training_args.triplet_loss_margin)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
            'but got {}'.format(training_args.metric_loss_type))   
    if training_args.label_smooth:
        xent = CrossEntropyLabelSmooth(num_classes=data_args.num_classes)
    def triplet_loss_func(feats, target):
        triplet_loss = triplet(feats, target, True)[0]
        return {
            "triplet_loss":triplet_loss * training_args.triplet_loss_weight
        }
        
    def id_loss_func(score, target):
        if training_args.label_smooth:
            id_loss = xent(score, target)
        else:
            id_loss = F.cross_entropy(score, target)
        preds = torch.argmax(score, dim=1)
        target = target.long()
        train_acc = (preds==target).float().mean()
        return {
            "id_loss":id_loss * training_args.id_loss_weight,
            "id_acc":train_acc
        }

    def match_loss_func(feats, pids, camids, modalities, logits_scale):
        """
        计算跨模态SDM-loss（全模态双向对齐）
        
        参数:
            feats: 特征张量 [bs, feat_dim]
            pids: 人员ID [bs]
            camids: 摄像头ID [bs]
            modalities(list): 模态标签 [bs], 取值: 'text', 'rgb', 'ir', 'sketch'
            logits_scale: 温度系数
            
        返回:
            {'sdm_loss': loss} 字典
        """
        # 按模态分组特征索引
        modality_indices = {
            'text': torch.tensor([i=='text' for i in modalities]),
            'rgb': torch.tensor([i=='RGB' for i in modalities]),
            'ir': torch.tensor([i=='IR' for i in modalities]),
            'sketch': torch.tensor([i=='sketch' for i in modalities]),
        }
        
        # 计算所有模态对之间的loss
        losses = {}
        # 1. text相关对齐
        if torch.sum(modality_indices['text']) > 0:
            text_feats = feats[modality_indices['text']]
            text_pids = pids[modality_indices['text']]
            # text_camids = camids[modality_indices['text']]
            # 1.1 text2rgb
            if torch.sum(modality_indices['rgb']) > 0:
                rgb_feats = feats[modality_indices['rgb']]
                rgb_pids = pids[modality_indices['rgb']]
                loss = compute_sdm_v2(
                    image_fetures=rgb_feats,
                    text_fetures=text_feats,
                    image_pids=rgb_pids,
                    text_pids=text_pids,
                    logit_scale=logits_scale,
                )
                losses['text_rgb'] = loss
            
            # 1.2 text2ir
            if torch.sum(modality_indices['ir']) > 0:
                ir_feats = feats[modality_indices['ir']]
                ir_pids = pids[modality_indices['ir']]
                loss = compute_sdm_v2(
                    image_fetures=ir_feats,
                    text_fetures=text_feats,
                    image_pids=ir_pids,
                    text_pids=text_pids,
                    logit_scale=logits_scale,
                )
                losses['text_ir'] = loss
            
            # 1.3 text2sketch
            if torch.sum(modality_indices['sketch']) > 0:
                sketch_feats = feats[modality_indices['sketch']]
                sketch_pids = pids[modality_indices['sketch']]
                loss = compute_sdm_v2(
                    image_fetures=sketch_feats,
                    text_fetures=text_feats,
                    image_pids=sketch_pids,
                    text_pids=text_pids,
                    logit_scale=logits_scale,
                )
                losses['text_sketch'] = loss
        
        # 2. rgb相关对齐 (除了已有的text2rgb)
        if torch.sum(modality_indices['rgb']) > 0:
            rgb_feats = feats[modality_indices['rgb']]
            rgb_pids = pids[modality_indices['rgb']]
            # rgb_camids = camids[modality_indices['rgb']]
            # 2.1 rgb2ir (与ir2rgb不同)
            if torch.sum(modality_indices['ir']) > 0:
                ir_feats = feats[modality_indices['ir']]
                ir_pids = pids[modality_indices['ir']]
                loss = compute_sdm_v2(
                    image_fetures=ir_feats,
                    text_fetures=rgb_feats,
                    image_pids=ir_pids,
                    text_pids=rgb_pids,
                    logit_scale=logits_scale,
                )
                losses['rgb_ir'] = loss
            
            # 2.2 rgb2sketch (与sketch2rgb不同)
            if torch.sum(modality_indices['sketch']) > 0:
                sketch_feats = feats[modality_indices['sketch']]
                sketch_pids = pids[modality_indices['sketch']]
                loss = compute_sdm_v2(
                    image_fetures=sketch_feats,
                    text_fetures=rgb_feats,
                    image_pids=sketch_pids,
                    text_pids=rgb_pids,
                    logit_scale=logits_scale,
                )
                losses['rgb_sketch'] = loss
        
        # 3. ir相关对齐
        if torch.sum(modality_indices['ir']) > 0:
            ir_feats = feats[modality_indices['ir']]
            ir_pids = pids[modality_indices['ir']]
            # ir_camids = camids[modality_indices['ir']]
            # 3.1 ir2sketch
            if torch.sum(modality_indices['sketch']) > 0:
                sketch_feats = feats[modality_indices['sketch']]
                sketch_pids = pids[modality_indices['sketch']]
                loss = compute_sdm_v2(
                    image_fetures=sketch_feats,
                    text_fetures=ir_feats,
                    image_pids=sketch_pids,
                    text_pids=ir_pids,
                    logit_scale=logits_scale,
                )
                losses['ir_sketch'] = loss
        
        all_losses = losses
        return all_losses
    
    if data_args.sampler_name=="softmax_triplet":
        return triplet_loss_func, id_loss_func
    elif data_args.sampler_name=="multimodalty":
        return triplet_loss_func, id_loss_func, match_loss_func
    else:
        ValueError("sampler_name should be softmax or softmax_triplet")
        
