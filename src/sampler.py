from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import math
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None
import torch
import pickle

def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, item in enumerate(self.data_source):
            self.index_dic[item['pid']].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                random.shuffle(batch_idxs)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)
        
    def __len__(self):
        return self.length
    

# class BanlancedMultiModalRandomIdentitySampler(Sampler):
#     """
#     改进版采样器，确保：
#     1. 每个batch包含N个不同的pid
#     2. 每个pid包含K个样本（K是4的倍数）
#     3. 每个pid的样本来自K//4个不同的camid
#     4. 每个camid对应4个连续样本（4个模态）
    
#     Args:
#         data_source (list): 包含(img_path, pid, camid)的列表
#         batch_size (int): 每个batch的样本总数（必须是num_instances的整数倍）
#         num_instances (int): 每个pid在每个batch中的样本数（必须是4的倍数）
#     """

#     def __init__(self, data_source, batch_size, num_instances, num_modality=4):
#         assert num_instances % num_modality == 0, "num_instances必须是4的倍数"
#         assert batch_size % num_instances == 0, "batch_size必须是num_instances的整数倍"
        
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.num_instances = num_instances
#         self.num_pids_per_batch = batch_size // num_instances
#         self.num_modality = num_modality
        
#         # 创建嵌套字典：{pid: {camid: [index1, index2,...]}}
#         self.pid_cam_indices = defaultdict(lambda: defaultdict(list))
#         for index, item in enumerate(data_source):
#             pid = item['pid']
#             camid = item['camid']
#             self.pid_cam_indices[pid][camid].append(index)
            
#         self.pids = list(self.pid_cam_indices.keys())
        
#         # 验证数据集是否满足每个camid至少有4个样本
#         for pid in self.pids:
#             for camid in self.pid_cam_indices[pid]:
#                 assert len(self.pid_cam_indices[pid][camid]) >= num_modality, \
#                     f"pid {pid}的camid {camid}样本不足{num_modality}个"
        
#         # 计算epoch长度（确保所有样本都被覆盖）
#         self.length = 0
#         for pid in self.pids:
#             num_cams = len(self.pid_cam_indices[pid])
#             # 每个pid需要的camid数
#             num_cams_needed = self.num_instances // num_modality
#             # 实际可用的camid数（取最小值）
#             usable_cams = min(num_cams, num_cams_needed)
#             self.length += usable_cams * num_modality  # 每个camid贡献4个样本

#     def __iter__(self):
#         # 复制原始数据以避免修改
#         pid_cam_indices = copy.deepcopy(self.pid_cam_indices)
        
#         # 打乱所有pid和每个pid下的camid顺序
#         random.shuffle(self.pids)
#         for pid in pid_cam_indices:
#             camids = list(pid_cam_indices[pid].keys())
#             random.shuffle(camids)
#             pid_cam_indices[pid] = {camid: pid_cam_indices[pid][camid] for camid in camids}
        
#         final_idxs = []
        
#         # 生成batch
#         for pid in self.pids:
#             camids = list(pid_cam_indices[pid].keys())
#             num_cams_needed = self.num_instances // self.num_modality
            
#             # 取前num_cams_needed个camid
#             selected_cams = camids[:num_cams_needed]
            
#             # 对每个选中的camid，取前4个样本
#             for camid in selected_cams:
#                 samples = pid_cam_indices[pid][camid][:self.num_modality]
#                 final_idxs.extend(samples)
                
#                 # 从原始数据中移除已使用的样本（避免重复采样）
#                 pid_cam_indices[pid][camid] = pid_cam_indices[pid][camid][self.num_modality:]
                
#                 # 如果该camid的样本已用完，则删除该camid
#                 if len(pid_cam_indices[pid][camid]) == 0:
#                     del pid_cam_indices[pid][camid]
        
#         # 将样本分成batch_size大小的块
#         batch_idxs = []
#         for i in range(0, len(final_idxs), self.batch_size):
#             batch = final_idxs[i:i+self.batch_size]
#             if len(batch) == self.batch_size:
#                 batch_idxs.append(batch)
        
#         # 打乱batch顺序
#         random.shuffle(batch_idxs)
        
#         # 展开所有batch
#         final_idxs = [idx for batch in batch_idxs for idx in batch]
        
#         return iter(final_idxs)
        
#     def __len__(self):
#         return self.length

import random
import copy
from collections import defaultdict

class BalancedMultiModalRandomIdentitySampler:
    def __init__(self, data_source, batch_size, num_instances, num_modality):
        assert num_instances % num_modality == 0, "num_instances必须是num_modality的整数倍"
        assert batch_size % num_instances == 0, "batch_size必须是num_instances的整数倍"
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.num_modality = num_modality
        
        # 构建嵌套字典：{pid: {camid: [index1, index2,...]}}
        if os.path.exists("pid_cam_indices.pth"):
            self.pid_cam_indices = torch.load("pid_cam_indices.pth", map_location='cpu',weights_only=False)
        else:
            self.pid_cam_indices = defaultdict(lambda: defaultdict(list))
            for index, item in tqdm(enumerate(data_source)):
                pid = item['pid']
                camid = item['camid']
                self.pid_cam_indices[pid][camid].append(index)
        
        self.pids = list(self.pid_cam_indices.keys())
        
        # 验证数据集是否满足每个camid至少有num_modality个样本
        for pid in self.pids:
            for camid in self.pid_cam_indices[pid]:
                assert len(self.pid_cam_indices[pid][camid]) >= num_modality, \
                    f"pid {pid}的camid {camid}样本不足{num_modality}个"
        
        # 计算epoch长度（覆盖所有camid所需的最小样本数）
        self.length = sum(
            len(camids)
            for pid in self.pids 
            for camids in self.pid_cam_indices[pid].values()
        )

    def __iter__(self):
        # 深拷贝原始数据并初始化camid使用状态
        pid_cam_indices = copy.deepcopy(self.pid_cam_indices)
        pid_cam_used = {pid: {camid: False for camid in camids} 
                        for pid, camids in pid_cam_indices.items()}
        
        # 计算每个pid需要的camid数量
        num_cams_per_pid_per_batch = self.num_instances // self.num_modality
        
        final_idxs = []
        
        while any(not all(used.values()) for used in pid_cam_used.values()):# 只有当还有未使用的camid(False)时继续
            batch_indices = []
            selected_pids = []
            
            # 优先选择有未使用camid的pid
            remaining_pids = [
                pid for pid in pid_cam_indices 
                if not all(pid_cam_used[pid].values())
            ]
            random.shuffle(remaining_pids)
            
            # 选择当前batch需要的pids
            for pid in remaining_pids[:self.num_pids_per_batch]:
                if len(batch_indices) >= self.batch_size:
                    break
                
                # 获取该pid未使用的camid并打乱顺序
                unused_camids = [
                    camid for camid, used in pid_cam_used[pid].items() 
                    if not used
                ]
                random.shuffle(unused_camids)
                
                # 补充已使用的camid以凑足num_cams_per_pid_per_batch
                if len(unused_camids) < num_cams_per_pid_per_batch:
                    used_camids = [
                        camid for camid, used in pid_cam_used[pid].items() 
                        if used
                    ]
                    random.shuffle(used_camids)
                    needed = num_cams_per_pid_per_batch - len(unused_camids)
                    unused_camids.extend(used_camids[:needed])
                
                # 选择camid
                selected_cams = unused_camids[:num_cams_per_pid_per_batch]
                
                # 收集样本并更新状态
                for camid in selected_cams:
                    samples = pid_cam_indices[pid][camid][:self.num_modality]
                    batch_indices.extend(samples)
                    pid_cam_used[pid][camid] = True  # 标记为已使用
                    
                    # 更新数据结构
                    pid_cam_indices[pid][camid] = pid_cam_indices[pid][camid][self.num_modality:]
                    if not pid_cam_indices[pid][camid]:
                        del pid_cam_indices[pid][camid]
                
                selected_pids.append(pid)
            
            final_idxs.extend(batch_indices)
        
        return iter(final_idxs)
    
    def __len__(self):
        return self.length
    
    
import random
import copy
from collections import defaultdict
import torch.distributed as dist
import os
import heapq
from tqdm import tqdm

class DistributedBalancedMultiModalRandomIdentitySampler:
    def __init__(self, data_source, batch_size, num_instances, num_modality, 
             rank, world_size, shuffle=True):
        assert num_instances % num_modality == 0, "num_instances必须是num_modality的整数倍"
        assert batch_size % num_instances == 0, "batch_size必须是num_instances的整数倍"
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.num_modality = num_modality
        self.shuffle = shuffle
        self.epoch = 0
        
        self.rank = rank
        self.world_size = world_size

        # 构建嵌套字典：{pid: {camid: [index1, index2,...]}}
        try:
            self.pid_cam_indices = torch.load(f"pid_cam_indices{self.rank}.pth", map_location='cpu',weights_only=False)
        except:
            print(f"pid_cam_indices.pth{self.rank}损坏，重新构建.")
            self.pid_cam_indices = defaultdict(lambda: defaultdict(list))
            for index, item in tqdm(enumerate(data_source)):
                pid = item['pid']
                camid = item['camid']
                self.pid_cam_indices[pid][camid].append(index)
            print("构建字典完成")
            self.pid_cam_indices = dict(self.pid_cam_indices)  # 转换为普通字典
            torch.save(self.pid_cam_indices, f"pid_cam_indices{self.rank}.pth")
        
        # ========== 新增：按样本数均衡分配 pid 到各 rank ==========
        import heapq
        # 1. 统计每个 pid 的总样本数
        pid_sample_counts = {}
        for pid, cam_dict in self.pid_cam_indices.items():
            total = sum(len(indices) for indices in cam_dict.values())
            pid_sample_counts[pid] = total
        
        # 2. 按样本数从大到小排序 pid
        sorted_pids = sorted(pid_sample_counts.keys(), key=lambda x: -pid_sample_counts[x])
        
        # 3. 贪心均衡分配到各 rank
        rank_pids = [[] for _ in range(self.world_size)]
        rank_sample_sums = [0] * self.world_size
        heap = []
        for r in range(self.world_size):
            heapq.heappush(heap, (0, r))  # (当前样本数, rank)
        
        for pid in sorted_pids:
            cnt = pid_sample_counts[pid]
            current_sum, r = heapq.heappop(heap)
            rank_pids[r].append(pid)
            heapq.heappush(heap, (current_sum + cnt, r))
        
        # 4. 当前 rank 的 pid 列表
        self.pids = rank_pids[self.rank]
        # ==========================================================

        # 验证数据集是否满足每个camid至少有num_modality个样本
        for pid in self.pids:
            for camid in self.pid_cam_indices[pid]:
                assert len(self.pid_cam_indices[pid][camid]) >= num_modality, \
                    f"pid {pid}的camid {camid}样本不足{num_modality}个"

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        pid_cam_indices = {gpu_pid: copy.deepcopy(self.pid_cam_indices[gpu_pid]) for gpu_pid in self.pids}
        
        pid_cam_used = {pid: {camid: False for camid in camids} 
                        for pid, camids in pid_cam_indices.items()}
        
        num_cams_per_pid_per_batch = self.num_instances // self.num_modality
        
        final_idxs = []
        
        if self.shuffle:
            random.seed(self.epoch)  # 同步随机种子
            random.shuffle(self.pids)  # 全局shuffle
        
        while any(not all(used.values()) for used in pid_cam_used.values()):
            batch_indices = []
            selected_pids = []
            
            remaining_pids = [
                pid for pid in self.pids 
                if not all(pid_cam_used[pid].values())
            ]
            if self.shuffle:
                random.shuffle(remaining_pids)
            
            for pid in remaining_pids[:self.num_pids_per_batch]:
                if len(batch_indices) >= self.batch_size:
                    break
                
                unused_camids = [
                    camid for camid, used in pid_cam_used[pid].items() 
                    if not used
                ]
                if self.shuffle:
                    random.shuffle(unused_camids)
                
                if len(unused_camids) < num_cams_per_pid_per_batch:
                    used_camids = [
                        camid for camid, used in pid_cam_used[pid].items() 
                        if used
                    ]
                    if self.shuffle:
                        random.shuffle(used_camids)
                    needed = num_cams_per_pid_per_batch - len(unused_camids)
                    unused_camids.extend(used_camids[:needed])
                
                selected_cams = unused_camids[:num_cams_per_pid_per_batch]
                
                for camid in selected_cams:
                    samples = pid_cam_indices[pid][camid][:self.num_modality]
                    batch_indices.extend(samples)
                    pid_cam_used[pid][camid] = True
                    pid_cam_indices[pid][camid] = pid_cam_indices[pid][camid][self.num_modality:]
                    if not pid_cam_indices[pid][camid]:
                        del pid_cam_indices[pid][camid]
                
                selected_pids.append(pid)
            
            final_idxs.extend(batch_indices)
        
        return iter(final_idxs)
    
    def __len__(self):
        return sum(
            len(camids) 
            for pid in self.pids
            for camids in self.pid_cam_indices[pid].values()
        )