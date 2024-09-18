import os
import sys
import torch
import torch.distributed as dist

class DfcclWrapper:
    def __init__(self, rank, local_rank, group_id, group_rank, group_size):
        self.rank = rank
        self.local_rank = local_rank
        self.group_id = group_id
        self.group_rank = group_rank
        self.group_size = group_size

        self.dfccl_ext = None
        self.coll_already_init_nccl_comm = {}

    def init_dfccl_ext(self):
    # 将 build 目录添加到 sys.path，以便导入 dfccl_extension 模块
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
        sys.path.append(build_dir)

        # 导入自定义扩展模块
        import dfccl_extension
        self.dfccl_ext = dfccl_extension.DfcclExtension(
            self.rank,
            self.local_rank,
            self.group_rank,
            self.group_size
        )

        return self.dfccl_ext

    def prepare_dfccl_ar(self, coll_id, parallel_type):
        assert parallel_type in ["DP", "TP"], "parallel_type must be either 'DP' or 'TP'"

        # 一个coll第一次被调用, 完成两件事: 
        # 1. 初始化nccl_comm
        # 2. 进行注册
        nccl_unique_id_str = f"{parallel_type}_AR_{self.group_id}_{self.group_rank}_{self.group_size}_{coll_id}"
        self.dfccl_ext.InitNcclComm(coll_id, nccl_unique_id_str)
        self.coll_already_init_nccl_comm[coll_id] = True
        print(f"rank {self.rank}, local_rank {self.local_rank}, init comm for coll_id: {coll_id} on group_id: {self.group_id}, group_rank: {self.group_rank}")

        # TODO: 注册

    def call_dfccl_ar(self, coll_id, tensor):
        print(f"rank {self.rank}, local_rank {self.local_rank}, call dfccl_ar for coll_id: {coll_id} on group_id: {self.group_id}, group_rank: {self.group_rank}")