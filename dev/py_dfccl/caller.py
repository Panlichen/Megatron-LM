import torch
import torch.distributed as dist
import time
import os

from dfccl_wrapper import DfcclWrapper

def main():
    # 设置当前进程使用的 CUDA 设备
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # 设置环境变量（如有需要，可以启用）
    # os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    # os.environ["NCCL_IB_DISABLE"] = "1"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

    # 获取当前进程的 rank 和总进程数
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 初始化默认进程组，需要指定 init_method，以确保所有进程同步创建进程组
    dist.init_process_group(backend="nccl")

    # 获取进程组列表，以确保所有进程在相同的顺序创建进程组
    group_ranks_list = [
        list(range(0, world_size // 2)),    # 前一半 GPU 的 ranks
        list(range(world_size // 2, world_size))  # 后一半 GPU 的 ranks
    ]

    # 创建进程组列表
    process_groups = []
    for idx, group_ranks in enumerate(group_ranks_list):
        group = dist.new_group(ranks=group_ranks) # 让所有进程按照相同的顺序创建进程组。即所有进程首先创建第一个组，然后创建第二个组。这样可以确保进程组的创建顺序一致。
        # 学到的经验是, 不同的进程不要搞 if-else的不同逻辑. 各个进程要创建的group都相同, 用不同的就好.
        process_groups.append(group)

    # 确定当前进程所在的组和组内的 rank
    group_id = 0 if rank < world_size // 2 else 1
    group = process_groups[group_id]
    group_ranks = group_ranks_list[group_id]
    group_rank = group_ranks.index(rank)
    group_size = len(group_ranks)

    # 在当前进程的 GPU 上创建一个随机张量
    tensors = [torch.randn(1000000).cuda() for _ in range(5)]

    dfccl_ext = None
    dfccl_wrapper = DfcclWrapper(rank, local_rank, group_id, group_rank, group_size, group)

    # 执行 AllReduce 操作
    for i in range(10):
        if dfccl_ext is None:
            dfccl_ext = dfccl_wrapper.init_dfccl_ext()  # 调用了InitOfcclRankCtx, 理论上, 一个进程只需要调用一次这个
        if not dfccl_wrapper.coll_already_init_nccl_comm:
            for coll_id, tensor in enumerate(tensors):
                dfccl_wrapper.prepare_dfccl_ar(coll_id=coll_id, parallel_type="DP", tensor=tensor)
            dfccl_wrapper.dfccl_finalize()
            
        # for coll_id, tensor in enumerate(tensors):
        #     dfccl_wrapper.call_dfccl_ar(coll_id=coll_id, tensor=tensor)
    # print(f"Global Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}, Group ID: {group_id}, Group Rank: {group_rank}, Group Size: {group_size}")

    # 清理所有进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()