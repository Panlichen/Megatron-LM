import torch
import torch.distributed as dist
import time
import os
import sys
# 添加包含 dfccl_wrapper 的目录到 Python 路径
dfccl_path = '/workspace/Megatron-LM/dev/py_dfccl'
sys.path.append(dfccl_path)
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
    # torch_ddp_tensors = [torch.ones(1000000).cuda(local_rank).to(torch.float32) for _ in range(1)]  # bug: 十分奇怪, 用了这里的话, daemonKernel里就卡住了, 放弃这种验证方法好了.
    # tensors = [torch.ones(1000000).cuda(local_rank).to(torch.float32) for _ in range(1)]
    tensors = [torch.randn(10000).cuda(local_rank).to(torch.float32) for _ in range(10)]
    # torch_ddp_tensors = [tensor.clone().to(tensor.device) for tensor in tensors]  # bug: 也TM十分奇怪, 明明是clone, 但是用dfccl把tensors进行ar之后, 这里的也跟着变了, 放弃这种验证方法好了.
    # for tensor, torch_ddp_tensor in zip(tensors, torch_ddp_tensors):
    #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, dfccl ori: {tensor[0]}, torch_ddp_ori: {torch_ddp_tensor[0]}, dfccl ptr: {tensor.data_ptr()}, torch_ddp_ptr: {torch_ddp_tensor.data_ptr()}")

    dfccl_ext = None
    dfccl_wrapper = DfcclWrapper(rank, local_rank, group_id, group_rank, group_size, group)

    # 执行 AllReduce 操作
    for i in range(100):
        # tensors = [torch.randn(1000000).cuda(local_rank).to(torch.float32) for _ in range(1)]
        if dfccl_ext is None:
            dfccl_ext = dfccl_wrapper.init_dfccl_ext()  # 调用了InitOfcclRankCtx, 理论上, 一个进程只需要调用一次这个
        if not dfccl_wrapper.coll_already_init_nccl_comm:
            for coll_id, tensor in enumerate(tensors):
                dfccl_wrapper.prepare_dfccl_ar(coll_id=coll_id, parallel_type="DP", tensor=tensor)
            dfccl_wrapper.dfccl_finalize()

        for coll_id, tensor in enumerate(tensors):
            dfccl_wrapper.call_dfccl_ar(coll_id=coll_id, tensor=tensor)
        # print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, call dfccl_ar done, before wait_dfccl_cqes")
        dfccl_wrapper.wait_dfccl_cqes()

        # if i % 100 == 0:
        print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, done {i} iters for {len(tensors)} tensors")


        # for coll_id, tensor in enumerate(tensors):
        #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, coll_id {coll_id}, dfccl result: {tensor[0]}")

        # for tensor, torch_ddp_tensor in zip(tensors, torch_ddp_tensors):
        #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, dfccl before check: {tensor[0]}, torch_ddp_before check: {torch_ddp_tensor[0]}, dfccl ptr: {tensor.data_ptr()}, torch_ddp_ptr: {torch_ddp_tensor.data_ptr()}")

        # for tensor, torch_ddp_tensor in zip(tensors, torch_ddp_tensors):
        #     dist.all_reduce(torch_ddp_tensor, group=group, op=dist.ReduceOp.SUM)

        #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, dfccl result: {tensor[0]}, torch_ddp_result: {torch_ddp_tensor[0]}.")

        #     # 检查结果是否一致
        #     # if not torch.allclose(tensor, torch_ddp_tensor):
        #     #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, detected a difference in tensor values after AllReduce.")
        #     # else:
        #     #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, tensor values are consistent after AllReduce.")

    # 清理所有进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()