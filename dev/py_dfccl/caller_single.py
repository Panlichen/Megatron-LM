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

    # a = torch.ones(1000000).cuda(local_rank).to(torch.float32)

    # 在当前进程的 GPU 上创建一个随机张量
    # torch_ddp_tensors = [torch.ones(1000000).cuda(local_rank).to(torch.float32) for _ in range(1)]  # bug: 十分奇怪, 用了这里的话, daemonKernel里就卡住了, 放弃这种验证方法好了.
    # tensors = [torch.ones(1000000).cuda(local_rank).to(torch.float32) for _ in range(1)]
    tensors = [torch.randn(1000000).cuda(local_rank).to(torch.float32) for _ in range(2)]
    # torch_ddp_tensors = [tensor.clone().to(tensor.device) for tensor in tensors]  # bug: 也TM十分奇怪, 明明是clone, 但是用dfccl把tensors进行ar之后, 这里的也跟着变了, 放弃这种验证方法好了.
    # for tensor, torch_ddp_tensor in zip(tensors, torch_ddp_tensors):
    #     print(f"Rank {rank}, group_id {group_id}, group_rank {group_rank}, dfccl ori: {tensor[0]}, torch_ddp_ori: {torch_ddp_tensor[0]}, dfccl ptr: {tensor.data_ptr()}, torch_ddp_ptr: {torch_ddp_tensor.data_ptr()}")

    # dfccl_ext = None
    # dfccl_wrapper = DfcclWrapper(rank, local_rank, group_id, group_rank, group_size, group)

    extern_coll_id_2_dfccl_ext = [None for _ in tensors]
    extern_coll_id_2_dfccl_wrapper = [DfcclWrapper(rank, local_rank, group_id, group_rank, group_size, group) for _ in tensors]

    # 执行 AllReduce 操作
    for i in range(1000):
        for extern_coll_id, tensor in enumerate(tensors):
            if extern_coll_id_2_dfccl_ext[extern_coll_id] is None:
                extern_coll_id_2_dfccl_ext[extern_coll_id] = extern_coll_id_2_dfccl_wrapper[extern_coll_id].init_dfccl_ext()
            if not extern_coll_id_2_dfccl_wrapper[extern_coll_id].coll_already_init_nccl_comm:
                extern_coll_id_2_dfccl_wrapper[extern_coll_id].prepare_dfccl_ar(coll_id=0, parallel_type="DP", tensor=tensor)
                extern_coll_id_2_dfccl_wrapper[extern_coll_id].dfccl_finalize()
                # print(f"{i}th iter, Rank {rank}, group_id {group_id}, group_rank {group_rank}, init comm for coll_id: {extern_coll_id}")

            extern_coll_id_2_dfccl_wrapper[extern_coll_id].call_dfccl_ar(coll_id=0, tensor=tensor)
            # print(f"{i}th iter, Rank {rank}, group_id {group_id}, group_rank {group_rank}, call dfccl_ar for coll_id: {extern_coll_id}")
            extern_coll_id_2_dfccl_wrapper[extern_coll_id].wait_dfccl_cqes()
            # print(f"{i}th iter, Rank {rank}, group_id {group_id}, group_rank {group_rank}, wait dfccl_cqes for coll_id: {extern_coll_id}")

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