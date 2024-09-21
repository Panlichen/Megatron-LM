import os
import sys
import torch
import torch.distributed as dist

def main():
    # 获取环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 将 build 目录添加到 sys.path，以便导入 dfccl_extension 模块
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
    sys.path.append(build_dir)

    # 导入自定义扩展模块
    import dfccl_extension

    # 设置当前进程使用的 CUDA 设备
    torch.cuda.set_device(local_rank)

    # 初始化默认的全局进程组
    dist.init_process_group(backend="nccl")

    # 获取 GPU 总数
    num_gpus = torch.cuda.device_count()

    # 确定进程所在的组
    if rank < world_size // 2:
        # 前一半 GPU，创建一个新的进程组
        group_id = 0
        group_ranks = list(range(0, world_size // 2))
    else:
        # 后一半 GPU，创建另一个新的进程组
        group_id = 1
        group_ranks = list(range(world_size // 2, world_size))

    # 创建新的进程组
    group = dist.new_group(ranks=group_ranks)

    # 获取组内的 rank 和大小
    group_rank = group_ranks.index(rank)
    group_size = len(group_ranks)
    
    # 创建 DfcclExtension 实例
    # dfccl_ext = dfccl_extension.DfcclExtension(
    #     global_rank=rank,
    #     local_rank=local_rank,
    #     group_rank=group_rank,
    #     group_rank_cnt=group_size
    # )  // PyBind11 默认不支持关键字参数，除非在绑定时显式指定。
    dfccl_ext = dfccl_extension.DfcclExtension(
        rank,
        local_rank,
        group_rank,
        group_size
    )

    # 调用 InitNcclComm，coll_id=0，nccl_unique_id_str="DP_AR_{coll_id}"
    coll_id = 0
    nccl_unique_id_str = f"DP_AR_{coll_id}"
    dfccl_ext.InitNcclComm(coll_id, nccl_unique_id_str)

    # 后续您的代码逻辑...

    # 示例：打印信息
    print(f"Global Rank: {rank}, Local Rank: {local_rank}, Group ID: {group_id}, Group Rank: {group_rank}, Group Size: {group_size}")

    # 清理分布式进程组
    dist.destroy_process_group()
    print("dist.destroy_process_group() called")

if __name__ == "__main__":
    main()