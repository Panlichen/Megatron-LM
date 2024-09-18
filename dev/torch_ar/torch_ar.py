import torch
import torch.distributed as dist
import time
import os

def main():
    # 初始化默认进程组
    dist.init_process_group("nccl")
    
    # 获取当前进程的 rank 和总进程数
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 创建 dp_group 进程组
    dp_group = dist.new_group(list(range(world_size)))

    # 在当前进程的 GPU 上创建一个随机张量
    tensor = torch.randn(1000000).cuda(rank)

    for i in range(10):
        # 同步所有进程
        torch.cuda.synchronize()
        dist.barrier(group=dp_group)
        
        start_time = time.time()
        
        # 在 dp_group 内执行 AllReduce
        dist.all_reduce(tensor, group=dp_group)
        
        # 再次同步以确保 AllReduce 完成
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        if rank == 0:
            print(f"Iteration {i+1}: AllReduce took {end_time - start_time:.6f} seconds")

    # 清理
    dist.destroy_process_group(dp_group)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()