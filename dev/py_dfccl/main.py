import os
import sys

def main():

    # 将 build 目录添加到 sys.path，以便导入 my_extension
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
    sys.path.append(build_dir)

    # 导入自定义扩展模块
    import my_extension

    # 初始化分布式进程组
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    
    # 获取当前进程的 rank 和总进程数
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)


    for i in range(10):
        # 调用接口，增加 class 的值，并获取当前值
        class_value = my_extension.increment_class_value()
        # 在 GPU 上创建一个张量，初始值为进程的 rank
        tensor = torch.ones(10).cuda() * rank

        # 在 Python 中将当前的 class 值加到张量上
        tensor += class_value

        # 执行 allreduce 操作
        dist.all_reduce(tensor)

        # 打印 rank、class 的值和张量的第一个元素
        print(f"Rank: {rank}, Iteration: {i+1}, Class Value: {class_value}, Tensor First Element: {tensor[0].item()}")

    # 清理分布式进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()