import ctypes
import torch

# 加载编译后的库
cuda_lib = ctypes.CDLL('./build/libvector_add.so')

# 设置函数参数类型
cuda_lib.launchVectorAdd.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int
]

def vector_add(a, b):
    # 确保输入是CUDA tensor，并且类型为float32
    if not a.is_cuda:
        a = a.cuda()
    if not b.is_cuda:
        b = b.cuda()
    
    a = a.float()
    b = b.float()
    
    # 创建输出tensor
    c = torch.zeros_like(a)
    
    # 调用CUDA函数
    cuda_lib.launchVectorAdd(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_int(a.numel())
    )
    
    return c

# 测试函数
if __name__ == "__main__":
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation.")
        exit()

    a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    b = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32).cuda()
    
    result = vector_add(a, b)
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result: {result}")
    
    # 验证结果
    expected_result = a + b
    print(f"PyTorch result: {expected_result}")
    print(f"Results match: {torch.allclose(result, expected_result)}")