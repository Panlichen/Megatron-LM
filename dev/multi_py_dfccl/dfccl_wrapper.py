import os
import sys
import torch
import torch.distributed as dist

class DfcclWrapper:
    def __init__(self, rank, local_rank, group_id, group_rank, group_size, group):
        self.rank = rank
        self.local_rank = local_rank
        self.group_id = group_id
        self.group_rank = group_rank
        self.group_size = group_size
        self.group_handle = group

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
            self.group_id,
            self.group_rank,
            self.group_size
        )

        return self.dfccl_ext

    def prepare_dfccl_ar(self, coll_id, parallel_type, tensor):
        assert parallel_type in ["DP", "TP"], "parallel_type must be either 'DP' or 'TP'"

        # 一个coll第一次被调用, 完成两件事: 
        # 1. 初始化nccl_comm
        # 2. 进行注册

        pid = os.getpid()
        pid_tensor = torch.tensor([pid], dtype=torch.long, device=torch.cuda.current_device())
        all_pids = [torch.tensor([0], dtype=torch.long, device=torch.cuda.current_device()) for _ in range(self.group_size)]
        dist.all_gather(all_pids, pid_tensor, group=self.group_handle)
        # print(f"rank {self.local_rank}, group_id {self.group_id}, group_rank {self.group_rank}, pid: {pid}")
        all_pid_list = [pid.item() for pid in all_pids]
        # for group_rank, pid in enumerate(all_pid_list):
        #     print(f"group_id {self.group_id}, group_rank {group_rank}, pid: {pid}")
        
        self.dfccl_ext.InitNcclComm(coll_id, self.group_rank, self.group_size, all_pid_list)
        self.coll_already_init_nccl_comm[coll_id] = True
        # print(f"rank {self.rank}, local_rank {self.local_rank}, init comm for coll_id: {coll_id} on group_id: {self.group_id}, group_rank: {self.group_rank}")

        # 注册
        dtype_to_dfccl = {
            torch.float32: "dfccl_float32",
            torch.float64: "dfccl_float64",
            torch.float16: "dfccl_float16",
            torch.bfloat16: "dfccl_bfloat16",
            torch.int8: "dfccl_int8",
            torch.uint8: "dfccl_uint8",
            torch.int32: "dfccl_int32",
            torch.int64: "dfccl_int64",
            # PyTorch 没有直接对应的 uint32 和 uint64 类型
            # 如果需要，你可以添加自定义的映射
        }

        count = tensor.numel() * tensor.element_size()
        datatype_str = dtype_to_dfccl[tensor.dtype]
        op_str = "dfccl_sum"
        self.dfccl_ext.PrepareAllReduce(count, datatype_str, op_str, coll_id)
        # print(f"call PrepareAllReduce for coll_id: {coll_id}, count: {count}, datatype_str: {datatype_str}, op_str: {op_str}")


    def dfccl_finalize(self):
        self.dfccl_ext.CallOfcclFinalize()

    def call_dfccl_ar(self, coll_id, tensor):
        # print(f"rank {self.rank}, local_rank {self.local_rank}, call dfccl_ar for coll_id: {coll_id} on group_id: {self.group_id}, group_rank: {self.group_rank}, tensor type: {tensor.dtype}, tensor size: {tensor.nbytes}")
        send_ptr = tensor.data_ptr()
        recv_ptr = tensor.data_ptr()
        # print(f"PYTHON rank {self.rank}, local_rank {self.local_rank}, call dfccl_ar for coll_id: {coll_id} on group_id: {self.group_id}, group_rank: {self.group_rank}, send_ptr: 0x{send_ptr:x}, recv_ptr: 0x{recv_ptr:x}")
        self.dfccl_ext.CallOfcclAllReduce(send_ptr, recv_ptr, coll_id)

    def wait_dfccl_cqes(self):
        self.dfccl_ext.WaitAllReduceCqes()