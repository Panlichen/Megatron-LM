#include "../include/dfccl_extension.h"

DfcclExtension::DfcclExtension(int32_t global_rank, int32_t local_rank, int32_t group_id, int32_t group_rank, int32_t group_rank_cnt) : global_rank_(global_rank), local_rank_(local_rank), group_id_(group_id), group_rank_(group_rank), group_rank_cnt_(group_rank_cnt), coll_cnt_(0) {
    cudaSetDevice(local_rank_);
    InitOfcclRankCtx();
}

DfcclExtension::~DfcclExtension() {
    for (auto& pair : coll_id2nccl_comm_) {
        ncclCommDestroy(pair.second);
    }
    pid_t pid = getpid();
    int cudaDev;
    cudaGetDevice(&cudaDev);
    std::cout << "pid: " << pid << ", cudaDev: " << cudaDev << ", group id: " << group_id_ << ", group rank: " << group_rank_ << ", calling ofcclDestroy" << std::endl;
    ofcclDestroy(ofccl_rank_ctx_);
}

std::string NcclUniqueIdToString(const ncclUniqueId& unique_id) {
  return std::string(unique_id.internal, NCCL_UNIQUE_ID_BYTES);
}

void NcclUniqueIdFromString(const std::string& str, ncclUniqueId* unique_id) {
//   assert(str.size() == NCCL_UNIQUE_ID_BYTES && "String size must match NCCL_UNIQUE_ID_BYTES");
  memcpy(unique_id->internal, str.data(), NCCL_UNIQUE_ID_BYTES);
}

void DfcclExtension::BroadcastUniqueId(const ncclUniqueId& nccl_unique_id, const std::vector<int>& pid_list) {
    // 发送者，group_rank_ == 0
    for (size_t i = 0; i < pid_list.size(); ++i) {
        if (static_cast<int32_t>(i) == group_rank_) continue;  // 跳过自己
        int target_pid = pid_list[i];
        std::string fifo_path = FIFO_PATH_PREFIX + std::to_string(target_pid);

        // 创建 FIFO，如果已存在则忽略错误
        if (mkfifo(fifo_path.c_str(), 0666) == -1 && errno != EEXIST) {
            std::cerr << "Failed to create FIFO: " << fifo_path << ", errno: " << errno << std::endl;
            continue;
        }

        // 打开 FIFO，写入数据
        int fd = open(fifo_path.c_str(), O_WRONLY);
        if (fd == -1) {
            std::cerr << "Failed to open FIFO for writing: " << fifo_path << ", errno: " << errno << std::endl;
            continue;
        }

        ssize_t bytes_written = write(fd, &nccl_unique_id, sizeof(ncclUniqueId));
        if (bytes_written != sizeof(ncclUniqueId)) {
            std::cerr << "Failed to write full ncclUniqueId to FIFO: " << fifo_path << std::endl;
        }

        close(fd);
        // 可选择在使用后删除 FIFO
        // unlink(fifo_path.c_str());
    }
}

ncclUniqueId DfcclExtension::ReceiveUniqueId() {
    // 接收者
    const std::string FIFO_PATH_PREFIX = "/tmp/dfccl_fifo_";
    std::string fifo_path = FIFO_PATH_PREFIX + std::to_string(getpid());

    // 创建 FIFO，如果已存在则忽略错误
    if (mkfifo(fifo_path.c_str(), 0666) == -1 && errno != EEXIST) {
        std::cerr << "Failed to create FIFO: " << fifo_path << ", errno: " << errno << std::endl;
        ncclUniqueId empty_id;
        memset(&empty_id, 0, sizeof(ncclUniqueId));
        return empty_id;
    }

    // 打开 FIFO，读取数据
    int fd = open(fifo_path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open FIFO for reading: " << fifo_path << ", errno: " << errno << std::endl;
        ncclUniqueId empty_id;
        memset(&empty_id, 0, sizeof(ncclUniqueId));
        return empty_id;
    }

    ncclUniqueId nccl_unique_id;
    ssize_t bytes_read = read(fd, &nccl_unique_id, sizeof(ncclUniqueId));
    if (bytes_read != sizeof(ncclUniqueId)) {
        std::cerr << "Failed to read full ncclUniqueId from FIFO: " << fifo_path << ", errno: " << errno << std::endl;
        memset(&nccl_unique_id, 0, sizeof(ncclUniqueId));
    }

    close(fd);
    // 删除 FIFO
    unlink(fifo_path.c_str());

    return nccl_unique_id;
}

void DfcclExtension::InitNcclComm(int32_t coll_id, int32_t group_rank, int32_t group_size, const std::vector<int>& pid_list) {

    assert(group_size == pid_list.size());

    ncclUniqueId nccl_unique_id;
    if (group_rank == 0) {
        // 发送者生成 ncclUniqueId
        ncclResult_t result = ncclGetUniqueId(&nccl_unique_id);
        if (result != ncclSuccess) {
            std::cerr << "Failed to get NCCL unique ID, error: " << ncclGetErrorString(result) << std::endl;
            return;
        }

        // 广播 ncclUniqueId
        BroadcastUniqueId(nccl_unique_id, pid_list);
    } else {
        // 接收者接收 ncclUniqueId
        nccl_unique_id = ReceiveUniqueId();
    }

    ncclComm_t nccl_comm;
    cudaSetDevice(local_rank_);
    ncclResult_t result = ncclCommInitRank(&nccl_comm, group_rank_cnt_, nccl_unique_id, group_rank_);
    if (result != ncclSuccess) {
        std::cout << "NCCL init failed: " << ncclGetErrorString(result) << std::endl;
        return;
    }
    coll_id2nccl_comm_[coll_id] = nccl_comm;
    
    // int cudaDev;
    // cudaGetDevice(&cudaDev);
    // pid_t pid = getpid();
    // std::cout << "在InitNcclComm中，cudaDev: " << cudaDev << ", pid: " << pid << ", nccl_comm地址: " << static_cast<void*>(nccl_comm) << std::endl;
}

ncclComm_t DfcclExtension::GetNcclComm(int32_t coll_id) {
    return coll_id2nccl_comm_[coll_id];
}

void DfcclExtension::InitOfcclRankCtx() {
    ofcclInitRankCtx(&ofccl_rank_ctx_, local_rank_);
}

void DfcclExtension::PrepareAllReduce(size_t count, std::string datatype_str, std::string op_str, int coll_id) {

    ++coll_cnt_;

    ncclDataType_t datatype;
    ncclRedOp_t op;
    const std::unordered_map<std::string, ncclDataType_t> datatype_map = {
        {"dfccl_float32", ncclFloat32},
        {"dfccl_float64", ncclFloat64},
        {"dfccl_float16", ncclFloat16},
        {"dfccl_bfloat16", ncclBfloat16},
        {"dfccl_int8", ncclInt8},
        {"dfccl_uint8", ncclUint8},
        {"dfccl_int32", ncclInt32},
        {"dfccl_uint32", ncclUint32},
        {"dfccl_int64", ncclInt64},
        {"dfccl_uint64", ncclUint64}
    };

    const std::unordered_map<std::string, ncclRedOp_t> op_map = {
        {"dfccl_sum", ncclSum},
        {"dfccl_prod", ncclProd},
        {"dfccl_min", ncclMin},
        {"dfccl_max", ncclMax},
        {"dfccl_avg", ncclAvg}
    };
    
    auto datatype_it = datatype_map.find(datatype_str);
    if (datatype_it == datatype_map.end()) {
        std::cerr << "Invalid datatype: " << datatype_str << std::endl;
        return;
    }
    datatype = datatype_it->second;

    auto op_it = op_map.find(op_str);
    if (op_it == op_map.end()) {
        std::cerr << "Invalid op: " << op_str << std::endl;
        return;
    }
    op = op_it->second;

    ncclComm_t comm = GetNcclComm(coll_id);

    // pid_t pid = getpid();
    // int cudaDev;
    // cudaGetDevice(&cudaDev);
    // std::cout << "in PrepareAllReduce, cudaDev: " << cudaDev << ", pid: " << pid << ", comm指针地址: " << static_cast<void*>(comm) << std::endl;
    // size_t count: 传参数
    // ncclDataType_t datatype: 目前dp观察到的是float32
    // ncclRedOp_t op: 使用SUM
    // ncclComm_t comm: 用成员变量
    // int collId: 传参数
    // ofcclRankCtx_t rankCtx: 用成员变量
    // std::cout << "在C++中: " << std::endl
    //           << "  集合ID: " << coll_id << std::endl
    //           << "  元素数量: " << count << std::endl
    //           << "  数据类型: " << datatype << " " << datatype_str << std::endl
    //           << "  操作类型: " << op << " " << op_str << std::endl;
    ofcclPrepareAllReduce(count, datatype, op, comm, coll_id, ofccl_rank_ctx_);
}

void DfcclExtension::CallOfcclFinalize() {
    ofcclFinalizeRankCtx7StartHostThrds(ofccl_rank_ctx_);
}

bool isValidDevicePointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    
    if (error != cudaSuccess) {
        cudaGetLastError(); // 重置最后的CUDA错误
        return false; // 如果发生错误，认为指针无效
    }
    
    // 检查指针类型
    #if CUDART_VERSION >= 10000
        // CUDA 10.0及以上版本
        return attributes.type == cudaMemoryTypeDevice;
    #else
        // CUDA 10.0之前的版本
        return attributes.memoryType == cudaMemoryTypeDevice;
    #endif
}

void DfcclExtension::CallOfcclAllReduce(const void* send_buff, void* recv_buff, int coll_id) {
    // , CallbackFunc callback, void *callback_args, ofcclRankCtx_t rank_ctx 这3个参数自行解决
    CallBackArgs *cb_args = new CallBackArgs();
    cb_arg_list_[coll_id] = cb_args;
    cb_args->coll_id = coll_id;
    cb_args->got_cqe = 0;
    pthread_mutex_init(&cb_args->mutex, NULL);

    auto my_call_back = [](int coll_id_from_cqe, void *args) -> int {
        pthread_mutex_lock(&((static_cast<CallBackArgs *>(args))->mutex));
        (static_cast<CallBackArgs *>(args))->got_cqe = 1;
        pthread_mutex_unlock(&((static_cast<CallBackArgs *>(args))->mutex));
        return 0;
    };
    CallbackFunc cb_func = my_call_back;

    pid_t pid = getpid();
    int cudaDev;
    cudaGetDevice(&cudaDev);
    std::cout << "in CallOfcclAllReduce, pid: " << pid << ", cudaDev: " << cudaDev << ", group id: " << group_id_ << ", group rank: " << group_rank_ << ", send_buff: " << send_buff << ", recv_buff: " << recv_buff << std::endl;

    // std::cout << "pid: " << pid << ", cudaDev: " << cudaDev << ", send_buff from pytorch valid: " << isValidDevicePointer(send_buff) << std::endl;
    // std::cout << "pid: " << pid << ", cudaDev: " << cudaDev << ", recv_buff from pytorch valid: " << isValidDevicePointer(recv_buff) << std::endl;

    // float *d_array;
    // size_t size = 1000000 * sizeof(float);
    // cudaMalloc((void**)&d_array, size);
    // send_buff = d_array;
    // recv_buff = d_array;
    // std::cout << "pid: " << pid << ", cudaDev: " << cudaDev << ", send_buff manual alloc valid: " << isValidDevicePointer(send_buff) << std::endl;
    // std::cout << "pid: " << pid << ", cudaDev: " << cudaDev << ", recv_buff manual alloc valid: " << isValidDevicePointer(recv_buff) << std::endl;

    ofcclRunAllReduce(send_buff, recv_buff, coll_id, cb_func, cb_args, ofccl_rank_ctx_);
}

void DfcclExtension::WaitAllReduceCqes() {
    int got_cqe_cnt = 0;
    while (got_cqe_cnt < coll_cnt_) {
        for (int i = 0; i < coll_cnt_; ++i) {
            pthread_mutex_lock(&(cb_arg_list_[i]->mutex));
            // pid_t pid = getpid();
            // int cudaDev;
            // cudaGetDevice(&cudaDev);
            // std::cout << "in WaitAllReduceCqes, pid: " << pid << ", cudaDev: " << cudaDev << ", group id: " << group_id_ << ", group rank: " << group_rank_ << ", cb_arg_list_[i]->got_cqe for coll_id: " << i << "is: " << cb_arg_list_[i]->got_cqe << std::endl;
            if (cb_arg_list_[i]->got_cqe == 1) {
                if (seen_cqe_[i] == 0) {
                    ++got_cqe_cnt;
                    seen_cqe_[i] = 1;
                    delete cb_arg_list_[i];
                    // std::cout << "in WaitAllReduceCqes, pid: " << pid << ", cudaDev: " << cudaDev << ", group id: " << group_id_ << ", group rank: " << group_rank_ << ", got cqe for coll_id: " << i << std::endl;
                }
            }
            pthread_mutex_unlock(&(cb_arg_list_[i]->mutex));
        }
    }
    for (int i = 0; i < coll_cnt_; ++i) {
        seen_cqe_[i] = 0;
    }
}

