
#ifndef DFCCL_EXTENSION_H
#define DFCCL_EXTENSION_H

#include "dfccl.h"
#include <cuda.h>
#include <unordered_map>
#include <cassert>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <sys/types.h>    // 新增
#include <sys/stat.h>     // 新增
#include <fcntl.h>        // 新增
#include <unistd.h>       // 新增
#include <errno.h>        // 新增


const std::string FIFO_PATH_PREFIX = "/tmp/dfccl_fifo_";

class DfcclExtension { // 这个按设计, 是每个进程一份, 一个进程可能维护多个coll, 所以搞一个coll id到comm的hashmap
public:
    DfcclExtension(int32_t global_rank, int32_t local_rank, int32_t group_id, int32_t group_rank, int32_t group_rank_cnt);
    ~DfcclExtension();
    void InitNcclComm(int32_t coll_id, int32_t group_rank, int32_t group_size, const std::vector<int>& pid_list);
    ncclComm_t GetNcclComm(int32_t coll_id);
    void InitOfcclRankCtx();
    void PrepareAllReduce(size_t count, std::string datatype_str, std::string op_str, int coll_id);
    void CallOfcclFinalize();

private:
    void BroadcastUniqueId(const ncclUniqueId& nccl_unique_id, const std::vector<int>& pid_list);
    ncclUniqueId ReceiveUniqueId();
    int32_t global_rank_; // 全体GPU, pytorch默认group里的rank, 多机时整体排序, 应该意义不大
    int32_t local_rank_; // 单机内, 对应oneflow里的local_device_id, pytorch用 os.environ['LOCAL_RANK']获取
    int32_t group_id_;
    int32_t group_rank_; // 似乎创建ncclComm, 以及集合通信内的rank, 应该用这个, 对应oneflow里的global_rank_
    int32_t group_rank_cnt_;
    ofcclRankCtx_t ofccl_rank_ctx_;
    std::unordered_map<int, ncclComm_t> coll_id2nccl_comm_;
};

#endif // DFCCL_EXTENSION_H
