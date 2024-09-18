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

class DfcclExtension { // 这个按设计, 是每个进程一份, 一个进程可能维护多个coll, 所以搞一个coll id到comm的hashmap
public:
    DfcclExtension(int32_t global_rank, int32_t local_rank, int32_t group_rank, int32_t group_rank_cnt);
    ~DfcclExtension();
    void InitNcclComm(int32_t coll_id, std::string nccl_unique_id_str); // 关于nccl_unique_id的获取, 其实可以手动在各个进程保持一致, 字符串{DP/TP}_AR_{$coll_id}
    ncclComm_t GetNcclComm(int32_t coll_id);
    void InitOfcclRankCtx();
private:
    int32_t global_rank_; // 全体GPU, pytorch默认group里的rank, 多机时整体排序, 应该意义不大
    int32_t local_rank_; // 单机内, 对应oneflow里的local_device_id, pytorch用 os.environ['LOCAL_RANK']获取
    int32_t group_rank_; // 似乎创建ncclComm, 以及集合通信内的rank, 应该用这个, 对应oneflow里的global_rank_
    int32_t group_rank_cnt_;
    ofcclRankCtx_t ofccl_rank_ctx;
    std::unordered_map<int, ncclComm_t> coll_id2nccl_comm;
};

#endif // DFCCL_EXTENSION_H