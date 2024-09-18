#include "../include/dfccl_extension.h"

DfcclExtension::DfcclExtension(int32_t global_rank, int32_t local_rank, int32_t group_rank, int32_t group_rank_cnt) : global_rank_(global_rank), local_rank_(local_rank), group_rank_(group_rank), group_rank_cnt_(group_rank_cnt) {
    cudaSetDevice(local_rank_);
    InitOfcclRankCtx();
}

void NcclUniqueIdFromString(const std::string& str, ncclUniqueId* unique_id) {

    std::string result(NCCL_UNIQUE_ID_BYTES, 'a');
    std::copy_n(str.begin(), 
                std::min(str.size(), static_cast<size_t>(NCCL_UNIQUE_ID_BYTES)), 
                result.begin());

    assert(result.size() == NCCL_UNIQUE_ID_BYTES && "String size must match NCCL_UNIQUE_ID_BYTES");
    memcpy(unique_id->internal, result.data(), NCCL_UNIQUE_ID_BYTES);
}

void DfcclExtension::InitNcclComm(int32_t coll_id, std::string nccl_unique_id_str) {
    ncclUniqueId nccl_unique_id;
    NcclUniqueIdFromString(nccl_unique_id_str, &nccl_unique_id);
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, group_rank_cnt_, nccl_unique_id, group_rank_);
    coll_id2nccl_comm[coll_id] = nccl_comm;
}

ncclComm_t DfcclExtension::GetNcclComm(int32_t coll_id) {
    return coll_id2nccl_comm[coll_id];
}

void DfcclExtension::InitOfcclRankCtx() {
    ofcclInitRankCtx(&ofccl_rank_ctx, local_rank_);
}

DfcclExtension::~DfcclExtension() {
    for (auto& pair : coll_id2nccl_comm) {
        ncclCommDestroy(pair.second);
    }
    // ofcclDestroy(ofccl_rank_ctx);
}