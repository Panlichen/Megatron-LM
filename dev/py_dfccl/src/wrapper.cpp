#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 支持 STL 容器，如 std::vector
#include "../include/dfccl_extension.h"

namespace py = pybind11;

// 准备将 ncclComm_t 包装为 Python 可用的对象（如果需要）
py::object ncclCommToPyObject(ncclComm_t comm) {
    // 由于 ncclComm_t 是一个指针，我们可以将其包装为 PyCapsule
    return py::capsule(comm, "ncclComm_t");
}

PYBIND11_MODULE(dfccl_extension, m) {
    py::class_<DfcclExtension>(m, "DfcclExtension")
        // 构造函数
        .def(py::init<int32_t, int32_t, int32_t, int32_t, int32_t>(),
             py::arg("global_rank"),
             py::arg("local_rank"),
             py::arg("group_id"),
             py::arg("group_rank"),
             py::arg("group_rank_cnt"))
        // InitNcclComm 方法，显式指定参数类型
        .def("InitNcclComm",
             (void (DfcclExtension::*)(int32_t, int32_t, int32_t, const std::vector<int>&))
                 &DfcclExtension::InitNcclComm,
             py::arg("coll_id"),
             py::arg("group_rank"),
             py::arg("group_size"),
             py::arg("pid_list"))
        // GetNcclComm 方法，返回值处理
        .def("GetNcclComm",
             [](DfcclExtension& self, int32_t coll_id) {
                 ncclComm_t comm = self.GetNcclComm(coll_id);
                 // 将 ncclComm_t 包装为 PyCapsule，以便在 Python 中传递
                 return py::capsule(comm, "ncclComm_t");
             },
             py::arg("coll_id"))
        // InitOfcclRankCtx 方法
        .def("InitOfcclRankCtx", &DfcclExtension::InitOfcclRankCtx)
        // PrepareAllReduce 方法，显式指定参数类型
        .def("PrepareAllReduce",
             (void (DfcclExtension::*)(size_t, std::string, std::string, int))
                 &DfcclExtension::PrepareAllReduce,
             py::arg("count"),
             py::arg("datatype_str"),
             py::arg("op_str"),
             py::arg("coll_id"))
        .def("CallOfcclFinalize", &DfcclExtension::CallOfcclFinalize)
        // CallOfcclAllReduce 方法，接受整数形式的地址参数
        .def("CallOfcclAllReduce",
             [](DfcclExtension& self, uint64_t send_ptr, uint64_t recv_ptr, int coll_id) {
                 const void* send_buff = reinterpret_cast<const void*>(send_ptr);
                 void* recv_buff = reinterpret_cast<void*>(recv_ptr);
                 // 调用 C++ 方法
                 self.CallOfcclAllReduce(send_buff, recv_buff, coll_id);
             },
             py::arg("send_ptr"),
             py::arg("recv_ptr"),
             py::arg("coll_id"))
        // WaitAllReduceCqes 方法
        .def("WaitAllReduceCqes", &DfcclExtension::WaitAllReduceCqes)
        // 添加 WaitCqe4Coll 方法
        .def("WaitCqe4Coll", &DfcclExtension::WaitCqe4Coll, py::arg("coll_id"))
        ;
}
