#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/dfccl_extension.h"

namespace py = pybind11;

PYBIND11_MODULE(dfccl_extension, m) {
    py::class_<DfcclExtension>(m, "DfcclExtension")
        .def(py::init<int32_t, int32_t, int32_t, int32_t>())
        .def("InitNcclComm", &DfcclExtension::InitNcclComm)
        .def("InitOfcclRankCtx", &DfcclExtension::InitOfcclRankCtx);
}