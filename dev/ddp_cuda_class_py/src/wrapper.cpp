#include <pybind11/pybind11.h>
#include "../include/my_class.h"

namespace py = pybind11;

// 全局的 MyClass 对象，保持状态
MyClass my_class;

// 增加 class 的值，并返回当前值
int increment_class_value() {
    return my_class.increment();
}

// 获取当前的 class 值
int get_class_value() {
    return my_class.get_value();
}

// 注册到 Python 模块
PYBIND11_MODULE(my_extension, m) {
    m.def("increment_class_value", &increment_class_value, "Increment and get class value");
    m.def("get_class_value", &get_class_value, "Get class value");
}