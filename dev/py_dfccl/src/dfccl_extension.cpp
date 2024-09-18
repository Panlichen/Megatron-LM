#include "../include/dfccl_extension.h"

DfcclExtesion::DfcclExtesion() : value(0) {}

int DfcclExtesion::increment() {
    value += 1;
    return value;
}

int DfcclExtesion::get_value() const {
    return value;
}