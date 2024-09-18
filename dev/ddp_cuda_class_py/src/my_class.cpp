#include "../include/my_class.h"

MyClass::MyClass() : value(0) {}

int MyClass::increment() {
    value += 1;
    return value;
}

int MyClass::get_value() const {
    return value;
}