#ifndef MY_CLASS_H
#define MY_CLASS_H

class MyClass {
public:
    MyClass();
    int increment();
    int get_value() const;
private:
    int value;
};

#endif // MY_CLASS_H