// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/mod.cppm \
// RUN:     -emit-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/use.cpp \
// RUN:     -fmodule-file=mod=%t/mod.pcm -emit-llvm \
// RUN:     -o - | FileCheck %t/use.cpp

//--- mod.cppm
export module mod;

export struct Thing {
    static const Thing One;
    explicit Thing(int raw) :raw(raw) { }
    int raw;
};

const Thing Thing::One = Thing(1);

export struct C {
    int value;
};
export const C ConstantValue = {1};

export const C *ConstantPtr = &ConstantValue;

C NonConstantValue = {1};
export const C &ConstantRef = NonConstantValue;

//--- use.cpp
import mod;

int consume(int);
int consumeC(C);

extern "C" __attribute__((noinline)) inline int unneeded() {
    return consume(43);
}

extern "C" __attribute__((noinline)) inline int needed() {
    return consume(43);
}

int use() {
    Thing t1 = Thing::One;
    return consume(t1.raw);
}

int use2() {
    if (ConstantValue.value)
        return consumeC(ConstantValue);
    return unneeded();
}

int use3() {
    if (ConstantPtr->value)
        return consumeC(*ConstantPtr);
    return needed();
}

int use4() {
    if (ConstantRef.value)
        return consumeC(ConstantRef);
    return needed();
}

// CHECK-NOT: @_ZNW3mod5Thing3OneE = {{.*}}constant
// CHECK: @_ZW3mod13ConstantValue ={{.*}}available_externally{{.*}} constant 

// Check that the middle end can optimize the program by the constant information.
// CHECK-NOT: @unneeded(

// Check that the use of ConstantPtr won't get optimized incorrectly.
// CHECK-LABEL: @_Z4use3v(
// CHECK: @needed(

// Check that the use of ConstantRef won't get optimized incorrectly.
// CHECK-LABEL: @_Z4use4v(
// CHECK: @needed(
