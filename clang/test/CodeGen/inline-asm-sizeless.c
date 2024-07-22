// RUN: not %clang_cc1 -S %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: Output operand is sizeless!
void foo(void) {
    extern long bar[];
    asm ("" : "=r"(bar));
}
