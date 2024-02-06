// RUN: %clang_cc1 -ffixed-point -S -emit-llvm %s -o - | FileCheck %s

// CHECK: @_ZL1a = internal constant [2 x i32] zeroinitializer
constexpr _Accum a[2] = {};

void func2(const _Accum *);
void func() {
  func2(a);
}
