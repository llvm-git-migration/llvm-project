// REQUIRES: arm-registered-target

// RUN: %clang_cc1 -triple=thumbv7m-unknown-unknown-eabi -msign-return-address=non-leaf %s -S -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce %s -S -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce -msign-return-address=all %s -S -emit-llvm -o - 2>&1 | FileCheck %s

// RUN: %clang_cc1 -flto -triple=thumbv7m-unknown-unknown-eabi -msign-return-address=non-leaf %s -S -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -flto -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce %s -S -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -flto -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce -msign-return-address=all %s -S -emit-llvm -o - 2>&1 | FileCheck %s

// RUN: %clang_cc1 -flto=thin -triple=thumbv7m-unknown-unknown-eabi -msign-return-address=non-leaf %s -S -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -flto=thin -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce  %s -S -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -flto=thin -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce -msign-return-address=all %s -S -emit-llvm -o - 2>&1 | FileCheck %s

void foo() {}

// Check there are branch protection function attributes.
// CHECK-LABEL: @foo() #[[#ATTR:]]
// CHECK: attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"{{.*}} "sign-return-address"
