// REQUIRES: arm-registered-target

// RUN: %clang -flto -target thumbv7m-unknown-unknown-eabi -mbranch-protection=pac-ret %s -S -o - 2>&1 | FileCheck %s
// RUN: %clang -flto -target thumbv7m-unknown-unknown-eabi -mbranch-protection=bti %s -S -o - 2>&1 | FileCheck %s
// RUN: %clang -flto -target thumbv7m-unknown-unknown-eabi -mbranch-protection=bti+pac-ret %s -S -o - 2>&1 | FileCheck %s


// RUN: %clang -flto=thin -target thumbv7m-unknown-unknown-eabi -mbranch-protection=pac-ret %s -S -o - 2>&1 | FileCheck %s
// RUN: %clang -flto=thin -target thumbv7m-unknown-unknown-eabi -mbranch-protection=bti %s -S -o - 2>&1 | FileCheck %s
// RUN: %clang -flto=thin -target thumbv7m-unknown-unknown-eabi -mbranch-protection=bti+pac-ret %s -S -o - 2>&1 | FileCheck %s

void foo() {}

/// Check there are branch protection function attributes while compiling for LTO
// CHECK-LABEL: @foo() #[[#ATTR:]]
// CHECK: attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"{{.*}} "sign-return-address"
