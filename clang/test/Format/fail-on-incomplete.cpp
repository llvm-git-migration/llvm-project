// RUN: cat %s | not clang-format --fail-on-incomplete-format | FileCheck %s
// RUN: cat %s | clang-format | FileCheck %s
int a([) {}

// CHECK: int a([) {}