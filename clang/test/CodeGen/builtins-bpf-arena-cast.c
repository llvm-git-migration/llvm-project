// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

#define __as __attribute__((address_space(7)))

void __as *cast_kern(void __as *ptr) {
  return __builtin_bpf_arena_cast(ptr, 1);
}

void __as *cast_user(void __as *ptr) {
  return __builtin_bpf_arena_cast(ptr, 2);
}
