// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

#if !__has_builtin(__builtin_bpf_arena_cast)
#error "no __builtin_bpf_arena_cast builtin"
#endif

void test(void) {}

// CHECK-LABEL: define {{.*}} @test()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret void
