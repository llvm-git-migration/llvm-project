// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -o - -emit-llvm -x cuda %s  | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple spirv32 -o - -emit-llvm -x cuda %s  | FileCheck %s

#define __global__ __attribute__((global))

__attribute__((reqd_work_group_size(128, 1, 1)))
__global__ void reqd_work_group_size_128_1_1() {}
// CHECK: define spir_kernel void @_Z28reqd_work_group_size_128_1_1v() #[[ATTR:[0-9]+]] !reqd_work_group_size ![[SIZE_128:.*]]

__attribute__((work_group_size_hint(2, 2, 2)))
__global__ void work_group_size_hint_2_2_2() {}
// CHECK: define spir_kernel void @_Z26work_group_size_hint_2_2_2v() #[[ATTR]] !work_group_size_hint ![[HINT_2:.*]]

__attribute__((vec_type_hint(int)))
__global__ void vec_type_hint_int() {}
// CHECK: define spir_kernel void @_Z17vec_type_hint_intv() #[[ATTR]] !vec_type_hint ![[VEC_HINT:.*]]

__attribute__((intel_reqd_sub_group_size(64)))
__global__ void intel_reqd_sub_group_size_64() {}
// CHECK: define spir_kernel void @_Z28intel_reqd_sub_group_size_64v() #[[ATTR]] !intel_reqd_sub_group_size ![[SUB_GROUP:.*]]

// CHECK: attributes #[[ATTR]] = { convergent mustprogress noinline norecurse nounwind optnone {{.*}} }

// CHECK: ![[SIZE_128]] = !{i32 128, i32 1, i32 1}
// CHECK: ![[HINT_2]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[VEC_HINT]] = !{i32 undef, i32 1}
// CHECK: ![[SUB_GROUP]] = !{i32 64}

