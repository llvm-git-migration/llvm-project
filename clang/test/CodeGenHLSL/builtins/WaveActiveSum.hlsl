// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_int
int test_int(int expr) {
  // CHECK-SPIRV: %[[#entry_tok:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.active.sum.i32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.active.sum.i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveSum(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.active.sum.i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.active.sum.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.active.sum.i64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok1]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.active.usum.i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveSum(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.active.usum.i64([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.active.sum.i64([[TY]]) #[[#attr:]]

// Test basic lowering to runtime function call with array and float value.

// CHECK-LABEL: test_floatv4
float4 test_floatv4(float4 expr) {
  // CHECK-SPIRV: %[[#entry_tok2:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET1:.*]] = call [[TY1:.*]] @llvm.spv.wave.active.sum.v4f32([[TY1]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok2]]) ]
  // CHECK-DXIL:  %[[RET1:.*]] = call [[TY1:.*]] @llvm.dx.wave.active.sum.v4f32([[TY1]] %[[#]])
  // CHECK:  ret [[TY1]] %[[RET1]]
  return WaveActiveSum(expr);
}

// CHECK-DXIL: declare [[TY1]] @llvm.dx.wave.active.sum.v4f32([[TY1]]) #[[#attr]]
// CHECK-SPIRV: declare [[TY1]] @llvm.spv.wave.active.sum.v4f32([[TY1]]) #[[#attr]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
