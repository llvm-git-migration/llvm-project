// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-vulkan-library %s -fnative-half-type -emit-llvm -O0 -o - | FileCheck %s --check-prefix=SPIRV



// CHECK: define {{.*}} i32 {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%.*]] = {{.*}} call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALD]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
// SPIRV: define spir_func {{.*}} i32 {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// SPIRV-NOT: @llvm.dx.splitdouble.i32
// SPIRV: [[REG:%.*]] = load double, ptr [[VALD]].addr, align 8
// SPIRV-NEXT: [[CAST:%.*]] = bitcast double [[REG]] to <2 x i32>
// SPIRV-NEXT: extractelement <2 x i32> [[CAST]], i64 0
// SPIRV-NEXT: extractelement <2 x i32> [[CAST]], i64 1
uint test_scalar(double D) {
  uint A, B;
  asuint(D, A, B);
  return A + B;
}


// CHECK: define {{.*}} <3 x i32> {{.*}}test_vector{{.*}}(<3 x double> {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%.*]] = {{.*}} call { <3 x i32>, <3 x i32> } @llvm.dx.splitdouble.v3i32(<3 x double> [[VALD]])
// CHECK-NEXT: extractvalue { <3 x i32>, <3 x i32> } [[VALRET]], 0
// CHECK-NEXT: extractvalue { <3 x i32>, <3 x i32> } [[VALRET]], 1
// SPIRV: define spir_func {{.*}} <3 x i32> {{.*}}test_vector{{.*}}(<3 x double> {{.*}} [[VALD:%.*]])
// SPIRV-NOT: @llvm.dx.splitdouble.i32
// SPIRV: [[REG:%.*]] = load <3 x double>, ptr [[VALD]].addr, align 32
// SPIRV-NEXT: [[VALRET1:%.*]] = shufflevector <3 x double> [[REG]], <3 x double> poison, <2 x i32> <i32 0, i32 1>
// SPIRV-NEXT: [[VALRET2:%.*]] = shufflevector <3 x double> [[REG]], <3 x double> poison, <2 x i32> <i32 2, i32 0>
// SPIRV-NEXT: [[CAST1:%.*]] = bitcast <2 x double> [[VALRET1]] to <4 x i32>
// SPIRV-NEXT: [[SHUF1:%.*]] = shufflevector <4 x i32> [[CAST1]], <4 x i32> poison, <2 x i32> <i32 0, i32 2>
// SPIRV-NEXT: [[SHUF2:%.*]] = shufflevector <4 x i32> [[CAST1]], <4 x i32> poison, <2 x i32> <i32 1, i32 3>
// SPIRV-NEXT: [[CAST2:%.*]] = bitcast <2 x double> [[VALRET2]] to <4 x i32>
// SPIRV-NEXT: [[SHUF3:%.*]] = shufflevector <4 x i32> [[CAST2]], <4 x i32> poison, <2 x i32> <i32 0, i32 2>
// SPIRV-NEXT: [[SHUF4:%.*]] = shufflevector <4 x i32> [[CAST2]], <4 x i32> poison, <2 x i32> <i32 1, i32 3>
// SPIRV-NEXT: shufflevector <2 x i32> [[SHUF1]], <2 x i32> [[SHUF3]], <3 x i32> <i32 0, i32 1, i32 2>
// SPIRV-NEXT: shufflevector <2 x i32> [[SHUF2]], <2 x i32> [[SHUF4]], <3 x i32> <i32 0, i32 1, i32 2>
uint3 test_vector(double3 D) {
  uint3 A, B;
  asuint(D, A, B);
  return A + B;
}
