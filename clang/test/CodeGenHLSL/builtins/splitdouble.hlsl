// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-vulkan-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s --check-prefix=SPIRV



// CHECK: define {{.*}} i32 {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%.*]] = {{.*}} call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALD]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
// SPIRV: define spir_func {{.*}} i32 {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// SPIRV-NOT: @llvm.dx.splitdouble.i32
// SPIRV: [[CAST:%.*]] = bitcast double [[VALD]] to <2 x i32>
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
// SPIRV: [[VALRET1:%.*]] = shufflevector <3 x double> [[VALD]], <3 x double> poison, <2 x i32> <i32 0, i32 1>
// SPIRV: [[VALRET2:%.*]] = shufflevector <3 x double> [[VALD]], <3 x double> poison, <2 x i32> <i32 2, i32 0>
// SPIRV: [[CAST1:%.*]] = bitcast <2 x double> [[VALRET1]] to <4 x i32>
// SPIRV: [[CAST2:%.*]] = bitcast <2 x double> [[VALRET2]] to <4 x i32>
// SPIRV: shufflevector <4 x i32> [[CAST1]], <4 x i32> [[CAST2]], <3 x i32> <i32 0, i32 2, i32 4>
// SPIRV: shufflevector <4 x i32> [[CAST1]], <4 x i32> [[CAST2]], <3 x i32> <i32 1, i32 3, i32 5>
uint3 test_vector(double3 D) {
  uint3 A, B;
  asuint(D, A, B);
  return A + B;
}
