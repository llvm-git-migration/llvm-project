// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s

// CHECK-LABEL: test_uint
// CHECK-NOT: bitcast
// CHECK: ret i32 %p0
export uint test_uint(uint p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_int
// CHECK-NOT: bitcast
// CHECK: ret i32 %p0
export uint test_int(int p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_float
// CHECK: bitcast float %p0 to i32
export uint test_float(float p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_vector_uint
// CHECK-NOT: bitcast
// CHECK: ret <4 x i32> %p0
export uint4 test_vector_uint(uint4 p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_vector_int
// CHECK-NOT: bitcast
// CHECK: ret <4 x i32> %p0
export uint4 test_vector_int(int4 p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_vector_float
// CHECK: bitcast <4 x float> %p0 to <4 x i32>
export uint4 test_vector_float(float4 p0) {
  return asuint(p0);
}
