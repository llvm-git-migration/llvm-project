// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// XFAIL: *

// CHECK-LABEL: builtin_lerp_half_scalar
// CHECK: %dx.lerp = call half @llvm.dx.lerp.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
// CHECK: ret half %dx.lerp
half builtin_lerp_half_scalar (half p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_float_scalar
// CHECK: %dx.lerp = call float @llvm.dx.lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %dx.lerp
float builtin_lerp_float_scalar ( float p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_half_vector
// CHECK: %dx.lerp = call <3 x half> @llvm.dx.lerp.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// CHECK: ret <3 x half> %dx.lerp
half3 builtin_lerp_half_vector (half3 p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_floar_vector
// CHECK: %dx.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %dx.lerp
float2 builtin_lerp_floar_vector ( float2 p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}
