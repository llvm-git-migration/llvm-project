// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -D__HLSL_ENABLE_16_BIT -o - | FileCheck %s --check-prefix=NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// CHECK: %dx.dot = mul i16 %0, %1
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = mul i16 %0, %1
// NO_HALF: ret i16 %dx.dot
int16_t test_dot_short ( int16_t p0, int16_t p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = call i16 @llvm.dx.dot.v2i16(<2 x i16> %0, <2 x i16> %1)
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = call i16 @llvm.dx.dot.v2i16(<2 x i16> %0, <2 x i16> %1)
// NO_HALF: ret i16 %dx.dot
int16_t test_dot_short2 ( int16_t2 p0, int16_t2 p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = call i16 @llvm.dx.dot.v3i16(<3 x i16> %0, <3 x i16> %1)
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = call i16 @llvm.dx.dot.v3i16(<3 x i16> %0, <3 x i16> %1)
// NO_HALF: ret i16 %dx.dot
int16_t test_dot_short3 ( int16_t3 p0, int16_t3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i16 @llvm.dx.dot.v4i16(<4 x i16> %0, <4 x i16> %1)
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = call i16 @llvm.dx.dot.v4i16(<4 x i16> %0, <4 x i16> %1)
// NO_HALF: ret i16 %dx.dot
int16_t test_dot_short4 ( int16_t4 p0, int16_t4 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = mul i16 %0, %1
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = mul i16 %0, %1
// NO_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort ( uint16_t p0, uint16_t p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = call i16 @llvm.dx.dot.v2i16(<2 x i16> %0, <2 x i16> %1)
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = call i16 @llvm.dx.dot.v2i16(<2 x i16> %0, <2 x i16> %1)
// NO_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort2 ( uint16_t2 p0, uint16_t2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i16 @llvm.dx.dot.v3i16(<3 x i16> %0, <3 x i16> %1)
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = call i16 @llvm.dx.dot.v3i16(<3 x i16> %0, <3 x i16> %1)
// NO_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort3 ( uint16_t3 p0, uint16_t3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i16 @llvm.dx.dot.v4i16(<4 x i16> %0, <4 x i16> %1)
// CHECK: ret i16 %dx.dot
// NO_HALF: %dx.dot = call i16 @llvm.dx.dot.v4i16(<4 x i16> %0, <4 x i16> %1)
// NO_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort4 ( uint16_t4 p0, uint16_t4 p1 ) {
  return dot ( p0, p1 );
}
#endif


// CHECK: %dx.dot = mul i32 %0, %1
// CHECK: ret i32 %dx.dot
int test_dot_int ( int p0, int p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v2i32(<2 x i32> %0, <2 x i32> %1)
// CHECK: ret i32 %dx.dot
int test_dot_int2 ( int2 p0, int2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v3i32(<3 x i32> %0, <3 x i32> %1)
// CHECK: ret i32 %dx.dot
int test_dot_int3 ( int3 p0, int3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v4i32(<4 x i32> %0, <4 x i32> %1)
// CHECK: ret i32 %dx.dot
int test_dot_int4 ( int4 p0, int4 p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = mul i32 %0, %1
// CHECK: ret i32 %dx.dot
uint test_dot_uint ( uint p0, uint p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v2i32(<2 x i32> %0, <2 x i32> %1)
// CHECK: ret i32 %dx.dot
uint test_dot_uint2 ( uint2 p0, uint2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v3i32(<3 x i32> %0, <3 x i32> %1)
// CHECK: ret i32 %dx.dot
uint test_dot_uint3 ( uint3 p0, uint3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v4i32(<4 x i32> %0, <4 x i32> %1)
// CHECK: ret i32 %dx.dot
uint test_dot_uint4 ( uint4 p0, uint4 p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = mul i64 %0, %1
// CHECK: ret i64 %dx.dot
int64_t test_dot_long ( int64_t p0, int64_t p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v2i64(<2 x i64> %0, <2 x i64> %1)
// CHECK: ret i64 %dx.dot
int64_t test_dot_long2 ( int64_t2 p0, int64_t2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v3i64(<3 x i64> %0, <3 x i64> %1)
// CHECK: ret i64 %dx.dot
int64_t test_dot_long3 ( int64_t3 p0, int64_t3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v4i64(<4 x i64> %0, <4 x i64> %1)
// CHECK: ret i64 %dx.dot
int64_t test_dot_long4 ( int64_t4 p0, int64_t4 p1) {
  return dot ( p0, p1 );
}


// CHECK:  %dx.dot = mul i64 %0, %1
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong ( uint64_t p0, uint64_t p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = call i64 @llvm.dx.dot.v2i64(<2 x i64> %0, <2 x i64> %1)
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong2 ( uint64_t2 p0, uint64_t2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v3i64(<3 x i64> %0, <3 x i64> %1)
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong3 ( uint64_t3 p0, uint64_t3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v4i64(<4 x i64> %0, <4 x i64> %1)
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong4 ( uint64_t4 p0, uint64_t4 p1) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = fmul half %0, %1
// CHECK: ret half %dx.dot
// NO_HALF: %dx.dot = fmul float %0, %1
// NO_HALF: ret float %dx.dot
half test_dot_half ( half p0, half p1 ) {
  return dot ( p0, p1 );
}


// CHECK: %dx.dot = call half @llvm.dx.dot.v2f16(<2 x half> %0, <2 x half> %1)
// CHECK: ret half %dx.dot
// NO_HALF: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %0, <2 x float> %1)
// NO_HALF: ret float %dx.dot
half test_dot_half2 ( half2 p0, half2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call half @llvm.dx.dot.v3f16(<3 x half> %0, <3 x half> %1)
// CHECK: ret half %dx.dot
// NO_HALF: %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %0, <3 x float> %1)
// NO_HALF: ret float %dx.dot
half test_dot_half3 ( half3 p0, half3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call half @llvm.dx.dot.v4f16(<4 x half> %0, <4 x half> %1)
// CHECK: ret half %dx.dot
// NO_HALF: %dx.dot = call float @llvm.dx.dot.v4f32(<4 x float> %0, <4 x float> %1)
// NO_HALF: ret float %dx.dot
half test_dot_half4 ( half4 p0, half4 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: define noundef float @
// CHECK: %dx.dot = fmul float %0, %1
// CHECK: ret float %dx.dot
float test_dot_float ( float p0, float p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %0, <2 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float2 ( float2 p0, float2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %0, <3 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float3 ( float3 p0, float3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call float @llvm.dx.dot.v4f32(<4 x float> %0, <4 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float4 ( float4 p0, float4 p1) {
  return dot ( p0, p1 );
}

// CHECK: define noundef double @
// CHECK: %dx.dot = fmul double %0, %1
// CHECK: ret double %dx.dot
double test_dot_double ( double p0, double p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call double @llvm.dx.dot.v2f64(<2 x double> %0, <2 x double> %1)
// CHECK: ret double %dx.dot
double test_dot_double2 ( double2 p0, double2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call double @llvm.dx.dot.v3f64(<3 x double> %0, <3 x double> %1)
// CHECK: ret double %dx.dot
double test_dot_double3 ( double3 p0, double3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call double @llvm.dx.dot.v4f64(<4 x double> %0, <4 x double> %1)
// CHECK: ret double %dx.dot
double test_dot_double4 ( double4 p0, double4 p1) {
  return dot ( p0, p1 );
}
