// DirectX target:
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=DX-CHECK,DX-NATIVE_HALF

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=DX-CHECK,DX-NO_HALF 



// Spirv target:
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=SPV-CHECK,SPV-NATIVE_HALF

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=SPV-CHECK,SPV-NO_HALF



// DX-NATIVE_HALF: define noundef half @
// DX-NATIVE_HALF: %dx.fmod = call half @llvm.dx.fmod.f16(
// DX-NATIVE_HALF: ret half %dx.fmod
// DX-NO_HALF: define noundef float @
// DX-NO_HALF: %dx.fmod = call float @llvm.dx.fmod.f32(
// DX-NO_HALF: ret float %dx.fmod
//
// SPV-NATIVE_HALF: define spir_func noundef half @
// SPV-NATIVE_HALF: %fmod = frem half
// SPV-NATIVE_HALF: ret half %fmod
// SPV-NO_HALF: define spir_func noundef float @
// SPV-NO_HALF: %fmod = frem float
// SPV-NO_HALF: ret float %fmod
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// DX-NATIVE_HALF: define noundef <2 x half> @
// DX-NATIVE_HALF: %dx.fmod = call <2 x half> @llvm.dx.fmod.v2f16
// DX-NATIVE_HALF: ret <2 x half> %dx.fmod
// DX-NO_HALF: define noundef <2 x float> @
// DX-NO_HALF: %dx.fmod = call <2 x float> @llvm.dx.fmod.v2f32(
// DX-NO_HALF: ret <2 x float> %dx.fmod
//
// SPV-NATIVE_HALF: define spir_func noundef <2 x half> @
// SPV-NATIVE_HALF: %fmod = frem <2 x half>
// SPV-NATIVE_HALF: ret <2 x half> %fmod
// SPV-NO_HALF: define spir_func noundef <2 x float> @
// SPV-NO_HALF: %fmod = frem <2 x float>
// SPV-NO_HALF: ret <2 x float> %fmod
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// DX-NATIVE_HALF: define noundef <3 x half> @
// DX-NATIVE_HALF: %dx.fmod = call <3 x half> @llvm.dx.fmod.v3f16
// DX-NATIVE_HALF: ret <3 x half> %dx.fmod
// DX-NO_HALF: define noundef <3 x float> @
// DX-NO_HALF: %dx.fmod = call <3 x float> @llvm.dx.fmod.v3f32(
// DX-NO_HALF: ret <3 x float> %dx.fmod
//
// SPV-NATIVE_HALF: define spir_func noundef <3 x half> @
// SPV-NATIVE_HALF: %fmod = frem <3 x half>
// SPV-NATIVE_HALF: ret <3 x half> %fmod
// SPV-NO_HALF: define spir_func noundef <3 x float> @
// SPV-NO_HALF: %fmod = frem <3 x float>
// SPV-NO_HALF: ret <3 x float> %fmod
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// DX-NATIVE_HALF: define noundef <4 x half> @
// DX-NATIVE_HALF: %dx.fmod = call <4 x half> @llvm.dx.fmod.v4f16
// DX-NATIVE_HALF: ret <4 x half> %dx.fmod
// DX-NO_HALF: define noundef <4 x float> @
// DX-NO_HALF: %dx.fmod = call <4 x float> @llvm.dx.fmod.v4f32(
// DX-NO_HALF: ret <4 x float> %dx.fmod
//
// SPV-NATIVE_HALF: define spir_func noundef <4 x half> @
// SPV-NATIVE_HALF: %fmod = frem <4 x half>
// SPV-NATIVE_HALF: ret <4 x half> %fmod
// SPV-NO_HALF: define spir_func noundef <4 x float> @
// SPV-NO_HALF: %fmod = frem <4 x float>
// SPV-NO_HALF: ret <4 x float> %fmod
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// DX-CHECK: define noundef float @
// DX-CHECK: %dx.fmod = call float @llvm.dx.fmod.f32(
// DX-CHECK: ret float %dx.fmod
//
// SPV-CHECK: define spir_func noundef float @
// SPV-CHECK: %fmod = frem float
// SPV-CHECK: ret float %fmod
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// DX-CHECK: define noundef <2 x float> @
// DX-CHECK: %dx.fmod = call <2 x float> @llvm.dx.fmod.v2f32
// DX-CHECK: ret <2 x float> %dx.fmod
//
// SPV-CHECK: define spir_func noundef <2 x float> @
// SPV-CHECK: %fmod = frem <2 x float>
// SPV-CHECK: ret <2 x float> %fmod
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// DX-CHECK: define noundef <3 x float> @
// DX-CHECK: %dx.fmod = call <3 x float> @llvm.dx.fmod.v3f32
// DX-CHECK: ret <3 x float> %dx.fmod
//
// SPV-CHECK: define spir_func noundef <3 x float> @
// SPV-CHECK: %fmod = frem <3 x float>
// SPV-CHECK: ret <3 x float> %fmod
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// DX-CHECK: define noundef <4 x float> @
// DX-CHECK: %dx.fmod = call <4 x float> @llvm.dx.fmod.v4f32
// DX-CHECK: ret <4 x float> %dx.fmod
//
// SPV-CHECK: define spir_func noundef <4 x float> @
// SPV-CHECK: %fmod = frem <4 x float>
// SPV-CHECK: ret <4 x float> %fmod
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

