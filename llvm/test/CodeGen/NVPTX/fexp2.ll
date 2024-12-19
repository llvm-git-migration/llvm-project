; RUN: llc < %s -march=nvptx64 -mcpu=sm_52 -mattr=+ptx86 | FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-12.6 %{ llc < %s -march=nvptx64 -mcpu=sm_52 -mattr=+ptx86 | %ptxas-verify -arch=sm_52 %}
source_filename = "fexp2.ll"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: exp2_test
define ptx_kernel void @exp2_test(ptr %a, ptr %res) local_unnamed_addr {
entry:
  %in = load float, ptr %a, align 4
  ; CHECK: ex2.approx.f32 [[D1:%f[0-9]+]], [[S1:%f[0-9]+]]
  %exp2 = call float @llvm.exp2.f32(float %in)
  ; CHECK: st.global.f32 {{.*}}, [[D1]]
  store float %exp2, ptr %res, align 4
  ret void
}

; CHECK-LABEL: exp2_ftz_test
define ptx_kernel void @exp2_ftz_test(ptr %a, ptr %res) local_unnamed_addr #0 {
entry:
  %in = load float, ptr %a, align 4
  ; CHECK: ex2.approx.ftz.f32 [[D1:%f[0-9]+]], [[S1:%f[0-9]+]]
  %exp2 = call float @llvm.exp2.f32(float %in)
  ; CHECK: st.global.f32 {{.*}}, [[D1]]
  store float %exp2, ptr %res, align 4
  ret void
}

; CHECK-LABEL: exp2_test_v
define ptx_kernel void @exp2_test_v(ptr %a, ptr %res) local_unnamed_addr {
entry:
  %in = load <4 x float>, ptr %a, align 16
  ; CHECK: ex2.approx.f32 [[D1:%f[0-9]+]], [[S1:%f[0-9]+]]
  ; CHECK: ex2.approx.f32 [[D2:%f[0-9]+]], [[S2:%f[0-9]+]]
  ; CHECK: ex2.approx.f32 [[D3:%f[0-9]+]], [[S3:%f[0-9]+]]
  ; CHECK: ex2.approx.f32 [[D4:%f[0-9]+]], [[S4:%f[0-9]+]]
  %exp2 = call <4 x float> @llvm.exp2.v4f32(<4 x float> %in)
  ; CHECK: st.global.v4.f32 {{.*}}, {{[{]}}[[D4]], [[D3]], [[D2]], [[D1]]{{[}]}}
  store <4 x float> %exp2, ptr %res, align 16
  ret void
}

declare float @llvm.exp2.f32(float %val)

declare <4 x float> @llvm.exp2.v4f32(<4 x float> %val)

attributes #0 = {"denormal-fp-math"="preserve-sign"}
