; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s

; Make sure dxil operation function calls for exp are generated for float and half.



target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; CHECK:fmul <4 x float> <float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000>,  %{{.*}}
; CHECK:call <4 x float> @llvm.exp2.v4f32(<4 x float>  %{{.*}})
; Function Attrs: noinline nounwind optnone
define noundef <4 x float> @exp_float4(<4 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %elt.exp = call <4 x float> @llvm.exp.v4f32(<4 x float> %0)
  ret <4 x float> %elt.exp
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.exp.v4f32(<4 x float>) #1
