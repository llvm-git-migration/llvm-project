; RUN: opt -S -scalarizer -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test that for vector values, WaveActiveSum scalarizes

define noundef <2 x half> @wave_active_sum_v2half(<2 x half> noundef %expr) {
entry:
; CHECK: call half @llvm.dx.wave.active.sum.f16(half %expr.i0)
; CHECK: call half @llvm.dx.wave.active.sum.f16(half %expr.i1)
  %ret = call <2 x half> @llvm.dx.wave.active.sum.f16(<2 x half> %expr)
  ret <2 x half> %ret
}

define noundef <3 x i32> @wave_active_sum_v3i32(<3 x i32> noundef %expr) {
entry:
; CHECK: call i32 @llvm.dx.wave.active.sum.i32(i32 %expr.i0)
; CHECK: call i32 @llvm.dx.wave.active.sum.i32(i32 %expr.i1)
; CHECK: call i32 @llvm.dx.wave.active.sum.i32(i32 %expr.i2)
  %ret = call <3 x i32> @llvm.dx.wave.active.sum(<3 x i32> %expr)
  ret <3 x i32> %ret
}

define noundef <4 x double> @wave_active_sum_v4f64(<4 x double> noundef %expr) {
entry:
; CHECK: call double @llvm.dx.wave.active.sum.f64(double %expr.i0)
; CHECK: call double @llvm.dx.wave.active.sum.f64(double %expr.i1)
; CHECK: call double @llvm.dx.wave.active.sum.f64(double %expr.i2)
; CHECK: call double @llvm.dx.wave.active.sum.f64(double %expr.i3)
  %ret = call <4 x double> @llvm.dx.wave.active.sum(<4 x double> %expr)
  ret <4 x double> %ret
}

declare <2 x half> @llvm.dx.wave.active.sum.v2f16(<2 x half>)
declare <3 x i32> @llvm.dx.wave.active.sum.v3i32(<3 x i32>)
declare <4 x double> @llvm.dx.wave.active.sum.v4f64(<4 x double>)
