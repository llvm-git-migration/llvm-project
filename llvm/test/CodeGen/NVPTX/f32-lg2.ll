; RUN: llc < %s -mcpu=sm_20 -mattr=+ptx32 | FileCheck --check-prefixes=CHECK %s
target triple = "nvptx-nvidia-cuda"

declare float @llvm.nvvm.lg2.approx.f32(float)
declare <2 x float> @llvm.nvvm.lg2.approx.v2f32(<2 x float>)

; CHECK-LABEL: log2_float
define float @log2_float(float %0) {
  %res = call float @llvm.nvvm.lg2.approx.f32(float %0)
  ret float %res
}

; CHECK-LABEL: log2_2xfloat
define <2 x float> @log2_2xfloat(<2 x float> %0) {
  %res = call <2 x float> @llvm.nvvm.lg2.approx.v2f32(<2 x float> %0)
  ret <2 x float> %res
}
