; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

define float @div_full(float %a, float %b) {
  ; CHECK: div.full.f32 {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}
  %1 = call float @llvm.nvvm.div.full(float %a, float %b)
  ; CHECK: div.full.ftz.f32 {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}
  %2 = call float @llvm.nvvm.div.full.ftz(float %1, float %b)
  ret float %2
}