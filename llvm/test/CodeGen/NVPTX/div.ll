; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

define float @div_full(float %a, float %b) {
  ; CHECK: div.full.f32 {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}
  %1 = call float @llvm.nvvm.div.full(float %a, float %b)
  ; CHECK: mov.f32 {{%f[0-9]+}}, 0f40400000
  ; CHECK: div.full.f32 {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}
  %2 = call float @llvm.nvvm.div.full(float %1, float 3.0)
  ; CHECK: div.full.ftz.f32 {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}
  %3 = call float @llvm.nvvm.div.full.ftz(float %2, float %b)
  ; CHECK: mov.f32 {{%f[0-9]+}}, 0f40800000
  ; CHECK: div.full.ftz.f32 {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}
  %4 = call float @llvm.nvvm.div.full.ftz(float %3, float 4.0)
  ret float %4
}
