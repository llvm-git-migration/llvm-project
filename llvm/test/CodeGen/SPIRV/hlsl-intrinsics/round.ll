; RUN: llc -O0 -mtriple=spirv-unknown-linux %s -o - | FileCheck %s

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @round_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Round %[[#]]
  %elt.round = call float @llvm.round.f32(float %a)
  ret float %elt.round
}

define noundef half @round_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Round %[[#]]
  %elt.round = call half @llvm.round.f16(half %a)
  ret half %elt.round
}

declare half @llvm.round.f16(half)
declare float @llvm.round.f32(float)
