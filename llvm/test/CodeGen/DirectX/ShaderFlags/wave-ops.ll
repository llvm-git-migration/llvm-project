; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
;
; CHECK: ; Shader Flags Value: 0x00080000
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Wave level operations
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: {{^;$}}

target triple = "dxil-pc-shadermodel6.7-library"

define noundef half @wave_rla_half(half noundef %expr, i32 noundef %idx) {
entry:
  %ret = call half @llvm.dx.wave.readlane.f16(half %expr, i32 %idx)
  ret half %ret
}
