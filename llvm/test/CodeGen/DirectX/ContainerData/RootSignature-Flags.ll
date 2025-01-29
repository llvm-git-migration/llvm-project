; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.rts0 = private constant [12 x i8]  c"{{.*}}", section "RTS0", align 4


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !4, !5 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
!5 = !{ !"RootConstants", i32 0, i32 1, i32 2, i32 3 }


; DXC:    - Name: RTS0
; DXC-NEXT: Size: 12
; DXC-NEXT: RootSignature:
; DXC-NEXT:   Size: 8
; DXC-NEXT:   Version: 1
; DXC-NEXT:   AllowInputAssemblerInputLayout: true
