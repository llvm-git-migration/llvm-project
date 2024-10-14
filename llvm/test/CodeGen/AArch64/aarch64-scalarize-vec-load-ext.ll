; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; FIXME: Currently, we avoid narrowing this v4i32 load, in the
; hopes of being able to fold the shift, despite it requiring stack
; storage + loads. Ideally, we should narrow here and load the i32
; directly from the variable offset e.g:
;
; add     x8, x0, x1, lsl #4
; and     x9, x2, #0x3
; ldr     w0, [x8, x9, lsl #2]
;
; The AArch64TargetLowering::shouldReduceLoadWidth heuristic should
; probably be updated to choose load-narrowing instead of folding the
; lsl in larger vector cases.
;
; CHECK-LABEL: narrow_load_v4_i32_single_ele_variable_idx:
; CHECK: sub  sp, sp, #16
; CHECK: ldr  q[[REG0:[0-9]+]], [x0, x1, lsl #4]
; CHECK: bfi  x[[REG1:[0-9]+]], x2, #2, #2
; CHECK: str  q[[REG0]], [sp]
; CHECK: ldr  w0, [x[[REG1]]]
; CHECK: add  sp, sp, #16
define i32 @narrow_load_v4_i32_single_ele_variable_idx(ptr %ptr, i64 %off, i32 %ele) {
entry:
  %idx = getelementptr inbounds <4 x i32>, ptr %ptr, i64 %off
  %x = load <4 x i32>, ptr %idx, align 8
  %res = extractelement <4 x i32> %x, i32 %ele
  ret i32 %res
}
