; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

@x_i8 = common dso_local global i8 0, align 8
@y_i8 = common dso_local global i8 0, align 8
@x_i16 = common dso_local global i16 0, align 8
@y_i16 = common dso_local global i16 0, align 8
@x_i32 = common dso_local global i32 0, align 8
@y_i32 = common dso_local global i32 0, align 8

define void @test() {
; CHECK: .literal_position
; CHECK-NEXT:  .literal .LCPI0_0, x_i8
; CHECK-NEXT:  .literal .LCPI0_1, y_i8
; CHECK-NEXT:  .literal .LCPI0_2, x_i16
; CHECK-NEXT:  .literal .LCPI0_3, y_i16
; CHECK-NEXT:  .literal .LCPI0_4, x_i32
; CHECK-NEXT:  .literal .LCPI0_5, y_i32
; CHECK-LABEL: test:
; CHECK:  # %bb.0:
; CHECK-NEXT:  l32r a8, .LCPI0_0
; CHECK-NEXT:  memw
; CHECK-NEXT:  l8ui a8, a8, 0
; CHECK-NEXT:  l32r a9, .LCPI0_1
; CHECK-NEXT:  memw
; CHECK-NEXT:  s8i a8, a9, 0
; CHECK-NEXT:  l32r a8, .LCPI0_2
; CHECK-NEXT:  memw
; CHECK-NEXT:  l16ui a8, a8, 0
; CHECK-NEXT:  l32r a9, .LCPI0_3
; CHECK-NEXT:  memw
; CHECK-NEXT:  s16i a8, a9, 0
; CHECK-NEXT:  l32r a8, .LCPI0_4
; CHECK-NEXT:  memw
; CHECK-NEXT:  l32i a8, a8, 0
; CHECK-NEXT:  l32r a9, .LCPI0_5
; CHECK-NEXT:  memw
; CHECK-NEXT:  s32i a8, a9, 0
; CHECK-NEXT:  ret

entry:
  %0 = load volatile i8, ptr @x_i8, align 4
  store volatile i8 %0, ptr @y_i8, align 4
  %1 = load volatile i16, ptr @x_i16, align 4
  store volatile i16 %1, ptr @y_i16, align 4
  %2 = load volatile i32, ptr @x_i32, align 4
  store volatile i32 %2, ptr @y_i32, align 4
  ret void
}
