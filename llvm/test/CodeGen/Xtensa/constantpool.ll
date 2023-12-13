; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Test placement of the i32,i64, float and double constants in constantpool

define dso_local i32 @const_i32() #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI0_0, 74565
; CHECK-LABEL: const_i32:
; CHECK: l32r a2, .LCPI0_0
  %1 = alloca i32, align 4
  store i32 74565, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

define dso_local i64 @const_int64() #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI1_0, 305419896
; CHECK-NEXT: .literal .LCPI1_1, -1859959449
; CHECK-LABEL: const_int64:
; CHECK: l32r a3, .LCPI1_0
; CHECK: l32r a2, .LCPI1_1
  %1 = alloca i64, align 8
  store i64 1311768467302729063, ptr %1, align 8
  %2 = load i64, ptr %1, align 8
  ret i64 %2
}

define dso_local float @const_fp() #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI2_0, 1092721050
; CHECK-LABEL: const_fp:
; CHECK: l32r a2, .LCPI2_0
  %1 = alloca float, align 4
  store float 0x4024333340000000, ptr %1, align 4
  %2 = load float, ptr %1, align 4
  ret float %2
}

define dso_local i64 @const_double() #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI3_0, 1070945621
; CHECK-NEXT: .literal .LCPI3_1, 1371607770
; CHECK-LABEL: const_double:
; CHECK: l32r a3, .LCPI3_0
; CHECK: l32r a2, .LCPI3_1
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  store double 0x3FD5555551C112DA, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  store double %3, ptr %1, align 8
  %4 = load i64, ptr %1, align 8
  ret i64 %4
}

; Test placement of the block address in constantpool

define dso_local i32 @const_blockaddress(i32 noundef %0) #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI4_0, .Ltmp0
; CHECK-LABEL: const_blockaddress:
; CHECK:       l32r a8, .LCPI4_0
; CHECK:       jx a8
; CHECK-NEXT:  .Ltmp0:
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 4
  store i32 %0, ptr %2, align 4
  store ptr blockaddress(@const_blockaddress, %5), ptr %3, align 4
  %4 = load ptr, ptr %3, align 4
  br label %7

5:                                                ; preds = %7
  %6 = load i32, ptr %2, align 4
  ret i32 %6

7:                                                ; preds = %1
  %8 = phi ptr [ %4, %1 ]
  indirectbr ptr %8, [label %5]
}
