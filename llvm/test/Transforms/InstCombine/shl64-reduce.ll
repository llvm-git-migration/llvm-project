; RUN: opt < %s -passes=instcombine -S | FileCheck %s


target triple = "amdgcn-amd-amdhsa"

define i64 @func_range(i64 noundef %arg0, ptr %arg1.ptr) {
  %shift.amt = load i64, ptr %arg1.ptr, !range !0
  %shl = shl i64 %arg0, %shift.amt
  ret i64 %shl

; CHECK:  define i64 @func_range(i64 noundef %arg0, ptr %arg1.ptr) {
; CHECK:  %shift.amt = load i64, ptr %arg1.ptr, align 8, !range !0
; CHECK:  %1 = trunc i64 %arg0 to i32
; CHECK:  %2 = trunc nuw nsw i64 %shift.amt to i32
; CHECK:  %3 = add nsw i32 %2, -32
; CHECK:  %4 = shl i32 %1, %3
; CHECK:  %5 = insertelement <2 x i32> <i32 0, i32 poison>, i32 %4, i64 1
; CHECK:  %shl = bitcast <2 x i32> %5 to i64
; CHECK:  ret i64 %shl

}
!0 = !{i64 32, i64 64}

define i64 @func_max(i64 noundef %arg0, i64 noundef %arg1) {
  %max = call i64 @llvm.umax.i64(i64 %arg1, i64 32)
  %min = call i64 @llvm.umin.i64(i64 %max,  i64 63)  
  %shl = shl i64 %arg0, %min
  ret i64 %shl
}
  

