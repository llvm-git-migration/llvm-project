; RUN: llc -O1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; REQUIRES: powerpc-registered-target

; Test that a negative parameter smaller than 64 bits (e.g., int)
; is correctly implemented with sign-extension when passed to
; a floating point libcall.

define double @ldexp_test(ptr %a, ptr %b) {
; CHECK-LABEL: ldexp_test: # @ldexp_test
; CHECK: # %bb.0:
; CHECK: lfd 1, 0(3)
; CHECK-NEXT: lwa 4, 0(4)
; CHECK-NEXT: bl ldexp
; CHECK-NEXT: nop
	%base = load double, ptr %a
	%exp = load i32, ptr %b
	%call = call double @llvm.ldexp.f64.i32(double %base, i32 signext %exp)
	ret double %call
}

define i64 @frexp_test(ptr %a) {
; CHECK-LABEL: frexp_test: # @frexp_test
; CHECK: # %bb.0:
; CHECK: bl frexp
; CHECK-NEXT: nop
; CHECK-NEXT: lwa 3, 124(1)
; CHECK-NEXT: addi 1, 1, 128
; CHECK-NEXT: ld 0, 16(1)
; CHECK-NEXT: mtlr 0
; CHECK-NEXT: blr

	%input = load double, ptr %a
	%call = call { double, i32 } @llvm.frexp.f64.i32(double %input)
	%exp_result = extractvalue { double, i32 } %call, 1
	%exp_extended = sext i32 %exp_result to i64
	ret i64 %exp_extended
}
