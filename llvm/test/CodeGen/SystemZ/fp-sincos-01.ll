; Test that combined sin/cos library call is emitted when appropriate

; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s --check-prefix=CHECK-OPT
; RUN: llc < %s -mtriple=s390x-linux-gnu -enable-unsafe-fp-math | FileCheck %s --check-prefix=CHECK-OPT

; Test f16 libcalls.
define half @f0(half %x) {
; CHECK-OPT-LABEL: f0:
; CHECK-OPT-NOT: brasl %r14, __extendhfsf2@PLT
; CHECK-OPT: brasl %r14, sinh@PLT
; CHECK-OPT: brasl %r14, cosh@PLT
; CHECK-OPT: brasl %r14, __extendhfsf2@PLT
; CHECK-OPT: brasl %r14, __extendhfsf2@PLT
; CHECK-OPT: aebr %f0, %f8
; CHECK-OPT: brasl %r14, __truncsfhf2@PLT
  %tmp1 = call half @sinh(half %x) readnone
  %tmp2 = call half @cosh(half %x) readnone
  %add = fadd half %tmp1, %tmp2
  ret half %add
}

define float @f1(float %x) {
; CHECK-OPT-LABEL: f1:
; CHECK-OPT: brasl %r14, sincosf@PLT
; CHECK-OPT: le %f0, 164(%r15)
; CHECK-OPT: aeb %f0, 160(%r15)
  %tmp1 = call float @sinf(float %x) readnone
  %tmp2 = call float @cosf(float %x) readnone
  %add = fadd float %tmp1, %tmp2
  ret float %add
}

define float @f1_errno(float %x) {
; CHECK-OPT-LABEL: f1_errno:
; CHECK-OPT: brasl %r14, sinf@PLT
; CHECK-OPT: ler %f9, %f0
; CHECK-OPT: brasl %r14, cosf@PLT
; CHECK-OPT: aebr %f0, %f9
  %tmp1 = call float @sinf(float %x)
  %tmp2 = call float @cosf(float %x)
  %add = fadd float %tmp1, %tmp2
  ret float %add
}

define double @f2(double %x) {
; CHECK-OPT-LABEL: f2:
; CHECK-OPT: brasl %r14, sincos@PLT
; CHECK-OPT: ld %f0, 168(%r15)
; CHECK-OPT: adb %f0, 160(%r15)
  %tmp1 = call double @sin(double %x) readnone
  %tmp2 = call double @cos(double %x) readnone
  %add = fadd double %tmp1, %tmp2
  ret double %add
}

define double @f2_errno(double %x) {
; CHECK-OPT-LABEL: f2_errno:
; CHECK-OPT: brasl %r14, sin@PLT
; CHECK-OPT: ldr %f9, %f0
; CHECK-OPT: brasl %r14, cos@PLT
; CHECK-OPT: adbr %f0, %f9
  %tmp1 = call double @sin(double %x)
  %tmp2 = call double @cos(double %x)
  %add = fadd double %tmp1, %tmp2
  ret double %add
}

define fp128 @f3(fp128 %x) {
; CHECK-OPT-LABEL: f3:
; CHECK-OPT: brasl %r14, sincosl@PLT
; CHECK-OPT: axbr
  %tmp1 = call fp128 @sinl(fp128 %x) readnone
  %tmp2 = call fp128 @cosl(fp128 %x) readnone
  %add = fadd fp128 %tmp1, %tmp2
  ret fp128 %add
}

define fp128 @f3_errno(fp128 %x) {
; CHECK-OPT-LABEL: f3_errno:
; CHECK-OPT: brasl %r14, sinl@PLT
; CHECK-OPT: brasl %r14, cosl@PLT
; CHECK-OPT: axbr
  %tmp1 = call fp128 @sinl(fp128 %x)
  %tmp2 = call fp128 @cosl(fp128 %x)
  %add = fadd fp128 %tmp1, %tmp2
  ret fp128 %add
}

declare half @sinh(half)
declare float @sinf(float)
declare double @sin(double)
declare fp128 @sinl(fp128)
declare half @cosh(half)
declare float @cosf(float)
declare double @cos(double)
declare fp128 @cosl(fp128)

