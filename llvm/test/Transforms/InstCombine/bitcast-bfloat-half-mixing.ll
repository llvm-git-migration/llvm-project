; RUN: opt -S %s | FileCheck %s

define double @F0([2 x bfloat] %P0) {
entry:
  %P0.extract = extractvalue [2 x bfloat] %P0, 1
  %conv0 = bitcast bfloat %P0.extract to half
  %0 = fpext half %conv0 to double
  ret double %0
}

; CHECK: fpext half %conv0 to double
; CHECK-NOT: fpext bfloat %P0.extract to double

define double @F1([2 x half] %P1) {
entry:
  %P1.extract = extractvalue [2 x half] %P1, 1
  %conv1 = bitcast half %P1.extract to bfloat
  %0 = fpext bfloat %conv1 to double
  ret double %0
}

; CHECK: fpext bfloat %conv1 to double
; CHECK-NOT: fpext bfloat %P1.extract to double

define i32 @F2([2 x bfloat] %P2) {
entry:
  %P2.extract = extractvalue [2 x bfloat] %P2, 1
  %conv2 = bitcast bfloat %P2.extract to half
  %0 = fptoui half %conv2 to i32
  ret i32 %0
}

; CHECK: fptoui half %conv2 to i32
; CHECK-NOT: fptoui bfloat %P2.extract to i32

define i32 @F3([2 x half] %P3) {
entry:
  %P3.extract = extractvalue [2 x half] %P3, 1
  %conv3 = bitcast half %P3.extract to bfloat
  %0 = fptoui bfloat %conv3 to i32
  ret i32 %0
}

; CHECK: fptoui bfloat %conv3 to i32
; CHECK-NOT: fptoui half %P3.extract to i32


define i32 @F4([2 x bfloat] %P4) {
entry:
  %P4.extract = extractvalue [2 x bfloat] %P4, 1
  %conv4 = bitcast bfloat %P4.extract to half
  %0 = fptosi half %conv4 to i32
  ret i32 %0
}

; CHECK: fptosi half %conv4 to i32
; CHECK-NOT: fptosi bfloat %P4.extract to i32

define i32 @F5([2 x half] %P5) {
entry:
  %P5.extract = extractvalue [2 x half] %P5, 1
  %conv5 = bitcast half %P5.extract to bfloat
  %0 = fptosi bfloat %conv5 to i32
  ret i32 %0
}

; CHECK: fptosi bfloat %conv5 to i32
; CHECK-NOT: fptosi half %P5.extract to i32


