; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

;; Check if fdiv -1, X lowers to fneg (rcp.rn X).

; CHECK-LABEL: .func{{.*}}test1
define double @test1(double %in) {
; CHECK: rcp.rn.f64 [[RCP:%.*]], [[X:%.*]];
; CHECK-NEXT: neg.f64 [[FNEG:%.*]], [[RCP]];
  %div = fdiv double 1.000000e+00, %in
  %neg = fsub double -0.000000e+00, %div
  ret double %neg
}
