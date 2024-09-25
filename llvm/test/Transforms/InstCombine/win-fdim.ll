; RUN: opt %s -passes=instcombine -S -mtriple=i386-pc-mingw32 | FileCheck %s --check-prefixes=CHECK,MINGW32

define double @fdim_double() {
; CHECK-LABEL: define double @fdim_double() {
; MINGW32:    ret double 2.500000e+00
;
  %dim = call double @fdim(double 10.5, double 8.0)
  ret double %dim
}

define float @fdim_float() {
; CHECK-LABEL: define float @fdim_float() {
; MINGW32:    ret float 0.000000e+00
;
  %dim = call float @fdimf(float 1.500000e+00, float 8.0)
  ret float %dim
}

declare double @fdim(double, double)
declare float @fdimf(float, float)
