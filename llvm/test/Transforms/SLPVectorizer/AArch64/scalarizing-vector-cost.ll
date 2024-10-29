; RUN: opt -mtriple=aarch64 -passes=slp-vectorizer -debug-only=SLP -S -disable-output < %s 2>&1 | FileCheck %s

define <4 x i8> @v4i8(<4 x i8> %a, <4 x i8> %b)
{
; CHECK: SLP: Found cost = 18 for VF=4
  %a0 = extractelement <4 x i8> %a, i64 0
  %a1 = extractelement <4 x i8> %a, i64 1
  %a2 = extractelement <4 x i8> %a, i64 2
  %a3 = extractelement <4 x i8> %a, i64 3
  %b0 = extractelement <4 x i8> %b, i64 0
  %b1 = extractelement <4 x i8> %b, i64 1
  %b2 = extractelement <4 x i8> %b, i64 2
  %b3 = extractelement <4 x i8> %b, i64 3
  %1 = sdiv i8 %a0, undef
  %2 = sdiv i8 %a1, 1
  %3 = sdiv i8 %a2, 2
  %4 = sdiv i8 %a3, 4
  %r0 = insertelement <4 x i8> poison, i8 %1, i32 0
  %r1 = insertelement <4 x i8> %r0, i8 %2, i32 1
  %r2 = insertelement <4 x i8> %r1, i8 %3, i32 2
  %r3 = insertelement <4 x i8> %r2, i8 %4, i32 3
  ret <4 x i8> %r3
}
