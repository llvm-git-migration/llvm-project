; RUN: opt -S -passes='indvars' -verify-scev < %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; We should not crash on multiplicative inverse called within SCEV's binomial
; coefficient function.
define i32 @pr87798() {
; CHECK-LABEL: pr87798 
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i32 [ 0, %bb ], [ %add4, %bb1 ]
  %phi2 = phi i32 [ 0, %bb ], [ %add, %bb1 ]
  %phi3 = phi i32 [ 0, %bb ], [ %add5, %bb1 ]
  %add = add i32 %phi2, %phi3
  %mul = mul i32 %phi2, %phi3
  %add4 = add i32 %mul, %phi
  %and = and i32 %phi, 1
  %add5 = add i32 %phi3, 1
  br i1 true, label %preheader, label %bb1

preheader:                                              ; preds = %bb1
  %phi9 = phi i32 [ %and, %bb1 ]
  br label %loop

loop:                                              ; preds = %preheader, %loop
  br label %loop

bb7:                                              ; No predecessors!
  %zext = zext i32 %phi9 to i64
  ret i32 0
}
