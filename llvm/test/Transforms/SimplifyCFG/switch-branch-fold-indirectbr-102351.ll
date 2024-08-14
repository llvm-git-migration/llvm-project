; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; CHECK: switch i32 %state.0, label %sw.epilog
; CHECK-NEXT: i32 0, label %sw.bb
; CHECK-NEXT: i32 1, label %VM__OP_1
; CHECK-NEXT: i32 2, label %sw.bb4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = private unnamed_addr constant [23 x i8] c"OP_1:(instruction=%d)\0A\00", align 1
@global.1 = private unnamed_addr constant [28 x i8] c"TERMINATE:(instruction=%d)\0A\00", align 1

define dso_local noundef i32 @main() {
bb:
  %alloca = alloca [2 x ptr], align 16
  store ptr blockaddress(@main, %bb4), ptr %alloca, align 16, !tbaa !0
  %getelementptr = getelementptr inbounds [2 x ptr], ptr %alloca, i64 0, i64 1
  store ptr blockaddress(@main, %bb10), ptr %getelementptr, align 8, !tbaa !0
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %phi = phi i32 [ 0, %bb ], [ %phi8, %bb7 ]
  %phi2 = phi i32 [ 0, %bb ], [ %phi9, %bb7 ]
  switch i32 %phi, label %bb7 [
    i32 0, label %bb3
    i32 1, label %bb4
    i32 2, label %bb6
  ]

bb3:                                              ; preds = %bb1
  br label %bb12

bb4:                                              ; preds = %bb12, %bb1
  %phi5 = phi i32 [ %phi13, %bb12 ], [ %phi2, %bb1 ]
  br label %bb7

bb6:                                              ; preds = %bb1
  %call = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @global, i32 noundef %phi2)
  %add = add nsw i32 %phi2, 1
  br label %bb12

bb7:                                              ; preds = %bb4, %bb1
  %phi8 = phi i32 [ %phi, %bb1 ], [ 2, %bb4 ]
  %phi9 = phi i32 [ %phi2, %bb1 ], [ %phi5, %bb4 ]
  br label %bb1, !llvm.loop !4

bb10:                                             ; preds = %bb12
  %call11 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @global.1, i32 noundef %phi13)
  ret i32 0

bb12:                                             ; preds = %bb6, %bb3
  %phi13 = phi i32 [ %add, %bb6 ], [ %phi2, %bb3 ]
  %sext = sext i32 %phi13 to i64
  %getelementptr14 = getelementptr inbounds [2 x ptr], ptr %alloca, i64 0, i64 %sext
  %load = load ptr, ptr %getelementptr14, align 8, !tbaa !0
  indirectbr ptr %load, [label %bb4, label %bb10]
}

declare i32 @printf(ptr noundef, ...)

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
