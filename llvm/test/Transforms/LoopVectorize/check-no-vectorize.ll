; This test asserts that we don't emit both
; successful and unsuccessful message about vectorization.

; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug -disable-output < %s 2>&1 | FileCheck %s
; CHECK-NOT: LV: We can vectorize this loop
; CHECK: LV: Not vectorizing: Cannot prove legality
; CHECK-NOT: LV: We can vectorize this loop

@a = global [32000 x i32] zeroinitializer, align 4
@b = global [32000 x i32] zeroinitializer, align 4

define void @foo() {
entry:
  %load_a_gep = load i32, ptr getelementptr inbounds (i8, ptr @a, i64 4), align 4
  %val_a = load i32, ptr @a, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %0 = phi i32 [ %val_a, %entry ], [ %add6, %for.body ]
  %1 = phi i32 [ %load_a_gep, %entry ], [ %2, %for.body ]
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [32000 x i32], ptr @a, i64 0, i64 %indvars.iv
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds [32000 x i32], ptr @a, i64 0, i64 %indvars.iv.next
  %2 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %2, %1
  %add6 = add nsw i32 %add3, %0
  store i32 %add6, ptr %arrayidx, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 31999
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
