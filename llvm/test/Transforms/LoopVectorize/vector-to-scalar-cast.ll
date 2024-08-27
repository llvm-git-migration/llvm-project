; RUN: opt --force-widen-divrem-via-safe-divisor=false -passes=loop-vectorize -S --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s

; CHECK: LV: Not vectorizing: Found unvectorizable type   %s2 = bitcast <2 x i16> %vec1 to i32
; CHECK-NOT: Assertion

; Function Attrs: willreturn
define void @__start() #0 {
entry:
  %vec0 = insertelement <2 x i16> undef, i16 0, i64 0
  %vec1 = insertelement <2 x i16> %vec0, i16 0, i64 1
  br label %bb0

bb0:
  %s0 = phi i32 [ %s1, %bb1 ], [ 1, %entry ]
  br i1 0, label %bb2, label %bb1

bb1:
  %s1 = add nuw nsw i32 %s0, 1
  %exitcond = icmp ne i32 %s1, 11
  br i1 %exitcond, label %bb0, label %bb3

bb2:
  %s2 = bitcast <2 x i16> %vec1 to i32
  %s3 = srem i32 0, %s2
  br label %bb1

bb3:
  ret void
}

attributes #0 = { willreturn }

