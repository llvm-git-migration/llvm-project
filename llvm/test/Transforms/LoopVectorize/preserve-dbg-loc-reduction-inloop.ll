; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -S | FileCheck %s -check-prefix DEBUGLOC

; Testing the debug locations of the generated vector intstruction are same as
; their scalar counterpart.

; DEBUGLOC-LABEL: define i32 @reduction_sum(
define i32 @reduction_sum(ptr %A, ptr %B) {
; DEBUGLOC: vector.body:
; DEBUGLOC:   %[[VecLoad:.*]] = load <4 x i32>, ptr %2, align 4, !dbg ![[LoadLoc0:[0-9]+]]
; DEBUGLOC:   %[[VecRed:.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %wide.load), !dbg ![[RedLoc0:[0-9]+]]
; DEBUGLOC: loop:
; DEBUGLOC:   %l3 = load i32, ptr %l2, align 4, !dbg ![[LoadLoc0]]
; DEBUGLOC:   %l7 = add i32 %sum.02, %l3, !dbg ![[RedLoc0]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum.02 = phi i32 [ 0, %entry ], [ %l7, %loop ]
  %l2 = getelementptr inbounds i32, ptr %A, i64 %iv
  %l3 = load i32, ptr %l2, align 4
  %l7 = add i32 %sum.02, %l3
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 256
  br i1 %exitcond, label %exit, label %loop

exit:
  %sum.0.lcssa = phi i32 [ %l7, %loop ]
  ret i32 %sum.0.lcssa
}
