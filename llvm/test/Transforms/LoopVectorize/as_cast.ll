; RUN: opt -passes=loop-vectorize %s
; ModuleID = '<bc file>'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr addrspace(1) %in) {
entry:
  br label %for.body.i.epil

for.body.i.epil:                                  ; preds = %for.body.i.epil, %entry
  %epil.iter = phi i64 [ %epil.iter.next, %for.body.i.epil ], [ 0, %entry ]
  %arrayidx.ascast.i.epil = addrspacecast ptr addrspace(1) %in to ptr
  %epil.iter.next = add i64 %epil.iter, 1
  %arrayidx = getelementptr inbounds i64, ptr %arrayidx.ascast.i.epil, i64 %epil.iter.next
  store i64 %epil.iter.next, ptr %arrayidx, align 4
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, 7
  br i1 %epil.iter.cmp.not, label %loop.exit, label %for.body.i.epil

loop.exit: ; preds = %for.body.i.epil
  ret void
}
