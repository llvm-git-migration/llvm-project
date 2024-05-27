; RUN: opt < %s -passes="loop-vectorize"
; ModuleID = 'reduced.ll'
source_filename = "reduced.ll"
;target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none)
define noalias noundef ptr @foo(ptr readonly %__first, ptr writeonly %__last) local_unnamed_addr #0 {
entry:
  %cmp.not1 = icmp eq ptr %__first, %__last
  br i1 %cmp.not1, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %__first.addr.02 = phi ptr [ %incdec.ptr, %for.body ], [ %__first, %for.body.preheader ]
  %0 = load ptr, ptr %__first.addr.02, align 8
  store ptr %0, ptr %__last, align 8
  %incdec.ptr = getelementptr inbounds i8, ptr %__first.addr.02, i64 16
  %cmp.not = icmp eq ptr %incdec.ptr, %__last
  br i1 %cmp.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) "target-cpu"="znver4" }
