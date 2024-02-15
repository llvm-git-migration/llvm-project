; RUN: opt -S -passes=loop-rotate < %s | FileCheck %s

declare void @bar1()

@threadlocalint = thread_local global i32 0, align 4

define void @foo() #0 {
; CHECK-LABEL: entry:
; CHECK: call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @threadlocalint)
; CHECK: br {{.*}} label %cond.end
entry:
  br label %while.cond

while.cond:
  %1 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @threadlocalint)
  %2 = load i32, ptr %1, align 4
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:
  call void @bar1()
  unreachable

; The address of threadlocalint must not be cached outside loops in presplit
; coroutines.
; CHECK-LABEL: cond.end:
; CHECK: call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @threadlocalint)
; CHECK: br {{.*}} label %cond.end
cond.end:
  call void @bar1()
  br label %while.cond
}

attributes #0 = { presplitcoroutine }
