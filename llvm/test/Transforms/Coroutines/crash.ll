; RUN: opt < %s -passes='cgscc(function<>(simplifycfg<>),function-attrs,coro-split,coro-annotation-elide)'  -S | FileCheck %s

; Function Attrs: presplitcoroutine
define void @bar() #0 personality ptr null {
entry:
  %0 = call token @llvm.coro.save(ptr null)
  br label %entry.invoke.cont9_crit_edge

entry.invoke.cont9_crit_edge:                     ; preds = %entry
  call void @foo(ptr null, ptr null, ptr null, ptr null) #4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #3

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: presplitcoroutine
define void @foo(ptr %agg.result, ptr %this, ptr %queryRange, ptr %setName) #0 personality ptr null {
entry:
  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %1 = call ptr @llvm.coro.begin(token %0, ptr null)
  %2 = call token @llvm.coro.save(ptr null)
  %3 = call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}

attributes #0 = { presplitcoroutine }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind }
attributes #3 = { nomerge nounwind }
attributes #4 = { coro_elide_safe }

; CHECK: attributes
