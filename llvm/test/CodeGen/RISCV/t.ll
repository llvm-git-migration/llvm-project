; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(ptr noundef %x) #0 {
entry:
  %x.addr = alloca ptr, align 8
  %y = alloca i16, align 2
  store ptr %x, ptr %x.addr, align 8
  %0 = load ptr, ptr %x.addr, align 8
  %1 = load i16, ptr %0, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %y, align 2
  %conv = sext i16 %2 to i32
  %cmp = icmp sgt i32 %conv, 15
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load i16, ptr %y, align 2
  %conv2 = sext i16 %3 to i32
  switch i32 %conv2, label %sw.epilog [
    i32 16, label %sw.bb
    i32 17, label %sw.bb3
    i32 18, label %sw.bb4
  ]

sw.bb:                                            ; preds = %if.then
  call void (...) @bar()
  br label %sw.epilog

sw.bb3:                                           ; preds = %if.then
  call void (...) @foo1()
  br label %sw.bb4

sw.bb4:                                           ; preds = %if.then, %sw.bb3
  call void (...) @foo2()
  br label %sw.epilog

sw.epilog:                                        ; preds = %if.then, %sw.bb4, %sw.bb
  br label %if.end

if.end:                                           ; preds = %sw.epilog, %entry
  call void (...) @bat()
  ret void
}

declare void @bar(...) #1

declare void @foo1(...) #1

declare void @foo2(...) #1

declare void @bat(...) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 16.0.6 (git@github.com:karouzakisp/llvm-project.git 41a95c5c390d1c2854d15f0ee9cd7ca37c4b6dcf)"}
