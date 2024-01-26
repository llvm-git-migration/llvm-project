; RUN: opt --bpf-check-and-opt-ir -S -mtriple=bpf-pc-linux < %s | FileCheck %s

; Generated from the following C code:
;
;   extern int __uptr *magic1();
;   extern int __uptr *magic2();
;
;   void test(long i) {
;     int __uptr *a;
;
;     if (i > 42)
;       a = magic1();
;     else
;       a = magic2();
;     a[5] = 7;
;   }
;
; Using the following command:
;
;   clang --target=bpf -O2 -S -emit-llvm -o t.ll t.c

; Function Attrs: nounwind
define dso_local void @test(i64 noundef %i) local_unnamed_addr #0 {
; CHECK-NOT:   @llvm.bpf.arena.cast
; CHECK:       if.end:
; CHECK-NEXT:    [[A_0:%.*]] = phi
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr addrspace(1) @llvm.bpf.arena.cast.p1.p1(ptr addrspace(1) [[A_0]], i32 1)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr addrspace(1) [[TMP0]], i64 5
; CHECK-NEXT:    store i32 7, ptr addrspace(1) [[TMP1]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp sgt i64 %i, 42
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = tail call ptr addrspace(1) @magic1() #2
  br label %if.end

if.else:                                          ; preds = %entry
  %call1 = tail call ptr addrspace(1) @magic2() #2
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %a.0 = phi ptr addrspace(1) [ %call, %if.then ], [ %call1, %if.else ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a.0, i64 5
  store i32 7, ptr addrspace(1) %arrayidx, align 4, !tbaa !3
  ret void
}

declare dso_local ptr addrspace(1) @magic1(...) local_unnamed_addr #1

declare dso_local ptr addrspace(1) @magic2(...) local_unnamed_addr #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"some clang version"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
